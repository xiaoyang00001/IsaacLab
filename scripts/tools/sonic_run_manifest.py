#!/usr/bin/env python3
"""Create a reproducibility manifest for a SONIC closed-loop evaluation run.

The helper also verifies that the Isaac Lab packages resolve from the selected
worktree.  This catches a stale editable install before a long simulation run
silently imports code from another checkout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ENVIRONMENT_KEYS = {
    "CONDA_DEFAULT_ENV",
    "CONDA_ENV_PREFIX",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "DEPLOY_POLICY_DIR",
    "DISPLAY",
    "GR00T_WBC_ROOT",
    "ISAACLAB_PATH",
    "ISAACLAB_ROOT",
    "JITTER_GUI",
    "JITTER_INPUT",
    "JITTER_OUT_ROOT",
    "JITTER_POSE_PROTOCOL",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "DEPLOY_LIBDDSC_SHA256_EXPECTED",
    "DEPLOY_LIBDDSCXX_SHA256_EXPECTED",
    "DEPLOY_FASTRTPS_PROFILE_SHA256_EXPECTED",
    "PROXY_LIBDDSC_SHA256_EXPECTED",
    "PROXY_LIBDDSCXX_SHA256_EXPECTED",
    "PYTHONPATH",
    "SONY_REPO",
    "UNITREE_DDS_DOMAIN_ID",
    "UNITREE_DDS_INTERFACE",
    "XR_RUNTIME_JSON",
}
ENVIRONMENT_PREFIXES = (
    "DEPLOY_",
    "DDS_",
    "ISAACLAB_",
    "JITTER_",
    "POSE_",
    "SONIC_",
    "UNITREE_",
)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _run_git_optional(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.rstrip("\n") if result.returncode == 0 else ""


def git_info(repo_arg: str) -> dict[str, Any]:
    repo = Path(repo_arg).expanduser().resolve()
    root = Path(_run_git(repo, "rev-parse", "--show-toplevel")).resolve()
    # "normal" still proves dirty state while collapsing large generated build
    # trees to one entry instead of bloating every run manifest with thousands
    # of untracked object paths.
    status_text = _run_git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    diff_result = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tracked_diff = diff_result.stdout
    branch = _run_git_optional(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    return {
        "path": str(root),
        "realpath": str(root.resolve()),
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "branch": branch or None,
        "dirty": bool(status_text),
        "status_porcelain": status_text.splitlines() if status_text else [],
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path_arg: str | None) -> dict[str, Any] | None:
    if not path_arg:
        return None
    input_path = Path(path_arg).expanduser()
    realpath = input_path.resolve(strict=True)
    stat = realpath.stat()
    return {
        "path": str(input_path),
        "realpath": str(realpath),
        "is_symlink": input_path.is_symlink(),
        "symlink_target": os.readlink(input_path) if input_path.is_symlink() else None,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(realpath),
    }


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def import_paths(isaaclab_root_arg: str) -> dict[str, dict[str, Any]]:
    root = Path(isaaclab_root_arg).expanduser().resolve()
    source_root = root / "source"
    packages: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for extension_dir in sorted(source_root.iterdir()):
        package_dir = extension_dir / extension_dir.name
        if not package_dir.is_dir() or not (package_dir / "__init__.py").is_file():
            continue

        package_name = extension_dir.name
        spec = importlib.util.find_spec(package_name)
        origin = None if spec is None else spec.origin
        locations = [] if spec is None else [str(Path(p).resolve()) for p in (spec.submodule_search_locations or [])]
        resolved_origin = None
        if origin and origin not in {"built-in", "frozen"}:
            resolved_origin = str(Path(origin).resolve())

        expected_root = package_dir.resolve()
        valid = bool(
            spec
            and (
                (resolved_origin and _is_below(Path(resolved_origin), expected_root))
                or any(_is_below(Path(location), expected_root) for location in locations)
            )
        )
        packages[package_name] = {
            "origin": resolved_origin or origin,
            "search_locations": locations,
            "expected_root": str(expected_root),
            "from_selected_worktree": valid,
        }
        if not valid:
            errors.append(
                f"{package_name}: resolved to {resolved_origin or origin or '<not found>'}, "
                f"expected below {expected_root}"
            )

    if not packages:
        errors.append(f"no importable Isaac Lab source packages found below {source_root}")
    if errors:
        raise RuntimeError("Isaac Lab import path validation failed:\n  " + "\n  ".join(errors))
    return packages


def captured_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in ENVIRONMENT_KEYS or key.startswith(ENVIRONMENT_PREFIXES)
    }


def key_value_pairs(values: list[str] | None) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values or []:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise RuntimeError(f"expected KEY=VALUE, got {value!r}")
        pairs[key] = item
    return dict(sorted(pairs.items()))


def print_import_paths(root: str) -> None:
    packages = import_paths(root)
    print(f"[jitter-import] python={sys.executable}", flush=True)
    print(f"[jitter-import] PYTHONPATH={os.environ.get('PYTHONPATH', '')}", flush=True)
    for package_name, details in packages.items():
        print(f"[jitter-import] {package_name}={details['origin']}", flush=True)


def component_contracts(args: argparse.Namespace) -> dict[str, Any]:
    sony_repo = Path(args.sony_repo).expanduser().resolve()
    deploy_root = Path(args.deploy_root).expanduser().resolve()
    teleop_python = str(Path(args.teleop_python).expanduser())
    runtime_arch = platform.machine()
    proxy_dds_dir = str(
        sony_repo
        / "gear_sonic_deploy"
        / "thirdparty/unitree_sdk2/thirdparty/lib"
        / runtime_arch
    )
    proxy_sdk_dir = str(
        sony_repo
        / "gear_sonic_deploy"
        / "thirdparty/unitree_sdk2/lib"
        / runtime_arch
    )
    deploy_dds_dir = str(
        deploy_root / "thirdparty/unitree_sdk2/thirdparty/lib" / runtime_arch
    )
    deploy_sdk_dir = str(deploy_root / "thirdparty/unitree_sdk2/lib" / runtime_arch)

    input_enabled = args.input == "bvh"
    if args.pose_protocol == 3:
        input_source_args = [
            "--source",
            "sony_pico",
            "--bvh-stream-host",
            "0.0.0.0",
            "--bvh-stream-port",
            "12352",
            "--bvh-stream-bonedata-position-scale",
            "1.0",
            "--bvh-stream-bonedata-input-quat-order",
            "xyzw",
            "--sony-pico-bonedata-basis",
            "zflip",
            "--sony-pico-smpl-joints-source",
            "pico_fk",
            "--pose-encoder-mode",
            "smpl",
        ]
        input_source = "sony_pico"
        pose_encoder = "smpl"
    else:
        input_source_args = [
            "--source",
            "bvh_stream",
            "--bvh-stream-host",
            "0.0.0.0",
            "--bvh-stream-port",
            "12352",
            "--bvh-stream-bonedata-coordinate-frame",
            "left_handed_yup",
            "--bvh-stream-bonedata-position-scale",
            "1.0",
            "--bvh-stream-bonedata-input-quat-order",
            "xyzw",
            "--bvh-stream-bonedata-rotation-mode",
            "input",
            "--pose-encoder-mode",
            "g1",
        ]
        input_source = "bvh_stream"
        pose_encoder = "g1"

    input_argv = [
        teleop_python,
        "-u",
        str(Path(args.mocap_manager).expanduser()),
        *input_source_args,
        "--control-mode",
        "pose",
        "--pose-window-size",
        "80",
        "--pose-protocol-version",
        str(args.pose_protocol),
        "--zmq-port",
        "5556",
        "--log-interval-s",
        "1.0",
    ]
    proxy_argv = [
        str(Path(args.proxy_bin).expanduser()),
        "--interface",
        "lo",
        "--domain-id",
        "0",
        "--lowstate-hz",
        "500.0",
        "--follow-alpha",
        "0.35",
        "--isaac-state-endpoint",
        "tcp://127.0.0.1:5560",
        "--isaac-state-topic",
        "sonic_state",
    ]
    deploy_argv = [
        str(Path(args.deploy_bin).expanduser()),
        "lo",
        str(Path(args.decoder).expanduser()),
        "reference/example",
        "--obs-config",
        str(Path(args.obs_config).expanduser()),
        "--encoder-file",
        str(Path(args.encoder).expanduser()),
        "--planner-file",
        str(Path(args.planner).expanduser()),
        "--input-type",
        "zmq_manager" if input_enabled else "keyboard",
    ]
    if input_enabled:
        deploy_argv.extend(
            [
                "--zmq-host",
                "localhost",
                "--zmq-port",
                "5556",
                "--zmq-topic",
                "pose",
            ]
        )
    deploy_argv.extend(
        [
            "--output-type",
            "all",
            "--zmq-out-port",
            "5557",
            "--zmq-out-topic",
            "g1_debug",
            "--disable-crc-check",
        ]
    )
    sender_argv = [
        teleop_python,
        "-u",
        str(Path(args.bvh_sender).expanduser()),
        "--bvh-file",
        str(Path(args.bvh).expanduser()) if args.bvh else "",
        "--host",
        "127.0.0.1",
        "--port",
        "12352",
        "--fps",
        "50",
        "--unit-scale",
        "0.01",
        "--format",
        "msgpack",
        "--loop",
        "--log-interval-s",
        "1.0",
    ]
    run_dir = Path(args.run_dir).expanduser().resolve()
    components = {
        "input": {
            "enabled": input_enabled,
            "window": "input",
            "cwd": str(sony_repo),
            "argv": input_argv if input_enabled else [],
            "executable_artifact": "teleop_python",
            "script_artifact": "mocap_manager",
            "environment_prefixes": {
                "PYTHONPATH": [str(sony_repo)],
            },
            "environment_equals": {"PYTHONUNBUFFERED": "1"},
            "environment_absent": ["LD_PRELOAD"],
            "log": str(run_dir / "input.log"),
        },
        "proxy": {
            "enabled": True,
            "window": "proxy",
            "cwd": str(sony_repo),
            "argv": proxy_argv,
            "executable_artifact": "proxy_binary",
            "environment_prefixes": {
                "LD_LIBRARY_PATH": [proxy_dds_dir, proxy_sdk_dir],
            },
            "environment_equals": {"DDS_INTERFACE": "lo"},
            "environment_absent": ["LD_PRELOAD"],
            "log": str(run_dir / "proxy.log"),
        },
        "deploy": {
            "enabled": True,
            "window": "deploy",
            "cwd": str(deploy_root),
            "argv": deploy_argv,
            "executable_artifact": "deploy_binary",
            "environment_prefixes": {
                "LD_LIBRARY_PATH": [deploy_dds_dir, deploy_sdk_dir],
            },
            "environment_equals": {
                "DDS_INTERFACE": "lo",
                "FASTRTPS_DEFAULT_PROFILES_FILE": str(
                    Path(args.deploy_fastrtps_profile).expanduser()
                ),
                "ROS_LOCALHOST_ONLY": "1",
            },
            "environment_absent": ["LD_PRELOAD"],
            "log": str(run_dir / "deploy.log"),
        },
        "bvh_sender": {
            "enabled": input_enabled,
            "window": "bvh_sender",
            "cwd": str(sony_repo),
            "argv": sender_argv if input_enabled else [],
            "executable_artifact": "teleop_python",
            "script_artifact": "bvh_sender",
            "environment_prefixes": {
                "PYTHONPATH": [str(sony_repo)],
            },
            "environment_equals": {"PYTHONUNBUFFERED": "1"},
            "environment_absent": ["LD_PRELOAD"],
            "log": str(run_dir / "bvh_sender.log"),
        },
    }
    return {
        "launch_mode": "direct_orchestrated",
        "launch_order": (
            ["input", "proxy", "deploy", "bvh_sender"]
            if input_enabled
            else ["proxy", "deploy"]
        ),
        "pose_contract": {
            "protocol": args.pose_protocol,
            "source": input_source if input_enabled else None,
            "encoder": pose_encoder if input_enabled else None,
        },
        "components": components,
    }


def create_manifest(args: argparse.Namespace) -> None:
    root = Path(args.isaaclab_root).expanduser().resolve()
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_local = now_utc.astimezone()
    imports = import_paths(str(root))
    command_argv = args.command_arg or []

    contracts = component_contracts(args)
    manifest = {
        "schema_version": 3,
        "created_at": {
            "utc": now_utc.isoformat(),
            "local": now_local.isoformat(),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
        "run": {
            "label": args.label,
            "run_dir": str(Path(args.run_dir).resolve()),
            "input": args.input,
            "pose_protocol": args.pose_protocol,
            "policy_dir": args.policy_dir,
            "policy_root": str(Path(args.policy_root).resolve()),
            "deploy_root": str(Path(args.deploy_root).resolve()),
            **({"seed": args.seed} if args.seed is not None else {}),
            "gui": args.gui,
            "session": args.session,
            "launch_mode": contracts["launch_mode"],
            "runtime_sidecar": str(Path(args.runtime_sidecar).resolve()),
            "outputs": {
                "npz": str(Path(args.out_npz).resolve()),
                "isaac_log": str(Path(args.isaac_log).resolve()),
                "manifest": str(Path(args.output).resolve()),
            },
        },
        "command": {
            "argv": command_argv,
            "shell": shlex.join(command_argv),
            "runner_argv": args.runner_command_arg or [],
            "runner_shell": shlex.join(args.runner_command_arg or []),
            "runner_extra_args": args.runner_arg or [],
        },
        "environment": captured_environment(),
        "runner_environment": key_value_pairs(args.runner_env),
        "environment_capture": {
            "explicit_keys": sorted(ENVIRONMENT_KEYS),
            "prefixes": list(ENVIRONMENT_PREFIXES),
            "note": "Runtime-relevant environment only; unrelated variables and credentials are excluded.",
        },
        "repositories": {
            "isaaclab": git_info(args.isaaclab_root),
            "sony": git_info(args.sony_repo),
            **(
                {"deploy_runtime": git_info(args.deploy_runtime_repo)}
                if args.deploy_runtime_repo
                else {}
            ),
        },
        "artifacts": {
            "bvh": file_info(args.bvh),
            "decoder_model": file_info(args.decoder),
            "encoder_model": file_info(args.encoder),
            "observation_config": file_info(args.obs_config),
            "planner_model": file_info(args.planner),
            "gr00t_43dof_import_asset": file_info(args.gr00t_43dof_usd),
            "proxy_binary": file_info(args.proxy_bin),
            "proxy_libddsc": file_info(args.proxy_libddsc),
            "proxy_libddscxx": file_info(args.proxy_libddscxx),
            "deploy_binary": file_info(args.deploy_bin),
            "deploy_fastrtps_profile": file_info(args.deploy_fastrtps_profile),
            "deploy_libddsc": file_info(args.deploy_libddsc),
            "deploy_libddscxx": file_info(args.deploy_libddscxx),
            "deploy_source": file_info(args.deploy_source),
            "teleop_python": file_info(args.teleop_python),
            "mocap_manager": file_info(args.mocap_manager),
            "bvh_sender": file_info(args.bvh_sender),
            "orchestrator": file_info(str(root / "scripts/tools/run_sonic_jitter_closed_loop.sh")),
            "verify_script": file_info(str(root / "scripts/tools/sonic_jitter_verify.py")),
            "report_script": file_info(str(root / "scripts/tools/sonic_jitter_report.py")),
            "manifest_helper": file_info(str(root / "scripts/tools/sonic_run_manifest.py")),
            "matrix_driver": file_info(str(root / "scripts/tools/run_sonic_eval_matrix.py")),
            "actions_module": file_info(
                str(
                    root
                    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                    "locomanipulation/pick_place/mdp/actions.py"
                )
            ),
            "environment_config": file_info(
                str(
                    root
                    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                    "locomanipulation/pick_place/locomanipulation_g1_env_cfg.py"
                )
            ),
            "action_config": file_info(
                str(
                    root
                    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                    "locomanipulation/pick_place/configs/action_cfg.py"
                )
            ),
        },
        "component_contract": contracts,
        "imports": imports,
        "python_path": list(sys.path),
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)

    print(f"[jitter-manifest] wrote {output.resolve()}", flush=True)
    for package_name, details in imports.items():
        print(f"[jitter-manifest] import {package_name}={details['origin']}", flush=True)


def _read_proc_environment(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        key_text = key.decode("utf-8", errors="replace")
        if key_text in {
            "DDS_INTERFACE",
            "FASTRTPS_DEFAULT_PROFILES_FILE",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONPATH",
            "PYTHONUNBUFFERED",
            "ROS_LOCALHOST_ONLY",
        }:
            values[key_text] = value.decode("utf-8", errors="replace")
    return values


def _read_proc_argv(pid: int) -> list[str]:
    return [
        item.decode("utf-8", errors="replace")
        for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if item
    ]


def _mapped_files(pid: int) -> set[str]:
    mapped: set[str] = set()
    for line in Path(f"/proc/{pid}/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        try:
            mapped.add(str(Path(fields[5]).resolve(strict=True)))
        except OSError:
            continue
    return mapped


def _dynamic_libraries(
    executable: str, environment: dict[str, str], pid: int
) -> dict[str, dict[str, Any]]:
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    if "LD_LIBRARY_PATH" in environment:
        env["LD_LIBRARY_PATH"] = environment["LD_LIBRARY_PATH"]
    result = subprocess.run(
        ["ldd", executable],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    mapped_files = _mapped_files(pid)
    libraries: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) >= 3 and fields[1] == "=>" and fields[2] != "not":
            realpath = str(Path(fields[2]).resolve())
            details: dict[str, Any] = {
                "realpath": realpath,
                "loaded_in_process": realpath in mapped_files,
            }
            if fields[0] in {"libddsc.so.0", "libddscxx.so.0"}:
                details.update(file_info(realpath) or {})
            libraries[fields[0]] = details
    return dict(sorted(libraries.items()))


def capture_runtime(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts = manifest.get("component_contract", {}).get("components", {})
    expected = {
        details["window"]: (name, details)
        for name, details in contracts.items()
        if isinstance(details, dict) and details.get("enabled") is True
    }
    reasons: list[str] = []
    observed: dict[str, int] = {}
    result: subprocess.CompletedProcess[str] | None = None
    deadline = time.monotonic() + max(0.0, float(args.wait_s))
    artifacts = manifest.get("artifacts", {})
    while True:
        result = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-s",
                "-t",
                f"={args.session}",
                "-F",
                "#{window_name}\t#{pane_pid}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        observed = {}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                window, separator, raw_pid = line.partition("\t")
                if not separator:
                    continue
                try:
                    observed[window] = int(raw_pid)
                except ValueError:
                    continue
        ready = set(observed) == set(expected)
        if ready:
            for window, (_, contract) in expected.items():
                artifact_name = contract.get("executable_artifact")
                expected_executable = (
                    artifacts.get(artifact_name, {}).get("realpath")
                    if isinstance(artifacts.get(artifact_name), dict)
                    else None
                )
                try:
                    actual_executable = str(
                        Path(f"/proc/{observed[window]}/exe").resolve(strict=True)
                    )
                except OSError:
                    ready = False
                    break
                if actual_executable != expected_executable:
                    ready = False
                    break
        if ready or time.monotonic() >= deadline:
            break
        time.sleep(0.25)

    assert result is not None
    if result.returncode != 0:
        reasons.append(f"tmux_list_panes_failed={result.stderr.strip()!r}")
    else:
        for line in result.stdout.splitlines():
            window, separator, raw_pid = line.partition("\t")
            if not separator:
                reasons.append(f"invalid_tmux_pane_row={line!r}")
                continue
            try:
                observed[window] = int(raw_pid)
            except ValueError:
                reasons.append(f"invalid_tmux_pane_pid={line!r}")
    if set(observed) != set(expected):
        reasons.append(
            "component_window_set_mismatch="
            f"expected:{sorted(expected)},actual:{sorted(observed)}"
        )

    runtime_artifacts: dict[str, Any] = {}
    for artifact_name, expected_artifact in artifacts.items():
        if not isinstance(expected_artifact, dict) or not expected_artifact.get(
            "realpath"
        ):
            continue
        try:
            actual_artifact = file_info(expected_artifact["realpath"])
        except OSError as exc:
            reasons.append(f"{artifact_name}_runtime_artifact_unreadable={exc}")
            continue
        runtime_artifacts[artifact_name] = actual_artifact
        if actual_artifact.get("realpath") != expected_artifact.get("realpath"):
            reasons.append(
                f"{artifact_name}_runtime_path_mismatch="
                f"{actual_artifact.get('realpath')!r}"
            )
        if actual_artifact.get("sha256") != expected_artifact.get("sha256"):
            reasons.append(
                f"{artifact_name}_runtime_hash_mismatch="
                f"{actual_artifact.get('sha256')!r}"
            )

    runtime_repositories: dict[str, Any] = {}
    for repository_name, expected_repository in manifest.get(
        "repositories", {}
    ).items():
        if not isinstance(expected_repository, dict) or not expected_repository.get(
            "realpath"
        ):
            continue
        try:
            actual_repository = git_info(expected_repository["realpath"])
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            reasons.append(f"{repository_name}_runtime_repository_unreadable={exc}")
            continue
        runtime_repositories[repository_name] = actual_repository
        for key in (
            "realpath",
            "commit",
            "dirty",
            "status_porcelain",
            "tracked_diff_sha256",
        ):
            if actual_repository.get(key) != expected_repository.get(key):
                reasons.append(
                    f"{repository_name}_runtime_repository_{key}_mismatch="
                    f"{actual_repository.get(key)!r}"
                )

    components: dict[str, Any] = {}
    for window, (name, contract) in expected.items():
        pid = observed.get(window)
        if pid is None:
            continue
        try:
            executable = str(Path(f"/proc/{pid}/exe").resolve(strict=True))
            cwd = str(Path(f"/proc/{pid}/cwd").resolve(strict=True))
            argv = _read_proc_argv(pid)
            environment = _read_proc_environment(pid)
            executable_info = file_info(executable)
        except OSError as exc:
            reasons.append(f"{name}_process_unreadable={exc}")
            continue
        if cwd != contract.get("cwd"):
            reasons.append(f"{name}_cwd_mismatch={cwd!r}")
        if argv != contract.get("argv"):
            reasons.append(f"{name}_argv_mismatch={argv!r}")
        artifact_name = contract.get("executable_artifact")
        expected_executable = (
            artifacts.get(artifact_name, {}).get("realpath")
            if isinstance(artifacts.get(artifact_name), dict)
            else None
        )
        if executable != expected_executable:
            reasons.append(f"{name}_executable_mismatch={executable!r}")
        for key, expected_value in contract.get("environment_equals", {}).items():
            if environment.get(key) != expected_value:
                reasons.append(
                    f"{name}_environment_{key}_mismatch={environment.get(key)!r}"
                )
        for key in contract.get("environment_absent", []):
            if key in environment:
                reasons.append(
                    f"{name}_environment_{key}_unexpected={environment.get(key)!r}"
                )
        for key, prefixes in contract.get("environment_prefixes", {}).items():
            actual_parts = environment.get(key, "").split(":")
            if actual_parts[: len(prefixes)] != prefixes:
                reasons.append(
                    f"{name}_environment_{key}_prefix_mismatch={actual_parts!r}"
                )
        libraries = _dynamic_libraries(executable, environment, pid)
        for soname, artifact_key in (
            ("libddsc.so.0", "deploy_libddsc" if name == "deploy" else "proxy_libddsc"),
            (
                "libddscxx.so.0",
                "deploy_libddscxx" if name == "deploy" else "proxy_libddscxx",
            ),
        ):
            if name not in {"deploy", "proxy"}:
                continue
            expected_artifact = artifacts.get(artifact_key, {})
            actual_library = libraries.get(soname, {})
            if actual_library.get("realpath") != expected_artifact.get("realpath"):
                reasons.append(
                    f"{name}_{soname}_path_mismatch={actual_library.get('realpath')!r}"
                )
            if actual_library.get("sha256") != expected_artifact.get("sha256"):
                reasons.append(
                    f"{name}_{soname}_hash_mismatch={actual_library.get('sha256')!r}"
                )
            if actual_library.get("loaded_in_process") is not True:
                reasons.append(
                    f"{name}_{soname}_not_loaded_in_process"
                )
        components[name] = {
            "window": window,
            "pid": pid,
            "cwd": cwd,
            "argv": argv,
            "environment": environment,
            "executable": executable_info,
            "dynamic_libraries": libraries,
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "session": args.session,
        "manifest": file_info(str(manifest_path)),
        "valid": not reasons,
        "reasons": reasons,
        "artifacts": runtime_artifacts,
        "repositories": runtime_repositories,
        "components": components,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(f"[jitter-runtime] wrote {output.resolve()} valid={payload['valid']}", flush=True)
    if reasons:
        raise RuntimeError("runtime component validation failed:\n  " + "\n  ".join(reasons))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    print_parser = subparsers.add_parser(
        "print-imports",
        help="print and validate the actual Isaac Lab package import paths",
    )
    print_parser.add_argument("--isaaclab-root", required=True)

    create_parser = subparsers.add_parser("create", help="write a new run manifest")
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument("--isaaclab-root", required=True)
    create_parser.add_argument("--sony-repo", required=True)
    create_parser.add_argument("--label", required=True)
    create_parser.add_argument("--run-dir", required=True)
    create_parser.add_argument("--out-npz", required=True)
    create_parser.add_argument("--isaac-log", required=True)
    create_parser.add_argument("--input", choices=("bvh", "keyboard"), required=True)
    create_parser.add_argument("--pose-protocol", choices=(1, 3), type=int, required=True)
    create_parser.add_argument("--policy-dir", required=True)
    create_parser.add_argument("--policy-root", required=True)
    create_parser.add_argument("--decoder", required=True)
    create_parser.add_argument("--encoder", required=True)
    create_parser.add_argument("--obs-config", required=True)
    create_parser.add_argument("--planner", required=True)
    create_parser.add_argument("--gr00t-43dof-usd", required=True)
    create_parser.add_argument("--proxy-bin", required=True)
    create_parser.add_argument("--proxy-libddsc", required=True)
    create_parser.add_argument("--proxy-libddscxx", required=True)
    create_parser.add_argument("--deploy-bin", required=True)
    create_parser.add_argument("--deploy-root", required=True)
    create_parser.add_argument("--deploy-fastrtps-profile", required=True)
    create_parser.add_argument("--deploy-libddsc", required=True)
    create_parser.add_argument("--deploy-libddscxx", required=True)
    create_parser.add_argument("--deploy-runtime-repo")
    create_parser.add_argument("--deploy-source")
    create_parser.add_argument("--seed", type=int)
    create_parser.add_argument("--teleop-python", required=True)
    create_parser.add_argument("--mocap-manager", required=True)
    create_parser.add_argument("--bvh-sender", required=True)
    create_parser.add_argument("--bvh")
    create_parser.add_argument("--gui", action="store_true")
    create_parser.add_argument("--session", required=True)
    create_parser.add_argument("--runtime-sidecar", required=True)
    create_parser.add_argument("--command-arg", action="append")
    create_parser.add_argument("--runner-command-arg", action="append")
    create_parser.add_argument("--runner-env", action="append")
    create_parser.add_argument("--runner-arg", action="append")

    runtime_parser = subparsers.add_parser(
        "capture-runtime",
        help="capture and validate the actual tmux component processes",
    )
    runtime_parser.add_argument("--output", required=True)
    runtime_parser.add_argument("--manifest", required=True)
    runtime_parser.add_argument("--session", required=True)
    runtime_parser.add_argument("--wait-s", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "print-imports":
            print_import_paths(args.isaaclab_root)
        elif args.command == "capture-runtime":
            capture_runtime(args)
        else:
            create_manifest(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[jitter-manifest] ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
