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


def create_manifest(args: argparse.Namespace) -> None:
    root = Path(args.isaaclab_root).expanduser().resolve()
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_local = now_utc.astimezone()
    imports = import_paths(str(root))
    command_argv = args.command_arg or []

    manifest = {
        "schema_version": 2,
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
            "robot_usd": file_info(args.robot_usd),
            "proxy_binary": file_info(args.proxy_bin),
            "deploy_binary": file_info(args.deploy_bin),
            "deploy_setup_env": file_info(args.deploy_setup_env),
            "deploy_source": file_info(args.deploy_source),
            "external_launcher": file_info(args.external_launcher),
            "external_wrapper": file_info(args.external_wrapper),
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
    create_parser.add_argument("--robot-usd", required=True)
    create_parser.add_argument("--proxy-bin", required=True)
    create_parser.add_argument("--deploy-bin", required=True)
    create_parser.add_argument("--deploy-root", required=True)
    create_parser.add_argument("--deploy-setup-env", required=True)
    create_parser.add_argument("--deploy-runtime-repo")
    create_parser.add_argument("--deploy-source")
    create_parser.add_argument("--seed", type=int)
    create_parser.add_argument("--external-launcher", required=True)
    create_parser.add_argument("--external-wrapper", required=True)
    create_parser.add_argument("--mocap-manager", required=True)
    create_parser.add_argument("--bvh-sender", required=True)
    create_parser.add_argument("--bvh")
    create_parser.add_argument("--gui", action="store_true")
    create_parser.add_argument("--session", required=True)
    create_parser.add_argument("--command-arg", action="append")
    create_parser.add_argument("--runner-command-arg", action="append")
    create_parser.add_argument("--runner-env", action="append")
    create_parser.add_argument("--runner-arg", action="append")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "print-imports":
            print_import_paths(args.isaaclab_root)
        else:
            create_manifest(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"[jitter-manifest] ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
