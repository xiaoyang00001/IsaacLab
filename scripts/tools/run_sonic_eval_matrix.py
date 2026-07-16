#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a balanced SONIC candidate matrix through the closed-loop evaluator.

The default matrix deliberately changes only two factors under one pinned
deploy runtime:

* deploy policy: ``policy/release`` or ``policy/low_latency``
* Isaac target consumption: env-step (50 Hz) or physics-substep (200 Hz)

Candidate-specific deploy binary/root overrides can instead compare complete
runtime bundles. Such a result selects the better bundle but is explicitly not
reported as the causal effect of policy or substep consumption alone.

Every other experimental switch is pinned to the current production baseline.
In particular, auto recovery is disabled: a fall is an outcome, not a new trial
silently spliced into the same recording.

Examples:

    # One screening block (each candidate once), v3 BVH motion.
    python3 scripts/tools/run_sonic_eval_matrix.py --repeats 1

    # Eight balanced confirmatory blocks for the two finalists.
    python3 scripts/tools/run_sonic_eval_matrix.py \
        --candidates release_base,release_substep --repeats 8 --free-seconds 120

    # Print the exact run order and environment without launching Isaac.
    python3 scripts/tools/run_sonic_eval_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    name: str
    policy_dir: str
    substep_consume: bool


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate("release_base", "policy/release", False),
        Candidate("release_substep", "policy/release", True),
        Candidate("lowlat_base", "policy/low_latency", False),
        Candidate("lowlat_substep", "policy/low_latency", True),
    )
}

# These switches have already been isolated as regressions/no-ops on the parent
# branch. All inherited SONIC_* variables are removed before these pins are
# applied, so an interactive tuning shell cannot silently alter the matrix.
PINNED_ENV = {
    "SONIC_DEPLOY_AUTO_RECOVER": "0",
    "SONIC_DEPLOY_ELASTIC_BAND": "0",
    "SONIC_G1_MUJOCO_TORQUE_PARITY": "0",
    "SONIC_G1_MUJOCO_NO_ARMATURE": "0",
    "SONIC_G1_MUJOCO_NO_VEL_LIMIT": "0",
}


class RunInterrupted(Exception):
    """Raised by the temporary SIGTERM handler while a candidate is running."""

    def __init__(self, signum: int):
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATES),
        help=f"Comma-separated candidate names. Choices: {','.join(CANDIDATES)}",
    )
    parser.add_argument(
        "--scenario",
        choices=("v3_bvh", "keyboard"),
        default="v3_bvh",
        help="v3_bvh measures motion latency/fidelity; keyboard measures idle standing stability.",
    )
    parser.add_argument("--bvh", default="/home/nolo/RAYNOS_Motion1.bvh")
    parser.add_argument("--repeats", type=int, default=1, help="Balanced blocks; each candidate runs once per block.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum attempts for one planned candidate/block before declaring that run invalid.",
    )
    parser.add_argument("--locked-seconds", type=float, default=15.0)
    parser.add_argument(
        "--free-seconds",
        type=float,
        default=55.56,
        help="True-free measurement length; default covers three 18.52s RAYNOS BVH cycles.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Matrix output root. Default: /tmp/sonic_eval/<UTC timestamp>.",
    )
    parser.add_argument("--sony-repo", default=None, help="Override the external GR00T/SONY checkout.")
    parser.add_argument(
        "--deploy-bin",
        default=None,
        help="Default deploy executable for candidates without --candidate-deploy-bin.",
    )
    parser.add_argument(
        "--deploy-root",
        default=None,
        help="Default gear_sonic_deploy runtime root for candidates without --candidate-deploy-root.",
    )
    parser.add_argument(
        "--candidate-deploy-bin",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Bind one candidate to a deploy executable. Repeat for mixed-runtime bundle comparisons.",
    )
    parser.add_argument(
        "--candidate-deploy-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Bind one candidate to a gear_sonic_deploy runtime root. Repeat as needed.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=20260716,
        help="Isaac RNG seed for block 1; block N uses seed-base+N-1, shared by all candidates/retries.",
    )
    parser.add_argument(
        "--stop-after-fall-grace-s",
        type=float,
        default=1.0,
        help="End a definitively fallen run after this grace period; 0 disables fall early-stop.",
    )
    parser.add_argument(
        "--unlock-source-index",
        type=int,
        default=800,
        help="For v3_bvh, keep root locked until the deploy source counter reaches this index.",
    )
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Allow a dirty IsaacLab worktree. Formal runs should leave this off.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue later candidates after a failed run; final exit remains non-zero.",
    )
    return parser.parse_args()


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = pathlib.Path(handle.name)
    os.replace(tmp_path, path)


def _file_fingerprint(path: pathlib.Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _git_snapshot(path: pathlib.Path) -> dict[str, Any]:
    root_result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if root_result.returncode != 0:
        raise ValueError(f"deploy root is not inside a git checkout: {path}")
    root = pathlib.Path(root_result.stdout.strip()).resolve()

    def git(*command: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *command],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.rstrip("\n")

    status = git("status", "--porcelain=v1", "--untracked-files=normal")
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    return {
        "path": str(root),
        "realpath": str(root),
        "commit": git("rev-parse", "HEAD"),
        "branch": branch or None,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "tracked_diff_bytes": len(diff),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _named_paths(values: list[str], option: str) -> dict[str, pathlib.Path]:
    parsed: dict[str, pathlib.Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        name = name.strip()
        path = path.strip()
        if not separator or not name or not path:
            raise ValueError(f"{option} expects NAME=PATH, got {raw!r}")
        if name in parsed:
            raise ValueError(f"{option} repeats candidate {name!r}")
        parsed[name] = pathlib.Path(path).expanduser().resolve()
    return parsed


def _block_seed(seed_base: int, block: int) -> int:
    return int(seed_base) + int(block) - 1


def _runner_extra_args(args: argparse.Namespace, block: int) -> list[str]:
    return [
        "--locked_seconds",
        str(args.locked_seconds),
        "--free_seconds",
        str(args.free_seconds),
        "--seed",
        str(_block_seed(args.seed_base, block)),
        "--stop_after_fall_grace_s",
        str(args.stop_after_fall_grace_s),
        *(
            ["--unlock_source_index", str(getattr(args, "unlock_source_index", 800))]
            if getattr(args, "scenario", "v3_bvh") == "v3_bvh"
            and getattr(args, "unlock_source_index", 800) >= 0
            else []
        ),
    ]


def _candidate_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("--candidates cannot be empty")
    unknown = [name for name in names if name not in CANDIDATES]
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("--candidates contains duplicates")
    return names


def _balanced_order(names: list[str], repeats: int) -> list[tuple[int, str]]:
    """Return Williams-style Latin-square blocks.

    Across a complete cycle of ``len(names)`` blocks, every candidate occupies
    every within-block position exactly once. Partial cycles are useful only as
    screening runs and are marked as such in ``matrix_plan.json``.
    """
    count = len(names)
    first_row = [0]
    for position in range(1, count):
        if position % 2:
            first_row.append((position + 1) // 2)
        else:
            first_row.append(count - position // 2)
    rows = [
        [names[(index + row) % count] for index in first_row]
        for row in range(count)
    ]
    runs: list[tuple[int, str]] = []
    for block in range(repeats):
        row = rows[block % count]
        runs.extend((block + 1, name) for name in row)
    return runs


def _extract_path(output: str, key: str) -> str | None:
    matches = re.findall(rf"^{re.escape(key)}=(.+)$", output, flags=re.MULTILINE)
    return matches[-1].strip() if matches else None


def _run_streaming(
    command: list[str], *, cwd: pathlib.Path, env: dict[str, str]
) -> tuple[int, str, int | None]:
    """Run one isolated process group, stream output, and reap it on INT/TERM."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )
    output_lines: list[str] = []

    def raise_interrupted(signum, _frame) -> None:
        raise RunInterrupted(signum)

    old_int = signal.signal(signal.SIGINT, raise_interrupted)
    old_term = signal.signal(signal.SIGTERM, raise_interrupted)
    interrupted_signal: int | None = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()
    except (KeyboardInterrupt, RunInterrupted) as exc:
        interrupted_signal = signal.SIGINT if isinstance(exc, KeyboardInterrupt) else exc.signum
        # Ignore repeated terminal signals while the child performs its own
        # trap cleanup; the child is in a separate process group.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process.poll() is None:
            try:
                os.killpg(process.pid, interrupted_signal)
            except ProcessLookupError:
                pass
        try:
            returncode = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                returncode = process.wait()
        if process.stdout is not None:
            tail = process.stdout.read()
            if tail:
                output_lines.append(tail)
                sys.stdout.write(tail)
                sys.stdout.flush()
        returncode = 128 + interrupted_signal
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
    return returncode, "".join(output_lines), interrupted_signal


def main() -> int:
    args = _parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.locked_seconds <= 0.0 or args.free_seconds <= 0.0:
        raise SystemExit("--locked-seconds and --free-seconds must be > 0")
    if args.stop_after_fall_grace_s < 0.0:
        raise SystemExit("--stop-after-fall-grace-s must be >= 0")

    try:
        names = _candidate_names(args.candidates)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    isaaclab_repository = _git_snapshot(repo_root)
    if isaaclab_repository["dirty"] and not args.allow_dirty_worktree:
        raise SystemExit(
            "IsaacLab worktree is dirty; commit/stash changes before a formal matrix "
            "or use --allow-dirty-worktree for diagnostics"
        )
    orchestrator = repo_root / "scripts/tools/run_sonic_jitter_closed_loop.sh"
    if not orchestrator.is_file():
        raise SystemExit(f"orchestrator not found: {orchestrator}")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    matrix_root = pathlib.Path(args.out_root or f"/tmp/sonic_eval/{stamp}").resolve()
    runs_root = matrix_root / "runs"
    plan_path = matrix_root / "matrix_plan.json"
    results_path = matrix_root / "matrix_results.json"

    order = _balanced_order(names, args.repeats)
    sony_repo = pathlib.Path(
        args.sony_repo
        or os.environ.get("SONY_REPO", "/home/nolo/GR00T-WholeBodyControl-sony-json-stream-20260702")
    ).expanduser().resolve()
    default_deploy_root = pathlib.Path(
        args.deploy_root
        or os.environ.get("DEPLOY_ROOT_OVERRIDE")
        or sony_repo / "gear_sonic_deploy"
    ).expanduser().resolve()
    default_deploy_bin = pathlib.Path(
        args.deploy_bin
        or os.environ.get("DEPLOY_BIN_OVERRIDE")
        or sony_repo / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref"
    ).expanduser().resolve()
    try:
        deploy_bin_overrides = _named_paths(args.candidate_deploy_bin, "--candidate-deploy-bin")
        deploy_root_overrides = _named_paths(args.candidate_deploy_root, "--candidate-deploy-root")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown_overrides = (set(deploy_bin_overrides) | set(deploy_root_overrides)) - set(names)
    if unknown_overrides:
        raise SystemExit(
            "candidate runtime override names are not selected candidates: "
            + ", ".join(sorted(unknown_overrides))
        )

    bvh_path = pathlib.Path(args.bvh).expanduser().resolve()
    planner_path = (
        sony_repo / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"
    ).resolve()
    proxy_path = (
        sony_repo / "gear_sonic_deploy/build/tools/sonic_unitree_lowstate_cpp_proxy"
    ).resolve()
    if not proxy_path.is_file():
        proxy_path = (
            sony_repo
            / "gear_sonic_deploy/prebuilt/linux-x86_64/sonic_unitree_lowstate_cpp_proxy"
        ).resolve()
    shared_paths = {
        "planner_model": planner_path,
        "proxy_binary": proxy_path,
        "external_launcher": (
            sony_repo / "scripts/launch_sonic_local_isaaclab_closed_loop.py"
        ).resolve(),
        "external_wrapper": (
            sony_repo / "scripts/launch_sonic_json_isaaclab_closed_loop.sh"
        ).resolve(),
        "mocap_manager": (
            sony_repo / "gear_sonic/scripts/mocap_manager_server.py"
        ).resolve(),
        "bvh_sender": (
            sony_repo / "gear_sonic/scripts/bvh_stream_sender.py"
        ).resolve(),
    }
    if args.scenario == "v3_bvh":
        shared_paths["bvh"] = bvh_path
    for artifact_name, artifact_path in shared_paths.items():
        if not artifact_path.is_file():
            raise SystemExit(f"shared artifact {artifact_name!r} does not exist: {artifact_path}")
    shared_artifacts = {
        name: _file_fingerprint(path) for name, path in shared_paths.items()
    }

    candidate_definitions: dict[str, dict[str, Any]] = {}
    for name in names:
        deploy_bin_path = deploy_bin_overrides.get(name, default_deploy_bin)
        deploy_root = deploy_root_overrides.get(name, default_deploy_root)
        setup_env = deploy_root / "scripts/setup_env.sh"
        if not deploy_bin_path.is_file() or not os.access(deploy_bin_path, os.X_OK):
            raise SystemExit(f"deploy binary for {name!r} is not executable: {deploy_bin_path}")
        if not deploy_root.is_dir():
            raise SystemExit(f"deploy root for {name!r} is not a directory: {deploy_root}")
        if not setup_env.is_file():
            raise SystemExit(f"deploy setup_env for {name!r} does not exist: {setup_env}")
        policy_dir = pathlib.Path(CANDIDATES[name].policy_dir)
        policy_root = (
            policy_dir.resolve()
            if policy_dir.is_absolute()
            else (sony_repo / "gear_sonic_deploy" / policy_dir).resolve()
        )
        policy_paths = {
            "decoder_model": policy_root / "model_decoder.onnx",
            "encoder_model": policy_root / "model_encoder.onnx",
            "observation_config": policy_root / "observation_config.yaml",
        }
        for artifact_name, artifact_path in policy_paths.items():
            if not artifact_path.is_file():
                raise SystemExit(
                    f"policy artifact {artifact_name!r} for {name!r} does not exist: "
                    f"{artifact_path}"
                )
        candidate_definitions[name] = {
            **asdict(CANDIDATES[name]),
            "policy_root": str(policy_root),
            "policy_artifacts": {
                artifact_name: _file_fingerprint(artifact_path)
                for artifact_name, artifact_path in policy_paths.items()
            },
            "deploy_binary": _file_fingerprint(deploy_bin_path),
            "deploy_runtime": {
                "root": str(deploy_root),
                "realpath": str(deploy_root.resolve()),
                "repository": _git_snapshot(deploy_root),
                "setup_env": _file_fingerprint(setup_env),
            },
        }

    bundle_identities = {
        (
            definition["deploy_binary"]["realpath"],
            definition["deploy_binary"]["sha256"],
            definition["deploy_runtime"]["realpath"],
            definition["deploy_runtime"]["setup_env"]["sha256"],
        )
        for definition in candidate_definitions.values()
    }
    shared_deploy_binary = (
        next(iter(candidate_definitions.values()))["deploy_binary"]
        if len({identity[:2] for identity in bundle_identities}) == 1
        else None
    )
    plan = {
        "schema_version": 2,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "isaaclab_repository": isaaclab_repository,
        "sony_repo": str(sony_repo),
        **({"deploy_binary": shared_deploy_binary} if shared_deploy_binary is not None else {}),
        "orchestrator": str(orchestrator),
        "matrix_root": str(matrix_root),
        "scenario": args.scenario,
        "bvh": str(bvh_path),
        "shared_artifacts": shared_artifacts,
        "locked_seconds": args.locked_seconds,
        "free_seconds": args.free_seconds,
        "repeats": args.repeats,
        "max_attempts": args.max_attempts,
        "stop_after_fall_grace_s": args.stop_after_fall_grace_s,
        "unlock_alignment": {
            "enabled": args.scenario == "v3_bvh" and args.unlock_source_index >= 0,
            "source_index_target": (
                args.unlock_source_index
                if args.scenario == "v3_bvh" and args.unlock_source_index >= 0
                else None
            ),
            "note": "Source index is the streamed v3 frame counter; alignment remains under locked root.",
        },
        "seed_policy": {
            "scope": "Isaac RNG only; deploy planner seed is separately fixed by the deploy runtime",
            "type": "paired_by_block",
            "seed_base": args.seed_base,
            "distinct_across_blocks": True,
            "retry_reuses_planned_seed": True,
        },
        "comparison_scope": {
            "unit": "candidate_bundle",
            "causal_attribution": len(bundle_identities) == 1,
            "note": (
                "All candidates share one deploy runtime; policy/substep factors are isolated."
                if len(bundle_identities) == 1
                else "Deploy binary/runtime bundle differs across candidates; ranking selects a complete bundle, "
                "not the causal effect of one factor."
            ),
        },
        "design": {
            "type": "Williams-style Latin square",
            "cycle_blocks": len(names),
            "complete_position_balance": args.repeats % len(names) == 0,
            "inference": (
                "confirmatory-capable; summary still requires paired exact-sign p<=0.05"
                if args.repeats % len(names) == 0 and args.repeats >= 8
                else "balanced directional screening"
                if args.repeats % len(names) == 0
                else "screening only; candidate is confounded with run position"
            ),
        },
        "candidates": [candidate_definitions[name] for name in names],
        "environment_policy": "drop inherited SONIC_* variables; force headless; apply pinned baseline",
        "command_template": [
            str(orchestrator),
            "<scenario>_<candidate>_b<block>",
            str(bvh_path),
            "--",
            "--locked_seconds",
            str(args.locked_seconds),
            "--free_seconds",
            str(args.free_seconds),
            "--seed",
            "<paired block seed>",
            "--stop_after_fall_grace_s",
            str(args.stop_after_fall_grace_s),
            *(
                ["--unlock_source_index", str(args.unlock_source_index)]
                if args.scenario == "v3_bvh" and args.unlock_source_index >= 0
                else []
            ),
        ],
        "pinned_env": PINNED_ENV,
        "order": [
            {
                "sequence": sequence,
                "block": block,
                "candidate": name,
                "seed": _block_seed(args.seed_base, block),
            }
            for sequence, (block, name) in enumerate(order, start=1)
        ],
    }

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    try:
        matrix_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(
            f"matrix output root already exists; choose a new --out-root: {matrix_root}"
        ) from exc
    _atomic_json(plan_path, plan)
    results: dict = {
        "schema_version": 2,
        "plan": str(plan_path),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": [],
    }
    _atomic_json(results_path, results)

    any_failed = False
    for sequence, (block, name) in enumerate(order, start=1):
        candidate = CANDIDATES[name]
        candidate_definition = candidate_definitions[name]
        seed = _block_seed(args.seed_base, block)
        deploy_bin_path = pathlib.Path(
            candidate_definition["deploy_binary"]["realpath"]
        )
        deploy_root = pathlib.Path(candidate_definition["deploy_runtime"]["realpath"])
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SONIC_") and key not in {"JITTER_GUI"}
        }
        env.update(PINNED_ENV)
        env.update(
            {
                "JITTER_GUI": "0",
                "JITTER_OUT_ROOT": str(runs_root),
                "JITTER_INPUT": "bvh" if args.scenario == "v3_bvh" else "keyboard",
                "JITTER_POSE_PROTOCOL": "3" if args.scenario == "v3_bvh" else "1",
                "DEPLOY_POLICY_DIR": candidate.policy_dir,
                "SONIC_DEPLOY_SUBSTEP_CONSUME": "1" if candidate.substep_consume else "0",
            }
        )
        if args.sony_repo:
            env["SONY_REPO"] = str(pathlib.Path(args.sony_repo).resolve())
        env["DEPLOY_BIN_OVERRIDE"] = str(deploy_bin_path)
        env["DEPLOY_ROOT_OVERRIDE"] = str(deploy_root)

        print(
            f"\n[matrix] {sequence}/{len(order)} block={block} candidate={name} "
            f"policy={candidate.policy_dir} substep={int(candidate.substep_consume)} "
            f"seed={seed} deploy={candidate_definition['deploy_binary']['sha256'][:12]}",
            flush=True,
        )
        attempts = []
        returncode = 1
        interrupted_signal = None
        run_dir = None
        npz = None
        npz_exists = False
        label = ""
        runner_extra_args = _runner_extra_args(args, block)
        for attempt in range(1, args.max_attempts + 1):
            label = f"{args.scenario}_{name}_b{block:02d}_a{attempt:02d}"
            command = [
                str(orchestrator),
                label,
                str(bvh_path),
                "--",
                *runner_extra_args,
            ]
            if attempt > 1:
                print(
                    f"[matrix] retry attempt={attempt}/{args.max_attempts} "
                    f"block={block} candidate={name}",
                    flush=True,
                )
            returncode, output, interrupted_signal = _run_streaming(
                command, cwd=repo_root, env=env
            )
            run_dir = _extract_path(output, "SONIC_JITTER_RUN_DIR")
            npz = _extract_path(output, "SONIC_JITTER_NPZ")
            npz_exists = bool(npz and pathlib.Path(npz).is_file())
            attempts.append(
                {
                    "attempt": attempt,
                    "label": label,
                    "seed": seed,
                    "deploy_binary_sha256": candidate_definition["deploy_binary"]["sha256"],
                    "deploy_runtime_root": candidate_definition["deploy_runtime"]["realpath"],
                    "command": command,
                    "returncode": returncode,
                    "run_dir": run_dir,
                    "npz": npz,
                    "npz_exists": npz_exists,
                    "interrupted_signal": interrupted_signal,
                    "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            if interrupted_signal is not None or (returncode == 0 and npz_exists):
                break

        run_result = {
            "sequence": sequence,
            "block": block,
            "candidate": name,
            "seed": seed,
            "deploy_binary_sha256": candidate_definition["deploy_binary"]["sha256"],
            "deploy_runtime_root": candidate_definition["deploy_runtime"]["realpath"],
            "label": label,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "returncode": returncode,
            "run_dir": run_dir,
            "npz": npz,
            "npz_exists": npz_exists,
            "interrupted_signal": interrupted_signal,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        results["runs"].append(run_result)
        _atomic_json(results_path, results)

        if returncode != 0 or not npz_exists:
            any_failed = True
            print(
                f"[matrix] FAILED candidate={name} returncode={returncode} "
                f"npz={npz or '<missing>'}",
                file=sys.stderr,
                flush=True,
            )
            if interrupted_signal is not None:
                results["interrupted_signal"] = interrupted_signal
                results["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                results["success"] = False
                _atomic_json(results_path, results)
                print(f"\nSONIC_EVAL_MATRIX_ROOT={matrix_root}")
                print(f"SONIC_EVAL_MATRIX_RESULTS={results_path}")
                return 128 + interrupted_signal
            if not args.keep_going:
                break

    results["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    results["success"] = not any_failed and len(results["runs"]) == len(order)
    summary_path = matrix_root / "matrix_summary.json"
    summary_command = [
        sys.executable,
        str(repo_root / "scripts/tools/sonic_eval_matrix_report.py"),
        str(results_path),
        "--out",
        str(summary_path),
    ]
    summary_status = subprocess.run(summary_command, cwd=repo_root).returncode
    results["summary"] = str(summary_path)
    results["summary_returncode"] = summary_status
    if summary_status != 0:
        any_failed = True
        results["success"] = False
    _atomic_json(results_path, results)
    print(f"\nSONIC_EVAL_MATRIX_ROOT={matrix_root}")
    print(f"SONIC_EVAL_MATRIX_RESULTS={results_path}")
    print(f"SONIC_EVAL_MATRIX_SUMMARY={summary_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
