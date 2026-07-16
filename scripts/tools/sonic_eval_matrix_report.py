#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate a SONIC evaluation matrix and select the best valid candidate.

Selection is deliberately lexicographic rather than a hidden weighted score:

1. valid/reproducible run fraction
2. no-fall run fraction
3. normalized survival time
4. healthy true-free fraction
5. reference-to-robot fidelity (correlation, tracking error)
6. XY drift and excess body jitter
7. internal reference-to-q following lag
8. target receive age (diagnostic tie-breaker after physical outcomes)

The output labels a winner ``confirmed`` only when the observed plan/results
really form a complete position-balanced design, every candidate has complete
valid blocks, and the winner beats every alternative with a one-sided exact
sign-test p-value <= 0.05.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from . import sonic_jitter_report as jitter_report
except ImportError:
    import sonic_jitter_report as jitter_report


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def _get(node: dict, dotted: str, default=None):
    value: Any = node
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    return None


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    sd = float(np.std(array, ddof=1)) if array.size >= 2 else 0.0
    half = (
        jitter_report.t_critical_95(int(array.size)) * sd / math.sqrt(float(array.size))
        if array.size >= 2
        else 0.0
    )
    mean = float(np.mean(array))
    return {
        "n": int(array.size),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _runtime_valid(
    report: dict,
    repo_root: pathlib.Path,
    *,
    expected_free_seconds: float,
    candidate_definition: dict,
    pinned_env: dict,
    expected_seed: int | None,
    expected_shared_artifacts: dict,
    expected_sony_repo: str | None,
    expected_unlock_source_index: int | None,
    expected_isaaclab_repository: dict | None,
    expected_launch_mode: str | None,
    expected_component_contract: dict | None,
) -> list[str]:
    reasons: list[str] = []
    meta = report.get("meta", {})
    schema_version = _finite(meta.get("schema_version"))
    if schema_version is None or schema_version < 2:
        reasons.append(f"unsupported_schema_version={meta.get('schema_version', '<missing>')}")
    if meta.get("status") != "ok":
        reasons.append(f"meta_status={meta.get('status', '<missing>')}")
    if meta.get("unlocked") is not True:
        reasons.append("runner_did_not_unlock")
    free = report.get("free", {})
    if not free:
        reasons.append("missing_free_phase")
    if _get(free, "coverage.invalid", True):
        reasons.append("coverage_invalid")
    free_seconds = _finite(free.get("seconds"))
    step_dt = _finite(meta.get("step_dt")) or 0.02
    duration_tolerance = max(0.10, 3.0 * step_dt)
    planned_free_seconds = _finite(meta.get("planned_free_seconds"))
    if (
        planned_free_seconds is not None
        and abs(planned_free_seconds - expected_free_seconds) > duration_tolerance
    ):
        reasons.append(
            f"planned_free_duration_mismatch={planned_free_seconds:.3f}"
            f"!={expected_free_seconds:.3f}"
        )

    fall_known_fraction = _finite(_get(free, "fall.known_frac"))
    if fall_known_fraction is None or fall_known_fraction < 0.98:
        reasons.append(f"insufficient_fall_observability={fall_known_fraction!r}")
    if _finite(_get(free, "fall.event_count")) is None:
        reasons.append("missing_fall_event_count")
    if _finite(_get(free, "fall.survival_s")) is None:
        reasons.append("missing_survival_time")
    fall_events = _finite(_get(free, "fall.event_count"))
    fall_survival = _finite(_get(free, "fall.survival_s"))
    termination_reason = meta.get("termination_reason")
    if free_seconds is None or free_seconds + duration_tolerance < expected_free_seconds:
        fall_grace = _finite(meta.get("stop_after_fall_grace_s")) or 0.0
        legitimate_fall_stop = (
            termination_reason == "fall_observed"
            and fall_events is not None
            and fall_events >= 1.0
            and fall_survival is not None
            and free_seconds is not None
            and free_seconds + duration_tolerance
            >= fall_survival + min(fall_grace, max(expected_free_seconds - fall_survival, 0.0))
        )
        if not legitimate_fall_stop:
            reasons.append(
                f"incomplete_free_duration={free_seconds!r}<{expected_free_seconds:.3f}"
            )
    recovery_count = _finite(_get(free, "fall.recovery_count"))
    if recovery_count is None:
        reasons.append("missing_recovery_count")
    elif recovery_count != 0.0:
        reasons.append(f"recovery_contamination={recovery_count:g}")
    if _finite(_get(free, "coverage.target_age_s.p95")) is None:
        reasons.append("missing_target_age")
    if expected_unlock_source_index is not None:
        target = _finite(meta.get("unlock_source_index_target"))
        actual = _finite(meta.get("unlock_source_index_actual"))
        if target != float(expected_unlock_source_index):
            reasons.append(f"unlock_source_target_mismatch={target!r}")
        if actual is None or actual < float(expected_unlock_source_index):
            reasons.append(f"unlock_source_not_reached={actual!r}")

    manifest = meta.get("run_manifest")
    expected_deploy_binary = candidate_definition.get("deploy_binary")
    if not isinstance(manifest, dict):
        reasons.append("missing_run_manifest")
    else:
        if expected_launch_mode is not None:
            manifest_schema = _finite(manifest.get("schema_version"))
            if manifest_schema is None or manifest_schema < 3:
                reasons.append(
                    f"unsupported_isolated_manifest_schema={manifest.get('schema_version', '<missing>')}"
                )
            if _get(manifest, "run.launch_mode") != expected_launch_mode:
                reasons.append(
                    f"launch_mode_mismatch={_get(manifest, 'run.launch_mode')!r}"
                )
            manifest_artifacts = manifest.get("artifacts", {})
            if isinstance(manifest_artifacts, dict) and any(
                name in manifest_artifacts
                for name in ("external_launcher", "external_wrapper")
            ):
                reasons.append("external_launcher_artifact_present")
        manifest_root = _get(manifest, "repositories.isaaclab.realpath")
        if manifest_root != str(repo_root):
            reasons.append(f"manifest_worktree_mismatch={manifest_root!r}")
        if isinstance(expected_isaaclab_repository, dict):
            for key in (
                "realpath",
                "commit",
                "dirty",
                "status_porcelain",
                "tracked_diff_sha256",
            ):
                actual = _get(manifest, f"repositories.isaaclab.{key}")
                expected = expected_isaaclab_repository.get(key)
                if actual != expected:
                    reasons.append(f"isaaclab_repository_{key}_mismatch={actual!r}")
        manifest_policy = _get(manifest, "run.policy_dir")
        if manifest_policy != candidate_definition.get("policy_dir"):
            reasons.append(f"manifest_policy_mismatch={manifest_policy!r}")
        manifest_policy_root = _get(manifest, "run.policy_root")
        expected_policy_root = candidate_definition.get("policy_root")
        if expected_policy_root is not None and manifest_policy_root != expected_policy_root:
            reasons.append(f"manifest_policy_root_mismatch={manifest_policy_root!r}")
        for artifact_name, expected_artifact in (
            candidate_definition.get("policy_artifacts") or {}
        ).items():
            actual_path = _get(manifest, f"artifacts.{artifact_name}.realpath")
            actual_hash = _get(manifest, f"artifacts.{artifact_name}.sha256")
            if actual_path != expected_artifact.get("realpath"):
                reasons.append(f"{artifact_name}_path_mismatch={actual_path!r}")
            if actual_hash != expected_artifact.get("sha256"):
                reasons.append(f"{artifact_name}_hash_mismatch={actual_hash!r}")
        for artifact_name, expected_artifact in expected_shared_artifacts.items():
            actual_path = _get(manifest, f"artifacts.{artifact_name}.realpath")
            actual_hash = _get(manifest, f"artifacts.{artifact_name}.sha256")
            if actual_path != expected_artifact.get("realpath"):
                reasons.append(f"shared_{artifact_name}_path_mismatch={actual_path!r}")
            if actual_hash != expected_artifact.get("sha256"):
                reasons.append(f"shared_{artifact_name}_hash_mismatch={actual_hash!r}")
        expected_import_asset = expected_shared_artifacts.get(
            "gr00t_43dof_import_asset"
        )
        if isinstance(expected_import_asset, dict):
            expected_import_path = expected_import_asset.get("realpath")
            expected_import_hash = expected_import_asset.get("sha256")
            configured_import_path = _get(
                meta, "gr00t_43dof_import_asset.realpath"
            )
            configured_import_hash = _get(
                meta, "gr00t_43dof_import_asset.sha256"
            )
            if configured_import_path != expected_import_path:
                reasons.append(
                    "configured_gr00t_43dof_path_mismatch="
                    f"{configured_import_path!r}"
                )
            if configured_import_hash != expected_import_hash:
                reasons.append(
                    "configured_gr00t_43dof_hash_mismatch="
                    f"{configured_import_hash!r}"
                )
            runner_import_path = _get(
                manifest, "runner_environment.SONIC_GR00T_43DOF_USD"
            )
            if runner_import_path != expected_import_path:
                reasons.append(
                    f"runner_gr00t_43dof_path_mismatch={runner_import_path!r}"
                )
            captured_import_path = _get(
                manifest, "environment.SONIC_GR00T_43DOF_USD"
            )
            if captured_import_path != expected_import_path:
                reasons.append(
                    f"captured_gr00t_43dof_path_mismatch={captured_import_path!r}"
                )
            if expected_sony_repo is not None:
                runner_root = _get(manifest, "runner_environment.GR00T_WBC_ROOT")
                captured_root = _get(manifest, "environment.GR00T_WBC_ROOT")
                if runner_root != expected_sony_repo:
                    reasons.append(f"runner_gr00t_root_mismatch={runner_root!r}")
                if captured_root != expected_sony_repo:
                    reasons.append(f"captured_gr00t_root_mismatch={captured_root!r}")
        if isinstance(expected_deploy_binary, dict):
            actual_deploy_path = _get(manifest, "artifacts.deploy_binary.realpath")
            actual_deploy_hash = _get(manifest, "artifacts.deploy_binary.sha256")
            if actual_deploy_path != expected_deploy_binary.get("realpath"):
                reasons.append(f"deploy_binary_path_mismatch={actual_deploy_path!r}")
            if actual_deploy_hash != expected_deploy_binary.get("sha256"):
                reasons.append(f"deploy_binary_hash_mismatch={actual_deploy_hash!r}")

        expected_runtime = candidate_definition.get("deploy_runtime")
        if isinstance(expected_runtime, dict):
            manifest_schema = _finite(manifest.get("schema_version"))
            if manifest_schema is None or manifest_schema < 2:
                reasons.append(
                    f"unsupported_manifest_schema={manifest.get('schema_version', '<missing>')}"
                )
            actual_root = _get(manifest, "run.deploy_root")
            if actual_root != expected_runtime.get("realpath"):
                reasons.append(f"deploy_runtime_root_mismatch={actual_root!r}")
            expected_setup = expected_runtime.get("setup_env", {})
            actual_setup_path = _get(manifest, "artifacts.deploy_setup_env.realpath")
            actual_setup_hash = _get(manifest, "artifacts.deploy_setup_env.sha256")
            if expected_setup and actual_setup_path != expected_setup.get("realpath"):
                reasons.append(f"deploy_setup_env_path_mismatch={actual_setup_path!r}")
            if expected_setup and actual_setup_hash != expected_setup.get("sha256"):
                reasons.append(f"deploy_setup_env_hash_mismatch={actual_setup_hash!r}")
            expected_profile = expected_runtime.get("fastrtps_profile", {})
            actual_profile_path = _get(
                manifest, "artifacts.deploy_fastrtps_profile.realpath"
            )
            actual_profile_hash = _get(
                manifest, "artifacts.deploy_fastrtps_profile.sha256"
            )
            if expected_profile and actual_profile_path != expected_profile.get(
                "realpath"
            ):
                reasons.append(
                    f"deploy_fastrtps_profile_path_mismatch={actual_profile_path!r}"
                )
            if expected_profile and actual_profile_hash != expected_profile.get(
                "sha256"
            ):
                reasons.append(
                    f"deploy_fastrtps_profile_hash_mismatch={actual_profile_hash!r}"
                )
            for artifact_name, expected_artifact in (
                expected_runtime.get("runtime_libraries") or {}
            ).items():
                actual_path = _get(manifest, f"artifacts.{artifact_name}.realpath")
                actual_hash = _get(manifest, f"artifacts.{artifact_name}.sha256")
                if actual_path != expected_artifact.get("realpath"):
                    reasons.append(
                        f"{artifact_name}_path_mismatch={actual_path!r}"
                    )
                if actual_hash != expected_artifact.get("sha256"):
                    reasons.append(
                        f"{artifact_name}_hash_mismatch={actual_hash!r}"
                    )
            expected_repository = expected_runtime.get("repository", {})
            for key in (
                "realpath",
                "commit",
                "dirty",
                "status_porcelain",
                "tracked_diff_sha256",
            ):
                actual = _get(manifest, f"repositories.deploy_runtime.{key}")
                expected = expected_repository.get(key)
                if actual != expected:
                    reasons.append(f"deploy_runtime_{key}_mismatch={actual!r}")

        manifest_contract = manifest.get("component_contract")
        if expected_component_contract is not None:
            if not isinstance(manifest_contract, dict):
                reasons.append("missing_component_contract")
            else:
                if manifest_contract.get("launch_mode") != expected_launch_mode:
                    reasons.append(
                        f"component_launch_mode_mismatch={manifest_contract.get('launch_mode')!r}"
                    )
                if manifest_contract.get("launch_order") != expected_component_contract.get(
                    "launch_order"
                ):
                    reasons.append(
                        "component_launch_order_mismatch="
                        f"{manifest_contract.get('launch_order')!r}"
                    )
                if manifest_contract.get("pose_contract") != expected_component_contract.get(
                    "pose"
                ):
                    reasons.append(
                        f"component_pose_contract_mismatch={manifest_contract.get('pose_contract')!r}"
                    )
                components = manifest_contract.get("components", {})
                enabled_components = sorted(
                    name
                    for name, details in components.items()
                    if isinstance(details, dict) and details.get("enabled") is True
                )
                if enabled_components != sorted(
                    expected_component_contract.get("launch_order", [])
                ):
                    reasons.append(
                        f"enabled_component_set_mismatch={enabled_components!r}"
                    )
                forbidden_tokens = ("launch_sonic_", "just", "target/release/g1_deploy_onnx_ref")
                for component_name, details in components.items():
                    if not isinstance(details, dict) or details.get("enabled") is not True:
                        continue
                    argv = details.get("argv")
                    if not isinstance(argv, list) or not all(
                        isinstance(item, str) for item in argv
                    ):
                        reasons.append(f"{component_name}_invalid_planned_argv")
                        continue
                    joined_argv = " ".join(argv)
                    for token in forbidden_tokens:
                        if token in joined_argv:
                            reasons.append(
                                f"{component_name}_forbidden_argv_token={token}"
                            )

                def option_value(component_name: str, option: str) -> str | None:
                    details = components.get(component_name)
                    argv = details.get("argv", []) if isinstance(details, dict) else []
                    try:
                        index = argv.index(option)
                    except (ValueError, AttributeError):
                        return None
                    return argv[index + 1] if index + 1 < len(argv) else None

                proxy_expected = expected_component_contract.get("proxy", {})
                for option, expected in (
                    ("--interface", proxy_expected.get("interface")),
                    ("--domain-id", proxy_expected.get("domain_id")),
                    ("--lowstate-hz", proxy_expected.get("lowstate_hz")),
                    ("--follow-alpha", proxy_expected.get("follow_alpha")),
                    (
                        "--isaac-state-endpoint",
                        proxy_expected.get("isaac_state_endpoint"),
                    ),
                    ("--isaac-state-topic", proxy_expected.get("isaac_state_topic")),
                ):
                    actual = option_value("proxy", option)
                    if actual != str(expected):
                        reasons.append(
                            f"proxy_contract_{option}_mismatch={actual!r}"
                        )
                pose_expected = expected_component_contract.get("pose", {})
                if "input" in expected_component_contract.get("launch_order", []):
                    input_expected = expected_component_contract.get("input", {})
                    for option, expected in (
                        ("--source", input_expected.get("source")),
                        (
                            "--bvh-stream-host",
                            input_expected.get("bvh_stream_host"),
                        ),
                        (
                            "--bvh-stream-port",
                            input_expected.get("bvh_stream_port"),
                        ),
                        (
                            "--bvh-stream-bonedata-position-scale",
                            input_expected.get("position_scale"),
                        ),
                        (
                            "--bvh-stream-bonedata-input-quat-order",
                            input_expected.get("input_quat_order"),
                        ),
                        (
                            "--sony-pico-bonedata-basis",
                            input_expected.get("bonedata_basis"),
                        ),
                        (
                            "--sony-pico-smpl-joints-source",
                            input_expected.get("smpl_joints_source"),
                        ),
                        ("--control-mode", input_expected.get("control_mode")),
                        (
                            "--pose-window-size",
                            input_expected.get("pose_window_size"),
                        ),
                        (
                            "--pose-encoder-mode",
                            input_expected.get("pose_encoder_mode"),
                        ),
                        (
                            "--pose-protocol-version",
                            input_expected.get("pose_protocol_version"),
                        ),
                        ("--zmq-port", input_expected.get("zmq_port")),
                        (
                            "--log-interval-s",
                            input_expected.get("log_interval_s"),
                        ),
                    ):
                        actual = option_value("input", option)
                        if actual != str(expected):
                            reasons.append(
                                f"input_contract_{option}_mismatch={actual!r}"
                            )
                    if (
                        input_expected.get("source") != pose_expected.get("source")
                        or input_expected.get("pose_encoder_mode")
                        != pose_expected.get("encoder")
                        or input_expected.get("pose_protocol_version")
                        != pose_expected.get("protocol")
                    ):
                        reasons.append("input_pose_contract_internal_mismatch")
                    sender_expected = expected_component_contract.get("sender", {})
                    for option, expected in (
                        ("--host", sender_expected.get("host")),
                        ("--port", sender_expected.get("port")),
                        ("--fps", sender_expected.get("fps")),
                        ("--unit-scale", sender_expected.get("unit_scale")),
                        ("--format", sender_expected.get("format")),
                        (
                            "--log-interval-s",
                            sender_expected.get("log_interval_s"),
                        ),
                    ):
                        actual = option_value("bvh_sender", option)
                        if actual != str(expected):
                            reasons.append(
                                f"sender_contract_{option}_mismatch={actual!r}"
                            )
                    sender_argv = components.get("bvh_sender", {}).get("argv", [])
                    sender_bvh = option_value("bvh_sender", "--bvh-file")
                    expected_bvh_path = _get(
                        manifest,
                        f"artifacts.{sender_expected.get('bvh_artifact')}.realpath",
                    )
                    if sender_bvh is None or str(
                        pathlib.Path(sender_bvh).resolve()
                    ) != expected_bvh_path:
                        reasons.append(
                            f"sender_contract_bvh_file_mismatch={sender_bvh!r}"
                        )
                    if sender_expected.get("loop") is True and "--loop" not in sender_argv:
                        reasons.append("sender_contract_missing_loop")
                deploy_expected = expected_component_contract.get("deploy", {})
                deploy_expected_input = deploy_expected.get("input_type")
                if option_value("deploy", "--input-type") != str(
                    deploy_expected_input
                ):
                    reasons.append(
                        "deploy_contract_input_type_mismatch="
                        f"{option_value('deploy', '--input-type')!r}"
                    )
                deploy_argv = components.get("deploy", {}).get("argv", [])
                expected_deploy_path = _get(manifest, "artifacts.deploy_binary.realpath")
                if (
                    not isinstance(deploy_argv, list)
                    or not deploy_argv
                    or str(pathlib.Path(deploy_argv[0]).resolve())
                    != expected_deploy_path
                ):
                    reasons.append("deploy_contract_executable_mismatch")
                if isinstance(deploy_argv, list) and len(deploy_argv) >= 4:
                    if deploy_argv[1] != str(
                        deploy_expected.get("interface_argument")
                    ):
                        reasons.append(
                            f"deploy_contract_interface_mismatch={deploy_argv[1]!r}"
                        )
                    if str(pathlib.Path(deploy_argv[2]).resolve()) != _get(
                        manifest,
                        f"artifacts.{deploy_expected.get('decoder_artifact')}.realpath",
                    ):
                        reasons.append("deploy_contract_decoder_mismatch")
                    if deploy_argv[3] != deploy_expected.get("motion_data"):
                        reasons.append(
                            f"deploy_contract_motion_data_mismatch={deploy_argv[3]!r}"
                        )
                else:
                    reasons.append("deploy_contract_positional_argv_missing")
                for option, artifact_name in (
                    (
                        "--obs-config",
                        deploy_expected.get("observation_config_artifact"),
                    ),
                    ("--encoder-file", deploy_expected.get("encoder_artifact")),
                    ("--planner-file", deploy_expected.get("planner_artifact")),
                ):
                    actual = option_value("deploy", option)
                    expected_path = _get(
                        manifest, f"artifacts.{artifact_name}.realpath"
                    )
                    if actual is None or str(pathlib.Path(actual).resolve()) != expected_path:
                        reasons.append(
                            f"deploy_contract_{option}_mismatch={actual!r}"
                        )
                for option, expected in (
                    ("--output-type", deploy_expected.get("output_type")),
                    ("--zmq-out-port", deploy_expected.get("debug_port")),
                    ("--zmq-out-topic", deploy_expected.get("debug_topic")),
                ):
                    actual = option_value("deploy", option)
                    if actual != str(expected):
                        reasons.append(
                            f"deploy_contract_{option}_mismatch={actual!r}"
                        )
                for option, expected in (
                    ("--zmq-host", deploy_expected.get("zmq_host")),
                    ("--zmq-port", deploy_expected.get("zmq_port")),
                    ("--zmq-topic", deploy_expected.get("zmq_topic")),
                ):
                    actual = option_value("deploy", option)
                    if expected is None:
                        if actual is not None:
                            reasons.append(
                                f"deploy_contract_unexpected_{option}={actual!r}"
                            )
                    elif actual != str(expected):
                        reasons.append(
                            f"deploy_contract_{option}_mismatch={actual!r}"
                        )
                if (
                    deploy_expected.get("disable_crc_check") is True
                    and "--disable-crc-check" not in deploy_argv
                ):
                    reasons.append("deploy_contract_missing_disable_crc_check")

                deploy_details = components.get("deploy", {})
                deploy_environment = (
                    deploy_details.get("environment_equals", {})
                    if isinstance(deploy_details, dict)
                    else {}
                )
                expected_deploy_environment = deploy_expected.get("environment", {})
                for key in ("DDS_INTERFACE", "ROS_LOCALHOST_ONLY"):
                    if deploy_environment.get(key) != expected_deploy_environment.get(key):
                        reasons.append(
                            f"deploy_contract_environment_{key}_mismatch="
                            f"{deploy_environment.get(key)!r}"
                        )
                expected_profile_path = _get(
                    manifest, "artifacts.deploy_fastrtps_profile.realpath"
                )
                if (
                    deploy_environment.get("FASTRTPS_DEFAULT_PROFILES_FILE")
                    != expected_profile_path
                ):
                    reasons.append(
                        "deploy_contract_environment_FASTRTPS_DEFAULT_PROFILES_FILE_mismatch"
                    )
                deploy_prefixes = (
                    deploy_details.get("environment_prefixes", {}).get(
                        "LD_LIBRARY_PATH", []
                    )
                    if isinstance(deploy_details, dict)
                    else []
                )
                deploy_dds_path = _get(manifest, "artifacts.deploy_libddsc.realpath")
                deploy_root = _get(manifest, "run.deploy_root")
                runtime_arch = _get(manifest, "host.machine")
                expected_deploy_prefixes = [
                    str(pathlib.Path(deploy_dds_path).parent)
                    if deploy_dds_path
                    else None,
                    str(
                        pathlib.Path(deploy_root)
                        / "thirdparty/unitree_sdk2/lib"
                        / str(runtime_arch)
                    )
                    if deploy_root and runtime_arch
                    else None,
                ]
                if deploy_prefixes != expected_deploy_prefixes:
                    reasons.append(
                        f"deploy_contract_LD_LIBRARY_PATH_mismatch={deploy_prefixes!r}"
                    )

                proxy_details = components.get("proxy", {})
                proxy_environment = (
                    proxy_details.get("environment_equals", {})
                    if isinstance(proxy_details, dict)
                    else {}
                )
                if proxy_environment.get("DDS_INTERFACE") != proxy_expected.get(
                    "interface"
                ):
                    reasons.append(
                        "proxy_contract_environment_DDS_INTERFACE_mismatch"
                    )
                proxy_prefixes = (
                    proxy_details.get("environment_prefixes", {}).get(
                        "LD_LIBRARY_PATH", []
                    )
                    if isinstance(proxy_details, dict)
                    else []
                )
                proxy_dds_path = _get(manifest, "artifacts.proxy_libddsc.realpath")
                sony_root = _get(manifest, "repositories.sony.realpath")
                expected_proxy_prefixes = [
                    str(pathlib.Path(proxy_dds_path).parent)
                    if proxy_dds_path
                    else None,
                    str(
                        pathlib.Path(sony_root)
                        / "gear_sonic_deploy/thirdparty/unitree_sdk2/lib"
                        / str(runtime_arch)
                    )
                    if sony_root and runtime_arch
                    else None,
                ]
                if proxy_prefixes != expected_proxy_prefixes:
                    reasons.append(
                        f"proxy_contract_LD_LIBRARY_PATH_mismatch={proxy_prefixes!r}"
                    )

        runtime_components = meta.get("runtime_components")
        if expected_launch_mode is not None:
            if not isinstance(runtime_components, dict):
                reasons.append("missing_runtime_components")
            else:
                manifest_session = _get(manifest, "run.session")
                if runtime_components.get("session") != manifest_session:
                    reasons.append(
                        "runtime_session_mismatch="
                        f"{runtime_components.get('session')!r}!={manifest_session!r}"
                    )
                run_manifest_path = meta.get("run_manifest_path")
                runtime_manifest_path = meta.get("runtime_manifest_path")
                if runtime_manifest_path != _get(manifest, "run.runtime_sidecar"):
                    reasons.append(
                        f"runtime_sidecar_path_mismatch={runtime_manifest_path!r}"
                    )
                if not isinstance(run_manifest_path, str):
                    reasons.append("missing_run_manifest_path")
                else:
                    try:
                        manifest_path = pathlib.Path(run_manifest_path).resolve(
                            strict=True
                        )
                        manifest_on_disk = json.loads(
                            manifest_path.read_text(encoding="utf-8")
                        )
                        if manifest_on_disk != manifest:
                            reasons.append("embedded_manifest_differs_from_disk")
                        if _get(
                            runtime_components, "manifest.realpath"
                        ) != str(manifest_path):
                            reasons.append("runtime_manifest_path_fingerprint_mismatch")
                        if _get(
                            runtime_components, "manifest.sha256"
                        ) != _sha256_file(manifest_path):
                            reasons.append("runtime_manifest_hash_fingerprint_mismatch")
                    except (OSError, json.JSONDecodeError) as exc:
                        reasons.append(f"run_manifest_recheck_failed={exc}")
                if not isinstance(runtime_manifest_path, str):
                    reasons.append("missing_runtime_manifest_path")
                else:
                    try:
                        runtime_path = pathlib.Path(runtime_manifest_path).resolve(
                            strict=True
                        )
                        runtime_on_disk = json.loads(
                            runtime_path.read_text(encoding="utf-8")
                        )
                        if runtime_on_disk != runtime_components:
                            reasons.append(
                                "embedded_runtime_components_differs_from_disk"
                            )
                    except (OSError, json.JSONDecodeError) as exc:
                        reasons.append(f"runtime_manifest_recheck_failed={exc}")
                if runtime_components.get("valid") is not True:
                    reasons.append(
                        f"runtime_components_invalid={runtime_components.get('reasons')!r}"
                    )
                runtime_artifacts = runtime_components.get("artifacts", {})
                for artifact_name, expected_artifact in manifest.get(
                    "artifacts", {}
                ).items():
                    if not isinstance(expected_artifact, dict):
                        continue
                    actual_artifact = (
                        runtime_artifacts.get(artifact_name)
                        if isinstance(runtime_artifacts, dict)
                        else None
                    )
                    if not isinstance(actual_artifact, dict):
                        reasons.append(
                            f"runtime_artifact_missing={artifact_name}"
                        )
                        continue
                    if actual_artifact.get("realpath") != expected_artifact.get(
                        "realpath"
                    ):
                        reasons.append(
                            f"runtime_artifact_{artifact_name}_path_mismatch"
                        )
                    if actual_artifact.get("sha256") != expected_artifact.get(
                        "sha256"
                    ):
                        reasons.append(
                            f"runtime_artifact_{artifact_name}_hash_mismatch"
                        )
                runtime_repositories = runtime_components.get("repositories", {})
                for repository_name, expected_repository in manifest.get(
                    "repositories", {}
                ).items():
                    if not isinstance(expected_repository, dict):
                        continue
                    actual_repository = (
                        runtime_repositories.get(repository_name)
                        if isinstance(runtime_repositories, dict)
                        else None
                    )
                    if not isinstance(actual_repository, dict):
                        reasons.append(
                            f"runtime_repository_missing={repository_name}"
                        )
                        continue
                    for key in (
                        "realpath",
                        "commit",
                        "dirty",
                        "status_porcelain",
                        "tracked_diff_sha256",
                    ):
                        if actual_repository.get(key) != expected_repository.get(key):
                            reasons.append(
                                f"runtime_repository_{repository_name}_{key}_mismatch"
                            )
                runtime_map = runtime_components.get("components", {})
                planned_components = (
                    manifest_contract.get("components", {})
                    if isinstance(manifest_contract, dict)
                    else {}
                )
                expected_enabled = {
                    name
                    for name, details in planned_components.items()
                    if isinstance(details, dict) and details.get("enabled") is True
                }
                if set(runtime_map) != expected_enabled:
                    reasons.append(
                        "runtime_component_set_mismatch="
                        f"expected:{sorted(expected_enabled)},actual:{sorted(runtime_map)}"
                    )
                artifacts = manifest.get("artifacts", {})
                for component_name in expected_enabled:
                    planned = planned_components.get(component_name, {})
                    actual = runtime_map.get(component_name, {})
                    if actual.get("argv") != planned.get("argv"):
                        reasons.append(f"runtime_{component_name}_argv_mismatch")
                    if actual.get("cwd") != planned.get("cwd"):
                        reasons.append(f"runtime_{component_name}_cwd_mismatch")
                    actual_environment = actual.get("environment", {})
                    if not isinstance(actual_environment, dict):
                        reasons.append(
                            f"runtime_{component_name}_environment_invalid"
                        )
                        actual_environment = {}
                    for key, expected_value in planned.get(
                        "environment_equals", {}
                    ).items():
                        if actual_environment.get(key) != expected_value:
                            reasons.append(
                                f"runtime_{component_name}_environment_{key}_mismatch"
                            )
                    for key in planned.get("environment_absent", []):
                        if key in actual_environment:
                            reasons.append(
                                f"runtime_{component_name}_environment_{key}_unexpected"
                            )
                    for key, expected_prefixes in planned.get(
                        "environment_prefixes", {}
                    ).items():
                        actual_parts = actual_environment.get(key, "").split(":")
                        if actual_parts[: len(expected_prefixes)] != expected_prefixes:
                            reasons.append(
                                f"runtime_{component_name}_environment_{key}_prefix_mismatch"
                            )
                    artifact_name = planned.get("executable_artifact")
                    expected_executable = _get(
                        artifacts, f"{artifact_name}.realpath"
                    )
                    expected_executable_hash = _get(
                        artifacts, f"{artifact_name}.sha256"
                    )
                    if _get(actual, "executable.realpath") != expected_executable:
                        reasons.append(
                            f"runtime_{component_name}_executable_path_mismatch"
                        )
                    if _get(actual, "executable.sha256") != expected_executable_hash:
                        reasons.append(
                            f"runtime_{component_name}_executable_hash_mismatch"
                        )
                    if component_name in {"proxy", "deploy"}:
                        prefix = "deploy" if component_name == "deploy" else "proxy"
                        for soname, artifact_name in (
                            ("libddsc.so.0", f"{prefix}_libddsc"),
                            ("libddscxx.so.0", f"{prefix}_libddscxx"),
                        ):
                            actual_libraries = actual.get("dynamic_libraries", {})
                            actual_library = (
                                actual_libraries.get(soname)
                                if isinstance(actual_libraries, dict)
                                else {}
                            )
                            if not isinstance(actual_library, dict):
                                reasons.append(
                                    f"runtime_{component_name}_{soname}_invalid"
                                )
                                continue
                            if actual_library.get("realpath") != _get(
                                artifacts, f"{artifact_name}.realpath"
                            ):
                                reasons.append(
                                    f"runtime_{component_name}_{soname}_path_mismatch"
                                )
                            if actual_library.get("sha256") != _get(
                                artifacts, f"{artifact_name}.sha256"
                            ):
                                reasons.append(
                                    f"runtime_{component_name}_{soname}_hash_mismatch"
                                )
                            if actual_library.get("loaded_in_process") is not True:
                                reasons.append(
                                    f"runtime_{component_name}_{soname}_not_loaded"
                                )

        if expected_seed is not None:
            if _finite(meta.get("seed_requested")) != float(expected_seed):
                reasons.append(f"seed_requested_mismatch={meta.get('seed_requested')!r}")
            if _finite(meta.get("seed_actual")) != float(expected_seed):
                reasons.append(f"seed_actual_mismatch={meta.get('seed_actual')!r}")
            if _finite(_get(manifest, "run.seed")) != float(expected_seed):
                reasons.append(f"manifest_seed_mismatch={_get(manifest, 'run.seed')!r}")

    sonic_env = meta.get("sonic_env")
    if not isinstance(sonic_env, dict):
        reasons.append("missing_sonic_environment")
    else:
        expected_sonic = {
            **{
                key: str(value)
                for key, value in pinned_env.items()
                if str(key).startswith("SONIC_")
            },
            "SONIC_DEPLOY_SUBSTEP_CONSUME": (
                "1" if candidate_definition.get("substep_consume") else "0"
            ),
        }
        expected_import_asset = expected_shared_artifacts.get(
            "gr00t_43dof_import_asset"
        )
        if isinstance(expected_import_asset, dict):
            expected_sonic["SONIC_GR00T_43DOF_USD"] = str(
                expected_import_asset.get("realpath")
            )
        for key, expected in expected_sonic.items():
            if sonic_env.get(key) != expected:
                reasons.append(f"sonic_env_mismatch_{key}={sonic_env.get(key)!r}")

    for key in ("isaaclab_tasks_file", "actions_module_file"):
        raw_path = meta.get(key)
        if not isinstance(raw_path, str):
            reasons.append(f"missing_{key}")
            continue
        try:
            pathlib.Path(raw_path).resolve().relative_to(repo_root)
        except (OSError, ValueError):
            reasons.append(f"{key}_outside_worktree={raw_path}")
    scene_robot_asset = meta.get("scene_robot_asset")
    if not isinstance(scene_robot_asset, dict):
        reasons.append("missing_scene_robot_asset")
    else:
        for key in ("asset_name", "prim_path", "usd_path"):
            value = scene_robot_asset.get(key)
            if not isinstance(value, str) or not value:
                reasons.append(f"invalid_scene_robot_asset_{key}")
    return reasons


def _run_metrics(
    report: dict, *, expected_free_seconds: float | None = None
) -> dict[str, float | None]:
    free = report["free"]
    seconds = _finite(free.get("seconds"))
    normalization_seconds = (
        float(expected_free_seconds)
        if expected_free_seconds is not None and expected_free_seconds > 0.0
        else seconds
    )
    fall_events = _finite(_get(free, "fall.event_count"))
    survival = _finite(_get(free, "fall.survival_s"))
    survival_fraction = (
        min(max(survival / normalization_seconds, 0.0), 1.0)
        if survival is not None
        and normalization_seconds is not None
        and normalization_seconds > 0.0
        else None
    )
    signed_lag = _finite(_get(free, "internal_following_lag.reference_to_q.lag_s"))
    lag = _finite(_get(free, "internal_following_lag.reference_to_q.abs_lag_s"))
    lag_corr = _finite(_get(free, "internal_following_lag.reference_to_q.corr"))
    if lag_corr is None or lag_corr < 0.50 or lag is None:
        lag = None

    jitter_values = [
        value
        for dotted in (
            "healthy.arms_hf_rms_deg",
            "healthy.waist_hf_rms_deg",
            "healthy.legs_hf_rms_deg",
            "healthy.tilt_hf_rms_deg",
        )
        if (value := _finite(_get(free, dotted))) is not None
    ]
    tracking_error_hf_values = [
        value
        for dotted in (
            "healthy.arms_tracking_error_hf_rms_deg",
            "healthy.waist_tracking_error_hf_rms_deg",
            "healthy.legs_tracking_error_hf_rms_deg",
        )
        if (value := _finite(_get(free, dotted))) is not None
    ]
    tracking_rms_values = [
        value
        for dotted in (
            "healthy.arms_track_rms_deg",
            "healthy.waist_track_rms_deg",
            "healthy.legs_track_rms_deg",
        )
        if (value := _finite(_get(free, dotted))) is not None
    ]
    healthy_seconds = _finite(_get(free, "healthy.seconds"))
    healthy_fraction = (
        min(max(healthy_seconds / normalization_seconds, 0.0), 1.0)
        if healthy_seconds is not None
        and normalization_seconds is not None
        and normalization_seconds > 0.0
        else _finite(_get(free, "healthy.frac"))
    )
    return {
        "fall_events": fall_events,
        "no_fall": None if fall_events is None else 1.0 if fall_events == 0.0 else 0.0,
        "survival_fraction": survival_fraction,
        "healthy_fraction": healthy_fraction,
        "reference_to_q_abs_lag_s": lag,
        "reference_to_q_signed_lag_s": signed_lag,
        "reference_to_q_corr": lag_corr,
        "target_age_p95_s": _finite(_get(free, "coverage.target_age_s.p95")),
        "healthy_jitter_mean_deg": float(np.mean(jitter_values)) if jitter_values else None,
        "healthy_tracking_error_hf_mean_deg": (
            float(np.mean(tracking_error_hf_values)) if tracking_error_hf_values else None
        ),
        "healthy_track_rms_mean_deg": (
            float(np.mean(tracking_rms_values)) if tracking_rms_values else None
        ),
        "xy_max_drift_m": _finite(_get(free, "root.xy_max_drift_m")),
        "source_index_start": _finite(_get(free, "source_index.start")),
        "update_coverage": _finite(_get(free, "coverage.update_coverage")),
        "packet_coverage": _finite(_get(free, "coverage.packet_coverage")),
    }


def _candidate_rank(candidate: dict) -> tuple:
    metrics = candidate["metrics"]

    def higher(name: str, missing: float = -math.inf) -> float:
        summary = metrics.get(name)
        return float(summary["mean"]) if summary else missing

    def lower(name: str, missing: float = math.inf) -> float:
        summary = metrics.get(name)
        return -float(summary["mean"]) if summary else -missing

    return (
        float(candidate["valid_fraction"]),
        higher("no_fall"),
        higher("survival_fraction"),
        higher("healthy_fraction"),
        higher("reference_to_q_corr"),
        lower("healthy_tracking_error_hf_mean_deg"),
        lower("healthy_track_rms_mean_deg"),
        lower("reference_to_q_abs_lag_s"),
        lower("xy_max_drift_m"),
        lower("healthy_jitter_mean_deg"),
        lower("target_age_p95_s"),
    )


def _single_run_rank(metrics: dict) -> tuple:
    def higher(name: str, missing: float = -math.inf) -> float:
        value = metrics.get(name)
        return float(value) if value is not None else missing

    def lower(name: str, missing: float = math.inf) -> float:
        value = metrics.get(name)
        return -float(value) if value is not None else -missing

    return (
        higher("no_fall"),
        higher("survival_fraction"),
        higher("healthy_fraction"),
        higher("reference_to_q_corr"),
        lower("healthy_tracking_error_hf_mean_deg"),
        lower("healthy_track_rms_mean_deg"),
        lower("reference_to_q_abs_lag_s"),
        lower("xy_max_drift_m"),
        lower("healthy_jitter_mean_deg"),
        lower("target_age_p95_s"),
    )


def _exact_sign_p_one_sided(winner_better: int, paired_blocks: int) -> float:
    """P(X >= winner_better), X~Binomial(p=.5), conservatively counting ties as non-wins."""
    if paired_blocks <= 0:
        return 1.0
    return float(
        sum(math.comb(paired_blocks, count) for count in range(winner_better, paired_blocks + 1))
        / (2**paired_blocks)
    )


def _validate_design(plan: dict, results: dict) -> dict:
    """Derive balance/completeness from observed rows instead of trusting plan flags."""
    reasons: list[str] = []
    definitions = plan.get("candidates", [])
    names = [
        definition.get("name")
        for definition in definitions
        if isinstance(definition, dict) and isinstance(definition.get("name"), str)
    ]
    if len(names) != len(definitions) or len(set(names)) != len(names) or not names:
        reasons.append("invalid_or_duplicate_candidate_definitions")
    try:
        repeats = int(plan.get("repeats"))
    except (TypeError, ValueError):
        repeats = 0
    if repeats < 1:
        reasons.append("invalid_repeats")

    order = plan.get("order")
    expected_total = repeats * len(names)
    seed_policy = plan.get("seed_policy", {})
    paired_seed_required = (
        isinstance(seed_policy, dict) and seed_policy.get("type") == "paired_by_block"
    )
    expected_by_sequence: dict[int, tuple[int, str, int | None]] = {}
    block_seeds: dict[int, int] = {}
    position_counts = {name: [0] * len(names) for name in names}
    if not isinstance(order, list) or len(order) != expected_total:
        reasons.append("plan_order_length_mismatch")
    else:
        parsed_order: list[tuple[int, int, str, int | None]] = []
        for row in order:
            try:
                sequence = int(row["sequence"])
                block = int(row["block"])
                candidate = str(row["candidate"])
                seed = int(row["seed"]) if paired_seed_required else None
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_plan_order_row")
                continue
            parsed_order.append((sequence, block, candidate, seed))
        parsed_order.sort()
        if [sequence for sequence, _, _, _ in parsed_order] != list(
            range(1, expected_total + 1)
        ):
            reasons.append("plan_sequence_not_contiguous")
        for sequence, block, candidate, seed in parsed_order:
            if sequence in expected_by_sequence:
                reasons.append("duplicate_plan_sequence")
            expected_by_sequence[sequence] = (block, candidate, seed)
        for block in range(1, repeats + 1):
            block_rows = [
                (sequence, candidate, seed)
                for sequence, row_block, candidate, seed in parsed_order
                if row_block == block
            ]
            block_rows.sort()
            if len(block_rows) != len(names) or {
                name for _, name, _ in block_rows
            } != set(names):
                reasons.append(f"block_{block}_candidate_set_invalid")
                continue
            if paired_seed_required:
                observed_seeds = {seed for _, _, seed in block_rows}
                if len(observed_seeds) != 1 or None in observed_seeds:
                    reasons.append(f"block_{block}_seed_not_paired")
                else:
                    block_seeds[block] = int(next(iter(observed_seeds)))
            for position, (_, candidate, _) in enumerate(block_rows):
                position_counts[candidate][position] += 1
        if (
            paired_seed_required
            and seed_policy.get("distinct_across_blocks") is True
            and len(set(block_seeds.values())) != repeats
        ):
            reasons.append("block_seeds_not_distinct")

    position_balance = False
    if names and repeats > 0 and repeats % len(names) == 0 and not any(
        reason.startswith("block_") or reason.startswith("plan_") or reason.startswith("invalid_plan")
        for reason in reasons
    ):
        expected_position_count = repeats // len(names)
        position_balance = all(
            counts == [expected_position_count] * len(names)
            for counts in position_counts.values()
        )
        if not position_balance:
            reasons.append("observed_plan_not_position_balanced")

    result_rows = results.get("runs", [])
    seen_sequences: set[int] = set()
    if not isinstance(result_rows, list) or len(result_rows) != expected_total:
        reasons.append("result_row_count_mismatch")
    if isinstance(result_rows, list):
        for row in result_rows:
            try:
                sequence = int(row["sequence"])
                block = int(row["block"])
                candidate = str(row["candidate"])
                seed = int(row["seed"]) if paired_seed_required else None
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_result_row")
                continue
            if sequence in seen_sequences:
                reasons.append("duplicate_result_sequence")
            seen_sequences.add(sequence)
            if expected_by_sequence.get(sequence) != (block, candidate, seed):
                reasons.append(f"result_plan_mismatch_sequence_{sequence}")
        if seen_sequences != set(expected_by_sequence):
            reasons.append("result_sequence_set_mismatch")

    return {
        "valid": not reasons,
        "position_balanced": position_balance,
        "paired_seed_required": paired_seed_required,
        "paired_seed_valid": (
            not paired_seed_required
            or (
                len(block_seeds) == repeats
                and not any("seed" in reason for reason in reasons)
            )
        ),
        "block_seeds": {str(block): seed for block, seed in sorted(block_seeds.items())},
        "reasons": sorted(set(reasons)),
        "expected_total_runs": expected_total,
        "observed_total_runs": len(result_rows) if isinstance(result_rows, list) else 0,
        "position_counts": position_counts,
    }


def _paired_comparison(winner: dict, alternative: dict, expected_blocks: int) -> dict:
    winner_by_block = {
        run["block"]: run
        for run in winner["runs"]
        if run["valid"] and run["metrics"] is not None
    }
    alternative_by_block = {
        run["block"]: run
        for run in alternative["runs"]
        if run["valid"] and run["metrics"] is not None
    }
    common_blocks = sorted(set(winner_by_block) & set(alternative_by_block))
    better = 0
    tied = 0
    for block in common_blocks:
        winner_rank = _single_run_rank(winner_by_block[block]["metrics"])
        alternative_rank = _single_run_rank(alternative_by_block[block]["metrics"])
        if winner_rank > alternative_rank:
            better += 1
        elif winner_rank == alternative_rank:
            tied += 1

    higher_better = (
        "no_fall",
        "survival_fraction",
        "healthy_fraction",
        "reference_to_q_corr",
    )
    lower_better = (
        "healthy_tracking_error_hf_mean_deg",
        "healthy_track_rms_mean_deg",
        "reference_to_q_abs_lag_s",
        "xy_max_drift_m",
        "healthy_jitter_mean_deg",
        "target_age_p95_s",
    )
    paired_improvements = {}
    for metric in (*higher_better, *lower_better):
        values = []
        for block in common_blocks:
            winner_value = winner_by_block[block]["metrics"].get(metric)
            alternative_value = alternative_by_block[block]["metrics"].get(metric)
            if winner_value is None or alternative_value is None:
                continue
            delta = (
                float(winner_value) - float(alternative_value)
                if metric in higher_better
                else float(alternative_value) - float(winner_value)
            )
            values.append(delta)
        summary = _summary(values)
        if summary is not None:
            paired_improvements[metric] = {
                **summary,
                "orientation": "positive_means_winner_better",
            }

    p_value = _exact_sign_p_one_sided(better, len(common_blocks))
    return {
        "winner": winner["name"],
        "alternative": alternative["name"],
        "paired_blocks": len(common_blocks),
        "winner_better_blocks": better,
        "tied_blocks": tied,
        "winner_worse_blocks": len(common_blocks) - better - tied,
        "direction_fraction": better / max(len(common_blocks), 1),
        "exact_sign_p_one_sided": p_value,
        "complete": len(common_blocks) == expected_blocks,
        "directional_75pct": (
            len(common_blocks) == expected_blocks
            and better >= math.ceil(0.75 * expected_blocks)
        ),
        "statistically_confirmed": (
            len(common_blocks) == expected_blocks
            and expected_blocks >= 8
            and p_value <= 0.05
        ),
        "paired_metric_improvements": paired_improvements,
    }


def _validate_pairing_context(
    candidate_summaries: list[dict],
    *,
    expected_blocks: int,
    scenario: str,
    source_index_tolerance: int = 2,
) -> dict:
    """Check that paired runs entered the measured motion at comparable source frames."""
    scene_assets: dict[str, dict] = {}
    for candidate in candidate_summaries:
        for run in candidate["runs"]:
            if run.get("valid") and isinstance(run.get("scene_robot_asset"), dict):
                scene_assets[
                    f"{candidate['name']}:block_{run.get('block')}"
                ] = run["scene_robot_asset"]
    scene_identities = {
        json.dumps(asset, sort_keys=True, ensure_ascii=True)
        for asset in scene_assets.values()
    }
    scene_asset_consistent = len(scene_identities) <= 1
    common_reasons = (
        [] if scene_asset_consistent else ["scene_robot_asset_mismatch_across_runs"]
    )
    if scenario != "v3_bvh":
        return {
            "valid": not common_reasons,
            "required": False,
            "source_index_tolerance": source_index_tolerance,
            "blocks": {},
            "scene_robot_assets": scene_assets,
            "scene_robot_asset_consistent": scene_asset_consistent,
            "reasons": common_reasons,
        }

    reasons: list[str] = list(common_reasons)
    block_details: dict[str, dict] = {}
    candidate_names = [candidate["name"] for candidate in candidate_summaries]
    for block in range(1, expected_blocks + 1):
        starts: dict[str, int] = {}
        for candidate in candidate_summaries:
            matches = [
                run
                for run in candidate["runs"]
                if run.get("block") == block and run.get("valid")
            ]
            if len(matches) != 1:
                continue
            value = _finite((matches[0].get("metrics") or {}).get("source_index_start"))
            if value is not None:
                starts[candidate["name"]] = int(value)
        spread = max(starts.values()) - min(starts.values()) if starts else None
        complete = set(starts) == set(candidate_names)
        comparable = complete and spread is not None and spread <= source_index_tolerance
        if not complete:
            reasons.append(f"block_{block}_missing_source_index_start")
        elif not comparable:
            reasons.append(f"block_{block}_source_index_spread={spread}")
        block_details[str(block)] = {
            "starts": starts,
            "spread": spread,
            "complete": complete,
            "comparable": comparable,
        }
    return {
        "valid": not reasons,
        "required": True,
        "source_index_tolerance": source_index_tolerance,
        "blocks": block_details,
        "scene_robot_assets": scene_assets,
        "scene_robot_asset_consistent": scene_asset_consistent,
        "reasons": reasons,
    }


def build_summary(results_path: pathlib.Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    plan_path = pathlib.Path(results["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo_root = pathlib.Path(plan["repo_root"]).resolve()
    expected_per_candidate = int(plan["repeats"])
    expected_free_seconds = float(plan["free_seconds"])
    legacy_deploy_binary = plan.get("deploy_binary")
    pinned_env = plan.get("pinned_env", {})
    expected_shared_artifacts = plan.get("shared_artifacts", {})
    expected_sony_repo = plan.get("sony_repo")
    expected_isaaclab_repository = plan.get("isaaclab_repository")
    expected_launch_mode = plan.get("runtime_launch_mode")
    expected_component_contract = plan.get("component_contract")
    unlock_alignment = plan.get("unlock_alignment", {})
    expected_unlock_source_index = (
        int(unlock_alignment["source_index_target"])
        if isinstance(unlock_alignment, dict)
        and unlock_alignment.get("enabled") is True
        and unlock_alignment.get("source_index_target") is not None
        else None
    )
    design_validation = _validate_design(plan, results)
    expected_seed_by_sequence = {
        int(row["sequence"]): int(row["seed"])
        for row in plan.get("order", [])
        if isinstance(row, dict) and "sequence" in row and "seed" in row
    }

    runs_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for run in results.get("runs", []):
        runs_by_candidate[str(run.get("candidate"))].append(run)

    candidate_summaries = []
    for raw_definition in plan["candidates"]:
        definition = dict(raw_definition)
        if "deploy_binary" not in definition and isinstance(legacy_deploy_binary, dict):
            definition["deploy_binary"] = legacy_deploy_binary
        name = definition["name"]
        run_rows = runs_by_candidate.get(name, [])
        valid_metrics: list[dict] = []
        run_summaries = []
        for row in run_rows:
            reasons = []
            npz_path = row.get("npz")
            sequence = row.get("sequence")
            expected_seed = (
                expected_seed_by_sequence.get(int(sequence))
                if isinstance(sequence, (int, np.integer))
                else None
            )
            if row.get("returncode") != 0:
                reasons.append(f"returncode={row.get('returncode')}")
            if not isinstance(npz_path, str) or not pathlib.Path(npz_path).is_file():
                reasons.append("missing_npz")

            report = None
            if not reasons:
                try:
                    report = jitter_report.load_report(npz_path)
                    reasons.extend(
                        _runtime_valid(
                            report,
                            repo_root,
                            expected_free_seconds=expected_free_seconds,
                            candidate_definition=definition,
                            pinned_env=pinned_env,
                            expected_seed=expected_seed,
                            expected_shared_artifacts=expected_shared_artifacts,
                            expected_sony_repo=expected_sony_repo,
                            expected_unlock_source_index=expected_unlock_source_index,
                            expected_isaaclab_repository=expected_isaaclab_repository,
                            expected_launch_mode=expected_launch_mode,
                            expected_component_contract=expected_component_contract,
                        )
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    reasons.append(f"report_error={exc}")
            metrics = (
                _run_metrics(report, expected_free_seconds=expected_free_seconds)
                if report is not None and not reasons
                else None
            )
            if metrics is not None:
                valid_metrics.append(metrics)
            run_summaries.append(
                {
                    "sequence": row.get("sequence"),
                    "block": row.get("block"),
                    "seed": row.get("seed"),
                    "deploy_binary_sha256": row.get("deploy_binary_sha256"),
                    "deploy_runtime_root": row.get("deploy_runtime_root"),
                    "npz": npz_path,
                    "valid": not reasons,
                    "invalid_reasons": reasons,
                    "metrics": metrics,
                    "scene_robot_asset": (
                        _get(report, "meta.scene_robot_asset")
                        if report is not None
                        else None
                    ),
                }
            )

        metric_names = (
            "fall_events",
            "no_fall",
            "survival_fraction",
            "healthy_fraction",
            "reference_to_q_abs_lag_s",
            "reference_to_q_signed_lag_s",
            "reference_to_q_corr",
            "target_age_p95_s",
            "healthy_jitter_mean_deg",
            "healthy_tracking_error_hf_mean_deg",
            "healthy_track_rms_mean_deg",
            "xy_max_drift_m",
            "source_index_start",
            "update_coverage",
            "packet_coverage",
        )
        aggregate = {
            metric: summary
            for metric in metric_names
            if (
                summary := _summary(
                    [
                        value
                        for run_metric in valid_metrics
                        if (value := run_metric.get(metric)) is not None
                    ]
                )
            )
            is not None
        }
        total = len(run_rows)
        valid = len(valid_metrics)
        observed_blocks = [
            int(run["block"])
            for run in run_summaries
            if isinstance(run.get("block"), (int, np.integer))
        ]
        expected_blocks = set(range(1, expected_per_candidate + 1))
        block_complete = (
            len(observed_blocks) == expected_per_candidate
            and set(observed_blocks) == expected_blocks
            and len(set(observed_blocks)) == expected_per_candidate
        )
        candidate_summaries.append(
            {
                **definition,
                "expected_runs": expected_per_candidate,
                "observed_runs": total,
                "valid_runs": valid,
                "valid_fraction": valid / max(expected_per_candidate, 1),
                "observed_blocks": sorted(observed_blocks),
                "complete": (
                    total == expected_per_candidate
                    and valid == expected_per_candidate
                    and block_complete
                ),
                "metrics": aggregate,
                "runs": run_summaries,
            }
        )

    eligible = [candidate for candidate in candidate_summaries if candidate["valid_runs"] > 0]
    ranking = sorted(eligible, key=_candidate_rank, reverse=True)
    winner = ranking[0]["name"] if ranking else None
    winner_row = ranking[0] if ranking else None
    all_candidates_complete = bool(candidate_summaries) and all(
        candidate["complete"] for candidate in candidate_summaries
    )
    pairing_context = _validate_pairing_context(
        candidate_summaries,
        expected_blocks=expected_per_candidate,
        scenario=str(plan["scenario"]),
    )
    pairwise_comparisons = []
    if winner_row is not None:
        pairwise_comparisons = [
            _paired_comparison(winner_row, alternative, expected_per_candidate)
            for alternative in candidate_summaries
            if alternative["name"] != winner_row["name"]
        ]
    all_directional = bool(pairwise_comparisons) and all(
        comparison["directional_75pct"] for comparison in pairwise_comparisons
    )
    all_statistically_confirmed = bool(pairwise_comparisons) and all(
        comparison["statistically_confirmed"] for comparison in pairwise_comparisons
    )
    confidence = "insufficient_valid_data"
    if winner_row is not None:
        complete_balanced = (
            design_validation["valid"]
            and design_validation["position_balanced"]
            and design_validation["paired_seed_valid"]
            and pairing_context["valid"]
            and all_candidates_complete
        )
        if len(candidate_summaries) < 2:
            confidence = "single_candidate_only"
        elif complete_balanced and all_statistically_confirmed:
            confidence = "confirmed"
        elif complete_balanced and expected_per_candidate >= 4 and all_directional:
            confidence = "balanced_directional_evidence"
        elif complete_balanced:
            confidence = "balanced_inconclusive"
        else:
            confidence = "provisional_screening"

    return {
        "schema_version": 2,
        "matrix_results": str(results_path.resolve()),
        "matrix_plan": str(plan_path.resolve()),
        "scenario": plan["scenario"],
        "isaaclab_repository": plan.get("isaaclab_repository"),
        "seed_policy": plan.get("seed_policy"),
        "comparison_scope": plan.get(
            "comparison_scope",
            {
                "unit": "candidate",
                "causal_attribution": True,
                "note": "Legacy plan with one globally pinned deploy binary.",
            },
        ),
        "selection_rule": [
            "valid run fraction (higher)",
            "no-fall run fraction (higher)",
            "normalized survival time (higher)",
            "healthy true-free fraction (higher)",
            "reference-to-q correlation (higher)",
            "healthy high-frequency tracking error (lower)",
            "healthy target tracking RMS (lower)",
            "reference-to-q absolute internal lag with corr>=0.50 (lower)",
            "XY max drift (lower)",
            "healthy body jitter mean (lower)",
            "target receive age p95 (lower; diagnostic tie-breaker)",
        ],
        "winner": winner,
        "confidence": confidence,
        "confirmation": {
            "design_validation": design_validation,
            "pairing_context": pairing_context,
            "all_candidates_complete": all_candidates_complete,
            "minimum_confirmatory_blocks": 8,
            "directional_evidence_rule": (
                "winner better than every alternative in at least 75% of paired blocks"
            ),
            "confirmation_rule": (
                "complete balanced design and one-sided exact sign p<=0.05 versus every alternative"
            ),
            "all_directional": all_directional,
            "all_statistically_confirmed": all_statistically_confirmed,
            "pairwise_comparisons": pairwise_comparisons,
        },
        "ranking": [candidate["name"] for candidate in ranking],
        "candidates": candidate_summaries,
    }


def _fmt_metric(candidate: dict, name: str, digits: int = 3) -> str:
    summary = candidate["metrics"].get(name)
    return "n/a" if not summary else f"{summary['mean']:.{digits}f}"


def print_summary(summary: dict) -> None:
    print(
        f"\n=== SONIC matrix summary: {summary['scenario']} ===\n"
        f"winner={summary['winner'] or '<none>'} confidence={summary['confidence']}"
    )
    print(
        f"{'candidate':<20}{'bundle':>14}{'valid':>9}{'no-fall':>10}{'survival':>11}"
        f"{'healthy':>10}{'corr':>8}{'trackHF':>10}{'|lag|(s)':>11}{'xy(m)':>9}"
    )
    by_name = {candidate["name"]: candidate for candidate in summary["candidates"]}
    ordered = summary["ranking"] + [
        name for name in by_name if name not in set(summary["ranking"])
    ]
    for name in ordered:
        candidate = by_name[name]
        deploy_binary = candidate.get("deploy_binary", {})
        bundle = str(deploy_binary.get("sha256", "legacy"))[:12]
        valid_text = f"{candidate['valid_runs']}/{candidate['expected_runs']}"
        print(
            f"{name:<20}"
            f"{bundle:>14}"
            f"{valid_text:>9}"
            f"{_fmt_metric(candidate, 'no_fall'):>10}"
            f"{_fmt_metric(candidate, 'survival_fraction'):>11}"
            f"{_fmt_metric(candidate, 'healthy_fraction'):>10}"
            f"{_fmt_metric(candidate, 'reference_to_q_corr'):>8}"
            f"{_fmt_metric(candidate, 'healthy_tracking_error_hf_mean_deg'):>10}"
            f"{_fmt_metric(candidate, 'reference_to_q_abs_lag_s'):>11}"
            f"{_fmt_metric(candidate, 'xy_max_drift_m'):>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path, help="matrix_results.json")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    results_path = args.results.resolve()
    output = args.out.resolve() if args.out else results_path.with_name("matrix_summary.json")
    summary = build_summary(results_path)
    _atomic_json(output, summary)
    print_summary(summary)
    print(f"SONIC_EVAL_MATRIX_SUMMARY={output}")
    return 0 if summary["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
