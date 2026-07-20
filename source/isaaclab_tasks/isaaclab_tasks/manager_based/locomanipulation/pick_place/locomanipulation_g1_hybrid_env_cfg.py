# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-G1 hybrid teleoperation: Pink upper body plus GR00T/MuJoCo lower body."""

import os
import re
from pathlib import Path

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter import (
    G1TriHandUpperBodyMotionControllerRetargeterCfg,
)
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import MuJoCoG1MirrorActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
)


_ENV_REF_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
_LOWER_BODY_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
]


def _expand_config_refs(values: dict[str, str]) -> dict[str, str]:
    expanded = dict(values)
    for _ in range(10):
        changed = False
        next_values: dict[str, str] = {}
        for key, value in expanded.items():
            next_value = _ENV_REF_RE.sub(
                lambda match: expanded.get(match.group(1) or match.group(2), ""),
                value,
            )
            next_values[key] = next_value
            changed |= next_value != value
        expanded = next_values
        if not changed:
            break
    return expanded


def _load_network_config() -> dict[str, str]:
    path = Path(__file__).resolve().parents[6] / "scripts/gr00t_wbc/g1_udp_network.env"
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
        values = _expand_config_refs(values)
        print(f"[INFO] Hybrid G1 config loaded: {path}")
    else:
        print(f"[WARN] Hybrid G1 config not found: {path}; using built-in network defaults.")
    return values


_NETWORK_CFG = _load_network_config()


def _cfg(name: str, default: str) -> str:
    return os.environ.get(name, _NETWORK_CFG.get(name, default))


def _robot_cfg(name: str, default: str) -> str:
    return _cfg(f"ISAACLAB_G1_1_{name}", _cfg(f"ISAACLAB_G1_{name}", default))


def _robot_int(name: str, default: int) -> int:
    try:
        return int(_robot_cfg(name, str(default)))
    except (TypeError, ValueError):
        return default


def _robot_float(name: str, default: float) -> float:
    try:
        return float(_robot_cfg(name, str(default)))
    except (TypeError, ValueError):
        return default


def _robot_bool(name: str, default: bool) -> bool:
    value = _robot_cfg(name, "1" if default else "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _lower_body_mirror_cfg() -> MuJoCoG1MirrorActionCfg:
    """Build robot-1 network mirroring with no ownership of upper-body joints."""

    sender_ip = _cfg("UBUNTU_ROBOT_1_SENDER_IP", "192.168.10.230")
    transport = _cfg("ISAACLAB_G1_TRANSPORT", "udp")
    return MuJoCoG1MirrorActionCfg(
        asset_name="robot",
        enabled=True,
        transport=transport,
        zmq_host=_robot_cfg("ZMQ_HOST", sender_ip),
        zmq_port=_robot_int("ZMQ_PORT", 5557),
        zmq_topic=_robot_cfg("ZMQ_TOPIC", "g1_debug"),
        zmq_joint_order=_robot_cfg("JOINT_ORDER", "mujoco"),
        zmq_pose_source=_robot_cfg("POSE_SOURCE", "auto"),
        state_write_pose_source=_robot_cfg("STATE_WRITE_POSE_SOURCE", "measured"),
        target_only_pose_source=_robot_cfg("TARGET_ONLY_POSE_SOURCE", "measured"),
        udp_bind_host=_robot_cfg("UDP_BIND_HOST", "0.0.0.0"),
        udp_port=_robot_int("UDP_PORT", 5557),
        udp_topic=_robot_cfg("UDP_TOPIC", "g1_1_debug"),
        udp_rcvbuf=_robot_int("UDP_RCVBUF", 262144),
        root_zmq_host=_robot_cfg("ROOT_ZMQ_HOST", sender_ip),
        root_zmq_port=_robot_int("ROOT_ZMQ_PORT", 5558),
        root_zmq_topic=_robot_cfg("ROOT_ZMQ_TOPIC", "g1_root"),
        root_udp_bind_host=_robot_cfg("ROOT_UDP_BIND_HOST", "0.0.0.0"),
        root_udp_port=_robot_int("ROOT_UDP_PORT", 5558),
        root_udp_topic=_robot_cfg("ROOT_UDP_TOPIC", "g1_1_root"),
        root_udp_rcvbuf=_robot_int("ROOT_UDP_RCVBUF", 262144),
        locomotion_sync_mode="custom",
        write_root_state=True,
        write_body_joint_state=True,
        write_hand_joint_state=False,
        use_source_joint_velocity=_robot_bool("USE_SOURCE_JOINT_VELOCITY", False),
        body_joint_target_max_delta=_robot_float("BODY_JOINT_TARGET_MAX_DELTA", 0.20),
        hold_default_until_first_packet=_robot_bool("HOLD_DEFAULT_UNTIL_FIRST_PACKET", True),
        no_packet_debug_interval_s=_robot_float("NO_PACKET_DEBUG_INTERVAL_S", 1.0),
        root_motion_mode=_robot_cfg("ROOT_MOTION_MODE", "source"),
        root_zmq_required=_robot_bool("ROOT_ZMQ_REQUIRED", False),
        root_position_mode=_robot_cfg("ROOT_POSITION_MODE", "relative"),
        root_orientation_mode=_robot_cfg("ROOT_ORIENTATION_MODE", "relative"),
        mirror_joint_names=list(_LOWER_BODY_JOINT_NAMES),
        body_state_write_joint_names=list(_LOWER_BODY_JOINT_NAMES),
        mirror_hands=False,
        controller_gripper_enabled=False,
        ground_lock=_robot_bool("GROUND_LOCK", False),
    )


@configclass
class ActionsCfg:
    """Pink owns waist/arms/hands; the network mirror owns root and legs."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG
    lower_body_mirror = _lower_body_mirror_cfg()


@configclass
class LocomanipulationG1HybridEnvCfg(LocomanipulationG1EnvCfg):
    """Single G1_29DOF environment with independent upper/lower-body controllers."""

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The inherited Agile observation group reads the removed action term
        # ``lower_body_joint_pos``. UDP mirroring does not run the Agile policy,
        # so this entire auxiliary observation group must stay disabled.
        self.observations.lower_body_policy = None

        # Keep the official single-robot VR anchor configured by the base class:
        # /World/envs/env_0/Robot/pelvis, fixed height, smoothed yaw following.
        self.teleop_devices = DevicesCfg(
            devices={
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyMotionControllerRetargeterCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

        print(
            "[INFO] Hybrid single-G1 control: robot=G1_29DOF_CFG, "
            "upper=Pink IK motion controllers, lower=GR00T/MuJoCo root+legs mirror, "
            f"transport={self.actions.lower_body_mirror.transport}, "
            f"body_port={self.actions.lower_body_mirror.udp_port}, "
            f"root_port={self.actions.lower_body_mirror.root_udp_port}, "
            f"XR anchor={self.xr.anchor_prim_path}"
        )
