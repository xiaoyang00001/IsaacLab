# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-G1 hybrid teleoperation: GR00T/Pink upper body plus Agile locomotion."""

import os
import re
from pathlib import Path

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.groot_wrist_openxr_gripper_device import (
    G1GrootWristOpenXRGripperDeviceCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
)


_ENV_REF_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")
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


def _int(name: str, default: int) -> int:
    try:
        return int(_cfg(name, str(default)))
    except (TypeError, ValueError):
        return default


def _upper_body_ik_cfg():
    cfg = G1_UPPER_BODY_IK_ACTION_CFG.copy()
    cfg.input_poses_are_base_relative = True
    return cfg


@configclass
class ActionsCfg:
    """Pink owns waist/arms/hands; Agile owns the complete locomotion path."""

    upper_body_ik = _upper_body_ik_cfg()
    lower_body_joint_pos = AgileBasedLowerBodyActionCfg(
        asset_name="robot",
        joint_names=[
            ".*hip.*_joint",
            ".*_knee_joint",
            ".*ankle.*_joint",
        ],
        policy_output_scale=0.25,
        obs_group_name="lower_body_policy",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt",
    )


@configclass
class LocomanipulationG1HybridEnvCfg(LocomanipulationG1EnvCfg):
    """Single G1_29DOF environment with independent upper/lower-body controllers."""

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Match the UDP scene's XR behavior, adapted from Robot_N to the single Robot.
        local_robot_prim_path = "/World/envs/env_0/Robot"
        self.xr.anchor_pos = (0.0, 0.0, 0.0)
        self.xr.anchor_rot = (1.0, 0.0, 0.0, 0.0)
        self.xr.anchor_prim_path = f"{local_robot_prim_path}/head_link"
        self.xr.anchor_rotation_prim_path = f"{local_robot_prim_path}/pelvis"
        self.xr.fixed_anchor_height = False
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        self.xr.recenter_yaw_button = ("/user/hand/right", "b")
        self.xr.recenter_yaw_button_event = "release"
        self.xr.recenter_anchor_forward_axis = (-1.0, 0.0, 0.0)
        self.xr.recenter_headset_forward_axis = (0.0, -1.0, 0.0)
        self.xr.recenter_headset_fallback_axis = (1.0, 0.0, 0.0)

        self.teleop_devices = DevicesCfg(
            devices={
                "motion_controllers": G1GrootWristOpenXRGripperDeviceCfg(
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                    cache_key="robot",
                    wrist_zmq_host=_cfg("ISAACLAB_G1_WRIST_ZMQ_HOST", _cfg("UBUNTU_ROBOT_1_SENDER_IP", "192.168.10.230")),
                    wrist_zmq_port=_int("ISAACLAB_G1_WRIST_ZMQ_PORT", 5556),
                    wrist_zmq_topic=_cfg("ISAACLAB_G1_WRIST_ZMQ_TOPIC", "pose"),
                    wrist_timeout_s=0.5,
                    debug_interval_s=1.0,
                    input_deadzone=0.04,
                    full_press_threshold=0.85,
                ),
            }
        )

        print(
            "[INFO] Hybrid single-G1 control: robot=G1_29DOF_CFG, "
            "upper=GR00T wrists + OpenXR buttons through Pink IK, "
            "lower=Agile locomotion policy (hips+knees+ankles), "
            f"policy={self.actions.lower_body_joint_pos.policy_path}, "
            f"wrist=tcp://{self.teleop_devices.devices['motion_controllers'].wrist_zmq_host}:"
            f"{self.teleop_devices.devices['motion_controllers'].wrist_zmq_port}/"
            f"{self.teleop_devices.devices['motion_controllers'].wrist_zmq_topic}, "
            f"XR anchor={self.xr.anchor_prim_path}, "
            f"XR rotation anchor={self.xr.anchor_rotation_prim_path}, "
            "gripper_inputs=trigger|grip"
        )
