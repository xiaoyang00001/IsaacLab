# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction, MuJoCoG1MirrorAction


@configclass
class AgileBasedLowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term that is based on Agile lower body RL policy."""

    class_type: type[ActionTerm] = AgileBasedLowerBodyAction
    """The class type for the lower body action term."""

    joint_names: list[str] = MISSING
    """The names of the joints to control."""

    obs_group_name: str = MISSING
    """The name of the observation group to use."""

    policy_path: str = MISSING
    """The path to the policy model."""

    policy_output_offset: float = 0.0
    """Offsets the output of the policy."""

    policy_output_scale: float = 1.0
    """Scales the output of the policy."""


@configclass
class MuJoCoG1MirrorActionCfg(ActionTermCfg):
    """Configuration for mirroring MuJoCo/SONIC G1 walking state into Isaac Lab."""

    class_type: type[ActionTerm] = MuJoCoG1MirrorAction

    enabled: bool = True
    """Whether to enable the ZMQ mirror. If no packets arrive, the action stays idle."""

    zmq_host: str = "192.168.10.230"
    """MuJoCo debug publisher host."""

    zmq_port: int = 5557
    """MuJoCo debug publisher port."""

    zmq_topic: str = "g1_debug"
    """MuJoCo debug publisher topic."""

    zmq_timeout: float = 0.5
    """Seconds before the last received ZMQ packet is considered stale."""

    zmq_joint_order: str = "mujoco"
    """Fallback joint order for incoming 29-DoF body joint vectors: ``mujoco`` or ``isaaclab``."""

    zmq_pose_source: str = "measured"
    """Which pose fields to mirror: ``measured``, ``target``, or ``auto``."""

    root_zmq: bool = True
    """Whether to also subscribe to a dedicated root-state stream."""

    root_zmq_host: str = "192.168.10.230"
    """Dedicated root-state publisher host."""

    root_zmq_port: int = 5558
    """Dedicated root-state publisher port."""

    root_zmq_topic: str = "g1_root"
    """Dedicated root-state publisher topic."""

    root_z_offset: float = 0.0
    """Additive offset applied to mirrored root height."""

    root_motion_mode: str = "source"
    """Root translation mode: ``source`` uses the dedicated root stream; ``auto``/``stance`` use foot fallback."""

    root_zmq_required: bool = True
    """Whether root motion must come from the dedicated root-state stream instead of falling back to debug packets."""

    root_position_mode: str = "relative"
    """Root position mapping: ``relative`` applies MuJoCo displacement to the Isaac start pose; ``absolute`` copies it."""

    root_debug_interval_s: float = 2.0
    """Seconds between root mirror status prints. Non-positive disables periodic root status prints."""

    source_root_motion_eps: float = 1.0e-3
    """Source root xy displacement threshold used by ``root_motion_mode='auto'``."""

    mirror_joint_names: list[str] = [
        ".*_hip_.*_joint",
        ".*_knee_joint",
        ".*_ankle_.*_joint",
        "waist_.*_joint",
        ".*_shoulder_.*_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
    ]
    """Regex list of 29-DoF MuJoCo body joints to mirror into Isaac Lab."""

    mirror_hands: bool = True
    """Whether to mirror hand joints from MuJoCo."""

    controller_gripper_enabled: bool = True
    """Whether the action consumes motion-controller gripper inputs for the G1 hands.

    When enabled, the action dimension is four:
    ``[left_index, left_middle, right_index, right_middle]``.
    """

    controller_gripper_finger_close_angle: float = 1.0
    """Maximum index/middle finger close angle in radians at full trigger/grip press."""

    controller_gripper_thumb_yaw_angle: float = 0.5
    """Maximum thumb base yaw offset in radians used to bias the thumb toward the active finger."""

    controller_gripper_thumb_1_angle: float = 0.4
    """Maximum thumb middle joint close angle in radians."""

    controller_gripper_thumb_2_angle: float = 0.7
    """Maximum thumb tip joint close angle in radians."""

    controller_gripper_action_alpha: float = 0.65
    """Low-pass smoothing factor applied to incoming controller gripper commands."""

    foot_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    """Foot bodies used for stance-root estimation and ground locking."""

    ground_lock: bool = True
    """Whether to keep the mirrored feet at or above the initial standing clearance."""

    ground_height: float = 0.0
    """World z height of the ground."""

    ground_lock_clearance: float = -1.0
    """Minimum foot-body z above ground. Negative means infer from the default pose."""

    stance_foot_height_tolerance: float = 0.045
    """Foot-body height tolerance above standing clearance for support-foot root estimation."""

    stance_foot_switch_margin: float = 0.015
    """Height margin for switching support foot during stance-root estimation."""

    stance_root_max_step: float = 0.035
    """Maximum xy correction per physics step for stance-root estimation. Non-positive disables clamping."""
