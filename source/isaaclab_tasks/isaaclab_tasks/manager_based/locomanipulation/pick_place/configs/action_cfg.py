# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction, G1GripperSyncAction, MuJoCoG1MirrorAction


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
    """Whether to enable the network mirror. If no packets arrive, the action stays idle."""

    transport: str = "zmq"
    """Network transport for mirror packets: ``zmq`` or ``udp``."""

    zmq_host: str = "192.168.10.230"
    """MuJoCo debug publisher host."""

    zmq_port: int = 5557
    """MuJoCo debug publisher port."""

    zmq_topic: str = "g1_debug"
    """MuJoCo debug publisher topic."""

    zmq_timeout: float = 0.5
    """Seconds before the last received network packet is considered stale."""

    zmq_joint_order: str = "mujoco"
    """Fallback joint order for incoming 29-DoF body joint vectors: ``mujoco`` or ``isaaclab``."""

    zmq_pose_source: str = "measured"
    """Which pose fields to mirror: ``measured``, ``target``, or ``auto``."""

    state_write_pose_source: str | None = None
    """Pose source for joints hard-written to PhysX state.

    ``None`` preserves the legacy behavior and uses :attr:`zmq_pose_source`. Set this to
    ``"measured"`` to mirror the MuJoCo lower-body state while other joints use targets.
    """

    target_only_pose_source: str | None = None
    """Pose source for joints driven through actuator targets instead of state writes.

    ``None`` preserves the legacy behavior and uses :attr:`zmq_pose_source`. Use
    ``"target"`` for reference-motion joint targets, or ``"action"`` for the final
    scaled WBC motor position command published as ``last_action``.
    """

    hand_pose_source: str | None = None
    """Pose source for mirrored hand joints; ``None`` uses :attr:`zmq_pose_source`."""

    locomotion_sync_mode: str = "mirror"
    """How MuJoCo locomotion is applied in Isaac Lab.

    ``mirror`` hard-syncs root and 29-DoF body joints for stable walking.
    ``hybrid`` hard-syncs root but drives body joints through actuator targets.
    ``physics`` drives body joints through actuator targets and leaves root to PhysX.
    ``custom`` honors the explicit ``write_*_state`` flags.
    """

    write_root_state: bool = True
    """Whether to write MuJoCo root pose/velocity directly into Isaac Lab."""

    write_body_joint_state: bool = True
    """Whether to write MuJoCo 29-DoF body joint position/velocity directly into Isaac Lab."""

    write_hand_joint_state: bool = False
    """Whether MuJoCo mirrored hand joints should also be written directly into Isaac Lab."""

    use_source_joint_velocity: bool = True
    """Whether to use MuJoCo joint velocity when writing or targeting mirrored joints."""

    body_joint_target_max_delta: float = 0.30
    """Maximum body drive position error in radians when body joints are not hard-written."""

    zero_target_only_body_velocity: bool = False
    """Whether to zero velocity targets for mirrored body joints that are driven only by actuator targets."""

    zero_target_only_hand_velocity: bool = True
    """Whether to zero velocity targets for mirrored hand joints that are driven only by actuator targets.

    Enabling this keeps noisy or contact-incompatible source hand velocities out of the damping term while the
    position target continues to mirror the source hand pose.
    """

    body_joint_target_scale_overrides: dict[str, float] | None = None
    """Regex-to-scale overrides applied to mirrored body joint position and velocity targets."""

    hand_joint_target_max_delta: float = 0.02
    """Maximum hand drive position error in radians when hand joints are not hard-written."""

    hold_default_until_first_packet: bool = True
    """Whether to hold the default standing pose until the first valid MuJoCo body packet arrives."""

    no_packet_debug_interval_s: float = 1.0
    """Seconds between warnings while waiting for the first valid MuJoCo body packet."""

    udp_bind_host: str = "0.0.0.0"
    """Local UDP address to bind for debug packets."""

    udp_port: int = 5557
    """Local UDP port for debug packets."""

    udp_topic: str = "g1_debug"
    """UDP debug packet topic prefix."""

    udp_rcvbuf: int = 262144
    """UDP receive socket ``SO_RCVBUF`` in bytes."""

    root_zmq: bool = True
    """Whether to also subscribe to a dedicated root-state stream."""

    root_zmq_host: str = "192.168.10.230"
    """Dedicated root-state publisher host."""

    root_zmq_port: int = 5558
    """Dedicated root-state publisher port."""

    root_zmq_topic: str = "g1_root"
    """Dedicated root-state publisher topic."""

    root_udp: bool = True
    """Whether to also receive a dedicated root-state UDP stream when ``transport='udp'``."""

    root_udp_bind_host: str = "0.0.0.0"
    """Local UDP address to bind for dedicated root-state packets."""

    root_udp_port: int = 5558
    """Local UDP port for dedicated root-state packets."""

    root_udp_topic: str = "g1_root"
    """Dedicated root-state UDP topic prefix."""

    root_udp_rcvbuf: int = 262144
    """Root-state UDP receive socket ``SO_RCVBUF`` in bytes."""

    root_z_offset: float = 0.0
    """Additive offset applied to mirrored root height."""

    root_motion_mode: str = "source"
    """Root translation mode: ``source`` uses the dedicated root stream; ``auto``/``stance`` use foot fallback."""

    root_zmq_required: bool = True
    """Whether root motion must come from the dedicated root-state stream instead of falling back to debug packets.

    This legacy field name applies to both ZMQ and UDP transports.
    """

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

    body_state_write_joint_names: list[str] | None = None
    """Regex list of mirrored body joints that may be written directly to PhysX state.

    When ``write_body_joint_state`` is true, only joints matching this list are hard-synced with
    ``write_joint_state_to_sim``. Mirrored joints that do not match are still driven through actuator targets.
    If this is ``None``, all mirrored body joints are hard-synced, preserving the legacy mirror behavior.
    Set this to an empty list to make all mirrored body joints target-only.
    """

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

    controller_gripper_target_max_delta: float = 0.20
    """Maximum per-step controller gripper target change in radians. Non-positive disables limiting."""

    controller_gripper_use_soft_limits: bool = True
    """Whether controller gripper targets are clamped to soft limits instead of hard joint limits."""

    controller_gripper_write_joint_state: bool = False
    """Whether controller gripper targets should also be written directly to the hand joint state."""

    controller_gripper_debug_interval_s: float = 0.0
    """Seconds between controller gripper debug prints. Non-positive disables periodic prints."""

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


@configclass
class G1GripperSyncActionCfg(ActionTermCfg):
    """Configuration for local OpenXR gripper control plus peer gripper synchronization."""

    class_type: type[ActionTerm] = G1GripperSyncAction

    enabled: bool = True
    """Whether this gripper action term is active."""

    mode: str = "local_publish"
    """Gripper sync mode: ``local_publish`` consumes actions and publishes, ``remote_subscribe`` receives peer state."""

    robot_id: int = 1
    """Robot ID represented by this action term."""

    transport: str = "zmq"
    """Transport for gripper synchronization. Currently only ``zmq`` is supported."""

    zmq_host: str = "127.0.0.1"
    """Publisher host for ``remote_subscribe`` mode. Ignored by ``local_publish`` mode."""

    zmq_port: int = 5571
    """ZMQ bind/connect port for the gripper stream."""

    zmq_topic: str = "g1_1_gripper"
    """ZMQ topic prefix for the gripper stream."""

    timeout: float = 0.5
    """Seconds before a remote gripper packet is considered stale. Stale packets hold the last pose."""

    publish_interval_s: float = 0.0
    """Minimum seconds between local gripper publishes. Non-positive publishes every apply step."""

    controller_gripper_finger_close_angle: float = 1.0
    """Maximum index/middle finger close angle in radians at full trigger/grip press."""

    controller_gripper_thumb_yaw_angle: float = 0.5
    """Maximum thumb base yaw offset in radians used to bias the thumb toward the active finger."""

    controller_gripper_thumb_1_angle: float = 0.4
    """Maximum thumb middle joint close angle in radians."""

    controller_gripper_thumb_2_angle: float = 0.7
    """Maximum thumb tip joint close angle in radians."""

    controller_gripper_action_alpha: float = 0.65
    """Low-pass smoothing factor applied to incoming local gripper commands."""

    controller_gripper_use_soft_limits: bool = True
    """Whether local gripper targets are clamped to soft limits instead of hard joint limits."""

    write_joint_state: bool = True
    """Whether gripper targets should also be written directly to joint state."""

    target_max_delta: float = 0.20
    """Maximum per-step gripper target change in radians when not directly writing joint state."""

    debug_interval_s: float = 0.0
    """Seconds between debug prints. Non-positive disables periodic prints."""
