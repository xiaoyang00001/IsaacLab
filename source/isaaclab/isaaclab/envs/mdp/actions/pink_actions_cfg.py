# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers.pink_ik import PinkIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import pink_task_space_actions


@configclass
class PinkInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for Pink inverse kinematics action term.

    This configuration is used to define settings for the Pink inverse kinematics action term,
    which is a inverse kinematics framework.
    """

    class_type: type[ActionTerm] = pink_task_space_actions.PinkInverseKinematicsAction
    """Specifies the action term class type for Pink inverse kinematics action."""

    pink_controlled_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by pink IK."""

    hand_joint_names: list[str] = MISSING
    """List of joint names or regular expression patterns that specify the joints controlled by hand retargeting."""

    controller: PinkIKControllerCfg = MISSING
    """Configuration for the Pink IK controller that will be used to solve the inverse kinematics."""

    enable_gravity_compensation: bool = True
    """Whether to compensate for gravity in the Pink IK controller."""

    target_eef_link_names: dict[str, str] = MISSING
    """Dictionary mapping task names to controlled link names for the Pink IK controller.

    This dictionary should map the task names (e.g., 'left_wrist', 'right_wrist') to the
    corresponding link names in the URDF that will be controlled by the IK solver.
    """

    enable_waist_yaw_assist: bool = False
    """Whether to apply a separate waist-yaw helper after the hand IK solve."""

    waist_yaw_joint_name: str = "waist_yaw_joint"
    """Joint name used for the separate waist-yaw helper."""

    waist_yaw_source: str = "hand"
    """Source for waist-yaw assist: ``hand`` or ``head``."""

    waist_yaw_task_indices: tuple[int, ...] = (0, 1)
    """Indices of frame tasks whose lateral target offsets drive the waist assist."""

    waist_yaw_primary_task_index: int | None = None
    """Optional fixed frame-task index that has priority for waist assist driving."""

    waist_yaw_lateral_axis: int = 1
    """Axis in the base-link frame used as the lateral offset signal."""

    waist_yaw_direction: float = 1.0
    """Sign applied to the lateral-to-yaw mapping."""

    waist_yaw_head_gain: float = 1.0
    """Gain applied when mapping headset yaw delta to waist-yaw assist."""

    waist_yaw_deadzone: float = 0.04
    """Small lateral offset deadzone before the waist starts turning."""

    waist_yaw_release_deadzone: float = 0.02
    """Smaller deadzone used to keep the waist helper from chattering near center."""

    waist_yaw_scale: float = 3.2
    """Scale factor mapping lateral hand offset to waist yaw in radians per meter."""

    waist_yaw_max_angle: float = 1.57
    """Maximum absolute waist-yaw assist angle in radians."""

    waist_yaw_signal_smoothing: float = 0.12
    """Low-pass smoothing applied to the lateral hand signal before waist mapping."""

    waist_yaw_turn_smoothing: float = 0.35
    """Blend factor used while the waist is actively turning toward the target."""

    waist_yaw_return_smoothing: float = 0.16
    """Blend factor used while the waist is slowly returning to center."""

    waist_yaw_max_step: float = 0.025
    """Maximum waist-yaw target change per control step in radians."""
