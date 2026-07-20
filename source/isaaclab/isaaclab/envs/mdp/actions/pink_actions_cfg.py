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

    relative_controller_targets: bool = False
    """Interpret incoming frame poses as controller poses relative to their first valid sample.

    When enabled, the action anchors each robot end-effector to its current pelvis-relative pose and applies
    subsequent controller motion as a delta. This avoids requiring the OpenXR and robot workspaces to share an
    absolute origin, and keeps targets attached to a moving robot base.
    """

    controller_position_scale: float = 1.0
    """Scale applied to relative controller translation before it is mapped to the robot base frame."""

    hand_action_alpha: float = 1.0
    """Low-pass smoothing factor for the hand-joint portion of the action."""

    hand_joint_target_max_delta: float = 0.0
    """Maximum hand target change from current joint position per action step; non-positive disables limiting."""

    hand_use_soft_limits: bool = True
    """Clamp hand targets to soft limits instead of hard articulation joint limits."""
