# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction


@configclass
class AgileBasedLowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term used by robot A walking."""

    class_type: type[ActionTerm] = AgileBasedLowerBodyAction
    joint_names: list[str] = MISSING
    obs_group_name: str = MISSING
    policy_path: str = MISSING
    hip_height: float = 0.72
    policy_output_offset: float = 0.0
    policy_output_scale: float = 0.5
    action_smoothing: float = 0.2
    command_scale: float = 0.4
    stand_command_deadzone: float = 0.035
    enable_policy_when_moving: bool = False
    root_motion_deadzone: float = 0.01
    root_motion_scale: float = 1.0
    root_motion_smoothing: float = 0.25
    stabilize_root_pose: bool = True
    root_anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    root_anchor_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
