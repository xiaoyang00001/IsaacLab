# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction, AutoWalkAction


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


@configclass
class AutoWalkActionCfg(ActionTermCfg):
    """第三个机器人自动行走的动作配置。点击 Play 后无需任何外部输入即可行走。"""

    class_type: type[ActionTerm] = AutoWalkAction

    joint_names: list[str] = MISSING
    """下半身关节名称列表，必须包含左右髋俯仰、膝、踝关节。"""

    forward_speed: float = 0.3
    """行走线速度（m/s）。"""

    walk_frequency: float = 0.8
    """步态频率（Hz），即每秒完成的完整步态周期数。"""

    hip_pitch_amplitude: float = 0.25
    """髋关节俯仰摆动幅度（rad）。"""

    knee_amplitude: float = 0.30
    """膝关节弯曲幅度（rad）。"""

    ankle_pitch_amplitude: float = 0.12
    """踝关节俯仰补偿幅度（rad）。"""
