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
    """模拟全身骨骼捕捉数据驱动的行走配置。

    点击 Play 后无需外部输入。内部用解析公式合成与 walking phase 同步的
    全身关节角度（腿+腰+手臂+手），模拟一个本地 mocap 流。
    """

    class_type: type[ActionTerm] = AutoWalkAction

    joint_names: list[str] = MISSING
    """需要驱动的关节名列表（建议包含全身：腿+腰+手臂+手）。不存在的关节会被跳过。"""

    forward_speed: float = 0.3
    """行走线速度（m/s）。"""

    walk_frequency: float = 0.8
    """步态频率（Hz），即每秒完成的完整步态周期数。"""

    body_bob_amplitude: float = 0.015
    """躯干竖向起伏幅度（m），模拟双腿支撑/单腿支撑时的重心高度变化。"""

    # ── 腿部 ────────────────────────────────────────────────
    hip_pitch_amplitude: float = 0.25
    """髋关节俯仰摆动幅度（rad）。"""

    knee_amplitude: float = 0.30
    """膝关节弯曲幅度（rad）。"""

    ankle_pitch_amplitude: float = 0.12
    """踝关节俯仰补偿幅度（rad）。"""

    # ── 手臂（摆臂） ────────────────────────────────────────
    arm_swing_amplitude: float = 0.35
    """肩关节俯仰前后摆动幅度（rad）。与同侧腿 180° 反相（腿前迈/臂后摆）。"""

    elbow_bend_amplitude: float = 0.15
    """肘关节随摆臂的弯曲幅度（rad）。"""

    # ── 腰部 ────────────────────────────────────────────────
    waist_yaw_amplitude: float = 0.06
    """腰关节 yaw 反向扭转幅度（rad），增加自然感。"""

    waist_roll_amplitude: float = 0.05
    """腰关节 roll 侧倾幅度（rad），模拟行走时的重心转移。"""

    # ── 髋部 ────────────────────────────────────────────────
    hip_yaw_amplitude: float = 0.03
    """髋关节 yaw 旋转幅度（rad），与腰部协同产生骨盆旋转。"""

    # ── 手部 ────────────────────────────────────────────────
    hand_curl_amount: float = 0.10
    """手指关节相对默认位置的轻微卷曲（rad），仿真放松握拳姿态。设为 0 关闭。"""
