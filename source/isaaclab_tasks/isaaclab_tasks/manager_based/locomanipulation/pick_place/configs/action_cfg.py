# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction, AutoWalkAction, SONICWholeBodyAction


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
    """全身骨骼捕捉数据驱动的物理行走配置。

    点击 Play 后无需外部输入。内部用解析公式合成与 walking phase 同步的
    全身关节角度（腿+腰+手臂+手），通过物理引擎实现真实行走。
    """

    class_type: type[ActionTerm] = AutoWalkAction

    joint_names: list[str] = MISSING
    """需要驱动的关节名列表（建议包含全身：腿+腰+手臂+手）。不存在的关节会被跳过。"""

    walk_frequency: float = 0.8
    """步态频率（Hz），即每秒完成的完整步态周期数。"""

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


@configclass
class SONICWholeBodyActionCfg(ActionTermCfg):
    """GEAR-SONIC encoder-decoder 全身控制（最小骨架版）配置。

    当前 SONICWholeBodyAction 用 zero-fill 观测推理 ONNX，仅验证 IsaacLab → ONNX → 关节写入
    这条 pipeline。真实 SONIC 部署的 multi-frame history + motion reference + mode 切换
    留待下阶段补完。
    """

    class_type: type[ActionTerm] = SONICWholeBodyAction

    encoder_path: str = MISSING
    """SONIC encoder ONNX 路径。建议指向 gear_sonic_deploy/policy/release/model_encoder.onnx。"""

    decoder_path: str = MISSING
    """SONIC decoder ONNX 路径。建议指向 gear_sonic_deploy/policy/release/model_decoder.onnx。"""

    joint_names: list[str] = MISSING
    """29 个 G1 关节名，必须按 SONIC 训练顺序传入。

    传入 `list(SONIC_G1_29DOF_JOINT_ORDER)`（位于 mdp/actions.py）即可。
    """

    sonic_action_dim: int = 29
    """SONIC decoder 输出维度（固定 29）。"""

    action_scale: float = 0.25
    """SONIC 输出（joint_pos_rel 偏移量）→ 绝对关节目标的缩放因子。

    SONIC 训练时 IsaacLab 默认 scale=1.0，但 zero-fill 观测推理时 decoder 输出可达 ±2 rad，
    直接 ×1 会让机器人剧烈晃动。最小骨架阶段保守用 0.25；接入真实观测后可调回 1.0。
    """

    mocap_path: str = ""
    """walking mocap PKL 路径（joblib 格式，{motion_name: {root_rot, dof, ...}}）。

    用作 SONIC encoder 的 motion reference 源，替代 self-reference。
    第一版仅用 root_rot 给 anchor_orientation；后续可加 forward kinematics 算 body_pos。
    设为空字符串则 fallback 到 self-ref + identity anchor。
    建议路径：`D:/src/Isaac/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl`
    """

    probe_encoder_mode: int = 0
    """探针：强制 encoder_index 值。0 = g1（默认）、1 = teleop、2 = smpl。"""

    force_zero_body_pos: bool = False
    """探针：强制将 body_pos 字段清零，隔离 body_pos 对 absmax 的贡献。"""

    force_zero_last_action_history: bool = False
    """探针：强制 decoder 输入的 his_last_actions (offset 674:964) 清零，
    隔离 _last_action 累积反馈是否是 step 3+ 爆炸的根因。"""

    force_zero_decoder_history: bool = False
    """探针：强制 decoder 输入的全部 history (offset 64:994，含 base_ang_vel /
    joint_pos / joint_vel / last_actions / gravity_dir) 清零，
    仅保留 token_state (offset 0:64)。如果还爆，说明根因在 encoder token 本身。"""
