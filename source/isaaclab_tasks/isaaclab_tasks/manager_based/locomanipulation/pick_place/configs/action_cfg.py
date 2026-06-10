# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import (
    AgileBasedLowerBodyAction,
    AutoWalkAction,
    SonicDeployTargetAction,
    SonicRobotStatePublisherAction,
    SONICWholeBodyAction,
    UnitreeDdsLowCmdAction,
    UnitreeLowStatePublisherAction,
)


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
class SonicDeployTargetActionCfg(ActionTermCfg):
    """Minimal GR00T deploy target receiver for sonic_robot."""

    class_type: type[ActionTerm] = SonicDeployTargetAction

    joint_names: list[str] = MISSING
    """29 G1 joint names in IsaacLab/SONIC order."""

    endpoint: str = "tcp://127.0.0.1:5557"
    """GR00T deploy ZMQ PUB endpoint."""

    topic: str = "g1_debug"
    """Topic prefix published by GR00T deploy."""

    target_field: str = "last_action"
    """Msgpack field to consume as the 29-DoF joint target.

    Direct GR00T/SONIC deploy publishes ``last_action`` as the scaled motor
    target with default offsets. ``body_q_target`` is the motion/reference
    visualization target and is kept as a fallback for simple proxy tools.
    """

    target_order: str = "mujoco"
    """Input joint order. GR00T deploy/debug joint arrays are in MuJoCo order."""

    target_rate_limit_rad_per_step: float = 0.16
    """Optional per-step target clamp to reduce abrupt deploy/sim startup jumps. 0 disables it."""

    stabilize_root_pose: bool = True
    """Hold the robot root at the post-reset pose while validating deploy joint targets."""

    lock_root_z: bool = True
    """When ``stabilize_root_pose`` is active, whether to also lock root Z.
    Set False in physics mode so PhysX can settle the robot to the correct ground height."""

    startup_settle_steps: int = 0
    """Number of steps after reset/unlock to hold the default pose before consuming deploy targets.
    In physics mode this lets PhysX settle the robot to the ground before joint tracking begins.
    Set to 0 to disable (default for fixed-root; ~50 for physics mode)."""

    stale_timeout_s: float = 0.5
    """Warn and hold the last target if no fresh deploy packet arrives for this long. 0 disables warning."""

    fallback_to_last_action: bool = False
    """If the preferred target field is absent, optionally consume deploy last_action."""

    fallback_to_body_q_target: bool = True
    """If the preferred target field is absent, optionally consume body_q_target."""

    fallback_to_measured: bool = False
    """If target fields are absent, optionally fall back to measured body_q fields."""

    reference_target_field: str = "body_q_target"
    """Optional deploy reference field used for lower-body and waist visualization."""

    blend_reference_lower_body: bool = True
    """Use reference_target_field for legs and waist while keeping target_field for arms."""

    hold_last_reference_target: bool = True
    """Hold the last valid lower-body reference when deploy sends an empty reference frame."""

    follow_base_yaw_target: bool = True
    """Rotate the fixed root yaw from deploy base_quat_target."""

    follow_base_translation_target: bool = True
    """Move the fixed root XY from deploy base_trans_target for visual walking."""

    base_quat_target_field: str = "base_quat_target"
    """Msgpack quaternion field used by follow_base_yaw_target."""

    base_trans_target_field: str = "base_trans_target"
    """Msgpack translation field used by follow_base_translation_target."""

    base_yaw_rate_limit_rad_per_step: float = 0.12
    """Optional per-step root yaw clamp. 0 disables it."""

    base_translation_rate_limit_m_per_step: float = 0.08
    """Optional per-step root XY clamp. 0 disables it."""

    base_translation_scale: float = 2.0
    """Scale applied to deploy base_trans_target XY deltas."""

    follow_base_height_target: bool = False
    """Lower/raise the fixed root Z from deploy base_trans_target so a squat shows as the body sinking instead of the feet lifting."""

    base_height_rate_limit_m_per_step: float = 0.05
    """Optional per-step root Z clamp. 0 disables it."""

    base_height_scale: float = 1.0
    """Scale applied to deploy base_trans_target Z deltas."""

    keep_feet_on_ground: bool = False
    """Lower the fixed root when the knees bend so a squat shows as the body sinking instead of the feet lifting. Driven by knee-joint angle (stable and bounded), not world foot positions (which get corrupted by floor contact in fixed-root mode)."""

    foot_ground_scale: float = 0.35
    """Meters the fixed root sinks per radian of average knee bend beyond the standing pose."""

    max_squat_drop_m: float = 0.45
    """Maximum root sink (m) from squat compensation, clamped to avoid going through the floor."""

    synthetic_base_motion_from_lower_body: bool = True
    """Generate visual root XY motion from leg activity when base_trans_target is static."""

    synthetic_base_motion_gain: float = 0.35
    """Meters of visual root travel per radian of lower-body target change."""

    synthetic_base_motion_deadzone: float = 0.002
    """Mean lower-body target delta below this value does not move the visual root."""

    synthetic_base_motion_max_step_m: float = 0.035
    """Maximum synthetic root translation per apply step."""

    debug_log_interval: int = 50
    """Print target statistics every N control steps. 0 disables periodic logging."""


@configclass
class UnitreeDdsLowCmdActionCfg(ActionTermCfg):
    """Unitree DDS low-level sim bridge for sonic_robot."""

    class_type: type[ActionTerm] = UnitreeDdsLowCmdAction

    joint_names: list[str] = MISSING
    """29 G1 joint names in IsaacLab/SONIC order."""

    domain_id: int = 0
    """DDS domain id used by Unitree SDK2."""

    network_interface: str = ""
    """Optional DDS network interface name. Leave empty to let CycloneDDS choose."""

    lowcmd_topic: str = "rt/lowcmd"
    """Unitree low-level command topic to subscribe."""

    lowstate_topic: str = "rt/lowstate"
    """Unitree low-level state topic to publish."""

    secondary_imu_topic: str = "rt/secondary_imu"
    """Unitree torso IMU topic to publish."""

    target_order: str = "mujoco"
    """LowCmd motor order. Unitree G1 lowcmd uses hardware/MuJoCo order."""

    target_rate_limit_rad_per_step: float = 0.08
    """Optional per-step target clamp to reduce abrupt deploy/sim startup jumps. 0 disables it."""

    stabilize_root_pose: bool = True
    """Hold the robot root at the post-reset pose while the DDS state bridge is still being validated."""

    stale_timeout_s: float = 0.5
    """Warn and hold the last command if no fresh LowCmd arrives for this long. 0 disables warning."""

    publish_lowstate_every_apply: bool = True
    """Publish LowState from IsaacLab at every action apply call."""

    mode_machine: int = 5
    """G1 mode_machine value reported in LowState so deploy can identify the robot variant."""

    debug_log_interval: int = 50
    """Print DDS bridge statistics every N control steps. 0 disables periodic logging."""


@configclass
class UnitreeLowStatePublisherActionCfg(ActionTermCfg):
    """Publish sonic_robot state on Unitree DDS rt/lowstate without consuming commands.

    Use this alongside the default ZMQ ``SonicDeployTargetActionCfg`` so GR00T/SONIC
    deploy can read IsaacLab's simulated robot state while joint targets are still driven
    over ZMQ. In DDS transport mode the lowstate is already published by
    ``UnitreeDdsLowCmdActionCfg``, so this term is not needed there.
    """

    class_type: type[ActionTerm] = UnitreeLowStatePublisherAction

    asset_name: str = "sonic_robot"
    """Articulation whose state is mirrored onto rt/lowstate."""

    joint_names: list[str] = MISSING
    """29 G1 joint names in IsaacLab/SONIC order."""

    domain_id: int = 0
    """DDS domain id used by Unitree SDK2."""

    network_interface: str = ""
    """Optional DDS network interface name. Leave empty to let CycloneDDS choose."""

    lowstate_topic: str = "rt/lowstate"
    """Unitree low-level state topic to publish."""

    secondary_imu_topic: str = "rt/secondary_imu"
    """Unitree torso IMU topic to publish."""

    publish_secondary_imu: bool = True
    """Also publish the torso IMU on secondary_imu_topic."""

    target_order: str = "mujoco"
    """Output motor order. Unitree G1 lowstate uses hardware/MuJoCo order."""

    mode_machine: int = 5
    """G1 mode_machine value reported in LowState so deploy can identify the robot variant."""

    debug_log_interval: int = 100
    """Print publish statistics every N control steps. 0 disables periodic logging."""


@configclass
class SonicRobotStatePublisherActionCfg(ActionTermCfg):
    """Publish simulated ``sonic_robot`` state over ZMQ/msgpack for the C++ LowState bridge."""

    class_type: type[ActionTerm] = SonicRobotStatePublisherAction

    asset_name: str = "sonic_robot"
    """Articulation whose state is published."""

    joint_names: list[str] = MISSING
    """29 G1 joint names in IsaacLab/SONIC order."""

    bind_endpoint: str = "tcp://127.0.0.1:5560"
    """ZMQ PUB bind endpoint consumed by ``sonic_unitree_lowstate_cpp_proxy``."""

    topic: str = "sonic_state"
    """Topic prefix for state packets."""

    target_order: str = "mujoco"
    """Output motor order. Unitree G1 lowstate uses hardware/MuJoCo order."""

    mode_machine: int = 5
    """G1 mode_machine value forwarded to the C++ LowState bridge."""

    debug_log_interval: int = 100
    """Print publish statistics every N control steps. 0 disables periodic logging."""


@configclass
class SONICWholeBodyActionCfg(ActionTermCfg):
    """GEAR-SONIC encoder-decoder 全身追踪控制配置。

    默认走确定性部署路径：mocap reference → encoder token → decoder action mean。
    探针用的 noise / random reset 保留为显式开关，不作为默认运行路径。
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
    """SONIC per-joint deploy action scale 外层的全局倍率。

    实际关节目标为 `default + raw_action * per_joint_action_scale * action_scale`。
    `per_joint_action_scale` 对齐 SONIC deploy 的 `0.25 * effort / stiffness`，
    本字段默认在任务配置中设为 1.0，仅作为后续整体降幅实验入口。
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

    obs_noise_enabled: bool = False
    """B1: 推理时给 obs history 注入 AdditiveUniformNoise，匹配训练时分布。

    训练 sonic_release/config.yaml policy obs 配置：
      - joint_pos:    n_min=-0.01,  n_max=0.01
      - joint_vel:    n_min=-0.5,   n_max=0.5
      - base_ang_vel: n_min=-0.2,   n_max=0.2
      - gravity_dir:  n_min=-0.05,  n_max=0.05
      - last_action:  无 noise

    推理无 noise 时 obs 比训练分布更"干净"，SONIC 学到"对 noise robust 的特征"
    在无 noise 输入上反 OOD，可能是 closed-loop 反馈循环根因之一。"""

    obs_noise_joint_pos: float = 0.01
    """joint_pos history noise 半幅 (rad)。"""

    obs_noise_joint_vel: float = 0.5
    """joint_vel history noise 半幅 (rad/s)。"""

    obs_noise_base_ang_vel: float = 0.2
    """base_ang_vel history noise 半幅 (rad/s)。"""

    obs_noise_gravity_dir: float = 0.05
    """gravity_dir history noise 半幅 (单位向量)。"""

    action_noise_enabled: bool = False
    """探针：在 ONNX raw action 上叠加 Normal(0, action_noise_std) 噪声后写关节。

    这会直接把随机关节位置偏移写入 PD target，部署时默认关闭。"""

    action_noise_std: float = 0.40
    """B2b: action noise 标准差 (scalar fallback)。

    取 ckpt 中 `std: (29,)` 中位数 ≈ 0.40。若 `action_noise_std_path` 非空则被覆盖。"""

    action_noise_std_path: str = ""
    """B2b-iter: per-joint std (29,) .npy 文件路径，覆盖 scalar `action_noise_std`。

    Why: B2b scalar=0.40 让腿部动起来但 r_arm absmax 仍 17~20（晃动倒下）。
    extract_sonic_action_std.py 提取 ckpt 真实 std (29,)：
      index 0-11 (legs) ≈ 0.30~0.40, 12-14 (waist) ≈ 0.42, 15-21 (l_arm) ≈ 0.31~0.41,
      22-24 (r_arm 前 3) ≈ 0.35~0.42, 25-28 (r_arm 末端 4) = 0.50（训练未有效更新，保持初始化）
    r_arm 末端 std 0.50 > scalar 0.40 → scalar 加得不够，per-joint 应能进一步压制 OOD。

    生产路径：source/isaaclab_tasks/.../pick_place/data/sonic_action_std_29d.npy"""

    reset_to_random_mocap_frame: bool = False
    """B3: reset 时把 robot 同步到 mocap[t_random] 而不是 mocap[0]。

    Why: 训练 motion_lib reset 在 episode 开始时从 motion 序列里 *random sample* 一帧设
    robot，让 actor 见到全帧分布；当前仅 mocap[0] 让推理时 obs 分布偏向 walking start，
    decoder 见过的训练分布是均匀全帧 → OOD。B2b/B2b-iter 已证 noise 不是全部根因，
    reset 帧分布对齐是下一个"硬"训练对齐项。

    第一版：所有 env 共享同一 random frame（避免 per-env 复杂度）。mocap 推进指针
    同步更新到该帧，确保 encoder ref 与 robot 起点一致。"""

    reset_mocap_frame: int = 0
    """确定性 reset 使用的 mocap 帧。`reset_to_random_mocap_frame=True` 时会忽略此值。"""

    loop_mocap: bool = False
    """是否把 mocap 当成首尾无缝循环片段播放。

    BVH/RAYNOS 这类离线转换片段通常不是 seamless loop，尾帧接第 0 帧会产生 root/reference
    突变，导致 SONIC reference 相位跳变后摔倒；默认按 finite clip clamp 到末尾。
    """

    seed_history_from_reset_pose: bool = True
    """reset 后用当前 mocap 姿态预热 decoder 10 帧 history。

    这样第一帧 decoder 看到的 proprioception 与实际 robot 起点一致，而不是 9 帧全零 + 1 帧当前值。"""

    align_root_height_to_mocap: bool = True
    """reset 同步 mocap 姿态时，把 root Z 按 mocap 相对第 0 帧的高度差轻微对齐。"""

    follow_mocap_root_xy: bool = False
    """诊断开关：每个 control step 将 root XY 对齐到 mocap root trajectory。

    默认关闭。用于验证 mocap 片段、encoder reference 和上/下肢 target 是否能组成完整可视化
    步态；不作为默认物理行走方案。"""

    follow_mocap_root_xy_rate_limit_mps: float = 2.0
    """`follow_mocap_root_xy=True` 时 root XY 每秒最大位移速度。小于等于 0 表示不限制。"""

    follow_mocap_root_z: bool = False
    """诊断开关：跟随 mocap root Z。通常只和 full root pose replay 一起打开。"""

    follow_mocap_root_rot: bool = False
    """诊断开关：跟随 mocap root rotation。通常只和 full root pose replay 一起打开。"""

    startup_blend_steps: int = 25
    """reset 后前 N 个 control step 将 SONIC target 从当前 reset target 平滑过渡出来。

    当前 sim/control 频率约 50Hz，因此 25 step 约等于 0.5s。用于压制 step 0 到
    step 50 的上肢启动瞬态；设为 0 可关闭。"""

    target_rate_limit_rad_per_step: float = 0.0
    """全身关节 target 每步最大变化量 (rad/step)。0 表示不限制全身。"""

    upper_body_target_rate_limit_rad_per_step: float = 0.06
    """shoulder / elbow / wrist 关节 target 每步最大变化量 (rad/step)。"""

    wrist_target_rate_limit_rad_per_step: float = 0.04
    """wrist 关节 target 每步最大变化量 (rad/step)，用于优先压制 wrist yaw 残余甩动。"""

    upper_body_mocap_target_blend: float = 0.0
    """shoulder / elbow / wrist 关节向 mocap DoF target 混合的比例。0 表示纯 SONIC target。"""

    wrist_mocap_target_blend: float = 0.0
    """wrist 关节向 mocap DoF target 混合的比例；会覆盖 upper body 的较小值。"""
