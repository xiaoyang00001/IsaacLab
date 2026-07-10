# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环物理调试专用极简场景。

py-spy 实测（2026-06-10）：完整 locomanipulation 场景 env_hz≈18.5（0.37× 实时），
开销构成 渲染 30% + 4 台机器人 actuator python 写入 24% + PhysX 17% + 杂项。
deploy 按墙钟 50Hz 推进步态相位，sim 必须接近实时行走才有意义。

本场景只保留：sonic_robot + 高摩擦 GroundPlane + 天光 + SONIC deploy/发布 action 项，
以及一个可关闭的轻量抱取演示物体（SONIC_SOLO_DEMO_OBJECT=0 关闭）。裁掉仓库 USD、
其余机器人、传送带、pick-place 全套机构。任务 id 含 "Locomanipulation" 以复用
teleop_se3_agent.py 的 deploy_target_mode 与 U 键回调。

移植说明（pickplace-g1-collision 基线）：主配置 ActionsCfg 承载的是镜像遥操/多人
同步动作，不再包含 SONIC deploy 项；SONIC 三个 action term（deploy target +
lowstate/state 发布）以本文件的 ``SonicSoloActionsCfg`` 为唯一定义来源，
SonicFullscene 直接导入复用，防止两场景漂移。机器人本体配置仍取
``_main.SONIC_G1_29DOF_CFG``（与 SONIC 训练对齐的 PD/armature 配方）。
抱取演示物体的配置（台座/纸箱/物理事件）同样以本文件的模块级常量为唯一来源。
"""

import os

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    SonicDeployTargetActionCfg,
    SonicRobotStatePublisherActionCfg,
    UnitreeDdsLowCmdActionCfg,
    UnitreeLowStatePublisherActionCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import (
    SONIC_G1_29DOF_JOINT_ORDER,
)

from . import locomanipulation_g1_env_cfg as _main

# 环境变量行为与主配置同源：直接复用 _main 的解析结果，保证 SONIC 机器人
# 本体（fix root / physics mode）与 action term 的模式判定永远一致。
_env_flag = _main._env_flag
SONIC_G1_FIX_ROOT = _main.SONIC_G1_FIX_ROOT
SONIC_G1_PHYSICS_MODE = _main.SONIC_G1_PHYSICS_MODE

# ---------------------------------------------------------------------------
# 抱取演示物体（移植自 sonic-hug-object-demo 分支 c7660b80f）
# 位置：SONIC 出生点 (-2.0, 11.008) 正前方 +X 1.05m、胸腰高度——solo 与
# fullscene 两场景的 SONIC 出生点同为主配置硬拷贝，故坐标可共用（fullscene
# 中该点位于行走通道，距传送带 y 向 4m+，无碰撞）。
# ---------------------------------------------------------------------------
_ENABLE_DEMO_OBJECT = _env_flag("SONIC_SOLO_DEMO_OBJECT", True)
_DEMO_OBJECT_POS = (-0.95, 11.008, 0.72)
_DEMO_STAND_POS = (-0.95, 11.008, 0.25)

# 台座：kinematic 深色方块，把纸箱垫到胸腰高度，机器人撞不动
HUG_BOX_STAND_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/HugBoxStand",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=_DEMO_STAND_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.CuboidCfg(
        size=(0.46, 0.56, 0.50),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.18, 0.18),
            roughness=0.85,
        ),
    ),
)

# 纸箱：复用主场景同款纸箱 USD，质量/阻尼在 prestartup 事件里调轻便于臂遥操
HUG_BOX_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/HugBox",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=_DEMO_OBJECT_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=3.0,
        ),
    ),
)

# USD 纸箱的刚体物理补齐（质量/阻尼/凸包碰撞），需 replicate_physics=False
# 才能对 /World/envs/env_{}/HugBox 逐 env 写 USD 属性
SETUP_HUG_BOX_PHYSICS_EVENT = EventTerm(
    func=locomanip_mdp.setup_usd_rigid_object_physics,
    mode="prestartup",
    params={
        "prim_path_template": "/World/envs/env_{}/HugBox",
        "mass": 0.8,
        "linear_damping": 3.0,
        "angular_damping": 0.4,
        "mesh_approximation": "convexHull",
    },
)


def build_sonic_xr_cfg() -> XrCfg:
    """按 SONIC_XR_VIEW 环境变量构造 XR 视角锚点配置（SonicSolo/SonicFullscene 共用）。

    - ``first``（默认）：头部第一视角。配方移植自 晓阳全身001 分支（head 锚 +
      朝向跟随）——注意本 g1.usd 里 head_link 嵌套在 torso_link 下（晓阳的
      GR00T 43dof USD 是根下 /Robot/head_link，路径不同，抄错会静默失效）。
      fixed_anchor_height=False 让高度也跟随头部；FOLLOW_PRIM_SMOOTHED 让房间
      朝向平滑跟随机器人转身（yaw-only，见 xr_anchor_utils.py）。
      位置锚点用 head_link，yaw 参考用 pelvis；启动拿到第一帧头显姿态后按
      晓阳全身001 的轴配置自动 recenter，修正头显视觉前向与机器人前向的 90 度差。
      若需要保留右手 B 键 release 手动 recenter，可设置 SONIC_XR_ENABLE_B_RECENTER=1；
      默认不绑定 B，避免 PICO/上游控制流同时消费该按键导致动作跳变。
      前提：SteamVR 驱动侧滤掉 HMD 平移（NOLO 驱动特性）。若 PICO 走标准串流
      不滤平移，佩戴者真实身高会叠加在 head_link 之上导致视点偏高，届时给
      anchor_pos 加负 Z 补偿（真机标定）。
    - ``third``：第三视角（2026-07-02 的 pelvis 方案）。OpenXR 房间地板原点
      对齐到机器人脚下（pelvis 下沉 -0.82），高度锁初始值、朝向 FIXED 不随
      转身，佩戴者以自身身高自由观察、走动。

    启动脚本 start_ubuntu_isaaclab_sonic.sh 用 SONIC_XR_VIEW=first|third 设置该变量。
    """
    view = os.environ.get("SONIC_XR_VIEW", "first").strip().lower()
    if view == "third":
        return XrCfg(
            anchor_pos=(0.0, 0.0, -0.82),
            anchor_rot=(1.0, 0.0, 0.0, 0.0),
            anchor_prim_path="/World/envs/env_0/SONICRobot/pelvis",
            fixed_anchor_height=True,
        )
    if view != "first":
        raise ValueError(
            f"SONIC_XR_VIEW={view!r} 无效，只支持 'first'（头部第一视角）或 'third'（pelvis 第三视角）。"
        )
    enable_b_recenter = _env_flag("SONIC_XR_ENABLE_B_RECENTER", False)
    return XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
        anchor_prim_path="/World/envs/env_0/SONICRobot/torso_link/head_link",
        anchor_rotation_prim_path="/World/envs/env_0/SONICRobot/pelvis",
        anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
        fixed_anchor_height=False,
        recenter_yaw_on_start=True,
        recenter_yaw_button=("/user/hand/right", "b") if enable_b_recenter else None,
        recenter_yaw_button_event="release",
        recenter_anchor_forward_axis=(-1.0, 0.0, 0.0),
        recenter_headset_forward_axis=(0.0, -1.0, 0.0),
        recenter_headset_fallback_axis=(1.0, 0.0, 0.0),
    )


@configclass
class SonicSoloSceneCfg(InteractiveSceneCfg):
    """极简场景：1 台 SONIC G1 + 地面 + 天光。"""

    sonic_robot: ArticulationCfg = _main.SONIC_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/SONICRobot"
    )

    # 高摩擦地面（μ=1.0 对齐 MuJoCo deploy 参考环境）
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="max",
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    if _ENABLE_DEMO_OBJECT:
        hug_box_stand = HUG_BOX_STAND_CFG
        hug_box = HUG_BOX_CFG


@configclass
class SonicSoloActionsCfg:
    """SONIC deploy target + 状态发布（两个 SONIC 场景的唯一定义来源）。

    Minimal bridge modes:
      SONIC_DEPLOY_TRANSPORT=zmq (default): GR00T deploy publishes debug body_q_target over ZMQ.
      SONIC_DEPLOY_TRANSPORT=dds: IsaacLab behaves like a virtual G1, subscribing rt/lowcmd
                                  and publishing rt/lowstate for GR00T deploy.
    """

    if os.environ.get("SONIC_DEPLOY_TRANSPORT", "zmq").lower() == "dds":
        sonic_wholebody = UnitreeDdsLowCmdActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            domain_id=int(os.environ.get("UNITREE_DDS_DOMAIN_ID", "0")),
            network_interface=os.environ.get("UNITREE_DDS_INTERFACE", ""),
            lowcmd_topic=os.environ.get("UNITREE_LOWCMD_TOPIC", "rt/lowcmd"),
            lowstate_topic=os.environ.get("UNITREE_LOWSTATE_TOPIC", "rt/lowstate"),
            secondary_imu_topic=os.environ.get("UNITREE_SECONDARY_IMU_TOPIC", "rt/secondary_imu"),
            target_order="mujoco",
            target_rate_limit_rad_per_step=0.08,
            stabilize_root_pose=_env_flag("SONIC_DEPLOY_STABILIZE_ROOT", SONIC_G1_FIX_ROOT),
            stale_timeout_s=0.5,
            publish_lowstate_every_apply=True,
            mode_machine=int(os.environ.get("UNITREE_G1_MODE_MACHINE", "5")),
            debug_log_interval=50,
        )
    else:
        sonic_wholebody = SonicDeployTargetActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            endpoint=os.environ.get("SONIC_DEPLOY_ENDPOINT", "tcp://127.0.0.1:5557"),
            topic=os.environ.get("SONIC_DEPLOY_TOPIC", "g1_debug"),
            target_field=os.environ.get("SONIC_DEPLOY_TARGET_FIELD", "last_action"),
            target_order="mujoco",
            target_rate_limit_rad_per_step=float(os.environ.get("SONIC_DEPLOY_TARGET_RATE_LIMIT", "0.16")),
            # 物理模式：解锁后旁路 rate limiter——软增益 policy 靠快甩目标偏置生成
            # 扭矩，slew limiter 在平衡环里是 100-250ms 人为迟滞（实测钉死 0.04 摔倒）
            rate_limit_only_while_root_locked=_env_flag(
                "SONIC_DEPLOY_RATE_LIMIT_ONLY_LOCKED", not SONIC_G1_FIX_ROOT
            ),
            stabilize_root_pose=_env_flag("SONIC_DEPLOY_STABILIZE_ROOT", SONIC_G1_FIX_ROOT),
            lock_root_z=SONIC_G1_FIX_ROOT,  # 物理模式放 Z 自由，让 PhysX settle 到正确地面高度
            startup_settle_steps=0 if SONIC_G1_FIX_ROOT else 50,  # 物理模式先 settle 再跟 deploy target
            # 物理模式 unlock 渐变释放（按物理步计数）。SONIC_DEPLOY_UNLOCK_BLEND_STEPS=0
            # 可做"瞬时交接"实验（最接近 MuJoCo eval 的自由根起始状态）
            unlock_blend_steps=int(
                os.environ.get("SONIC_DEPLOY_UNLOCK_BLEND_STEPS", "0" if SONIC_G1_FIX_ROOT else "50")
            ),
            hold_after_unlock=_env_flag("SONIC_DEPLOY_HOLD_AFTER_UNLOCK", False),  # 诊断：设1则unlock后保持站立不跟deploy
            # 摔倒自动恢复（对齐 MuJoCo base_sim.check_fall：root 高度 <0.2m 即自动
            # 扶正）。SONIC_DEPLOY_AUTO_RECOVER=0 恢复纯手动（J 键）；settle 后是否
            # 自动重新解锁由 SONIC_DEPLOY_AUTO_UNLOCK_AFTER_RECOVER 控制
            auto_recover_on_fall=_env_flag("SONIC_DEPLOY_AUTO_RECOVER", True),
            fall_root_height_m=float(os.environ.get("SONIC_DEPLOY_FALL_HEIGHT", "0.2")),
            auto_unlock_after_recover=_env_flag("SONIC_DEPLOY_AUTO_UNLOCK_AFTER_RECOVER", True),
            stale_timeout_s=0.5,
            fallback_to_last_action=True,
            fallback_to_body_q_target=True,
            reference_target_field=os.environ.get("SONIC_DEPLOY_REFERENCE_TARGET_FIELD", "body_q_target"),
            # 物理模式需要 policy 腿部平衡补偿，默认不 blend 掉
            blend_reference_lower_body=_env_flag("SONIC_DEPLOY_BLEND_REFERENCE_LOWER_BODY", SONIC_G1_FIX_ROOT),
            hold_last_reference_target=_env_flag("SONIC_DEPLOY_HOLD_LAST_REFERENCE", True),
            # 物理模式下 deploy base_trans/quat 非物理真实值，默认不跟随
            follow_base_yaw_target=_env_flag("SONIC_DEPLOY_FOLLOW_BASE_YAW", SONIC_G1_FIX_ROOT),
            follow_base_translation_target=_env_flag("SONIC_DEPLOY_FOLLOW_BASE_TRANSLATION", SONIC_G1_FIX_ROOT),
            base_quat_target_field=os.environ.get("SONIC_DEPLOY_BASE_QUAT_FIELD", "base_quat_target"),
            base_trans_target_field=os.environ.get("SONIC_DEPLOY_BASE_TRANS_FIELD", "base_trans_target"),
            base_yaw_rate_limit_rad_per_step=float(os.environ.get("SONIC_DEPLOY_BASE_YAW_RATE_LIMIT", "0.12")),
            base_translation_rate_limit_m_per_step=float(
                os.environ.get("SONIC_DEPLOY_BASE_TRANSLATION_RATE_LIMIT", "0.08")
            ),
            base_translation_scale=float(os.environ.get("SONIC_DEPLOY_BASE_TRANSLATION_SCALE", "2.0")),
            follow_base_height_target=_env_flag("SONIC_DEPLOY_FOLLOW_BASE_HEIGHT", False),
            base_height_rate_limit_m_per_step=float(os.environ.get("SONIC_DEPLOY_BASE_HEIGHT_RATE_LIMIT", "0.05")),
            base_height_scale=float(os.environ.get("SONIC_DEPLOY_BASE_HEIGHT_SCALE", "1.0")),
            keep_feet_on_ground=_env_flag("SONIC_DEPLOY_KEEP_FEET_ON_GROUND", False),
            foot_ground_scale=float(os.environ.get("SONIC_DEPLOY_FOOT_GROUND_SCALE", "0.35")),
            max_squat_drop_m=float(os.environ.get("SONIC_DEPLOY_MAX_SQUAT_DROP", "0.45")),
            # Synthetic base motion 是固定根可视化功能；物理模式 root 由 PhysX 驱动，默认关
            synthetic_base_motion_from_lower_body=_env_flag(
                "SONIC_DEPLOY_SYNTHETIC_BASE_MOTION", not SONIC_G1_PHYSICS_MODE
            ),
            synthetic_base_motion_gain=float(os.environ.get("SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_GAIN", "0.35")),
            synthetic_base_motion_deadzone=float(
                os.environ.get("SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_DEADZONE", "0.002")
            ),
            synthetic_base_motion_max_step_m=float(
                os.environ.get("SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_MAX_STEP", "0.035")
            ),
            debug_log_interval=50,
        )

    # 默认 ZMQ 链路下，关节由 SonicDeployTargetAction 驱动，但不会回传机器人状态。
    # 设 SONIC_PUBLISH_LOWSTATE=1（或 teleop 的 --publish_lowstate）时额外开一路 DDS，
    # 把 sonic_robot 的 sim 状态发到 rt/lowstate，供 GR00T/SONIC deploy 当状态源。
    # DDS 传输模式（SONIC_DEPLOY_TRANSPORT=dds）已由 UnitreeDdsLowCmdAction 发布 lowstate，
    # 因此这里仅在非 dds 模式下挂载，避免重复发布与 DDS 重复初始化。
    if os.environ.get("SONIC_DEPLOY_TRANSPORT", "zmq").lower() != "dds" and _env_flag(
        "SONIC_PUBLISH_LOWSTATE", False
    ):
        sonic_lowstate_pub = UnitreeLowStatePublisherActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            domain_id=int(os.environ.get("UNITREE_DDS_DOMAIN_ID", "0")),
            network_interface=os.environ.get("UNITREE_DDS_INTERFACE", ""),
            lowstate_topic=os.environ.get("UNITREE_LOWSTATE_TOPIC", "rt/lowstate"),
            secondary_imu_topic=os.environ.get("UNITREE_SECONDARY_IMU_TOPIC", "rt/secondary_imu"),
            publish_secondary_imu=_env_flag("SONIC_PUBLISH_SECONDARY_IMU", True),
            target_order="mujoco",
            mode_machine=int(os.environ.get("UNITREE_G1_MODE_MACHINE", "5")),
            debug_log_interval=100,
        )

    # 真实物理闭环桥：IsaacLab 用简单 ZMQ/msgpack 发布 sonic_robot 真实状态，
    # C++ proxy 再用 Unitree C++ SDK 转成 rt/lowstate，避开 Python DDS 与 C++ deploy 不互通的问题。
    if _env_flag("SONIC_PUBLISH_STATE_ZMQ", False):
        sonic_state_pub = SonicRobotStatePublisherActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            bind_endpoint=os.environ.get("SONIC_STATE_ZMQ_BIND", "tcp://127.0.0.1:5560"),
            topic=os.environ.get("SONIC_STATE_ZMQ_TOPIC", "sonic_state"),
            target_order="mujoco",
            mode_machine=int(os.environ.get("UNITREE_G1_MODE_MACHINE", "5")),
            debug_log_interval=100,
        )


@configclass
class SonicSoloObservationsCfg:
    """最小观测组（ManagerBasedRLEnv 需要 policy 组存在）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("sonic_robot")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SonicSoloTerminationsCfg:
    # teleop_se3_agent.py 会把 time_out 置 None；字段必须存在
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SonicSoloEventsCfg:
    """USD 演示道具的启动物理补齐 + R 键全场景复位。"""

    # R 键 env.reset() 依赖 reset 事件恢复实体状态：Articulation.reset() 只清
    # actuator/内部 buffer，不写姿态。没有这条，摔倒后按 R 机器人仍躺在原地，
    # 只有 action term 状态机被复位（root 在摔倒处重新锁定）。
    # 原地扶正不回出生点用 J 键（SonicDeployTargetAction.recover_standing；
    # 不用 H：Isaac Sim Edit 菜单 H = Toggle Visibility，会把选中 prim 隐藏）。
    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    if _ENABLE_DEMO_OBJECT:
        setup_hug_box_physics = SETUP_HUG_BOX_PHYSICS_EVENT


@configclass
class SonicSoloLocomanipulationEnvCfg(ManagerBasedRLEnvCfg):
    """SONIC 闭环物理调试极简环境。"""

    # replicate_physics=False：prestartup 事件要逐 env 写 HugBox 的 USD 物理属性
    scene: SonicSoloSceneCfg = SonicSoloSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=False)
    observations: SonicSoloObservationsCfg = SonicSoloObservationsCfg()
    actions: SonicSoloActionsCfg = SonicSoloActionsCfg()
    events: SonicSoloEventsCfg = SonicSoloEventsCfg()
    terminations: SonicSoloTerminationsCfg = SonicSoloTerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # 物理步频对齐参考 plant：SONIC 训练 = 200Hz/decimation4，MuJoCo deploy
        # sim2sim = 500Hz（默认 timestep 0.002）。此前 100Hz 的接触脉冲/joint_vel
        # 噪声特性与训练分布不同——慢动作时代被时间稀释掩盖，50Hz 实时下全带宽
        # 进入 policy 观测。控制率不变：4 × 1/200 = 0.02s = 50Hz env。
        # ⚠️ GPU pipeline 下 4 物理子步的同步开销 ≈11ms 会掉出实时；
        # 配合启动参数 --device cpu（单机器人 CPU PhysX ~1-2ms，绰绰有余）。
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4  # 每 env 步渲染一次（时序均匀），勿设 >decimation

        # XR 视角锚点：按 SONIC_XR_VIEW 环境变量选 first（头部第一视角，默认）
        # 或 third（pelvis 第三视角），配方细节与真机标定注意事项见
        # build_sonic_xr_cfg 的 docstring。
        # 只有真正构造出 OpenXRDevice 时锚点才会生效（见 openxr_device.py
        # __init__），所以必须同时挂一个 "handtracking" teleop device；
        # 启动时还需要 --teleop_device handtracking 才会选中它。
        self.xr = build_sonic_xr_cfg()
        self.teleop_devices = DevicesCfg(devices={"handtracking": OpenXRDeviceCfg(xr_cfg=self.xr)})
