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

SONIC 相关 action 项直接从主配置 ``ActionsCfg`` 实例摘取（主任务场景也挂了
sonic_robot + SONIC deploy 驱动），保证三个任务的环境变量行为（transport 选择、
发布开关、全部调参）永远一致，不会漂移。机器人本体配置同样取
``_main.SONIC_G1_29DOF_CFG``（与 SONIC 训练对齐的 PD/armature 配方）。
抱取演示物体的配置（台座/纸箱/物理事件）以本文件的模块级常量为唯一来源，
SonicFullscene 直接导入引用。
"""

import os

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters import G1GripperMotionControllerRetargeterCfg
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
    G1GripperSyncActionCfg,
)

from . import locomanipulation_g1_env_cfg as _main

# 环境变量行为与主配置同源：直接复用 _main 的解析结果。
_env_flag = _main._env_flag

# 主配置的 ActionsCfg 在类体执行时已按环境变量决定 transport 与发布 term；
# 实例化一份并摘取 SONIC 相关项（dataclass 实例化时 deepcopy 默认值，互不影响）。
_MAIN_ACTIONS = _main.ActionsCfg()

# ---------------------------------------------------------------------------
# 抱取演示物体（移植自 sonic-hug-object-demo 分支 c7660b80f）
# 位置：SONIC 出生点 (-2.0, 11.008) 正前方 +X 1.05m、胸腰高度——solo 与
# fullscene 两场景的 SONIC 出生点同为主配置硬拷贝，故坐标可共用（fullscene
# 中该点位于行走通道，距传送带 y 向 4m+，无碰撞）。
# ---------------------------------------------------------------------------
_ENABLE_DEMO_OBJECT = _env_flag("SONIC_SOLO_DEMO_OBJECT", True)
_DEMO_OBJECT_POS = (-0.95, 11.008, 0.92)
_DEMO_STAND_POS = (-0.95, 11.008, 0.45)

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


# ---------------------------------------------------------------------------
# XR 手柄 → 三指手（夹爪）遥操链路
#
# 与主任务 Isaac-PickPlace-Locomanipulation-G1-Abs-v0 同一条通路：
#   OpenXR motion_controllers 设备
#     → G1GripperMotionControllerRetargeter：扳机/握把（右手另加 A/B 键）
#       retarget 成 4 维闭合率 [左食指, 左中指, 右食指, 右中指]
#     → G1GripperSyncAction：展开成 14 个手指关节目标写进 sonic_robot
#
# SONIC policy 只驱动 29 个身体关节（腿+腰+臂，见 SONIC_G1_29DOF_JOINT_ORDER），
# 手指不在其中，两条通路互不干扰。不挂本 term 时手指没有任何驱动源，永远停在
# 默认张开姿态——这正是 SonicSolo/SonicFullscene 下"夹爪不受遥操作控制"的根因。
#
# ⚠️ 动作空间宽度随之从 0 变成 4，teleop_se3_agent.py 主循环不再走零动作分支，
#    因此 teleop 设备必须带 retargeter（无 retargeter 的 OpenXRDevice.advance()
#    返回 raw dict，主循环 action.repeat() 直接 AttributeError）——
#    见 build_sonic_teleop_devices()。
# ---------------------------------------------------------------------------
_ENABLE_GRIPPER_TELEOP = _env_flag("SONIC_GRIPPER_TELEOP", True)


def build_sonic_gripper_action_cfg() -> G1GripperSyncActionCfg:
    """构造 sonic_robot 的三指手遥操 action 配置（SonicSolo/SonicFullscene 共用）。

    夹持角度配方与主任务 local_gripper 一致（实测可抓取的档位）。ZMQ 发布端口
    默认 5573，刻意避开主任务的 5571/5572：同机同时跑主任务与 SONIC 场景时
    bind 冲突会被 action term 的 try 捕获并静默关掉整个夹爪链路。
    """
    return G1GripperSyncActionCfg(
        asset_name="sonic_robot",
        mode="local_publish",
        robot_id=int(os.environ.get("SONIC_GRIPPER_ROBOT_ID", "1")),
        transport="zmq",
        zmq_port=int(os.environ.get("SONIC_GRIPPER_ZMQ_PORT", "5573")),
        zmq_topic=os.environ.get("SONIC_GRIPPER_ZMQ_TOPIC", "sonic_gripper"),
        controller_gripper_finger_close_angle=1.8,
        controller_gripper_thumb_1_angle=1.1,
        controller_gripper_thumb_2_angle=1.8,
        controller_gripper_action_alpha=1.0,
        controller_gripper_use_soft_limits=False,
        write_joint_state=True,
    )


def build_sonic_teleop_devices(xr_cfg: XrCfg) -> DevicesCfg:
    """构造 SONIC 场景的 teleop 设备表（SonicSolo/SonicFullscene 共用）。

    只有真正构造出 OpenXRDevice 时 XR 锚点才会生效（见 openxr_device.py
    __init__），所以无论是否用夹爪都必须挂一个 teleop device。

    - 夹爪开启（默认）：挂 ``motion_controllers`` + G1GripperMotionControllerRetargeter，
      advance() 输出 4 维 tensor，与动作空间宽度匹配。
    - 夹爪关闭（SONIC_GRIPPER_TELEOP=0）：回到原先的 ``handtracking``（无 retargeter），
      动作空间宽度 0，主循环走零动作分支。

    右手 B 键归属：B 被 SONIC_XR_ENABLE_B_RECENTER=1 借去做 recenter 时不再兼做
    右中指闭合，避免一个按键触发两件事。
    """
    if not _ENABLE_GRIPPER_TELEOP:
        return DevicesCfg(devices={"handtracking": OpenXRDeviceCfg(xr_cfg=xr_cfg)})

    sim_device = os.environ.get("SONIC_GRIPPER_RETARGETER_DEVICE", "cpu")
    return DevicesCfg(
        devices={
            "motion_controllers": OpenXRDeviceCfg(
                retargeters=[
                    G1GripperMotionControllerRetargeterCfg(
                        sim_device=sim_device,
                        use_right_b_button=not _env_flag("SONIC_XR_ENABLE_B_RECENTER", False),
                    ),
                ],
                sim_device=sim_device,
                xr_cfg=xr_cfg,
            ),
        }
    )


def configure_sonic_physx(physx_cfg) -> None:
    """Tune PhysX defaults for the free-root SONIC closed loop.

    ⚠️ 两个旗标默认关闭：2026-07-15 闭环实测（CPU 物理 + stabilize_root 逐子步
    根位姿写回）开启后 settle 期 root 高度数值发散（0.76 → -324 → -1.2e6 → NaN，
    一秒内爆炸），机器人从未站起来过。该组合源自 2026-07-06 stash（当年即未实测）。
    仅供后续在 GPU 物理或去掉逐子步根写回的配置下重新评估时手动开启。
    """

    physx_cfg.enable_external_forces_every_iteration = _main._env_flag(
        "SONIC_PHYSX_EXTERNAL_FORCES_EVERY_ITERATION", False
    )
    min_velocity_iterations = int(os.environ.get("SONIC_PHYSX_MIN_VELOCITY_ITERATIONS", "0"))
    physx_cfg.min_velocity_iteration_count = max(
        int(physx_cfg.min_velocity_iteration_count),
        min_velocity_iterations,
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
    """SONIC deploy target + 状态发布（与主配置同一来源，见 _main.ActionsCfg）+ 夹爪遥操。

    前三项 action_dim 均为 0（自行消费网络包）；local_gripper 是唯一吃 env action
    的 term，动作空间宽度 = 4 = XR 手柄 retargeter 的输出维度。
    """

    sonic_wholebody = _MAIN_ACTIONS.sonic_wholebody

    if hasattr(_MAIN_ACTIONS, "sonic_state_pub"):
        sonic_state_pub = _MAIN_ACTIONS.sonic_state_pub
    if hasattr(_MAIN_ACTIONS, "sonic_lowstate_pub"):
        sonic_lowstate_pub = _MAIN_ACTIONS.sonic_lowstate_pub

    # XR 手柄扳机/握把 → 三指手（详见 build_sonic_gripper_action_cfg 上方注释块）
    if _ENABLE_GRIPPER_TELEOP:
        local_gripper = build_sonic_gripper_action_cfg()

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
        configure_sonic_physx(self.sim.physx)

        # XR 视角锚点：按 SONIC_XR_VIEW 环境变量选 first（头部第一视角，默认）
        # 或 third（pelvis 第三视角），配方细节与真机标定注意事项见
        # build_sonic_xr_cfg 的 docstring。
        # 只有真正构造出 OpenXRDevice 时锚点才会生效（见 openxr_device.py
        # __init__），所以必须同时挂一个 teleop device；默认设备名是
        # "motion_controllers"（带夹爪 retargeter），见 build_sonic_teleop_devices。
        # 启动脚本会传 --teleop_device motion_controllers；即便传的是旧的
        # handtracking，teleop_se3_agent.py 也会自动回退到 motion_controllers。
        self.xr = build_sonic_xr_cfg()
        self.teleop_devices = build_sonic_teleop_devices(self.xr)
