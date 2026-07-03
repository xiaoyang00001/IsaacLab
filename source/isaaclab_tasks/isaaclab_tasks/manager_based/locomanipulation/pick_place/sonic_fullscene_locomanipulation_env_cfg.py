# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环 Phase 1：完整仓库场景回归（fullscene）。

完整 locomanipulation 场景实测 env_hz≈18.5（0.37× 实时；py-spy 2026-06-10：
渲染 30% + 4 台 G1 actuator python 簿记 24% + PhysX 17% + 杂项）。闭环 deploy
按墙钟 50Hz 推进步态相位，对时间畸变零容忍——直接在主配置上跑闭环必复发
"时间基准三死法"（慢动作 / render 抖动 / 超实时）。

本场景 = 仓库视觉与交互保留，陪跑算力裁除：
  保留：warehouse.usd 背景（含传送带视觉）+ packing_table + 转向盘 +
        sonic_obstacle + 高摩擦地面 + sonic_robot（唯一 articulation）+
        抱取演示物体（SONIC_FULLSCENE_DEMO_OBJECT=0 关闭，配置与 SonicSolo 同源）
  裁掉：robot / remote_robot / walker_robot 三台陪跑 G1（actuator 簿记大头、
        upper_body_ik pinocchio、双机 ZMQ 同步的宿主全在它们身上）
  按需：传送带物理 + 流水箱子——SONIC_FULLSCENE_CONVEYOR=1 时加载，默认关

物理对齐闭环七条件：200Hz / decimation 4（主配置 100Hz/2 的接触脉冲与
joint_vel 噪声特性偏离训练分布）；render_interval=4 = 每 env 步渲染一次。
⚠️ render_interval 勿设 >decimation：渲染步/非渲染步墙钟交替会让墙钟驱动的
deploy 看到"冻结-跳变"状态流 → 站立失稳（主配置 post_init 有 2026-06-10 实测）。

任务 id 含 "Locomanipulation" 以复用 teleop_se3_agent.py 的 deploy_target_mode
与 U 键回调。场景实体/事件从主配置实例摘取、Actions/Observations/Terminations
直接复用 SonicSolo 的类——保证三个场景的环境变量行为永远一致，不会漂移。
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp

from . import locomanipulation_g1_env_cfg as _main
from .sonic_solo_locomanipulation_env_cfg import (
    HUG_BOX_CFG,
    HUG_BOX_STAND_CFG,
    SETUP_HUG_BOX_PHYSICS_EVENT,
    SonicSoloActionsCfg,
    SonicSoloObservationsCfg,
    SonicSoloTerminationsCfg,
    build_sonic_xr_cfg,
)

# 传送带按需加载（带体物理 + 流水箱子 + 箱子对齐/驱动事件）。
# 默认关：闭环验证期减负——drive_test_box 是每步 interval 事件。
SONIC_FULLSCENE_CONVEYOR = _main._env_flag("SONIC_FULLSCENE_CONVEYOR", False)

# 抱取演示物体（台座 + 轻量纸箱，配置常量与 SonicSolo 同源导入）。
# 默认开：一个 kinematic 台座 + 一个 0.8kg 刚体，对 env_hz 影响可忽略；
# 事件仅 prestartup 一次性执行。位置在 SONIC 出生点正前方 1.05m 行走通道上。
SONIC_FULLSCENE_DEMO_OBJECT = _main._env_flag("SONIC_FULLSCENE_DEMO_OBJECT", True)

# 主配置的场景/事件在类体执行时已按环境变量定型；实例化一份摘取所需项
# （dataclass 实例化时 deepcopy 默认值，互不影响），与 SonicSolo 摘取
# ActionsCfg 同一原则。
_MAIN_SCENE = _main.LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
_MAIN_EVENTS = _main.EventsCfg()


@configclass
class SonicFullsceneSceneCfg(InteractiveSceneCfg):
    """仓库场景：1 台 SONIC G1 + warehouse 背景 + 打包台道具，无陪跑机器人。"""

    sonic_robot: ArticulationCfg = _MAIN_SCENE.sonic_robot

    # 仓库背景（warehouse.usd，传送带视觉模型在其中）
    background = _MAIN_SCENE.background

    packing_table = _MAIN_SCENE.packing_table

    # 打包台上的转向盘（可抓取道具）
    object = _MAIN_SCENE.object

    # sonic_verify.py --sonic_obstacle 用的障碍物（默认停在远处，开销可忽略）
    sonic_obstacle = _MAIN_SCENE.sonic_obstacle

    if SONIC_FULLSCENE_CONVEYOR:
        test_box = _MAIN_SCENE.test_box
        test_box1 = _MAIN_SCENE.test_box1

    if SONIC_FULLSCENE_DEMO_OBJECT:
        hug_box_stand = HUG_BOX_STAND_CFG
        hug_box = HUG_BOX_CFG

    # 高摩擦地面 μ=1.0/combine=max（warehouse 碰撞区域外的兜底，闭环七条件之一）
    ground = _MAIN_SCENE.ground


@configclass
class SonicFullsceneEventsCfg:
    """主配置事件子集：viewer 对齐 +（按需）传送带物理与流水箱子。

    不摘取的项及原因：robot/remote_robot/test_box 对齐事件（实体不存在）、
    fix_walker_articulation_root（walker 不存在）。sonic 的传送带对齐事件也不
    摘取——主配置 SONIC init_state.pos 即对齐结果的硬拷贝（见 _main 行 113 注释），
    少一个 startup 依赖。
    """

    # 非物理模式下钉死 SONIC 根（物理模式时主配置类体不会生成此字段）
    if hasattr(_MAIN_EVENTS, "fix_sonic_articulation_root"):
        fix_sonic_articulation_root = _MAIN_EVENTS.fix_sonic_articulation_root

    # warehouse.usd 自带 CollisionPlane/CollisionMesh 两块地板碰撞体且未绑物理材质
    # （PhysX 默认 μ=0.5/average），与 μ=1.0/max 的 GroundPlane 在 z=0 共面叠放——
    # 脚底接触落在哪块上摩擦就跟谁，侧移/蹬地随机打滑。prestartup 补绑对齐（2026-06-11
    # USD 排查实锤：USD 内仅传送带/分拣箱有材质，地板两件均无绑定）。
    bind_warehouse_floor_friction = EventTerm(
        func=locomanip_mdp.bind_floor_physics_material,
        mode="prestartup",
        params={
            "prim_path_templates": (
                "/World/envs/env_{}/Background/GroundPlane/CollisionPlane",
                "/World/envs/env_{}/Background/GroundPlane/CollisionMesh",
            ),
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0,
            "friction_combine_mode": "max",
        },
    )

    # viewer 开机对准 SONIC 正面；reset（R 键摔倒恢复）后重对一次
    align_viewer_to_sonic_front_startup = _MAIN_EVENTS.align_viewer_to_sonic_front_startup
    clear_viewer_front_once_flag_reset = _MAIN_EVENTS.clear_viewer_front_once_flag_reset
    align_viewer_to_sonic_front_interval = _MAIN_EVENTS.align_viewer_to_sonic_front_interval

    if SONIC_FULLSCENE_CONVEYOR:
        setup_conveyor_belt_physics = _MAIN_EVENTS.setup_conveyor_belt_physics
        print_conveyor_bbox = _MAIN_EVENTS.print_conveyor_bbox
        setup_test_box_physics = _MAIN_EVENTS.setup_test_box_physics
        setup_test_box1_physics = _MAIN_EVENTS.setup_test_box1_physics
        align_test_boxes_to_conveyor_startup = _MAIN_EVENTS.align_test_boxes_to_conveyor_startup
        align_test_boxes_to_conveyor_reset = _MAIN_EVENTS.align_test_boxes_to_conveyor_reset
        drive_test_box = _MAIN_EVENTS.drive_test_box
        drive_test_box1 = _MAIN_EVENTS.drive_test_box1

    if SONIC_FULLSCENE_DEMO_OBJECT:
        setup_hug_box_physics = SETUP_HUG_BOX_PHYSICS_EVENT


@configclass
class SonicFullsceneLocomanipulationEnvCfg(ManagerBasedRLEnvCfg):
    """SONIC 闭环 Phase 1 仓库场景环境。"""

    scene: SonicFullsceneSceneCfg = SonicFullsceneSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
    observations: SonicSoloObservationsCfg = SonicSoloObservationsCfg()
    actions: SonicSoloActionsCfg = SonicSoloActionsCfg()
    events: SonicFullsceneEventsCfg = SonicFullsceneEventsCfg()
    terminations: SonicSoloTerminationsCfg = SonicSoloTerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # 物理步频对齐参考 plant：SONIC 训练 = 200Hz/decimation4（同 SonicSolo，
        # 注释详见 sonic_solo_locomanipulation_env_cfg.__post_init__）。
        # 大场景 200Hz CPU 物理的可行性正是本配置要实测的主战场。
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4  # 每 env 步渲染一次（时序均匀），勿设 >decimation

        # XR 视角锚点：与 SonicSolo 共用 build_sonic_xr_cfg（sonic_robot 在两个
        # 场景里是同一个 SONIC_G1_29DOF_CFG，prim_path 同为
        # {ENV_REGEX_NS}/SONICRobot）。SONIC_XR_VIEW=first|third 切换第一/第三
        # 视角，配方细节见该函数 docstring。
        self.xr = build_sonic_xr_cfg()
        self.teleop_devices = DevicesCfg(devices={"handtracking": OpenXRDeviceCfg(xr_cfg=self.xr)})
