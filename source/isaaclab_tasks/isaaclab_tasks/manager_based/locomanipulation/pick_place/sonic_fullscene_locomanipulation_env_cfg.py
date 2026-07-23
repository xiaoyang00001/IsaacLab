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
  裁掉：主配置的镜像遥操 G1 / 推车与箱堆（actuator 簿记与多人同步的宿主）
  按需：传送带物理 + 流水箱子——SONIC_FULLSCENE_CONVEYOR=1 时加载，默认关

物理对齐闭环七条件：200Hz / decimation 4；render_interval=4 = 每 env 步渲染一次。
⚠️ render_interval 勿设 >decimation：渲染步/非渲染步墙钟交替会让墙钟驱动的
deploy 看到"冻结-跳变"状态流 → 站立失稳（2026-06-10 实测）。

任务 id 含 "Locomanipulation" 以复用 teleop_se3_agent.py 的 deploy_target_mode
与 U 键回调。Actions/Observations/Terminations 直接复用 SonicSolo 的类——
保证两个 SONIC 场景的环境变量行为永远一致，不会漂移。

移植说明（pickplace-g1-collision 基线）：主配置场景已是宣传片仓库
（warehouse-simple6_v48.usd + 推车箱堆），本场景不再从主配置摘取实体，
而是自带 sonic 分支验证过的 warehouse.usd 背景与传送带/地板事件参数
（ConveyorBelt_A08_06 等 prim 路径、CollisionPlane/CollisionMesh 补绑均
以该 USD 的内部结构为准）；packing_table / 转向盘仍取主配置实例。
"""

import os

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.sim as sim_utils
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp

from . import locomanipulation_g1_env_cfg as _main
from .sonic_solo_locomanipulation_env_cfg import (
    HUG_BOX_CFG,
    HUG_BOX_STAND_CFG,
    SETUP_HUG_BOX_PHYSICS_EVENT,
    SonicSoloActionsCfg,
    SonicSoloObservationsCfg,
    SonicSoloTerminationsCfg,
    build_sonic_teleop_devices,
    build_sonic_xr_cfg,
    configure_sonic_physx,
)

# 传送带按需加载（带体物理 + 流水箱子 + 箱子对齐/驱动事件）。
# 默认关：闭环验证期减负——drive_test_box 是每步 interval 事件。
SONIC_FULLSCENE_CONVEYOR = _main._env_flag("SONIC_FULLSCENE_CONVEYOR", False)

# 抱取演示物体（台座 + 轻量纸箱，配置常量与 SonicSolo 同源导入）。
# 默认开：一个 kinematic 台座 + 一个 0.8kg 刚体，对 env_hz 影响可忽略；
# 事件仅 prestartup 一次性执行。位置在 SONIC 出生点正前方 1.05m 行走通道上。
SONIC_FULLSCENE_DEMO_OBJECT = _main._env_flag("SONIC_FULLSCENE_DEMO_OBJECT", True)

# 主配置的 packing_table / 转向盘沿用（实例化摘取，dataclass deepcopy 互不影响）
_MAIN_SCENE = _main.LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


@configclass
class SonicFullsceneSceneCfg(InteractiveSceneCfg):
    """仓库场景：1 台 SONIC G1 + warehouse 背景 + 打包台道具，无陪跑机器人。"""

    sonic_robot: ArticulationCfg = _main.SONIC_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/SONICRobot")

    # 仓库背景（warehouse.usd，传送带视觉模型在其中；沿用 sonic 分支验证过的
    # 摆位——传送带/地板事件的 prim 路径与坐标都以该 USD 为准）
    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.68, 14.39363, 0], rot=[0.7071, 0.0, 0.0, 0.7071]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse.usd"),
        ),
    )

    packing_table = _MAIN_SCENE.packing_table

    # 打包台上的转向盘（可抓取道具）
    object = _MAIN_SCENE.object

    # sonic_verify.py --sonic_obstacle 用的障碍物（默认停在远处，开销可忽略）
    sonic_obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SONICObstacle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[1000.0, 1000.0, -10.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.90, 0.22),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.12, 0.02), roughness=0.85),
        ),
    )

    if SONIC_FULLSCENE_CONVEYOR:
        # 流水箱子：落在 ConveyorBelt_A08_06 传送带入料端，沿 -y 流动
        # （坐标/物理参数移植自 sonic 分支主场景，单机模式无 ZMQ 订阅侧）
        test_box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TestBox",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.78886, 1.17033, 0.845],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    max_depenetration_velocity=10.0,
                ),
            ),
        )
        test_box1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TestBox1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.42787, 1.67696, 0.845],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8,
                    max_depenetration_velocity=10.0,
                ),
            ),
        )

    if SONIC_FULLSCENE_DEMO_OBJECT:
        hug_box_stand = HUG_BOX_STAND_CFG
        hug_box = HUG_BOX_CFG

    # 高摩擦地面 μ=1.0/combine=max（warehouse 碰撞区域外的兜底，闭环七条件之一）
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


@configclass
class SonicFullsceneEventsCfg:
    """SONIC 场景事件：viewer 对齐 + 地板摩擦补绑 +（按需）传送带物理与流水箱子。

    事件定义在本文件内自带（主配置事件承载的是镜像遥操场景，不再摘取）；
    参数与 sonic 分支验证结果保持一致。
    """

    # R 键 env.reset() 依赖 reset 事件恢复实体状态：Articulation.reset() 只清
    # actuator/内部 buffer，不写姿态（与 SonicSolo 同注释）。原地扶正用 J 键。
    reset_scene_to_default = EventTerm(func=isaaclab_mdp.reset_scene_to_default, mode="reset")

    # 非物理模式下钉死 SONIC 根（物理模式时不生成此字段）
    if _main.SONIC_G1_FIX_ROOT:
        fix_sonic_articulation_root = EventTerm(
            func=locomanip_mdp.fix_nested_articulation_roots,
            mode="prestartup",
            params={
                "prim_path_templates": ("/World/envs/env_{}/SONICRobot",),
                "fix_root_link": True,
                "disable_gravity": True,
            },
        )

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

    # Viewer 放在 sonic_robot 当前朝向的正前方。startup 先给初始位置，interval 只在
    # reset 后校正一次，避免持续覆盖鼠标中键/滚轮操作。
    align_viewer_to_sonic_front_startup = EventTerm(
        func=locomanip_mdp.align_viewer_to_asset_front,
        mode="startup",
        params={
            "viewer_asset_name": "sonic_robot",
            "front_axis": "+x",
            "distance": 4.0,
            "eye_height": 1.55,
            "lookat_height": 0.9,
            "track_asset_position": False,
        },
    )

    clear_viewer_front_once_flag_reset = EventTerm(
        func=locomanip_mdp.clear_viewer_alignment_once_flag,
        mode="reset",
        params={
            "once_key": "_sonic_front_viewer_aligned",
        },
    )

    align_viewer_to_sonic_front_interval = EventTerm(
        func=locomanip_mdp.align_viewer_to_asset_front,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "viewer_asset_name": "sonic_robot",
            "front_axis": "+x",
            "distance": 4.0,
            "eye_height": 1.55,
            "lookat_height": 0.9,
            "track_asset_position": False,
            "log_viewer": False,
            "once_key": "_sonic_front_viewer_aligned",
        },
    )

    if SONIC_FULLSCENE_CONVEYOR:
        setup_conveyor_belt_physics = EventTerm(
            func=locomanip_mdp.setup_conveyor_belt_physics,
            mode="prestartup",
            params={
                "velocity": (-0.5, 0.0, 0.0),
                "prim_name_patterns": ("ConveyorBelt_A08_06", "ConveyorBelt_A08_07", "ConveyorBelt_A08_08"),
                "roller_radius": 0.028951416,
                "rotation_axis": "X",
                "keep_rollers_parent_collision": False,
            },
        )

        # 启动时打印 ConveyorBelt_A08_06 的世界包围盒，用于校准 test_box 坐标。
        print_conveyor_bbox = EventTerm(
            func=locomanip_mdp.print_conveyor_world_bbox,
            mode="startup",
            params={"prim_name": "ConveyorBelt_A08_06"},
        )

        # prestartup：在 PhysX 初始化前为 SM_CardBoxD_05.usd 注入刚体物理 API。
        # UsdFileCfg 只会 modify（而非 define）RigidBodyAPI，此事件补上 define 步骤。
        setup_test_box_physics = EventTerm(
            func=locomanip_mdp.setup_usd_rigid_object_physics,
            mode="prestartup",
            params={
                "prim_path_template": "/World/envs/env_{}/TestBox",
                "mass": 1.5,
                "linear_damping": 5.0,
                "angular_damping": 0.1,
                "kinematic_enabled": False,
                "disable_gravity": False,
            },
        )

        setup_test_box1_physics = EventTerm(
            func=locomanip_mdp.setup_usd_rigid_object_physics,
            mode="prestartup",
            params={
                "prim_path_template": "/World/envs/env_{}/TestBox1",
                "mass": 1.5,
                "linear_damping": 5.0,
                "angular_damping": 0.1,
                "kinematic_enabled": False,
                "disable_gravity": False,
            },
        )

        align_test_boxes_to_conveyor_startup = EventTerm(
            func=locomanip_mdp.place_test_boxes_from_conveyor_bbox,
            mode="startup",
            params={"conveyor_prim_name": "ConveyorBelt_A08_06"},
        )

        align_test_boxes_to_conveyor_reset = EventTerm(
            func=locomanip_mdp.place_test_boxes_from_conveyor_bbox,
            mode="reset",
            params={"conveyor_prim_name": "ConveyorBelt_A08_06"},
        )

        # 状态感知速度覆写：仅在箱子处于传送带范围内时驱动，
        # 离带后（被提起/掉落/移出）自动停止，不再对抗机器人抓取力。
        drive_test_box = EventTerm(
            func=locomanip_mdp.drive_object_on_conveyor,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={"object_name": "test_box", "velocity_x": 0.0, "velocity_y": -0.5},
        )

        drive_test_box1 = EventTerm(
            func=locomanip_mdp.drive_object_on_conveyor,
            mode="interval",
            interval_range_s=(0.01, 0.01),
            params={"object_name": "test_box1", "velocity_x": 0.0, "velocity_y": -0.5},
        )

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
        configure_sonic_physx(self.sim.physx)

        # XR 视角锚点：与 SonicSolo 共用 build_sonic_xr_cfg（sonic_robot 在两个
        # 场景里是同一个 SONIC_G1_29DOF_CFG，prim_path 同为
        # {ENV_REGEX_NS}/SONICRobot）。SONIC_XR_VIEW=first|third 切换第一/第三
        # 视角，配方细节见该函数 docstring。
        # teleop 设备表与 SonicSolo 同源（默认 motion_controllers + 夹爪 retargeter，
        # SONIC_GRIPPER_TELEOP=0 回退到无 retargeter 的 handtracking）。
        self.xr = build_sonic_xr_cfg()
        self.teleop_devices = build_sonic_teleop_devices(self.xr)
