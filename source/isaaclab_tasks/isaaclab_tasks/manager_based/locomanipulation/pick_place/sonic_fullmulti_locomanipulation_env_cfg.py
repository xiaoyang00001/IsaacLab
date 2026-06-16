# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环：完整主配置场景（4 台 G1 全保留）+ 闭环七条件对齐。

与 SonicFullscene（裁掉 3 台陪跑 G1 的瘦身仓库）相反——本场景**全部保留**主配置
`LocomanipulationG1EnvCfg` 的实体，主配置文件一行不动：

  4 台 G1（robot / remote_robot / walker_robot / sonic_robot）+ 双机 ZMQ 同步 +
  upper_body_ik（pinocchio）+ walker 骨骼行走 + 传送带物理与流水箱子 + warehouse
  背景 + packing_table + 全部道具 + OpenXR 头显/手柄遥操设备。

实现 = 直接继承主配置类（继承自动带上 scene 全部资产、ActionsCfg 全部 action、
EventsCfg 全部 19 个事件、xr/xr2、teleop_devices）。相对主配置改三处：

  ① 物理步频 100Hz/dec2 → 闭环七条件的 200Hz/dec4（主配置为遥操 FPS 故意降的，
     见 _main.__post_init__ 注释；100Hz 的接触脉冲/joint_vel 噪声特性偏离 SONIC
     训练分布）。⚠️ env 控制率不变：4 × 1/200 = 0.02s = 50Hz（= 主配置 2 × 1/100），
     所以 SonicDeployTargetAction 的全部 per-step 参数（rate_limit 等）语义不变。
  ② 补 warehouse 自带两块地板碰撞体的物理材质（主配置只绑了 GroundPlane μ=1.0，
     warehouse.usd 的 CollisionPlane/CollisionMesh 未绑 → PhysX 默认 μ0.5/average，
     与 μ=1.0 GroundPlane 共面打滑；同 SonicFullscene @43dcf0aad）。
  ③ __post_init__ 禁用传送带/对齐动态事件（CPU 物理模式不兼容、与 SONIC 闭环无关）：
     align robots/boxes/walker + drive boxes（对 kinematic 刚体写 root 速度 → "Body
     must be non-kinematic" 累积停机）、setup_conveyor_belt_physics（989°/s 滚轮致
     Illegal BroadPhase）。详见 __post_init__；实体全保留、仅停驱动，GPU 跑全动态删该段。

⚠️ 性能（本配置存在的全部风险）：4 台 G1 + 200Hz/dec4 比主配置 100Hz/dec2
   （实测 env_hz≈18.5）更重，env_hz 预期更低。闭环 deploy 按墙钟 50Hz 推步态相位，
   sim 一旦掉出实时即复发"时间基准三死法"（慢动作 / render 抖动 / 超实时）。
   能否实时是核心未知数，需 py-spy 实测裁决；减负杠杆（按需）：CPU governor 切
   performance、评估陪跑 G1 的 walker/双机同步是否必需、关传送带 drive 事件。

任务 id 含 "Locomanipulation" 以复用 teleop_se3_agent.py 的 deploy_target_mode
判定与 U 键回调（同 SonicSolo / SonicFullscene）。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp

from . import locomanipulation_g1_env_cfg as _main


@configclass
class SonicFullMultiEventsCfg(_main.EventsCfg):
    """主配置全部事件 + warehouse 地板摩擦补绑。

    继承 `_main.EventsCfg` 的全部 17 个事件（fix_sonic/walker root、传送带物理
    setup/drive、4 台 G1 与流水箱子的 conveyor 对齐、viewer 对齐）。仅新增一项：

    warehouse.usd 自带的 CollisionPlane/CollisionMesh 两块地板碰撞体未绑物理材质
    （PhysX 默认 μ=0.5/average），与 μ=1.0/max 的 GroundPlane 在 z=0 共面叠放——
    脚底接触落在哪块上摩擦就跟谁，侧移/蹬地随机打滑。prestartup 补绑对齐
    （2026-06-11 USD 排查实锤，同 SonicFullsceneEventsCfg.bind_warehouse_floor_friction
    @43dcf0aad）。
    """

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


@configclass
class SonicFullMultiLocomanipulationEnvCfg(_main.LocomanipulationG1EnvCfg):
    """完整主配置场景（4 台 G1 全保留）+ SONIC 闭环七条件（200Hz/dec4）。"""

    # 覆盖事件集：主配置全部事件 + warehouse 地板摩擦补绑（其余字段全部继承主配置）
    events: SonicFullMultiEventsCfg = SonicFullMultiEventsCfg()

    def __post_init__(self):
        # 先执行主配置全部 post_init：urdf 路径、xr/xr2 anchor、teleop_devices、
        # 双机 OpenXR/motion_controllers 设备等一律不动地继承。
        super().__post_init__()
        # 物理步频对齐 SONIC 闭环七条件：200Hz / decimation 4。env 控制率不变
        # （4 × 1/200 = 0.02s = 50Hz = 主配置 2 × 1/100），per-step 参数语义不变。
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4  # 每 env 步渲染一次（时序均匀），勿设 >decimation

        # —— 禁用与 SONIC 物理模式(CPU) 不兼容的传送带/对齐动态 ——
        # 这 8 个事件用 write_root_velocity_to_sim 对 kinematic 刚体（fixed 陪跑 G1
        # robot/remote_robot/walker_robot、kinematic 流水箱 test_box/test_box1）写 root
        # 速度；--device cpu 下 PhysX 严格报 "PxRigidDynamic::setLinearVelocity: Body
        # must be non-kinematic" 并每步累积，~27s 触发 too many errors 停机（GPU pipeline
        # 会静默——这正是 v2 在主配置+GPU 跑通 SONIC 的原因）。它们是 pick-place 数据
        # 采集的场景动态，与 SONIC 闭环行走无关：实体（4 台 G1 + 传送带 + 流水箱）全部
        # 保留在场景里，仅停掉运行时驱动。align_sonic_* 保留（sonic 物理模式非 kinematic，
        # 写速度安全）。若改用 GPU 跑全动态，删除本段即可。
        for _disabled in (
            "align_robots_to_conveyor_startup",      # robot / remote_robot (fixed root)
            "align_robots_to_conveyor_reset",
            "align_test_boxes_to_conveyor_startup",   # test_box / test_box1 (kinematic)
            "align_test_boxes_to_conveyor_reset",
            "align_walker_startup",                    # walker_robot (fixed root)
            "align_walker_reset",
            "drive_test_box",                          # 每步驱动流水箱 (kinematic)
            "drive_test_box1",
        ):
            setattr(self.events, _disabled, None)

        # —— 禁用#2：传送带滚轮高速旋转致 Illegal BroadPhaseUpdateData ——
        # setup_conveyor_belt_physics 把 39×N 个 kinematic 滚轮设成 989°/s（17.27 rad/s）
        # 持续旋转；--device cpu 下高速旋转 kinematic 体的碰撞 AABB 更新非法 → 每步刷
        # "Illegal BroadPhaseUpdateData"（实测 1680+ 次/场；传送带 roller kinematic 在
        # GPU/CPU 走不同路径，CPU 扛不住）。传送带送货动态与 SONIC 闭环无关，fullscene
        # 同样用 SONIC_FULLSCENE_CONVEYOR 默认关闭它。warehouse USD 的传送带视觉模型
        # 保留，仅停滚轮旋转物理。GPU 跑全动态时连同上面 for 段一并删除本行。
        self.events.setup_conveyor_belt_physics = None
