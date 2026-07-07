# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""完整仓库场景（移植自 sonic-windows-xr-ar-anchor 分支的 SonicFullscene，只移植 USD 场景资产）。

移植原则：只要场景的 USD 资源，不移植源分支的机器人配置（SONIC/陪跑 G1）、
events 事件系统（地板摩擦补绑 / 传送带滚轮物理 / 流水箱驱动 / viewer 对齐等）。
机器人/动作/观测/XR/teleop 全部继承本分支主配置 ``LocomanipulationG1EnvCfg``。

场景 = 主场景 + warehouse.usd 背景（含传送带视觉模型，静态）+ packing_table
（USD 版替换主场景隐藏的方块占位）+ 转向盘可抓道具（替换主场景隐藏的方块）。

HugBox 不移植到这里：它在 Fullscene 用 UsdFileCfg + 默认 replicate_physics=True
时，prim 路径被模板化成 ``/World/envs/env_.*/HugBox``，USD 内的 RigidBodyAPI
不会被 replicate 到每个 env 实例，导致 ``Failed to find a rigid body when
resolving '/World/envs/env_.*/HugBox'``。源分支靠 ``replicate_physics=False``
+ prestartup 事件 ``setup_usd_rigid_object_physics`` 逐 env 写 USD 物理属性
绕开，本移植不带 events。Solo 场景（无 warehouse，num_envs=1）暂时未
暴露此问题，HugBox 仍可用。

坐标系沿用源分支：机器人出生 (-2.0, 11.008)，warehouse 背景 (-4.68, 14.39363)，
所有道具坐标原样照搬，零重新校准。

未移植说明：
- 传送带只有视觉（warehouse.usd 自带），不会转动——滚轮物理与流水箱驱动都在
  源分支 events 里，按"只要 USD 资源"原则未移植；
- warehouse 地板碰撞体无物理材质（默认 μ=0.5），源分支靠 prestartup 事件补绑
  μ=1.0——本分支机器人根姿态由 UDP 镜像直写，地面摩擦不影响镜像效果。
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import locomanipulation_g1_env_cfg as _main
from .sonic_solo_locomanipulation_env_cfg import ROBOT_SPAWN_POS


@configclass
class SonicFullsceneSceneCfg(_main.LocomanipulationG1SceneCfg):
    """主场景 + warehouse 背景 + 打包台道具（无 HugBox，见模块 docstring）。"""

    # 仓库背景（warehouse.usd，传送带视觉模型在其中；坐标为源分支校准值）
    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.68, 14.39363, 0], rot=[0.7071, 0.0, 0.0, 0.7071]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse-simple6_v48.usd"),
        ),
    )

    # 打包台（USD 版，覆盖主场景隐藏的方块占位；源分支坐标）
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    # 打包台上的转向盘（可抓取道具，覆盖主场景隐藏的方块；源分支坐标）
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-4.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )


@configclass
class SonicFullsceneLocomanipulationEnvCfg(_main.LocomanipulationG1EnvCfg):
    """完整仓库场景环境：主环境只换场景，其余（动作/观测/XR/teleop）不动。"""

    scene: SonicFullsceneSceneCfg = SonicFullsceneSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # 出生点移到源分支仓库通道坐标（与背景/道具坐标配套，见模块 docstring）
        self.scene.robot.init_state.pos = ROBOT_SPAWN_POS
