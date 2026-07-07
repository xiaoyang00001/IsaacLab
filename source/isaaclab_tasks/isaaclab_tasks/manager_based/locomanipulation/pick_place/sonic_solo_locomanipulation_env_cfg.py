# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""极简场景（移植自 sonic-windows-xr-ar-anchor 分支的 SonicSolo，只移植 USD 场景资产）。

移植原则：只要场景的 USD 资源，不移植源分支的机器人配置（SONIC 29dof）、
events 事件系统（HugBox 物理补齐 / 传送带驱动 / viewer 对齐等）。
机器人/动作/观测/XR/teleop 全部继承本分支主配置 ``LocomanipulationG1EnvCfg``，
仅把出生点移到源分支的仓库通道坐标，使 HugBox 等道具坐标可与源分支原样共用。

场景 = 主场景（镜像 G1 + 地面 + 天光 + 隐藏的占位道具）+ 抱取演示物体
（HugBox 台座 + 纸箱）。纸箱使用 USD 自带默认质量（源分支靠 prestartup 事件
调轻到 0.8kg，本移植不带事件；如抱取偏重可后续再调）。

抱取演示物体的配置以本文件的模块级常量为唯一来源，SonicFullscene 直接导入
引用，防止两场景漂移（沿用源分支的设计）。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import locomanipulation_g1_env_cfg as _main

# ---------------------------------------------------------------------------
# 机器人出生点：源分支 SONIC 出生点（仓库行走通道内）。UDP 镜像 action 的
# root_position_mode="relative"（世界系位移叠加到出生位姿），出生点任意可行。
# HugBox 位于出生点正前方 +X 1.05m、胸腰高度（源分支布局，假定 UDP 源朝向 +X；
# 若实际朝向不同，调整 _DEMO_OBJECT_POS/_DEMO_STAND_POS 方位即可）。
# ---------------------------------------------------------------------------
ROBOT_SPAWN_POS = (-2.0, 11.008, 0.78)
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

# 纸箱：仓库同款纸箱 USD（质量/阻尼用 USD 默认值）
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


@configclass
class SonicSoloSceneCfg(_main.LocomanipulationG1SceneCfg):
    """主场景 + 抱取演示物体（机器人/地面/天光/隐藏占位道具原样继承）。"""

    hug_box_stand = HUG_BOX_STAND_CFG
    hug_box = HUG_BOX_CFG


@configclass
class SonicSoloLocomanipulationEnvCfg(_main.LocomanipulationG1EnvCfg):
    """极简场景环境：主环境只换场景，其余（动作/观测/XR/teleop）不动。"""

    scene: SonicSoloSceneCfg = SonicSoloSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=False)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # 出生点移到源分支仓库通道坐标（与 HugBox/fullscene 道具坐标配套）
        self.scene.robot.init_state.pos = ROBOT_SPAWN_POS
