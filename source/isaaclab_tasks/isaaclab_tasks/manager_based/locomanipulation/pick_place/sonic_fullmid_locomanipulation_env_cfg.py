# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环中档场景：warehouse + sonic + N 台静止陪跑 G1（逼近实时）。

介于两端之间，用于找"完整感"与"实时性"的折中点：
  - SonicFullscene：单 sonic + warehouse，50Hz 实时（陪跑 G1 全裁）
  - SonicFullMulti：4 台 G1 全遥操 + warehouse，headless 仅 22-23Hz（物理墙；
    2026-06 实测 GUI 16-17 / headless 22-23，渲染仅占 ~6Hz，物理本身扛不住实时）

用途：在"带陪跑 G1 的完整仓库场景"里逐档加陪跑、逼近实时测 SONIC 行走。
陪跑数量由 env flag SONIC_FULLMID_COMPANIONS 控制，从 1 台往上加直到 env_hz 掉出
可接受区间，定出能跑实时行走的最大陪跑数。

实现 = **继承 SonicFullscene**（已是 warehouse + 单 sonic + solo 极简 obs/action +
200Hz/dec4 + bind_floor，且 fullscene 摘取式天生不含 align/drive/roller 冲突动态），
仅在 scene 加 N 台 FIXED 陪跑 G1：
  - 陪跑用主配置 FIXED_G1_29DOF_CFG（fix_root_link=True + disable_gravity，行 96-97），
    spawn 即固定根静止站立，**无需** driving/align/fix 事件，也不会触发 fullmulti 那两个
    崩溃根因（对 kinematic 写速度 / 高速滚轮 broad phase）
  - 陪跑不接任何 obs/action/sync/IK —— 纯增 PhysX 物理负载（正是要测的变量），
    同时避开 SonicFullMulti 继承的遥操 obs(robot/remote/eef/object) 与 pinocchio IK
    开销。**因此同样台数下 fullmid 比 fullmulti 更快**。

SONIC_FULLMID_COMPANIONS：0=纯 fullscene；1=+robot；2=+robot+remote_robot；
3=再+walker_robot（默认 1）。⚠️ 第 3 档 walker 仅在 ENABLE_WALKER_ROBOT 未开时为
fixed-root（否则它无驱动会倒）；逼实时用前两档（robot/remote 恒为 FIXED）即可。
陪跑 G1 静止在各自 init_state.pos，与 sonic(-2,+11) 不重叠。

任务 id 含 "Locomanipulation" 复用 teleop deploy_target_mode / U 键回调。
启动命令同 fullscene/fullmulti，仅换 --task 为 Isaac-SonicFullMid-Locomanipulation-G1-v0，
配 SONIC_FULLMID_COMPANIONS=N 调档。
"""

import os

from isaaclab.utils import configclass

from . import locomanipulation_g1_env_cfg as _main
from .sonic_fullscene_locomanipulation_env_cfg import (
    SonicFullsceneLocomanipulationEnvCfg,
    SonicFullsceneSceneCfg,
)

# 保留几台静止陪跑 G1（0=纯 fullscene；1/2/3 逐档加 robot / remote_robot / walker_robot）
SONIC_FULLMID_COMPANIONS = int(os.environ.get("SONIC_FULLMID_COMPANIONS", "1"))

# 摘取主配置场景实例，取陪跑 G1 的 ArticulationCfg（dataclass 实例化时 deepcopy 默认值，
# 与主配置互不影响；同 SonicSolo/SonicFullscene 摘取原则）。
_MAIN_SCENE = _main.LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)


@configclass
class SonicFullMidSceneCfg(SonicFullsceneSceneCfg):
    """fullscene 场景 + N 台静止 fixed 陪跑 G1（按 SONIC_FULLMID_COMPANIONS 档位）。"""

    if SONIC_FULLMID_COMPANIONS >= 1:
        robot = _MAIN_SCENE.robot
    if SONIC_FULLMID_COMPANIONS >= 2:
        remote_robot = _MAIN_SCENE.remote_robot
    if SONIC_FULLMID_COMPANIONS >= 3:
        walker_robot = _MAIN_SCENE.walker_robot


@configclass
class SonicFullMidLocomanipulationEnvCfg(SonicFullsceneLocomanipulationEnvCfg):
    """warehouse + sonic 闭环 + N 静止陪跑 G1（obs/action/events/物理全继承 fullscene）。"""

    scene: SonicFullMidSceneCfg = SonicFullMidSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
