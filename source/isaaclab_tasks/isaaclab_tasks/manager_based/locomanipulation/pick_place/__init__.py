# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import gymnasium as gym
import os

from . import (
    agents,
    fixed_base_upper_body_ik_g1_env_cfg,
    locomanipulation_g1_env_cfg,
    sonic_fullscene_locomanipulation_env_cfg,
    sonic_pickplace_ref_locomanipulation_env_cfg,
    sonic_solo_locomanipulation_env_cfg,
)

# SONIC 闭环物理调试极简场景（id 含 "Locomanipulation" 以复用 teleop 脚本的
# deploy_target_mode 判定与 U 键回调；场景只有 sonic_robot + 地面 + 天光）
gym.register(
    id="Isaac-SonicSolo-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_solo_locomanipulation_env_cfg.SonicSoloLocomanipulationEnvCfg,
    },
    disable_env_checker=True,
)

# SONIC 闭环 Phase 1：完整仓库场景（warehouse 背景 + 单台 SONIC G1，陪跑机器人
# 裁除，物理 200Hz/dec4 对齐闭环七条件；SONIC_FULLSCENE_CONVEYOR=1 加载传送带）
gym.register(
    id="Isaac-SonicFullscene-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_fullscene_locomanipulation_env_cfg.SonicFullsceneLocomanipulationEnvCfg,
    },
    disable_env_checker=True,
)

# 跟踪 origin/0716-校验 的 pickplace 参考场景（打包桌 + 3 箱），机器人栈沿用本分支
# 主配置（robot_1 43dof 镜像 / robot_2 / sonic_robot，靠 LOCOMANIP_SONIC_REPLACE_ROBOT1
# 与 LOCOMANIP_ENABLE_ROBOT2 切换）。id 含 "Locomanipulation-G1" 以复用 teleop 脚本的
# motion_controllers 默认 + 强制 pinocchio + deploy/U 键回调链路。
gym.register(
    id="Isaac-PickPlaceRef-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_pickplace_ref_locomanipulation_env_cfg.SonicPickPlaceRefLocomanipulationEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": locomanipulation_g1_env_cfg.LocomanipulationG1EnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": fixed_base_upper_body_ik_g1_env_cfg.FixedBaseUpperBodyIKG1EnvCfg,
    },
    disable_env_checker=True,
)
