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
    sonic_fullmid_locomanipulation_env_cfg,
    sonic_fullmulti_locomanipulation_env_cfg,
    sonic_fullscene_locomanipulation_env_cfg,
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

# SONIC 闭环：完整主配置场景（4 台 G1 + 双机遥操 + 传送带全保留，主配置文件一行不动，
# 继承 LocomanipulationG1EnvCfg 覆盖物理 200Hz/dec4 对齐闭环七条件 + 补 warehouse 地板摩擦）
gym.register(
    id="Isaac-SonicFullMulti-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_fullmulti_locomanipulation_env_cfg.SonicFullMultiLocomanipulationEnvCfg,
    },
    disable_env_checker=True,
)

# SONIC 闭环中档：fullscene + N 台静止陪跑 G1（env flag SONIC_FULLMID_COMPANIONS 控制档位），
# 在带陪跑 G1 的完整仓库场景里逼近实时测行走（继承 fullscene 的 200Hz/dec4 + 干净配置）
gym.register(
    id="Isaac-SonicFullMid-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_fullmid_locomanipulation_env_cfg.SonicFullMidLocomanipulationEnvCfg,
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
