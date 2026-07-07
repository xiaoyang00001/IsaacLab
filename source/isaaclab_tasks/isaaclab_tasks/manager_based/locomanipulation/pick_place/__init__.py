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
    sonic_solo_locomanipulation_env_cfg,
)

# 极简场景（移植自 sonic-windows-xr-ar-anchor 分支，只移植 USD 场景资产；
# 主环境 + HugBox 抱取演示物体，机器人/动作/观测/XR 原样继承主配置）
gym.register(
    id="Isaac-SonicSolo-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_solo_locomanipulation_env_cfg.SonicSoloLocomanipulationEnvCfg,
    },
    disable_env_checker=True,
)

# 完整仓库场景（主环境 + warehouse 背景 + USD 打包台 + 转向盘 + HugBox；
# 传送带仅视觉模型，滚轮物理/流水箱驱动等 events 未移植）
gym.register(
    id="Isaac-SonicFullscene-Locomanipulation-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": sonic_fullscene_locomanipulation_env_cfg.SonicFullsceneLocomanipulationEnvCfg,
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
