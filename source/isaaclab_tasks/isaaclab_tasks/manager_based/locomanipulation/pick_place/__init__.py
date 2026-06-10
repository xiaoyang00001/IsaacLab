# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import gymnasium as gym
import os

from . import agents, fixed_base_upper_body_ik_g1_env_cfg, locomanipulation_g1_env_cfg, sonic_solo_locomanipulation_env_cfg

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
