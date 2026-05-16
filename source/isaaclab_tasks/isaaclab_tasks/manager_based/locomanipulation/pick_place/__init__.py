# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import os
import gymnasium as gym

from . import (
    agents,
    fixed_base_upper_body_ik_g1_env_cfg,
    locomanipulation_g1_cafe_handover_env_cfg,
    locomanipulation_g1_cafe_handover_kitchenroom_env_cfg,
    locomanipulation_g1_cafe_handover_template_env_cfg,
    locomanipulation_g1_env_cfg,
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

gym.register(
    id="Isaac-CafeHandover-Locomanipulation-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": locomanipulation_g1_cafe_handover_env_cfg.CafeHandoverG1EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-CafeHandover-Locomanipulation-G1-Template-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": locomanipulation_g1_cafe_handover_template_env_cfg.CafeHandoverG1TemplateEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-CafeHandover-Locomanipulation-G1-KitchenRoom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": locomanipulation_g1_cafe_handover_kitchenroom_env_cfg.CafeHandoverG1KitchenRoomEnvCfg,
    },
    disable_env_checker=True,
)
