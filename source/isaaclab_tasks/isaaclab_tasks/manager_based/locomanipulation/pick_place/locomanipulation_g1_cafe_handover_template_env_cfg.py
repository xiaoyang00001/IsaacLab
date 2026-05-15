# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from .locomanipulation_g1_cafe_handover_env_cfg import (
    CafeHandoverG1EnvCfg as BaseCafeHandoverG1EnvCfg,
    CafeHandoverG1SceneCfg as BaseCafeHandoverG1SceneCfg,
)


@configclass
class CafeHandoverG1TemplateSceneCfg(BaseCafeHandoverG1SceneCfg):
    """Scene cfg that loads a minimal external USD carrying logical task anchors."""

    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "cafe_handover_scene_template.usda"),
        ),
    )


@configclass
class CafeHandoverG1TemplateEnvCfg(BaseCafeHandoverG1EnvCfg):
    """Drop-in env cfg for scene-template based cafe handover integration."""

    scene: CafeHandoverG1TemplateSceneCfg = CafeHandoverG1TemplateSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
    )
