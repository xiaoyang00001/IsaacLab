# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from .locomanipulation_g1_cafe_handover_env_cfg import (
    CafeHandoverG1EnvCfg as BaseCafeHandoverG1EnvCfg,
    CafeHandoverG1SceneCfg as BaseCafeHandoverG1SceneCfg,
)


def _resolve_lightwheel_kitchen_room_usd_path() -> str:
    """Resolve the Lightwheel KitchenRoom USD path from env var or default download location."""
    root_dir = os.environ.get(
        "LIGHTWHEEL_OPEN_SOURCE_ROOT_DIR",
        r"D:\Downloads\Lightwheel_OpenSource\Lightwheel_OpenSource",
    )
    return os.path.join(root_dir, "Locomotion", "KitchenRoom", "KitchenRoom.usd")


LIGHTWHEEL_KITCHEN_ROOM_USD_PATH = _resolve_lightwheel_kitchen_room_usd_path()


@configclass
class CafeHandoverG1KitchenRoomSceneCfg(BaseCafeHandoverG1SceneCfg):
    """Scene cfg that layers the Lightwheel KitchenRoom USD behind the cafe handover task."""

    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=LIGHTWHEEL_KITCHEN_ROOM_USD_PATH,
        ),
    )


@configclass
class CafeHandoverG1KitchenRoomEnvCfg(BaseCafeHandoverG1EnvCfg):
    """Cafe handover env cfg using the downloaded Lightwheel KitchenRoom background."""

    scene: CafeHandoverG1KitchenRoomSceneCfg = CafeHandoverG1KitchenRoomSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
    )
