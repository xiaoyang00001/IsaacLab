# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .cafe_handover_events import _resolve_anchor_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def named_prim_pos(
    env: ManagerBasedRLEnv,
    prim_name: str,
    fallback_pos: tuple[float, float, float],
) -> torch.Tensor:
    """Return a named prim position in environment-local coordinates."""
    positions = torch.zeros((env.scene.num_envs, 3), device=env.device, dtype=torch.float32)
    for env_id in range(env.scene.num_envs):
        world_pos, _, _ = _resolve_anchor_pose(env, env_id, prim_name, fallback_pos)
        positions[env_id, 0] = world_pos[0] - env.scene.env_origins[env_id, 0]
        positions[env_id, 1] = world_pos[1] - env.scene.env_origins[env_id, 1]
        positions[env_id, 2] = world_pos[2] - env.scene.env_origins[env_id, 2]
    return positions
