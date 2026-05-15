# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .cafe_handover_events import _resolve_anchor_pose

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cup_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.55,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Terminate if the cup falls below a minimum world height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height


def cup_tilt_exceeded(
    env: ManagerBasedRLEnv,
    max_tilt_deg: float = 55.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Terminate if the cup tilts too far away from the world up-axis."""
    asset: RigidObject = env.scene[asset_cfg.name]
    world_up = torch.tensor((0.0, 0.0, 1.0), device=asset.device, dtype=asset.data.root_quat_w.dtype).repeat(
        asset.data.root_quat_w.shape[0], 1
    )
    cup_up = quat_apply(asset.data.root_quat_w, world_up)
    cos_threshold = torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, device=asset.device, dtype=cup_up.dtype)))
    return cup_up[:, 2] < cos_threshold


def cup_in_serve_zone(
    env: ManagerBasedRLEnv,
    serve_zone_prim_name: str = "ServeZone",
    fallback_target_pos: tuple[float, float, float] = (1.0, 0.48, 0.95),
    max_xy_error: float = 0.12,
    max_z_error: float = 0.10,
    max_tilt_deg: float = 25.0,
    max_linear_speed: float = 0.30,
    max_angular_speed: float = 1.50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
) -> torch.Tensor:
    """Return success when the cup is upright, stable, and inside the serve zone."""
    asset: RigidObject = env.scene[asset_cfg.name]
    asset_pos = asset.data.root_pos_w
    asset_lin_speed = torch.linalg.vector_norm(asset.data.root_lin_vel_w, dim=1)
    asset_ang_speed = torch.linalg.vector_norm(asset.data.root_ang_vel_w, dim=1)

    world_up = torch.tensor((0.0, 0.0, 1.0), device=asset.device, dtype=asset.data.root_quat_w.dtype).repeat(
        asset.data.root_quat_w.shape[0], 1
    )
    cup_up = quat_apply(asset.data.root_quat_w, world_up)
    cos_threshold = torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, device=asset.device, dtype=cup_up.dtype)))

    done = torch.zeros((env.scene.num_envs,), device=asset.device, dtype=torch.bool)
    for env_id in range(env.scene.num_envs):
        target_pos, _, _ = _resolve_anchor_pose(env, env_id, serve_zone_prim_name, fallback_target_pos)
        dx = torch.abs(asset_pos[env_id, 0] - target_pos[0])
        dy = torch.abs(asset_pos[env_id, 1] - target_pos[1])
        dz = torch.abs(asset_pos[env_id, 2] - target_pos[2])
        done[env_id] = (
            (dx < max_xy_error)
            & (dy < max_xy_error)
            & (dz < max_z_error)
            & (cup_up[env_id, 2] > cos_threshold)
            & (asset_lin_speed[env_id] < max_linear_speed)
            & (asset_ang_speed[env_id] < max_angular_speed)
        )
    return done
