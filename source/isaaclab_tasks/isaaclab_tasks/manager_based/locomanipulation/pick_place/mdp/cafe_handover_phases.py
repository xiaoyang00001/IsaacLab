# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .cafe_handover_events import _resolve_anchor_pose
from .cafe_handover_terminations import cup_in_serve_zone

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NUM_CAFE_HANDOVER_PHASES = 5
PHASE_ID_TO_NAME = {
    -1: "uninitialized",
    0: "initialized",
    1: "pickup_success",
    2: "handover_zone_reached",
    3: "handover_success",
    4: "serve_success",
}


def _named_prim_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    prim_name: str,
    fallback_pos: tuple[float, float, float],
) -> torch.Tensor:
    """Return named prim positions expressed in environment-local coordinates."""
    positions = torch.zeros((env.scene.num_envs, 3), device=env.device, dtype=torch.float32)
    for env_id in range(env.scene.num_envs):
        world_pos, _, _ = _resolve_anchor_pose(env, env_id, prim_name, fallback_pos)
        world_pos_tensor = torch.tensor(world_pos, device=env.device, dtype=positions.dtype)
        positions[env_id] = world_pos_tensor - env.scene.env_origins[env_id].to(dtype=positions.dtype)
    return positions


def _robot_eef_pos_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> torch.Tensor:
    """Return left and right end-effector positions in environment-local coordinates."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_pos = robot.data.body_pos_w - env.scene.env_origins[:, None, :]
    left_idx = robot.data.body_names.index(left_eef_link_name)
    right_idx = robot.data.body_names.index(right_eef_link_name)
    return torch.stack((body_pos[:, left_idx], body_pos[:, right_idx]), dim=1)


def _cup_upright_mask(
    asset: RigidObject,
    max_tilt_deg: float,
) -> torch.Tensor:
    """Return whether the cup tilt stays within the provided threshold."""
    world_up = torch.tensor((0.0, 0.0, 1.0), device=asset.device, dtype=asset.data.root_quat_w.dtype).repeat(
        asset.data.root_quat_w.shape[0], 1
    )
    cup_up = quat_apply(asset.data.root_quat_w, world_up)
    cos_threshold = torch.cos(torch.deg2rad(torch.tensor(max_tilt_deg, device=asset.device, dtype=cup_up.dtype)))
    return cup_up[:, 2] > cos_threshold


def _compute_phase_flags(
    env: ManagerBasedRLEnv,
    cup_spawn_prim_name: str,
    fallback_cup_spawn_pos: tuple[float, float, float],
    handover_zone_prim_name: str,
    fallback_handover_zone_pos: tuple[float, float, float],
    serve_zone_prim_name: str,
    fallback_serve_zone_pos: tuple[float, float, float],
    min_pickup_height: float,
    max_handover_xy_error: float,
    max_handover_z_error: float,
    max_upright_tilt_deg: float,
    receiver_advantage_margin: float,
    handover_progress_margin: float,
    cup_asset_cfg: SceneEntityCfg,
    giver_asset_cfg: SceneEntityCfg,
    receiver_asset_cfg: SceneEntityCfg,
    left_eef_link_name: str,
    right_eef_link_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer cumulative task milestones from the current scene state."""
    cup: RigidObject = env.scene[cup_asset_cfg.name]
    cup_pos = cup.data.root_pos_w - env.scene.env_origins

    cup_spawn_pos = _named_prim_pos_in_env_frame(env, cup_spawn_prim_name, fallback_cup_spawn_pos)
    handover_zone_pos = _named_prim_pos_in_env_frame(env, handover_zone_prim_name, fallback_handover_zone_pos)
    serve_zone_pos = _named_prim_pos_in_env_frame(env, serve_zone_prim_name, fallback_serve_zone_pos)

    giver_eef_pos = _robot_eef_pos_in_env_frame(env, giver_asset_cfg, left_eef_link_name, right_eef_link_name)
    receiver_eef_pos = _robot_eef_pos_in_env_frame(env, receiver_asset_cfg, left_eef_link_name, right_eef_link_name)

    cup_lifted_now = cup_pos[:, 2] > (cup_spawn_pos[:, 2] + min_pickup_height)
    cup_upright_now = _cup_upright_mask(cup, max_upright_tilt_deg)

    handover_delta = cup_pos - handover_zone_pos
    in_handover_zone_now = (
        (torch.abs(handover_delta[:, 0]) < max_handover_xy_error)
        & (torch.abs(handover_delta[:, 1]) < max_handover_xy_error)
        & (torch.abs(handover_delta[:, 2]) < max_handover_z_error)
    )

    giver_dist = torch.linalg.vector_norm(giver_eef_pos - cup_pos[:, None, :], dim=2).amin(dim=1)
    receiver_dist = torch.linalg.vector_norm(receiver_eef_pos - cup_pos[:, None, :], dim=2).amin(dim=1)
    receiver_closer_now = receiver_dist + receiver_advantage_margin < giver_dist

    handover_dist = torch.linalg.vector_norm(cup_pos - handover_zone_pos, dim=1)
    serve_dist = torch.linalg.vector_norm(cup_pos - serve_zone_pos, dim=1)
    moved_beyond_handover_now = serve_dist + handover_progress_margin < handover_dist

    serve_success = cup_in_serve_zone(
        env,
        serve_zone_prim_name=serve_zone_prim_name,
        fallback_target_pos=fallback_serve_zone_pos,
        asset_cfg=cup_asset_cfg,
    )
    handover_success = serve_success | (
        cup_upright_now
        & receiver_closer_now
        & (in_handover_zone_now | moved_beyond_handover_now)
    )
    handover_zone_reached = handover_success | (cup_upright_now & in_handover_zone_now)
    pickup_success = handover_zone_reached | cup_lifted_now

    return pickup_success, handover_zone_reached, handover_success, serve_success


def _flag_to_obs(flag: torch.Tensor) -> torch.Tensor:
    return flag.unsqueeze(-1).to(dtype=torch.float32)


def task_phase_index(
    env: ManagerBasedRLEnv,
    cup_spawn_prim_name: str = "CupSpawn",
    fallback_cup_spawn_pos: tuple[float, float, float] = (0.2, 0.42, 0.95),
    handover_zone_prim_name: str = "HandoverZone",
    fallback_handover_zone_pos: tuple[float, float, float] = (0.62, 0.42, 0.98),
    serve_zone_prim_name: str = "ServeZone",
    fallback_serve_zone_pos: tuple[float, float, float] = (1.0, 0.48, 0.95),
    min_pickup_height: float = 0.08,
    max_handover_xy_error: float = 0.14,
    max_handover_z_error: float = 0.12,
    max_upright_tilt_deg: float = 35.0,
    receiver_advantage_margin: float = 0.04,
    handover_progress_margin: float = 0.02,
    cup_asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    giver_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    receiver_asset_cfg: SceneEntityCfg = SceneEntityCfg("remote_robot"),
    left_eef_link_name: str = "left_wrist_yaw_link",
    right_eef_link_name: str = "right_wrist_yaw_link",
) -> torch.Tensor:
    """Return the current task phase as a single scalar observation.

    Phase ids:
        0 = initialized / idle
        1 = pickup success
        2 = cup reached handover zone
        3 = receiver-side handover success
        4 = cup stably served
    """
    pickup_success, handover_zone_reached, handover_success, serve_success = _compute_phase_flags(
        env=env,
        cup_spawn_prim_name=cup_spawn_prim_name,
        fallback_cup_spawn_pos=fallback_cup_spawn_pos,
        handover_zone_prim_name=handover_zone_prim_name,
        fallback_handover_zone_pos=fallback_handover_zone_pos,
        serve_zone_prim_name=serve_zone_prim_name,
        fallback_serve_zone_pos=fallback_serve_zone_pos,
        min_pickup_height=min_pickup_height,
        max_handover_xy_error=max_handover_xy_error,
        max_handover_z_error=max_handover_z_error,
        max_upright_tilt_deg=max_upright_tilt_deg,
        receiver_advantage_margin=receiver_advantage_margin,
        handover_progress_margin=handover_progress_margin,
        cup_asset_cfg=cup_asset_cfg,
        giver_asset_cfg=giver_asset_cfg,
        receiver_asset_cfg=receiver_asset_cfg,
        left_eef_link_name=left_eef_link_name,
        right_eef_link_name=right_eef_link_name,
    )

    phase = torch.zeros((env.scene.num_envs,), device=env.device, dtype=torch.long)
    phase[pickup_success] = 1
    phase[handover_zone_reached] = 2
    phase[handover_success] = 3
    phase[serve_success] = 4
    return phase.unsqueeze(-1).to(dtype=torch.float32)


def task_phase_one_hot(
    env: ManagerBasedRLEnv,
    **kwargs,
) -> torch.Tensor:
    """Return the current task phase as a one-hot observation vector."""
    phase_index = task_phase_index(env, **kwargs).squeeze(-1).to(dtype=torch.long)
    return F.one_hot(phase_index, num_classes=NUM_CAFE_HANDOVER_PHASES).to(dtype=torch.float32)


def pickup_success_flag(
    env: ManagerBasedRLEnv,
    **kwargs,
) -> torch.Tensor:
    """Return whether pickup has been achieved under the current heuristic."""
    pickup_success, _, _, _ = _compute_phase_flags(env=env, **kwargs)
    return _flag_to_obs(pickup_success)


def handover_zone_reached_flag(
    env: ManagerBasedRLEnv,
    **kwargs,
) -> torch.Tensor:
    """Return whether the cup has reached the handover zone under the current heuristic."""
    _, handover_zone_reached, _, _ = _compute_phase_flags(env=env, **kwargs)
    return _flag_to_obs(handover_zone_reached)


def handover_success_flag(
    env: ManagerBasedRLEnv,
    **kwargs,
) -> torch.Tensor:
    """Return whether receiver-side handover has been achieved under the current heuristic."""
    _, _, handover_success, _ = _compute_phase_flags(env=env, **kwargs)
    return _flag_to_obs(handover_success)


def serve_success_flag(
    env: ManagerBasedRLEnv,
    **kwargs,
) -> torch.Tensor:
    """Return whether the cup has been stably placed in the serve zone."""
    _, _, _, serve_success = _compute_phase_flags(env=env, **kwargs)
    return _flag_to_obs(serve_success)


class log_phase_transitions(ManagerTermBase):
    """Log cafe handover phase transitions at runtime without spamming every step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_phase = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_phase[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        cup_spawn_prim_name: str = "CupSpawn",
        fallback_cup_spawn_pos: tuple[float, float, float] = (0.2, 0.42, 0.95),
        handover_zone_prim_name: str = "HandoverZone",
        fallback_handover_zone_pos: tuple[float, float, float] = (0.62, 0.42, 0.98),
        serve_zone_prim_name: str = "ServeZone",
        fallback_serve_zone_pos: tuple[float, float, float] = (1.0, 0.48, 0.95),
        min_pickup_height: float = 0.08,
        max_handover_xy_error: float = 0.14,
        max_handover_z_error: float = 0.12,
        max_upright_tilt_deg: float = 35.0,
        receiver_advantage_margin: float = 0.04,
        handover_progress_margin: float = 0.02,
        cup_asset_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
        giver_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        receiver_asset_cfg: SceneEntityCfg = SceneEntityCfg("remote_robot"),
        left_eef_link_name: str = "left_wrist_yaw_link",
        right_eef_link_name: str = "right_wrist_yaw_link",
        log_env_ids: tuple[int, ...] = (0,),
    ) -> None:
        phase = (
            task_phase_index(
                env=env,
                cup_spawn_prim_name=cup_spawn_prim_name,
                fallback_cup_spawn_pos=fallback_cup_spawn_pos,
                handover_zone_prim_name=handover_zone_prim_name,
                fallback_handover_zone_pos=fallback_handover_zone_pos,
                serve_zone_prim_name=serve_zone_prim_name,
                fallback_serve_zone_pos=fallback_serve_zone_pos,
                min_pickup_height=min_pickup_height,
                max_handover_xy_error=max_handover_xy_error,
                max_handover_z_error=max_handover_z_error,
                max_upright_tilt_deg=max_upright_tilt_deg,
                receiver_advantage_margin=receiver_advantage_margin,
                handover_progress_margin=handover_progress_margin,
                cup_asset_cfg=cup_asset_cfg,
                giver_asset_cfg=giver_asset_cfg,
                receiver_asset_cfg=receiver_asset_cfg,
                left_eef_link_name=left_eef_link_name,
                right_eef_link_name=right_eef_link_name,
            )
            .squeeze(-1)
            .to(dtype=torch.long)
        )

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if len(log_env_ids) > 0:
            allowed_env_ids = torch.tensor(log_env_ids, device=self.device, dtype=torch.long)
            mask = (env_ids[:, None] == allowed_env_ids[None, :]).any(dim=1)
            env_ids = env_ids[mask]

        if len(env_ids) == 0:
            return

        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            previous_phase = int(self._previous_phase[env_id].item())
            current_phase = int(phase[env_id].item())
            if previous_phase != current_phase:
                print(
                    f"[cafe_handover] env_{env_id} phase: "
                    f"{PHASE_ID_TO_NAME.get(previous_phase, 'unknown')}({previous_phase}) -> "
                    f"{PHASE_ID_TO_NAME.get(current_phase, 'unknown')}({current_phase})"
                )

        self._previous_phase[env_ids] = phase[env_ids]
