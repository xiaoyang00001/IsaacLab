# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_DUPLICATE_ANCHOR_WARNINGS: set[tuple[int, str]] = set()


def _select_named_prim_candidate(stage: Usd.Stage, env_id: int, prim_name: str, matches: list[Usd.Prim]) -> Usd.Prim | None:
    """Select the most appropriate prim when multiple prims share the same logical anchor name."""
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        return matches[0]

    background_matches = [prim for prim in matches if f"/World/envs/env_{env_id}/Background/" in prim.GetPath().pathString]
    selected = background_matches[0] if len(background_matches) > 0 else matches[0]

    warning_key = (env_id, prim_name)
    if warning_key not in _DUPLICATE_ANCHOR_WARNINGS:
        all_paths = [prim.GetPath().pathString for prim in matches]
        print(
            f"[cafe_handover] env_{env_id} duplicate anchor prims for {prim_name}: "
            f"selected={selected.GetPath().pathString}, candidates={all_paths}"
        )
        _DUPLICATE_ANCHOR_WARNINGS.add(warning_key)

    return selected


def _find_named_prim_in_env(stage: Usd.Stage, env_id: int, prim_name: str) -> Usd.Prim | None:
    """Find the first prim with the provided name under the environment root."""
    env_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
    if not (env_prim and env_prim.IsValid()):
        return None
    matches = []
    for prim in Usd.PrimRange(env_prim):
        if prim.GetName() == prim_name:
            matches.append(prim)
    return _select_named_prim_candidate(stage, env_id, prim_name, matches)


def _resolve_anchor_pose(
    env: ManagerBasedEnv,
    env_id: int,
    prim_name: str,
    fallback_pos: tuple[float, float, float],
    fallback_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], bool]:
    """Resolve a world pose from a named prim, or fall back to a relative pose."""
    stage = get_current_stage()
    if stage is not None:
        prim = _find_named_prim_in_env(stage, env_id, prim_name)
        if prim is not None and prim.IsValid():
            world_tf = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
            translation = world_tf.ExtractTranslation()
            rotation = world_tf.ExtractRotationQuat()
            return (
                (float(translation[0]), float(translation[1]), float(translation[2])),
                (float(rotation.GetReal()), *[float(v) for v in rotation.GetImaginary()]),
                True,
            )

    env_origin = env.scene.env_origins[env_id]
    return (
        (
            float(env_origin[0].item() + fallback_pos[0]),
            float(env_origin[1].item() + fallback_pos[1]),
            float(env_origin[2].item() + fallback_pos[2]),
        ),
        fallback_quat,
        False,
    )


def place_robots_from_named_prims(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_a_name: str = "robot",
    robot_b_name: str = "remote_robot",
    robot_a_prim_name: str = "RobotSpawnA",
    robot_b_prim_name: str = "RobotSpawnB",
    fallback_robot_a_pos: tuple[float, float, float] = (0.0, 0.0, 0.75),
    fallback_robot_a_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    fallback_robot_b_pos: tuple[float, float, float] = (1.15, 0.0, 0.75),
    fallback_robot_b_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
):
    """Place the two robots from named anchor prims or fallback poses."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    robot_a = env.scene[robot_a_name]
    robot_b = env.scene[robot_b_name]

    robot_a_state = robot_a.data.default_root_state[env_ids].clone()
    robot_b_state = robot_b.data.default_root_state[env_ids].clone()

    for i, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        robot_a_pos, robot_a_quat, a_found = _resolve_anchor_pose(
            env, env_id, robot_a_prim_name, fallback_robot_a_pos, fallback_robot_a_quat
        )
        robot_b_pos, robot_b_quat, b_found = _resolve_anchor_pose(
            env, env_id, robot_b_prim_name, fallback_robot_b_pos, fallback_robot_b_quat
        )

        robot_a_state[i, 0:3] = torch.tensor(robot_a_pos, device=env.device, dtype=robot_a_state.dtype)
        robot_a_state[i, 3:7] = torch.tensor(robot_a_quat, device=env.device, dtype=robot_a_state.dtype)
        robot_a_state[i, 7:13] = 0.0

        robot_b_state[i, 0:3] = torch.tensor(robot_b_pos, device=env.device, dtype=robot_b_state.dtype)
        robot_b_state[i, 3:7] = torch.tensor(robot_b_quat, device=env.device, dtype=robot_b_state.dtype)
        robot_b_state[i, 7:13] = 0.0

        print(
            f"[cafe_handover] env_{env_id} robot anchors: "
            f"{robot_a_prim_name}={'scene' if a_found else 'fallback'} pos={robot_a_pos}, "
            f"{robot_b_prim_name}={'scene' if b_found else 'fallback'} pos={robot_b_pos}"
        )

    robot_a.write_root_pose_to_sim(robot_a_state[:, :7], env_ids=env_ids)
    robot_b.write_root_pose_to_sim(robot_b_state[:, :7], env_ids=env_ids)
    robot_a.write_root_velocity_to_sim(robot_a_state[:, 7:], env_ids=env_ids)
    robot_b.write_root_velocity_to_sim(robot_b_state[:, 7:], env_ids=env_ids)


def report_named_prim_status(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_names: tuple[str, ...] = (
        "RobotSpawnA",
        "RobotSpawnB",
        "CupSpawn",
        "HandoverZone",
        "ServeZone",
        "ViewerAnchor",
    ),
):
    """Print whether the expected task anchor prims exist in the scene."""
    stage = get_current_stage()
    if stage is None:
        print("[cafe_handover] stage unavailable, cannot inspect anchor prims")
        return

    env_id = int(env_ids[0].item()) if env_ids is not None and len(env_ids) > 0 else 0
    found = []
    missing = []
    for prim_name in prim_names:
        prim = _find_named_prim_in_env(stage, env_id, prim_name)
        if prim is not None and prim.IsValid():
            found.append(prim_name)
        else:
            missing.append(prim_name)

    print(
        f"[cafe_handover] env_{env_id} anchor status: "
        f"found={found if found else '[]'}, missing={missing if missing else '[]'}"
    )


def place_rigid_asset_from_named_prim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_name: str = "cup",
    anchor_prim_name: str = "CupSpawn",
    fallback_pos: tuple[float, float, float] = (0.2, 0.42, 0.95),
    fallback_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    """Place a rigid asset from a named anchor prim or fallback pose."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    asset = env.scene[asset_name]
    asset_pose = asset.data.default_root_state[env_ids, :7].clone()
    zero_vel = torch.zeros((len(env_ids), 6), device=env.device, dtype=asset_pose.dtype)

    for i, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        pos, quat, found = _resolve_anchor_pose(env, env_id, anchor_prim_name, fallback_pos, fallback_quat)
        asset_pose[i, 0:3] = torch.tensor(pos, device=env.device, dtype=asset_pose.dtype)
        asset_pose[i, 3:7] = torch.tensor(quat, device=env.device, dtype=asset_pose.dtype)
        print(
            f"[cafe_handover] env_{env_id} asset anchor: "
            f"{asset_name}@{anchor_prim_name}={'scene' if found else 'fallback'} pos={pos}"
        )

    asset.write_root_pose_to_sim(asset_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


def align_viewer_to_named_prim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_name: str = "ViewerAnchor",
    fallback_target: tuple[float, float, float] = (0.62, 0.45, 1.0),
    eye_offset: tuple[float, float, float] = (2.2, 1.8, 1.2),
):
    """Align the GUI viewer to a named prim or a fallback target pose."""
    if not env.sim.has_gui():
        return

    env_id = int(env_ids[0].item()) if env_ids is not None and len(env_ids) > 0 else 0
    target, _, found = _resolve_anchor_pose(env, env_id, prim_name, fallback_target)
    eye = (
        float(target[0] + eye_offset[0]),
        float(target[1] + eye_offset[1]),
        float(target[2] + eye_offset[2]),
    )
    lookat = target

    env.cfg.viewer.origin_type = "world"
    env.cfg.viewer.asset_name = None
    env.cfg.viewer.body_name = None
    env.cfg.viewer.eye = eye
    env.cfg.viewer.lookat = lookat

    if env.viewport_camera_controller is not None:
        env.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
    else:
        env.sim.set_camera_view(eye=eye, target=lookat)

    print(
        f"[cafe_handover] viewer anchor {prim_name}={'scene' if found else 'fallback'} "
        f"eye={eye} lookat={lookat}"
    )
