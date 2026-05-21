# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_DIAG_LOG_PATH = os.environ.get("ISAACLAB_DIAG_LOG", r"D:\reboot\diagnostics\isaaclab_diagnostics.log")


def _diag_print(message: str):
    print(message)
    try:
        log_dir = os.path.dirname(_DIAG_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(_DIAG_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(message)
            log_file.write("\n")
    except OSError:
        pass


def _find_named_prim_under_background(stage: Usd.Stage, env_id: int, prim_name: str) -> Usd.Prim | None:
    """Find the first prim with the given name under /Background for an environment."""
    bg_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Background")
    if not (bg_prim and bg_prim.IsValid()):
        return None
    for prim in Usd.PrimRange(bg_prim):
        if prim.GetName() == prim_name:
            return prim
    return None


def setup_usd_rigid_object_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_template: str = "/World/envs/env_{}/TestBox",
    mass: float = 0.5,
    linear_damping: float = 0.1,
    angular_damping: float = 0.1,
    mesh_approximation: str = "convexHull",
    kinematic_enabled: bool = False,
    disable_gravity: bool = False,
):
    """Ensure the target USD prim has rigid-body APIs defined before simulation starts."""
    stage = get_current_stage()
    if stage is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    for env_id in env_ids.tolist():
        prim_path = prim_path_template.format(env_id)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            _diag_print(f"[locomanip_event] setup_usd_rigid_object_physics: prim not found: {prim_path}")
            continue

        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
        if not rigid_api:
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
        if not rigid_enabled_attr:
            rigid_enabled_attr = rigid_api.CreateRigidBodyEnabledAttr()
        rigid_enabled_attr.Set(True)
        kinematic_attr = rigid_api.GetKinematicEnabledAttr()
        if not kinematic_attr:
            kinematic_attr = rigid_api.CreateKinematicEnabledAttr()
        kinematic_attr.Set(bool(kinematic_enabled))

        collision_api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if not collision_api:
            UsdPhysics.CollisionAPI.Apply(prim)

        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if not mass_api:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(float(mass))

        physx_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
        if not physx_api:
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        disable_gravity_attr = physx_api.GetDisableGravityAttr()
        if not disable_gravity_attr:
            disable_gravity_attr = physx_api.CreateDisableGravityAttr()
        disable_gravity_attr.Set(bool(disable_gravity))
        # 注意：CreateLinearDampingAttr(value) 在属性已存在时不更新值，
        # 必须通过 GetXAttr().Set() 确保值写入生效。
        lin_damping = physx_api.GetLinearDampingAttr()
        if not lin_damping:
            lin_damping = physx_api.CreateLinearDampingAttr()
        lin_damping.Set(float(linear_damping))
        ang_damping = physx_api.GetAngularDampingAttr()
        if not ang_damping:
            ang_damping = physx_api.CreateAngularDampingAttr()
        ang_damping.Set(float(angular_damping))

        # For dynamic rigid bodies, triangle mesh collision is not supported.
        # Force mesh collision approximation on the root prim and all child mesh prims.
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath())
        if not mesh_collision_api:
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api.GetApproximationAttr().Set(mesh_approximation)
        for child_prim in Usd.PrimRange(prim):
            if not child_prim.IsA(UsdGeom.Mesh):
                continue
            child_collision_api = UsdPhysics.CollisionAPI.Get(stage, child_prim.GetPath())
            if not child_collision_api:
                UsdPhysics.CollisionAPI.Apply(child_prim)
            child_mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, child_prim.GetPath())
            if not child_mesh_collision_api:
                child_mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
            child_mesh_collision_api.GetApproximationAttr().Set(mesh_approximation)

        _diag_print(
            f"[locomanip_event] setup_usd_rigid_object_physics applied on {prim_path} "
            f"(mass={mass}, lin_damp={linear_damping}, ang_damp={angular_damping}, "
            f"mesh={mesh_approximation}, kinematic={kinematic_enabled}, disable_gravity={disable_gravity})"
        )


def drop_two_balls_with_random_colors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    drop_center_y: float = 1.2,
    left_x: float = -0.45,
    right_x: float = 0.45,
    drop_height_z: float = -0.02,
    lane_jitter_xy: float = 0.02,
    drop_at_conveyor_end: str = "min_x",
):
    """Drop two balls above the conveyor at fixed intervals and randomize their colors.

    Notes:
        - Ball pose/velocity is reset each trigger, so they fall again under gravity.
        - Color randomization is applied to the root sphere prim display color.
    """
    # 中文说明：
    # 该事件用于在 reset / interval 时重新投放红蓝球。
    # 投放点基于 ConveyorTrack 的世界包围盒自动计算，避免硬编码高度。
    # 球体初速度清零，后续由传送带接触运动驱动。
    _diag_print("[locomanip_event] drop_two_balls_with_random_colors triggered")

    red_ball = env.scene["red_ball"]
    blue_ball = env.scene["blue_ball"]
    device = red_ball.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)

    stage = get_current_stage()
    if stage is None:
        return

    # 增加小幅随机扰动，防止每次投放完全重合导致不自然堆叠。
    num_envs = len(env_ids)
    x_jitter = torch.empty((num_envs, 2), device=device).uniform_(-lane_jitter_xy, lane_jitter_xy)
    y_jitter = torch.empty((num_envs, 2), device=device).uniform_(-lane_jitter_xy, lane_jitter_xy)

    red_pose = red_ball.data.default_root_state[env_ids, :7].clone()
    blue_pose = blue_ball.data.default_root_state[env_ids, :7].clone()

    red_pose[:, 0] = left_x + x_jitter[:, 0]
    red_pose[:, 1] = drop_center_y + y_jitter[:, 0]
    red_pose[:, 2] = drop_height_z
    red_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    blue_pose[:, 0] = right_x + x_jitter[:, 1]
    blue_pose[:, 1] = drop_center_y + y_jitter[:, 1]
    blue_pose[:, 2] = drop_height_z
    blue_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # 将相对 env 原点的位姿转换到世界坐标系。
    env_origins = env.scene.env_origins[env_ids]
    red_pose[:, :3] += env_origins
    blue_pose[:, :3] += env_origins

    # 基于传送带包围盒计算“起点端 + 带面上方”的投放点。
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    for i, env_id in enumerate(env_ids.tolist()):
        conveyor_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/ConveyorTrack")
        if not conveyor_prim or not conveyor_prim.IsValid():
            continue
        conveyor_geom = UsdGeom.Imageable(conveyor_prim)
        world_bound = bbox_cache.ComputeWorldBound(conveyor_geom.GetPrim())
        aligned_box = world_bound.ComputeAlignedBox()
        min_pt = aligned_box.GetMin()
        max_pt = aligned_box.GetMax()

        if drop_at_conveyor_end == "max_x":
            start_x = max_pt[0] - 0.06
        else:
            start_x = min_pt[0] + 0.06
        top_z = max_pt[2]

        # x 轴传送带模式：
        # - 两个球从同一侧 x 端投放
        # - 通过 y 方向分道避免初始重叠
        red_pose[i, 0] = start_x + x_jitter[i, 0]
        blue_pose[i, 0] = start_x + x_jitter[i, 1]
        red_pose[i, 1] = drop_center_y - 0.06 + y_jitter[i, 0]
        blue_pose[i, 1] = drop_center_y + 0.06 + y_jitter[i, 1]
        red_pose[i, 2] = top_z + 0.14
        blue_pose[i, 2] = top_z + 0.14

    red_ball.write_root_pose_to_sim(red_pose, env_ids=env_ids)
    blue_ball.write_root_pose_to_sim(blue_pose, env_ids=env_ids)

    # 初速度清零，让球体仅受重力和带面接触作用。
    zero_vel = torch.zeros((num_envs, 6), device=device)
    red_ball.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    blue_ball.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)

    _diag_print(
        f"[locomanip_event] placed red at {red_pose[0, :3].tolist()} blue at {blue_pose[0, :3].tolist()}"
    )

    # 说明：这里不做运行时材质/颜色修改，避免触发某些版本下的图执行不稳定。


def place_robots_from_conveyor_bbox(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    conveyor_prim_name: str = "ConveyorBelt_A08_06",
    robot1_name: str = "robot",
    robot2_name: str = "remote_robot",
    reference_conveyor_center_x: float = 0.62,
    reference_conveyor_min_y: float = 0.98,
    reference_robot1_xy: tuple[float, float] = (0.0, 0.0),
    reference_robot2_xy: tuple[float, float] = (1.25, 0.0),
):
    """Place the two robots using change6-relative offsets from the conveyor bbox."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    stage = get_current_stage()
    if stage is None:
        return

    robot1 = env.scene[robot1_name]
    robot2 = env.scene[robot2_name]

    robot1_state = robot1.data.default_root_state[env_ids].clone()
    robot2_state = robot2.data.default_root_state[env_ids].clone()

    robot1_center_x_offset = reference_robot1_xy[0] - reference_conveyor_center_x
    robot2_center_x_offset = reference_robot2_xy[0] - reference_conveyor_center_x
    robot1_min_y_offset = reference_robot1_xy[1] - reference_conveyor_min_y
    robot2_min_y_offset = reference_robot2_xy[1] - reference_conveyor_min_y

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    for i, env_id in enumerate(env_ids.tolist()):
        conveyor_prim = _find_named_prim_under_background(stage, env_id, conveyor_prim_name)
        if conveyor_prim is None or not conveyor_prim.IsValid():
            _diag_print(f"[locomanip_event] conveyor prim '{conveyor_prim_name}' not found for env_{env_id}")
            continue

        world_bound = bbox_cache.ComputeWorldBound(UsdGeom.Imageable(conveyor_prim).GetPrim())
        aligned_box = world_bound.ComputeAlignedBox()
        min_pt = aligned_box.GetMin()
        max_pt = aligned_box.GetMax()
        center_x = 0.5 * (min_pt[0] + max_pt[0])

        robot1_state[i, 0] = center_x + robot1_center_x_offset
        robot1_state[i, 1] = min_pt[1] + robot1_min_y_offset
        robot2_state[i, 0] = center_x + robot2_center_x_offset
        robot2_state[i, 1] = min_pt[1] + robot2_min_y_offset

        if i == 0:
            _diag_print(
                f"[locomanip_event] aligned robots from {conveyor_prim_name}: "
                f"bbox_min=({min_pt[0]:.4f}, {min_pt[1]:.4f}, {min_pt[2]:.4f}), "
                f"bbox_max=({max_pt[0]:.4f}, {max_pt[1]:.4f}, {max_pt[2]:.4f}), "
                f"robot0=({robot1_state[i, 0]:.4f}, {robot1_state[i, 1]:.4f}, {robot1_state[i, 2]:.4f}), "
                f"robot1=({robot2_state[i, 0]:.4f}, {robot2_state[i, 1]:.4f}, {robot2_state[i, 2]:.4f})"
            )

    robot1.write_root_pose_to_sim(robot1_state[:, :7], env_ids=env_ids)
    robot2.write_root_pose_to_sim(robot2_state[:, :7], env_ids=env_ids)
    robot1.write_root_velocity_to_sim(robot1_state[:, 7:], env_ids=env_ids)
    robot2.write_root_velocity_to_sim(robot2_state[:, 7:], env_ids=env_ids)


def place_test_boxes_from_conveyor_bbox(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    conveyor_prim_name: str = "ConveyorBelt_A08_06",
    test_box_name: str = "test_box",
    test_box1_name: str = "test_box1",
    reference_conveyor_center_x: float = 0.62,
    reference_conveyor_min_y: float = 0.98,
    reference_test_box_xy: tuple[float, float] = (0.78886, 1.17033),
    reference_test_box1_xy: tuple[float, float] = (0.42787, 1.67696),
    box_half_height: float = 0.1,
):
    """Place the two test boxes using change6-relative offsets from the conveyor bbox.

    The old hard-coded box poses were calibrated against change6. This event preserves the
    same relative placement by anchoring each box to the live conveyor bbox instead of the
    world frame, so simple7 can reuse the same intent even if the conveyor moved.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    stage = get_current_stage()
    if stage is None:
        return

    test_box = env.scene[test_box_name]
    test_box1 = env.scene[test_box1_name]
    device = test_box.device

    test_box_pose = test_box.data.default_root_state[env_ids, :7].clone()
    test_box1_pose = test_box1.data.default_root_state[env_ids, :7].clone()
    zero_vel = torch.zeros((len(env_ids), 6), device=device)

    test_box_center_x_offset = reference_test_box_xy[0] - reference_conveyor_center_x
    test_box1_center_x_offset = reference_test_box1_xy[0] - reference_conveyor_center_x
    test_box_min_y_offset = reference_test_box_xy[1] - reference_conveyor_min_y
    test_box1_min_y_offset = reference_test_box1_xy[1] - reference_conveyor_min_y

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    for i, env_id in enumerate(env_ids.tolist()):
        conveyor_prim = _find_named_prim_under_background(stage, env_id, conveyor_prim_name)
        if conveyor_prim is None or not conveyor_prim.IsValid():
            _diag_print(f"[locomanip_event] conveyor prim '{conveyor_prim_name}' not found for env_{env_id}")
            continue

        world_bound = bbox_cache.ComputeWorldBound(UsdGeom.Imageable(conveyor_prim).GetPrim())
        aligned_box = world_bound.ComputeAlignedBox()
        min_pt = aligned_box.GetMin()
        max_pt = aligned_box.GetMax()
        center_x = 0.5 * (min_pt[0] + max_pt[0])
        spawn_z = max_pt[2] + box_half_height

        test_box_pose[i, 0] = center_x + test_box_center_x_offset
        test_box_pose[i, 1] = min_pt[1] + test_box_min_y_offset
        test_box_pose[i, 2] = spawn_z

        test_box1_pose[i, 0] = center_x + test_box1_center_x_offset
        test_box1_pose[i, 1] = min_pt[1] + test_box1_min_y_offset
        test_box1_pose[i, 2] = spawn_z

        if i == 0:
            _diag_print(
                f"[locomanip_event] aligned boxes from {conveyor_prim_name}: "
                f"bbox_min=({min_pt[0]:.4f}, {min_pt[1]:.4f}, {min_pt[2]:.4f}), "
                f"bbox_max=({max_pt[0]:.4f}, {max_pt[1]:.4f}, {max_pt[2]:.4f}), "
                f"box0=({test_box_pose[i, 0]:.4f}, {test_box_pose[i, 1]:.4f}, {test_box_pose[i, 2]:.4f}), "
                f"box1=({test_box1_pose[i, 0]:.4f}, {test_box1_pose[i, 1]:.4f}, {test_box1_pose[i, 2]:.4f})"
            )

    test_box.write_root_pose_to_sim(test_box_pose, env_ids=env_ids)
    test_box1.write_root_pose_to_sim(test_box1_pose, env_ids=env_ids)
    test_box.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    test_box1.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


def align_viewer_to_conveyor_bbox(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    conveyor_prim_name: str = "ConveyorBelt_A08_06",
    reference_conveyor_center_x: float = 0.62,
    reference_conveyor_min_y: float = 0.98,
    reference_viewer_eye: tuple[float, float, float] = (7.5, 7.5, 7.5),
    reference_viewer_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
    viewer_origin_type: str | None = None,
    viewer_asset_name: str | None = None,
    viewer_body_name: str | None = None,
    reference_viewer_target_xy: tuple[float, float] | None = None,
    lock_viewer_to_asset: bool = False,
):
    """Align the initial viewport camera to preserve the change6 first view."""
    if not env.sim.has_gui():
        return

    stage = get_current_stage()
    if stage is None:
        return

    env_id = int(env_ids[0]) if env_ids is not None and len(env_ids) > 0 else 0
    conveyor_prim = _find_named_prim_under_background(stage, env_id, conveyor_prim_name)
    if conveyor_prim is None or not conveyor_prim.IsValid():
        _diag_print(f"[locomanip_event] conveyor prim '{conveyor_prim_name}' not found for env_{env_id}")
        return

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    world_bound = bbox_cache.ComputeWorldBound(UsdGeom.Imageable(conveyor_prim).GetPrim())
    aligned_box = world_bound.ComputeAlignedBox()
    min_pt = aligned_box.GetMin()
    max_pt = aligned_box.GetMax()
    center_x = 0.5 * (min_pt[0] + max_pt[0])

    eye = (
        center_x + (reference_viewer_eye[0] - reference_conveyor_center_x),
        min_pt[1] + (reference_viewer_eye[1] - reference_conveyor_min_y),
        float(reference_viewer_eye[2]),
    )
    lookat = (
        center_x + (reference_viewer_lookat[0] - reference_conveyor_center_x),
        min_pt[1] + (reference_viewer_lookat[1] - reference_conveyor_min_y),
        float(reference_viewer_lookat[2]),
    )

    world_eye = tuple(float(value) for value in eye)
    world_lookat = tuple(float(value) for value in lookat)

    if viewer_asset_name is not None and reference_viewer_target_xy is not None:
        if viewer_origin_type == "asset_body":
            if viewer_body_name is None:
                raise ValueError("viewer_body_name must be provided when viewer_origin_type='asset_body'.")
            asset = env.scene[viewer_asset_name]
            body_id, _ = asset.find_bodies(viewer_body_name)
            viewer_origin = asset.data.body_pos_w[env_id, body_id].view(3)
        else:
            viewer_origin = env.scene[viewer_asset_name].data.root_pos_w[env_id]

        eye_xy_offset = (
            float(reference_viewer_eye[0] - reference_viewer_target_xy[0]),
            float(reference_viewer_eye[1] - reference_viewer_target_xy[1]),
        )
        lookat_xy_offset = (
            float(reference_viewer_lookat[0] - reference_viewer_target_xy[0]),
            float(reference_viewer_lookat[1] - reference_viewer_target_xy[1]),
        )

        world_eye = (
            float(viewer_origin[0].item() + eye_xy_offset[0]),
            float(viewer_origin[1].item() + eye_xy_offset[1]),
            float(reference_viewer_eye[2]),
        )
        world_lookat = (
            float(viewer_origin[0].item() + lookat_xy_offset[0]),
            float(viewer_origin[1].item() + lookat_xy_offset[1]),
            float(reference_viewer_lookat[2]),
        )

    env.cfg.viewer.origin_type = "world"
    env.cfg.viewer.asset_name = None
    env.cfg.viewer.body_name = None
    env.cfg.viewer.eye = world_eye
    env.cfg.viewer.lookat = world_lookat

    if (
        lock_viewer_to_asset
        and viewer_asset_name is not None
        and env.viewport_camera_controller is not None
        and viewer_origin_type in ("asset_root", "asset_body")
    ):
        viewer_origin = None
        if viewer_origin_type == "asset_body":
            if viewer_body_name is None:
                raise ValueError("viewer_body_name must be provided when viewer_origin_type='asset_body'.")
            asset = env.scene[viewer_asset_name]
            body_id, _ = asset.find_bodies(viewer_body_name)
            viewer_origin = asset.data.body_pos_w[env_id, body_id].view(3)
            env.viewport_camera_controller.update_view_to_asset_body(viewer_asset_name, viewer_body_name)
            env.cfg.viewer.body_name = viewer_body_name
        else:
            viewer_origin = env.scene[viewer_asset_name].data.root_pos_w[env_id]
            env.viewport_camera_controller.update_view_to_asset_root(viewer_asset_name)

        rel_eye = tuple(float(world_eye[i] - viewer_origin[i].item()) for i in range(3))
        rel_lookat = tuple(float(world_lookat[i] - viewer_origin[i].item()) for i in range(3))

        env.cfg.viewer.origin_type = viewer_origin_type
        env.cfg.viewer.asset_name = viewer_asset_name
        env.cfg.viewer.eye = rel_eye
        env.cfg.viewer.lookat = rel_lookat
        env.viewport_camera_controller.update_view_location(eye=env.cfg.viewer.eye, lookat=env.cfg.viewer.lookat)
    elif env.viewport_camera_controller is not None:
        env.viewport_camera_controller.update_view_location(eye=env.cfg.viewer.eye, lookat=env.cfg.viewer.lookat)
    else:
        env.sim.set_camera_view(eye=env.cfg.viewer.eye, target=env.cfg.viewer.lookat)

    _diag_print(
        f"[locomanip_event] aligned viewer from {conveyor_prim_name}: "
        f"bbox_min=({min_pt[0]:.4f}, {min_pt[1]:.4f}, {min_pt[2]:.4f}), "
        f"bbox_max=({max_pt[0]:.4f}, {max_pt[1]:.4f}, {max_pt[2]:.4f}), "
        f"lock_to_asset={lock_viewer_to_asset}, "
        f"origin_type={env.cfg.viewer.origin_type}, "
        f"asset={env.cfg.viewer.asset_name}, "
        f"eye=({env.cfg.viewer.eye[0]:.4f}, {env.cfg.viewer.eye[1]:.4f}, {env.cfg.viewer.eye[2]:.4f}), "
        f"lookat=({env.cfg.viewer.lookat[0]:.4f}, {env.cfg.viewer.lookat[1]:.4f}, {env.cfg.viewer.lookat[2]:.4f})"
    )


def print_conveyor_world_bbox(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_name: str = "ConveyorBelt_A08_06",
):
    """Print the world bounding box of a prim in the Background for coordinate calibration.

    Run this as a startup event to get the exact world-space position of a warehouse
    conveyor belt embedded in the background USD.
    """
    stage = get_current_stage()
    if stage is None:
        return

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    for env_id in (env_ids.tolist() if env_ids is not None else [0]):
        prim = _find_named_prim_under_background(stage, env_id, prim_name)
        if prim is None or not prim.IsValid():
            _diag_print(f"[conveyor_bbox] prim '{prim_name}' not found under Background for env_{env_id}")
            continue
        try:
            world_bound = bbox_cache.ComputeWorldBound(UsdGeom.Imageable(prim).GetPrim())
            box = world_bound.ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
            cx = 0.5 * (mn[0] + mx[0])
            cy = 0.5 * (mn[1] + mx[1])
            cz = 0.5 * (mn[2] + mx[2])
            _diag_print(
                f"[conveyor_bbox] {prim_name} world bbox:\n"
                f"  min=({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})\n"
                f"  max=({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})\n"
                f"  center=({cx:.4f}, {cy:.4f}, {cz:.4f})\n"
                f"  belt_surface_z={mx[2]:.4f}\n"
                f"  box_spawn_z (0.1m box half-height)={mx[2]+0.1:.4f}"
            )
        except Exception as e:
            _diag_print(f"[conveyor_bbox] Error computing bbox for {prim.GetPath()}: {e}")
        break  # 只打印第一个 env


def drive_object_on_conveyor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_name: str = "test_box",
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
):
    """Maintain constant linear velocity on an object to simulate conveyor belt motion.

    Called on an interval; overrides x/y velocity each tick so friction cannot slow
    the object down.  z and angular velocities are left unchanged (z) or zeroed (angular).

    Deprecated: Use setup_conveyor_belt_physics (PhysxSurfaceVelocityAPI) instead,
    which drives objects through contact forces rather than direct velocity override.
    """
    obj = env.scene[object_name]
    device = obj.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)

    vel = obj.data.root_vel_w[env_ids].clone()  # (N, 6) [lin_xyz, ang_xyz]
    vel[:, 0] = velocity_x
    vel[:, 1] = velocity_y
    vel[:, 3:] = 0.0          # 清零角速度，防止滚动
    obj.write_root_velocity_to_sim(vel, env_ids=env_ids)

def setup_conveyor_belt_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity: tuple[float, float, float] = (0.0, -0.5, 0.0),
    prim_name_patterns: tuple[str, ...] = ("ConveyorBelt",),
    rollers_name: str = "Rollers",
):
    """对匹配任意 prim_name_patterns 的 ConveyorBelt 的 Rollers 施加 PhysxSurfaceVelocityAPI。

    ConveyorBelt_A08 是 ROLLER 类型——滚轮即承载面。
    直接在 Rollers（已有 RigidBodyAPI + kinematic）上应用表面速度。
    不修改根 Xform，避免层级冲突。

    Args:
        prim_name_patterns: 要匹配的 prim 名称模式列表（substring 匹配，任一匹配即生效）。
    """
    import math
    stage = get_current_stage()
    if stage is None:
        return

    speed = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
    if speed < 1e-8:
        _diag_print("[locomanip_event] setup_conveyor_belt_physics: zero velocity, skipping")
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    for env_id in env_ids.tolist():
        # 递归遍历 env 和 Background，下兼容不同场景里传送带的层级差异。
        conveyor_roots = []
        seen_paths = set()
        for search_root_path in (
            f"/World/envs/env_{env_id}/Background",
            f"/World/envs/env_{env_id}",
        ):
            search_root = stage.GetPrimAtPath(search_root_path)
            if not (search_root and search_root.IsValid()):
                continue
            for prim in Usd.PrimRange(search_root):
                if prim == search_root:
                    continue
                path_str = str(prim.GetPath())
                if path_str in seen_paths:
                    continue
                if any(p in prim.GetName() for p in prim_name_patterns):
                    conveyor_roots.append(prim)
                    seen_paths.add(path_str)

        if not conveyor_roots:
            _diag_print(f"[locomanip_event] No conveyor prims (patterns={prim_name_patterns}) found for env_{env_id}")
            continue

        for root_prim in conveyor_roots:
            root_path = str(root_prim.GetPath())

            # 递归找 Rollers，兼容引用资产内部层级变化。
            rollers_prim = None
            for descendant in Usd.PrimRange(root_prim):
                if descendant == root_prim:
                    continue
                if descendant.GetName() == rollers_name:
                    rollers_prim = descendant
                    break

            if not rollers_prim or not rollers_prim.IsValid():
                _diag_print(f"[locomanip_event]  Rollers prim not found under {root_path}")
                continue

            # 确保 Rollers 有 RigidBodyAPI + kinematic（原始 USD 通常已有）
            if not rollers_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI.Apply(rollers_prim)
            rigid_api = UsdPhysics.RigidBodyAPI(rollers_prim)
            rigid_api.GetKinematicEnabledAttr().Set(True)
            rigid_api.CreateRigidBodyEnabledAttr(True)

            # The conveyor asset stores the visible/top belt contact meshes under
            # children that may be prefixed with "M_". If we skip those meshes,
            # boxes can fall through once this machine becomes the physics source.
            if not rollers_prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(rollers_prim)
            if not rollers_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(rollers_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(rollers_prim)
            mesh_collision_api.GetApproximationAttr().Set("convexHull")

            for mesh_child in Usd.PrimRange(rollers_prim):
                if not mesh_child.IsA(UsdGeom.Mesh):
                    continue
                if not mesh_child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(mesh_child)
                if not mesh_child.HasAPI(UsdPhysics.MeshCollisionAPI):
                    child_mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_child)
                else:
                    child_mesh_collision_api = UsdPhysics.MeshCollisionAPI(mesh_child)
                child_mesh_collision_api.GetApproximationAttr().Set("convexHull")

            # 施加 PhysxSurfaceVelocityAPI 到 Rollers
            # 速度在 prim 局部空间。不需要 world→local 变换，
            # 因为我们只驱动 Rollers，而 Rollers 局部轴与根一致（无中间旋转）。
            if not rollers_prim.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI):
                PhysxSchema.PhysxSurfaceVelocityAPI.Apply(rollers_prim)
            surf_api = PhysxSchema.PhysxSurfaceVelocityAPI(rollers_prim)
            surf_api.GetSurfaceVelocityEnabledAttr().Set(True)
            surf_api.CreateSurfaceVelocityAttr().Set(Gf.Vec3d(*velocity))

            _diag_print(
                f"[locomanip_event]  Rollers SurfaceVelocity "
                f"vel={velocity} -> {str(rollers_prim.GetPath())}"
            )
