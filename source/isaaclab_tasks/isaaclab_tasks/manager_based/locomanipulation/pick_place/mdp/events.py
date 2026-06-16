# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf,PhysxSchema, Usd, UsdGeom, UsdPhysics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
            print(f"[locomanip_event] setup_usd_rigid_object_physics: prim not found: {prim_path}")
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
        lin_damping = physx_api.GetLinearDampingAttr()
        if not lin_damping:
            lin_damping = physx_api.CreateLinearDampingAttr()
        lin_damping.Set(float(linear_damping))
        ang_damping = physx_api.GetAngularDampingAttr()
        if not ang_damping:
            ang_damping = physx_api.CreateAngularDampingAttr()
        ang_damping.Set(float(angular_damping))
        if not kinematic_enabled:
            try:
                ccd_attr = physx_api.GetCcdEnabledAttr()
                if not ccd_attr:
                    ccd_attr = physx_api.CreateCcdEnabledAttr()
                ccd_attr.Set(True)
            except Exception:
                pass

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

        print(
            f"[locomanip_event] setup_usd_rigid_object_physics applied on {prim_path} "
            f"(mass={mass}, lin_damp={linear_damping}, ang_damp={angular_damping}, "
            f"mesh={mesh_approximation}, kinematic={kinematic_enabled}, disable_gravity={disable_gravity})"
        )

def stop_box_motion_after_leaving_conveyor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_name: str = "test_box",
    conveyor_prim_name: str = "ConveyorBelt_A08_06",
    xy_margin: float = 0.02,
):
    obj = env.scene[object_name]
    device = obj.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)

    stage = get_current_stage()
    if stage is None:
        return

    bbox_cache = UsdGeom.BBoxCache(
        0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    )

    pos = obj.data.root_pos_w[env_ids].clone()
    vel = obj.data.root_vel_w[env_ids].clone()

    for i, env_id in enumerate(env_ids.tolist()):
        prim_path = f"/World/envs/env_{env_id}/Background/{conveyor_prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue

        world_bound = bbox_cache.ComputeWorldBound(UsdGeom.Imageable(prim).GetPrim())
        box = world_bound.ComputeAlignedBox()
        mn, mx = box.GetMin(), box.GetMax()

        x = pos[i, 0].item()
        y = pos[i, 1].item()

        on_conveyor_xy = (
            (mn[0] - xy_margin) <= x <= (mx[0] + xy_margin)
            and (mn[1] - xy_margin) <= y <= (mx[1] + xy_margin)
        )

        if not on_conveyor_xy:
            vel[i, 0] = 0.0
            vel[i, 1] = 0.0
            vel[i, 3:] = 0.0

    obj.write_root_velocity_to_sim(vel, env_ids=env_ids)


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
    print("[locomanip_event] drop_two_balls_with_random_colors triggered")

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

    print(
        f"[locomanip_event] placed red at {red_pose[0, :3].tolist()} blue at {blue_pose[0, :3].tolist()}"
    )

    # 说明：这里不做运行时材质/颜色修改，避免触发某些版本下的图执行不稳定。


def align_robots_to_conveyor_center_x(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot1_name: str = "robot",
    robot2_name: str = "robot2",
):
    """Align both robots' x position to conveyor bbox center x on reset."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    stage = get_current_stage()
    if stage is None:
        return

    robot1 = env.scene[robot1_name]
    robot2 = env.scene[robot2_name]

    robot1_pose = robot1.data.root_state_w[env_ids, :7].clone()
    robot2_pose = robot2.data.root_state_w[env_ids, :7].clone()

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
        center_x = 0.5 * (min_pt[0] + max_pt[0])
        if i == 0:
            print(
                f"[locomanip_event] conveyor x-range=({min_pt[0]:.4f}, {max_pt[0]:.4f}), "
                f"length={max_pt[0] - min_pt[0]:.4f}, center_x={center_x:.4f}"
            )

        robot1_pose[i, 0] = center_x
        robot2_pose[i, 0] = center_x

    robot1.write_root_pose_to_sim(robot1_pose, env_ids=env_ids)
    robot2.write_root_pose_to_sim(robot2_pose, env_ids=env_ids)


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
        prim_path = f"/World/envs/env_{env_id}/Background/{prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            print(f"[conveyor_bbox] prim not found: {prim_path}")
            continue
        try:
            world_bound = bbox_cache.ComputeWorldBound(UsdGeom.Imageable(prim).GetPrim())
            box = world_bound.ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
            cx = 0.5 * (mn[0] + mx[0])
            cy = 0.5 * (mn[1] + mx[1])
            cz = 0.5 * (mn[2] + mx[2])
            print(
                f"[conveyor_bbox] {prim_name} world bbox:\n"
                f"  min=({mn[0]:.4f}, {mn[1]:.4f}, {mn[2]:.4f})\n"
                f"  max=({mx[0]:.4f}, {mx[1]:.4f}, {mx[2]:.4f})\n"
                f"  center=({cx:.4f}, {cy:.4f}, {cz:.4f})\n"
                f"  belt_surface_z={mx[2]:.4f}\n"
                f"  box_spawn_z (0.1m box half-height)={mx[2]+0.1:.4f}"
            )
        except Exception as e:
            print(f"[conveyor_bbox] Error computing bbox for {prim_path}: {e}")
        break  # 只打印第一个 env


def drive_object_on_conveyor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_name: str = "test_box",
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
):
    """Only drive the box while it is still on the conveyor."""
    obj = env.scene[object_name]
    device = obj.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)

    if len(env_ids) == 0:
        return

    root_pos = obj.data.root_pos_w[env_ids]

    ref_key = f"_conveyor_ref_{object_name}"
    if not hasattr(env, ref_key):
        min_z_key = f"_conveyor_min_z_{object_name}"
        init_z_key = f"_conveyor_init_z_{object_name}"
        count_key = f"_conveyor_stable_count_{object_name}"
        timeout_key = f"_conveyor_timeout_{object_name}"

        if not hasattr(env, min_z_key):
            setattr(env, min_z_key, root_pos[:, 2].clone())
            setattr(env, init_z_key, root_pos[:, 2].clone())
            setattr(env, count_key, 0)
            print(f"[drive_conveyor] {object_name}: settling detect from z={root_pos[0, 2].item():.3f}")
            return

        prev_min_z = getattr(env, min_z_key)
        new_min_z = torch.min(prev_min_z, root_pos[:, 2])
        setattr(env, min_z_key, new_min_z)
        initial_z = getattr(env, init_z_key)

        if new_min_z[0] < prev_min_z[0]:
            setattr(env, count_key, 0)
        else:
            count = getattr(env, count_key, 0) + 1
            setattr(env, count_key, count)
            if count >= 3 and (initial_z[0] - new_min_z[0]) >= 0.03:
                setattr(env, ref_key, root_pos.clone())
                print(
                    f"[drive_conveyor] {object_name}: settled at conveyor reference: "
                    f"z={root_pos[0, 2].item():.3f}, fall={initial_z[0].item() - new_min_z[0].item():.3f}m"
                )

        timeout = getattr(env, timeout_key, 0) + 1
        setattr(env, timeout_key, timeout)
        if timeout > 100:
            setattr(env, ref_key, root_pos.clone())
            print(f"[drive_conveyor] {object_name}: timeout forcing reference at z={root_pos[0, 2].item():.3f}")
        return

    ref_pos = getattr(env, ref_key)

    z_on_belt = (root_pos[:, 2] >= ref_pos[:, 2] - 0.15) & (root_pos[:, 2] <= ref_pos[:, 2] + 0.15)
    x_on_belt = (root_pos[:, 0] >= ref_pos[:, 0] - 2.0) & (root_pos[:, 0] <= ref_pos[:, 0] + 2.0)
    y_on_belt = root_pos[:, 1] >= ref_pos[:, 1] - 9.0
    on_belt = z_on_belt & x_on_belt & y_on_belt

    if not on_belt.any():
        return

    env_ids_to_drive = env_ids[on_belt]
    vel = obj.data.root_vel_w[env_ids_to_drive].clone()
    vel[:, 0] = velocity_x
    vel[:, 1] = velocity_y
    obj.write_root_velocity_to_sim(vel, env_ids=env_ids_to_drive)

def setup_conveyor_belt_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity: tuple[float, float, float] = (0.0, -0.5, 0.0),
    prim_name_patterns: tuple[str, ...] = ("ConveyorBelt",),
    rollers_name: str = "Rollers",
    roller_radius: float = 0.028951416,
    rotation_axis: str = "X",
    keep_rollers_parent_collision: bool = False,
):
    """Configure the conveyor as a solid collider without PhysxSurfaceVelocityAPI."""
    import math

    stage = get_current_stage()
    if stage is None:
        return

    speed = math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
    if speed < 1e-8:
        print("[locomanip_event] setup_conveyor_belt_physics: zero velocity, skipping")
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    for env_id in env_ids.tolist():
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
                if any(pattern in prim.GetName() for pattern in prim_name_patterns):
                    conveyor_roots.append(prim)
                    seen_paths.add(path_str)

        if not conveyor_roots:
            print(f"[locomanip_event] No conveyor prims found for env_{env_id}")
            continue

        for root_prim in conveyor_roots:
            rollers_prim = None
            for descendant in Usd.PrimRange(root_prim):
                if descendant == root_prim:
                    continue
                if descendant.GetName() == rollers_name:
                    rollers_prim = descendant
                    break

            if not rollers_prim or not rollers_prim.IsValid():
                print(f"[locomanip_event] Rollers prim not found under {str(root_prim.GetPath())}")
                continue

            if rollers_prim.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI):
                surf_api = PhysxSchema.PhysxSurfaceVelocityAPI(rollers_prim)
                try:
                    surf_api.GetSurfaceVelocityEnabledAttr().Set(False)
                except Exception:
                    pass

            if not keep_rollers_parent_collision:
                if rollers_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision_api = UsdPhysics.CollisionAPI(rollers_prim)
                    collision_api.GetCollisionEnabledAttr().Set(False)
                if rollers_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_api = UsdPhysics.RigidBodyAPI(rollers_prim)
                    rigid_enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
                    if rigid_enabled_attr:
                        rigid_enabled_attr.Set(False)
                    kinematic_attr = rigid_api.GetKinematicEnabledAttr()
                    if kinematic_attr:
                        kinematic_attr.Set(False)

            configured_meshes = 0
            for mesh_child in Usd.PrimRange(rollers_prim):
                if not mesh_child.IsA(UsdGeom.Mesh):
                    continue
                if mesh_child.GetName().startswith("M_"):
                    continue
                if not mesh_child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(mesh_child)
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, mesh_child.GetPath())
                if not mesh_collision_api:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_child)
                mesh_collision_api.GetApproximationAttr().Set("convexHull")
                configured_meshes += 1

            print(
                f"[locomanip_event] GPU-safe conveyor collider configured: "
                f"{str(rollers_prim.GetPath())}, meshes={configured_meshes}, velocity_ref={velocity}"
            )
