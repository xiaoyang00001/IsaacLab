# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import PhysxSchema, UsdGeom, UsdPhysics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def setup_usd_rigid_object_physics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path_template: str = "/World/envs/env_{}/TestBox",
    mass: float = 0.5,
    linear_damping: float = 0.1,
    angular_damping: float = 1000.0,
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
        rigid_api.CreateRigidBodyEnabledAttr(True)

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
        physx_api.CreateLinearDampingAttr(float(linear_damping))
        physx_api.CreateAngularDampingAttr(float(angular_damping))

        print(
            f"[locomanip_event] setup_usd_rigid_object_physics applied on {prim_path} "
            f"(mass={mass}, lin_damp={linear_damping}, ang_damp={angular_damping})"
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
    """Maintain constant linear velocity on an object to simulate conveyor belt motion.

    Called on an interval; overrides x/y velocity each tick so friction cannot slow
    the object down.  z and angular velocities are left unchanged (z) or zeroed (angular).
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
