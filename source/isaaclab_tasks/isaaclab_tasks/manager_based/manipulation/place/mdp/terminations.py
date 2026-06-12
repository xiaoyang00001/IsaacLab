# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the place task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_placed_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_height: float = 0.927,
    euler_xy_threshold: float = 0.10,
):
    """Check if an object placed upright by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Compute mug euler angles of X, Y axis, to check if it is placed upright
    object_euler_x, object_euler_y, _ = math_utils.euler_xyz_from_quat(object.data.root_quat_w)  # (N,4) [0, 2*pi]

    object_euler_x_err = torch.abs(math_utils.wrap_to_pi(object_euler_x))  # (N,)
    object_euler_y_err = torch.abs(math_utils.wrap_to_pi(object_euler_y))  # (N,)

    success = torch.logical_and(object_euler_x_err < euler_xy_threshold, object_euler_y_err < euler_xy_threshold)

    # Check if current mug height is greater than target height
    height_success = object.data.root_pos_w[:, 2] > target_height

    success = torch.logical_and(height_success, success)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        success = torch.logical_and(suction_cup_is_open, success)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return success


def object_a_is_into_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.03,  # xy_distance_threshold
    height_threshold: float = 0.04,  # height_distance_threshold
    height_diff: float = 0.0,  # expected height_diff
) -> torch.Tensor:
    """Check if an object a is put into another object b by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    # check object a is into object b
    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    success = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        success = torch.logical_and(suction_cup_is_open, success)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return success


def objects_are_inside_box(
    env: ManagerBasedRLEnv,
    object_cfgs: tuple[SceneEntityCfg, ...] = (
        SceneEntityCfg("cube_1"),
        SceneEntityCfg("cube_2"),
        SceneEntityCfg("cube_3"),
    ),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    xy_threshold: float | tuple[float, float] = 0.14,
    z_min: float = -0.08,
    z_max: float = 0.16,
) -> torch.Tensor:
    """Check that all configured objects are inside the target box volume.

    The check is intentionally geometric and does not require the gripper to be open.
    This supports teleoperation tasks where success means that every cube has been
    placed into the container.
    """

    box: RigidObject = env.scene[box_cfg.name]
    box_pos_w = box.data.root_pos_w

    if isinstance(xy_threshold, tuple):
        x_threshold, y_threshold = xy_threshold
    else:
        x_threshold = y_threshold = xy_threshold

    success = torch.ones(box_pos_w.shape[0], dtype=torch.bool, device=box_pos_w.device)
    for object_cfg in object_cfgs:
        obj: RigidObject = env.scene[object_cfg.name]
        delta = obj.data.root_pos_w - box_pos_w
        inside_xy = torch.logical_and(torch.abs(delta[:, 0]) < x_threshold, torch.abs(delta[:, 1]) < y_threshold)
        inside_z = torch.logical_and(delta[:, 2] > z_min, delta[:, 2] < z_max)
        success = torch.logical_and(success, torch.logical_and(inside_xy, inside_z))

    return success
