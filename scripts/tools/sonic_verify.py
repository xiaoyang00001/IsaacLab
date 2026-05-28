# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal launcher to verify SONICWholeBodyAction pipeline on sonic_robot.

`pick_place` 在 isaaclab_tasks 的 _BLACKLIST_PKGS 里，自动注册会跳过它，需要手动 import 触发
gym.register。该脚本基于 zero_agent.py + 手动 import + SONIC 进度日志。
"""

import argparse
import csv
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SONIC pipeline verification (zero action driver).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0", help="Task name."
)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N env steps (0 = run forever).")
parser.add_argument("--metrics_interval", type=int, default=50, help="Print SONIC metrics every N steps (0 = off).")
parser.add_argument("--metrics_csv", type=str, default="", help="Optional CSV path for per-interval SONIC metrics.")
parser.add_argument(
    "--sonic_reset_frame",
    type=int,
    default=None,
    help="Override SONIC deterministic mocap reset frame (in the 50fps resampled mocap timeline).",
)
parser.add_argument(
    "--disable_test_boxes",
    action="store_true",
    help="Disable conveyor test boxes and their events/actions for SONIC-only diagnostics.",
)
parser.add_argument(
    "--follow_mocap_root_xy",
    action="store_true",
    help="Enable diagnostic SONIC root XY following from the mocap trajectory.",
)
parser.add_argument(
    "--follow_mocap_root_pose",
    action="store_true",
    help="Enable diagnostic full SONIC root pose replay (XY, Z, and rotation) from the mocap trajectory.",
)
parser.add_argument(
    "--follow_mocap_root_xy_rate_limit_mps",
    type=float,
    default=None,
    help="Optional root XY follow speed limit in m/s; <=0 disables limiting.",
)
parser.add_argument(
    "--target_rate_limit_rad_per_step",
    type=float,
    default=None,
    help="Override SONIC global target rate limit in rad/control-step for diagnostics.",
)
parser.add_argument(
    "--sonic_obstacle",
    action="store_true",
    help="Place a static collision block in front of sonic_robot to verify physical interaction while walking.",
)
parser.add_argument(
    "--sonic_obstacle_forward_m",
    type=float,
    default=2.8,
    help="Distance from the reset root pose to the diagnostic obstacle along mocap walking direction.",
)
parser.add_argument(
    "--sonic_obstacle_lateral_m",
    type=float,
    default=0.0,
    help="Lateral offset for the diagnostic obstacle relative to mocap walking direction.",
)
parser.add_argument(
    "--sonic_obstacle_size",
    type=float,
    nargs=3,
    default=(0.25, 0.90, 0.22),
    metavar=("X", "Y", "Z"),
    help="Diagnostic obstacle cuboid size in meters: forward thickness, lateral width, height.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

# pick_place 在 isaaclab_tasks 的 blacklist 里，必须手动 import 才会触发 gym.register
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils import parse_env_cfg


def _get_sonic_term(env):
    try:
        return env.unwrapped.action_manager.get_term("sonic_wholebody")
    except Exception:
        return None


def _compute_sonic_forward_xy(term, device: torch.device) -> torch.Tensor:
    forward_xy = torch.tensor([1.0, 0.0], device=device)
    mocap_root = getattr(term, "_mocap_root_trans", None)
    mocap_frames = int(getattr(term, "_mocap_num_frames", 0))
    if mocap_root is None or mocap_frames <= 2:
        return forward_xy

    frame0 = int(getattr(term, "_mocap_frame", 0)) % mocap_frames
    frame1 = min(frame0 + max(1, int(round(getattr(term, "_mocap_fps", 50.0)))), mocap_frames - 1)
    delta_xy = mocap_root[frame1, :2].to(device=device) - mocap_root[frame0, :2].to(device=device)
    delta_norm = torch.linalg.norm(delta_xy)
    if float(delta_norm.item()) > 1.0e-4:
        forward_xy = delta_xy / delta_norm
    return forward_xy


def _set_sonic_front_view(env) -> None:
    """Place the GUI camera in front of sonic_robot, looking back at the robot."""
    unwrapped = env.unwrapped
    if not unwrapped.sim.has_gui():
        return

    term = _get_sonic_term(env)
    if term is None:
        return
    try:
        asset = unwrapped.scene[term.cfg.asset_name]
    except Exception:
        return

    env_idx = 0
    device = unwrapped.device
    root_pos = asset.data.root_pos_w[env_idx]
    forward_xy = _compute_sonic_forward_xy(term, device=device)
    eye_distance = 4.0
    lookat_forward = 0.15
    eye = (
        float(root_pos[0].item() + forward_xy[0].item() * eye_distance),
        float(root_pos[1].item() + forward_xy[1].item() * eye_distance),
        1.55,
    )
    lookat = (
        float(root_pos[0].item() + forward_xy[0].item() * lookat_forward),
        float(root_pos[1].item() + forward_xy[1].item() * lookat_forward),
        0.90,
    )

    unwrapped.cfg.viewer.origin_type = "world"
    unwrapped.cfg.viewer.asset_name = None
    unwrapped.cfg.viewer.body_name = None
    unwrapped.cfg.viewer.eye = eye
    unwrapped.cfg.viewer.lookat = lookat
    if unwrapped.viewport_camera_controller is not None:
        unwrapped.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
    else:
        unwrapped.sim.set_camera_view(eye=eye, target=lookat)
    print(
        "[sonic_viewer] front view "
        f"eye=({eye[0]:+.3f},{eye[1]:+.3f},{eye[2]:+.3f}) "
        f"lookat=({lookat[0]:+.3f},{lookat[1]:+.3f},{lookat[2]:+.3f}) "
        f"forward=({forward_xy[0].item():+.3f},{forward_xy[1].item():+.3f})",
        flush=True,
    )


def _place_sonic_obstacle(env, forward_m: float, lateral_m: float, obstacle_size: tuple[float, float, float]) -> None:
    """Move the pre-spawned kinematic collider into sonic_robot's walking path."""
    term = _get_sonic_term(env)
    if term is None:
        print("[sonic_obstacle] skipped: SONIC action term is unavailable", flush=True)
        return
    if getattr(term, "_reset_root_pos_w", None) is None:
        print("[sonic_obstacle] skipped: SONIC reset root pose is unavailable", flush=True)
        return
    try:
        obstacle = env.unwrapped.scene["sonic_obstacle"]
    except Exception:
        print("[sonic_obstacle] skipped: scene asset 'sonic_obstacle' is unavailable", flush=True)
        return

    device = env.unwrapped.device
    env_ids = torch.arange(env.unwrapped.scene.num_envs, device=device, dtype=torch.long)
    root_pos = term._reset_root_pos_w[env_ids].to(device=device)  # noqa: SLF001

    forward_xy = _compute_sonic_forward_xy(term, device=device)
    lateral_xy = torch.stack((-forward_xy[1], forward_xy[0]))
    yaw = math.atan2(float(forward_xy[1].item()), float(forward_xy[0].item()))
    half_yaw = 0.5 * yaw

    pose = obstacle.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = root_pos
    pose[:, 0] += forward_xy[0] * forward_m + lateral_xy[0] * lateral_m
    pose[:, 1] += forward_xy[1] * forward_m + lateral_xy[1] * lateral_m
    pose[:, 2] = max(0.0, float(obstacle_size[2]) * 0.5)
    pose[:, 3] = math.cos(half_yaw)
    pose[:, 4] = 0.0
    pose[:, 5] = 0.0
    pose[:, 6] = math.sin(half_yaw)

    zero_vel = torch.zeros((len(env_ids), 6), device=device)
    obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    print(
        "[sonic_obstacle] placed static collider "
        f"size=({obstacle_size[0]:.2f},{obstacle_size[1]:.2f},{obstacle_size[2]:.2f}) "
        f"forward=({forward_xy[0].item():+.3f},{forward_xy[1].item():+.3f}) "
        f"env0_pos=({pose[0, 0].item():+.3f},{pose[0, 1].item():+.3f},{pose[0, 2].item():+.3f}) "
        f"yaw={math.degrees(yaw):+.1f}deg",
        flush=True,
    )


def _joint_group_absmax(values: torch.Tensor, joint_names: list[str], *, prefix: str | None = None, contains: tuple[str, ...]) -> float:
    indices = [
        i for i, name in enumerate(joint_names[: values.numel()])
        if (prefix is None or name.startswith(prefix)) and any(s in name for s in contains)
    ]
    if not indices:
        return 0.0
    return float(values[indices].abs().max().item())


def _joint_indices(joint_names: list[str], *, prefix: str | None = None, contains: tuple[str, ...]) -> list[int]:
    return [
        i for i, name in enumerate(joint_names)
        if (prefix is None or name.startswith(prefix)) and any(s in name for s in contains)
    ]


def _joint_absmax(values: torch.Tensor, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(values[indices].abs().max().item())


def _joint_value(values: torch.Tensor, joint_names: list[str], joint_name: str) -> float:
    try:
        return float(values[joint_names.index(joint_name)].item())
    except ValueError:
        return math.nan


def _print_mocap_root_report(env) -> None:
    term = _get_sonic_term(env)
    if term is None:
        return
    mocap_root = getattr(term, "_mocap_root_trans", None)
    if mocap_root is None or int(getattr(term, "_mocap_num_frames", 0)) <= 0:
        return

    n = int(getattr(term, "_mocap_num_frames", 0))
    frame0 = int(getattr(term, "_mocap_frame", 0)) % n
    base = mocap_root[frame0]
    probe_frames = [1, 50, 100, 250, 500, 1000, 1500, 2000]
    parts = []
    for offset in probe_frames:
        frame = (frame0 + offset) % n
        delta = mocap_root[frame] - base
        parts.append(
            f"+{offset}:xy=({delta[0].item():+.3f},{delta[1].item():+.3f}) z={delta[2].item():+.3f}"
        )
    rel = mocap_root - mocap_root[0]
    step_xy = torch.linalg.norm(mocap_root[1:, :2] - mocap_root[:-1, :2], dim=-1)
    mocap_fps = float(getattr(term, "_mocap_fps", 50.0))
    total_delta = mocap_root[-1] - mocap_root[0]
    print(
        "[sonic_mocap] "
        f"root_trans frames={n} reset_frame={frame0} "
        f"total_xy=({total_delta[0].item():+.3f},{total_delta[1].item():+.3f}) "
        f"total_z={total_delta[2].item():+.3f} "
        f"xy_range=x[{rel[:, 0].min().item():+.3f},{rel[:, 0].max().item():+.3f}] "
        f"y[{rel[:, 1].min().item():+.3f},{rel[:, 1].max().item():+.3f}] "
        f"max_step_xy={step_xy.max().item():.3f} max_xy_speed={step_xy.max().item() * mocap_fps:.3f} | "
        + " ".join(parts),
        flush=True,
    )


def _compute_sonic_metrics(env, step: int) -> dict[str, float | str] | None:
    """Compute compact diagnostics for the first SONIC env."""
    term = _get_sonic_term(env)
    if term is None:
        return None

    asset = env.unwrapped.scene[term.cfg.asset_name]
    env_idx = 0

    body_pos_b = term._compute_self_ref_body_pos_b()[env_idx]  # noqa: SLF001
    mocap_body = getattr(term, "_mocap_body_pos_b", None)
    mocap_frames = int(getattr(term, "_mocap_num_frames", 0))
    if mocap_body is not None and mocap_frames > 0:
        frame = int(term._mocap_frame) % mocap_frames  # noqa: SLF001
        body_err = torch.linalg.norm(body_pos_b - mocap_body[frame], dim=-1)
        max_body_idx = int(torch.argmax(body_err).item())
        max_body_name = str(asset.data.body_names[term._sonic_body_ids[max_body_idx]])  # noqa: SLF001
        mpjpe_mm = float(body_err.mean().item() * 1000.0)
        max_body_err_mm = float(body_err.max().item() * 1000.0)
        feet_body_err_mm = float(body_err[[3, 6]].mean().item() * 1000.0)
    else:
        frame = -1
        max_body_name = ""
        mpjpe_mm = math.nan
        max_body_err_mm = math.nan
        feet_body_err_mm = math.nan

    foot_ids = torch.tensor([term._sonic_body_ids[3], term._sonic_body_ids[6]], device=env.unwrapped.device)  # noqa: SLF001
    foot_pos_w = asset.data.body_pos_w[env_idx, foot_ids]
    foot_vel_w = asset.data.body_lin_vel_w[env_idx, foot_ids]
    root_quat_w = asset.data.root_quat_w[env_idx].unsqueeze(0)
    roll, pitch, _ = euler_xyz_from_quat(root_quat_w)
    root_pos_w = asset.data.root_pos_w[env_idx]
    root_origin_w = getattr(term, "_reset_root_pos_w", None)
    if root_origin_w is not None:
        root_delta_xy = root_pos_w[:2] - root_origin_w[env_idx, :2]
    else:
        root_delta_xy = torch.zeros(2, device=env.unwrapped.device)

    obstacle_x = math.nan
    obstacle_y = math.nan
    obstacle_z = math.nan
    obstacle_xy_distance = math.nan
    try:
        obstacle = env.unwrapped.scene["sonic_obstacle"]
        obstacle_pos_all = getattr(obstacle.data, "root_pos_w", None)
        if obstacle_pos_all is None:
            obstacle_state_all = getattr(obstacle.data, "root_state_w", None)
            obstacle_pos_all = obstacle_state_all[:, :3] if obstacle_state_all is not None else None
        if obstacle_pos_all is not None:
            obstacle_pos_w = obstacle_pos_all[env_idx]
            if float(obstacle_pos_w[2].item()) > -1.0:
                obstacle_x = float(obstacle_pos_w[0].item())
                obstacle_y = float(obstacle_pos_w[1].item())
                obstacle_z = float(obstacle_pos_w[2].item())
                obstacle_xy_distance = float(torch.linalg.norm(root_pos_w[:2] - obstacle_pos_w[:2]).item())
    except Exception:
        pass

    action = term._last_action[env_idx]  # noqa: SLF001
    target_delta = term.processed_actions[env_idx] - term._default_joint_pos[env_idx]  # noqa: SLF001
    joint_pos_rel = asset.data.joint_pos[env_idx, term._joint_ids] - term._default_joint_pos[env_idx]  # noqa: SLF001
    joint_names = list(term.cfg.joint_names)
    left_leg_ids = _joint_indices(joint_names, prefix="left", contains=("_hip_", "_knee_", "_ankle_"))
    right_leg_ids = _joint_indices(joint_names, prefix="right", contains=("_hip_", "_knee_", "_ankle_"))
    foot_xy_speed = torch.linalg.norm(foot_vel_w[:, :2], dim=-1)
    target_step_delta = getattr(term, "_last_target_step_delta_absmax", None)
    if target_step_delta is not None:
        target_step_delta_absmax = float(target_step_delta[env_idx].item())
    else:
        target_step_delta_absmax = math.nan

    mocap_delta_xy = torch.zeros(2, device=env.unwrapped.device)
    root_vs_mocap_xy_lag = math.nan
    mocap_joint_delta = torch.zeros_like(joint_pos_rel)
    mocap_dof = getattr(term, "_mocap_dof", None)
    mocap_root = getattr(term, "_mocap_root_trans", None)
    reset_mocap_root = getattr(term, "_reset_mocap_root_trans", None)
    if mocap_frames > 0 and frame >= 0:
        if mocap_dof is not None:
            mocap_joint_delta = mocap_dof[frame, : joint_pos_rel.numel()] - term._default_joint_pos[env_idx]  # noqa: SLF001
        if mocap_root is not None and reset_mocap_root is not None:
            mocap_delta_xy = mocap_root[frame, :2] - reset_mocap_root[env_idx, :2]
            root_vs_mocap_xy_lag = float(torch.linalg.norm(root_delta_xy - mocap_delta_xy).item())

    return {
        "step": float(step),
        "mocap_frame": float(frame),
        "mpjpe_mm": mpjpe_mm,
        "max_body_err_mm": max_body_err_mm,
        "max_body_name": max_body_name,
        "feet_body_err_mm": feet_body_err_mm,
        "root_z": float(asset.data.root_pos_w[env_idx, 2].item()),
        "root_x_delta": float(root_delta_xy[0].item()),
        "root_y_delta": float(root_delta_xy[1].item()),
        "root_xy_delta": float(torch.linalg.norm(root_delta_xy).item()),
        "obstacle_x": obstacle_x,
        "obstacle_y": obstacle_y,
        "obstacle_z": obstacle_z,
        "obstacle_xy_distance": obstacle_xy_distance,
        "mocap_root_x_delta": float(mocap_delta_xy[0].item()),
        "mocap_root_y_delta": float(mocap_delta_xy[1].item()),
        "mocap_root_xy_delta": float(torch.linalg.norm(mocap_delta_xy).item()),
        "root_vs_mocap_xy_lag": root_vs_mocap_xy_lag,
        "root_roll_deg": float(torch.rad2deg(roll)[0].item()),
        "root_pitch_deg": float(torch.rad2deg(pitch)[0].item()),
        "left_foot_z": float(foot_pos_w[0, 2].item()),
        "right_foot_z": float(foot_pos_w[1, 2].item()),
        "left_foot_xy_speed": float(foot_xy_speed[0].item()),
        "right_foot_xy_speed": float(foot_xy_speed[1].item()),
        "foot_z_diff_abs": float((foot_pos_w[0, 2] - foot_pos_w[1, 2]).abs().item()),
        "foot_xy_distance": float(torch.linalg.norm(foot_pos_w[0, :2] - foot_pos_w[1, :2]).item()),
        "foot_z_min": float(foot_pos_w[:, 2].min().item()),
        "foot_z_max": float(foot_pos_w[:, 2].max().item()),
        "foot_xy_speed_max": float(foot_xy_speed.max().item()),
        "raw_action_absmax": float(action.abs().max().item()),
        "target_step_delta_absmax": target_step_delta_absmax,
        "raw_legs_absmax": _joint_group_absmax(action, joint_names, contains=("_hip_", "_knee_", "_ankle_")),
        "raw_waist_absmax": _joint_group_absmax(action, joint_names, prefix="waist", contains=("_joint",)),
        "target_left_leg_absmax": _joint_absmax(target_delta, left_leg_ids),
        "target_right_leg_absmax": _joint_absmax(target_delta, right_leg_ids),
        "joint_left_leg_absmax": _joint_absmax(joint_pos_rel, left_leg_ids),
        "joint_right_leg_absmax": _joint_absmax(joint_pos_rel, right_leg_ids),
        "mocap_left_leg_absmax": _joint_absmax(mocap_joint_delta, left_leg_ids),
        "mocap_right_leg_absmax": _joint_absmax(mocap_joint_delta, right_leg_ids),
        "left_leg_target_joint_err_absmax": _joint_absmax(target_delta - joint_pos_rel, left_leg_ids),
        "right_leg_target_joint_err_absmax": _joint_absmax(target_delta - joint_pos_rel, right_leg_ids),
        "left_leg_target_mocap_err_absmax": _joint_absmax(target_delta - mocap_joint_delta, left_leg_ids),
        "right_leg_target_mocap_err_absmax": _joint_absmax(target_delta - mocap_joint_delta, right_leg_ids),
        "left_knee_target": _joint_value(target_delta, joint_names, "left_knee_joint"),
        "right_knee_target": _joint_value(target_delta, joint_names, "right_knee_joint"),
        "left_knee_joint": _joint_value(joint_pos_rel, joint_names, "left_knee_joint"),
        "right_knee_joint": _joint_value(joint_pos_rel, joint_names, "right_knee_joint"),
        "left_ankle_pitch_target": _joint_value(target_delta, joint_names, "left_ankle_pitch_joint"),
        "right_ankle_pitch_target": _joint_value(target_delta, joint_names, "right_ankle_pitch_joint"),
        "left_ankle_pitch_joint": _joint_value(joint_pos_rel, joint_names, "left_ankle_pitch_joint"),
        "right_ankle_pitch_joint": _joint_value(joint_pos_rel, joint_names, "right_ankle_pitch_joint"),
        "left_knee_mocap": _joint_value(mocap_joint_delta, joint_names, "left_knee_joint"),
        "right_knee_mocap": _joint_value(mocap_joint_delta, joint_names, "right_knee_joint"),
        "left_ankle_pitch_mocap": _joint_value(mocap_joint_delta, joint_names, "left_ankle_pitch_joint"),
        "right_ankle_pitch_mocap": _joint_value(mocap_joint_delta, joint_names, "right_ankle_pitch_joint"),
        "raw_left_arm_absmax": _joint_group_absmax(
            action, joint_names, prefix="left", contains=("_shoulder_", "_elbow_", "_wrist_")
        ),
        "raw_right_arm_absmax": _joint_group_absmax(
            action, joint_names, prefix="right", contains=("_shoulder_", "_elbow_", "_wrist_")
        ),
        "target_delta_absmax": float(target_delta.abs().max().item()),
        "target_legs_absmax": _joint_group_absmax(target_delta, joint_names, contains=("_hip_", "_knee_", "_ankle_")),
        "target_waist_absmax": _joint_group_absmax(target_delta, joint_names, prefix="waist", contains=("_joint",)),
        "target_left_arm_absmax": _joint_group_absmax(
            target_delta, joint_names, prefix="left", contains=("_shoulder_", "_elbow_", "_wrist_")
        ),
        "target_right_arm_absmax": _joint_group_absmax(
            target_delta, joint_names, prefix="right", contains=("_shoulder_", "_elbow_", "_wrist_")
        ),
        "joint_pos_absmax": float(joint_pos_rel.abs().max().item()),
    }


def _print_sonic_metrics(metrics: dict[str, float | str]) -> None:
    print(
        "[sonic_metrics] "
        f"step={int(metrics['step'])} frame={int(metrics['mocap_frame'])} "
        f"mpjpe={metrics['mpjpe_mm']:.1f}mm max_body={metrics['max_body_err_mm']:.1f}mm "
        f"({metrics['max_body_name']}) "
        f"feet_err={metrics['feet_body_err_mm']:.1f}mm root_z={metrics['root_z']:.3f} "
        f"roll={metrics['root_roll_deg']:+.1f}deg pitch={metrics['root_pitch_deg']:+.1f}deg "
        f"foot_z=[L{metrics['left_foot_z']:.3f},R{metrics['right_foot_z']:.3f}] "
        f"foot_xy_speed=[L{metrics['left_foot_xy_speed']:.3f},R{metrics['right_foot_xy_speed']:.3f}] "
        f"raw_abs={metrics['raw_action_absmax']:.3f} target_delta_abs={metrics['target_delta_absmax']:.3f} "
        f"target_step_delta={metrics['target_step_delta_absmax']:.3f} joint_abs={metrics['joint_pos_absmax']:.3f}",
        flush=True,
    )
    print(
        "[sonic_metrics] "
        f"raw_group legs={metrics['raw_legs_absmax']:.3f} waist={metrics['raw_waist_absmax']:.3f} "
        f"l_arm={metrics['raw_left_arm_absmax']:.3f} r_arm={metrics['raw_right_arm_absmax']:.3f} | "
        f"target_group legs={metrics['target_legs_absmax']:.3f} waist={metrics['target_waist_absmax']:.3f} "
        f"l_arm={metrics['target_left_arm_absmax']:.3f} r_arm={metrics['target_right_arm_absmax']:.3f}",
        flush=True,
    )
    print(
        "[sonic_lower] "
        f"root_xy=({metrics['root_x_delta']:+.3f},{metrics['root_y_delta']:+.3f}) "
        f"mocap_xy=({metrics['mocap_root_x_delta']:+.3f},{metrics['mocap_root_y_delta']:+.3f}) "
        f"lag={metrics['root_vs_mocap_xy_lag']:.3f} "
        f"foot_z_diff={metrics['foot_z_diff_abs']:.3f} foot_xy_dist={metrics['foot_xy_distance']:.3f} "
        f"leg_abs target=[L{metrics['target_left_leg_absmax']:.3f},R{metrics['target_right_leg_absmax']:.3f}] "
        f"joint=[L{metrics['joint_left_leg_absmax']:.3f},R{metrics['joint_right_leg_absmax']:.3f}] "
        f"mocap=[L{metrics['mocap_left_leg_absmax']:.3f},R{metrics['mocap_right_leg_absmax']:.3f}] "
        f"err_t-j=[L{metrics['left_leg_target_joint_err_absmax']:.3f},R{metrics['right_leg_target_joint_err_absmax']:.3f}] "
        f"err_t-m=[L{metrics['left_leg_target_mocap_err_absmax']:.3f},R{metrics['right_leg_target_mocap_err_absmax']:.3f}]",
        flush=True,
    )
    print(
        "[sonic_lower] "
        f"knee target=[L{metrics['left_knee_target']:+.3f},R{metrics['right_knee_target']:+.3f}] "
        f"joint=[L{metrics['left_knee_joint']:+.3f},R{metrics['right_knee_joint']:+.3f}] "
        f"mocap=[L{metrics['left_knee_mocap']:+.3f},R{metrics['right_knee_mocap']:+.3f}] | "
        f"ankle_pitch target=[L{metrics['left_ankle_pitch_target']:+.3f},R{metrics['right_ankle_pitch_target']:+.3f}] "
        f"joint=[L{metrics['left_ankle_pitch_joint']:+.3f},R{metrics['right_ankle_pitch_joint']:+.3f}] "
        f"mocap=[L{metrics['left_ankle_pitch_mocap']:+.3f},R{metrics['right_ankle_pitch_mocap']:+.3f}]",
        flush=True,
    )
    if math.isfinite(float(metrics["obstacle_xy_distance"])):
        print(
            "[sonic_obstacle] "
            f"pos=({metrics['obstacle_x']:+.3f},{metrics['obstacle_y']:+.3f},{metrics['obstacle_z']:+.3f}) "
            f"root_xy_distance={metrics['obstacle_xy_distance']:.3f}",
            flush=True,
        )


def _write_metrics_csv(path: str, rows: list[dict[str, float | str]]) -> None:
    if not path or not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sonic_verify] metrics saved: {path}", flush=True)


def _print_metrics_summary(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return

    def _finite_values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if isinstance(row[key], float) and math.isfinite(row[key])]

    mpjpe = _finite_values("mpjpe_mm")
    joint_abs = _finite_values("joint_pos_absmax")
    foot_speed = _finite_values("foot_xy_speed_max")
    target_delta = _finite_values("target_delta_absmax")
    target_step_delta = _finite_values("target_step_delta_absmax")
    root_lag = _finite_values("root_vs_mocap_xy_lag")
    mocap_root = _finite_values("mocap_root_xy_delta")
    obstacle_distance = _finite_values("obstacle_xy_distance")
    if not mpjpe:
        return
    print(
        "[sonic_summary] "
        f"samples={len(rows)} mpjpe_mean={sum(mpjpe) / len(mpjpe):.1f}mm mpjpe_max={max(mpjpe):.1f}mm "
        f"joint_abs_max={max(joint_abs):.3f} target_delta_abs_max={max(target_delta):.3f} "
        f"target_step_delta_max={max(target_step_delta):.3f} foot_xy_speed_max={max(foot_speed):.3f} "
        f"mocap_root_xy_max={max(mocap_root):.3f} root_lag_max={max(root_lag):.3f}",
        flush=True,
    )
    if obstacle_distance:
        print(f"[sonic_summary] obstacle_xy_distance_min={min(obstacle_distance):.3f}", flush=True)


def main():
    print(f"[sonic_verify] task={args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.sonic_reset_frame is not None and hasattr(env_cfg.actions, "sonic_wholebody"):
        env_cfg.actions.sonic_wholebody.reset_to_random_mocap_frame = False
        env_cfg.actions.sonic_wholebody.reset_mocap_frame = int(args_cli.sonic_reset_frame)
        print(f"[sonic_verify] override SONIC reset_mocap_frame={args_cli.sonic_reset_frame}", flush=True)
    if (args_cli.follow_mocap_root_xy or args_cli.follow_mocap_root_pose) and hasattr(
        env_cfg.actions, "sonic_wholebody"
    ):
        env_cfg.actions.sonic_wholebody.follow_mocap_root_xy = True
        if args_cli.follow_mocap_root_pose:
            env_cfg.actions.sonic_wholebody.follow_mocap_root_z = True
            env_cfg.actions.sonic_wholebody.follow_mocap_root_rot = True
        if args_cli.follow_mocap_root_xy_rate_limit_mps is not None:
            env_cfg.actions.sonic_wholebody.follow_mocap_root_xy_rate_limit_mps = float(
                args_cli.follow_mocap_root_xy_rate_limit_mps
            )
        print(
            "[sonic_verify] enabled diagnostic SONIC root follow "
            f"(pose={args_cli.follow_mocap_root_pose}, "
            f"rate_limit={env_cfg.actions.sonic_wholebody.follow_mocap_root_xy_rate_limit_mps:.3f}m/s)",
            flush=True,
        )
    if args_cli.target_rate_limit_rad_per_step is not None and hasattr(env_cfg.actions, "sonic_wholebody"):
        env_cfg.actions.sonic_wholebody.target_rate_limit_rad_per_step = float(
            args_cli.target_rate_limit_rad_per_step
        )
        print(
            "[sonic_verify] override SONIC target_rate_limit_rad_per_step="
            f"{env_cfg.actions.sonic_wholebody.target_rate_limit_rad_per_step:.3f}",
            flush=True,
        )
    if hasattr(env_cfg.scene, "sonic_obstacle") and getattr(env_cfg.scene.sonic_obstacle, "spawn", None) is not None:
        env_cfg.scene.sonic_obstacle.spawn.size = tuple(float(v) for v in args_cli.sonic_obstacle_size)
    if args_cli.disable_test_boxes:
        for name in ("test_box", "test_box1"):
            if hasattr(env_cfg.scene, name):
                setattr(env_cfg.scene, name, None)
        for name in ("object_sync", "object_sync1"):
            if hasattr(env_cfg.actions, name):
                setattr(env_cfg.actions, name, None)
        for name in (
            "setup_test_box_physics",
            "setup_test_box1_physics",
            "align_test_boxes_to_conveyor_startup",
            "align_test_boxes_to_conveyor_reset",
            "drive_test_box",
            "drive_test_box1",
        ):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        print("[sonic_verify] disabled conveyor test boxes for SONIC-only diagnostics", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[sonic_verify] env created; action_space={env.action_space}", flush=True)

    env.reset()
    _print_mocap_root_report(env)
    if args_cli.sonic_obstacle:
        _place_sonic_obstacle(
            env,
            forward_m=float(args_cli.sonic_obstacle_forward_m),
            lateral_m=float(args_cli.sonic_obstacle_lateral_m),
            obstacle_size=tuple(float(v) for v in args_cli.sonic_obstacle_size),
        )
    _set_sonic_front_view(env)
    print("[sonic_verify] reset done; entering step loop (press Ctrl+C to stop)", flush=True)

    step = 0
    metric_rows = []
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
        step += 1
        if args_cli.metrics_interval and step % args_cli.metrics_interval == 0:
            metrics = _compute_sonic_metrics(env, step)
            if metrics is not None:
                metric_rows.append(metrics)
                _print_sonic_metrics(metrics)
        if step % 100 == 0:
            print(f"[sonic_verify] step={step}", flush=True)
        if args_cli.max_steps and step >= args_cli.max_steps:
            print(f"[sonic_verify] reached max_steps={args_cli.max_steps}, exiting", flush=True)
            break

    _print_metrics_summary(metric_rows)
    _write_metrics_csv(args_cli.metrics_csv, metric_rows)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
