# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the ACTUAL teleop contact-freeze path against a box.

Reproduces the user's report ("the moment a finger touches the box it flies
off") by driving the fingers through the REAL MuJoCoG1MirrorAction freeze
method -- the same clamp -> _apply_hand_contact_freeze -> set_target sequence
the VR mirror uses -- while a box sits in the pinch pocket. Logs, every 30
steps: finger travel, box displacement, freeze state.

Run: ./isaaclab.sh -p scripts/diagnostics/verify_freeze.py --headless --device cpu
"""

import argparse
import os

os.environ["ISAACLAB_SCENE_SYNC_ROLE"] = "none"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="verify teleop contact-freeze against a box")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import functools  # noqa: E402
import builtins  # noqa: E402
print = functools.partial(builtins.print, flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ARM_JOINT_PATTERNS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                      ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
LOWER_BODY_PATTERNS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
for term_name in ("mujoco_g1_mirror_2", "scene_state_sync", "env_reset_sync", "box_success_reset"):
    if hasattr(env_cfg.actions, term_name):
        setattr(env_cfg.actions, term_name, None)
for term_name in ("box_dropped", "time_out", "success"):
    if hasattr(env_cfg.terminations, term_name):
        setattr(env_cfg.terminations, term_name, None)

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.reset()

sim = env.sim
scene = env.scene
robot = scene["robot_1"]
box = scene["small_box_1"]
dt = env.physics_dt
device = env.device

# the REAL mirror action term (carries the real freeze method + state)
term = env.action_manager.get_term("mujoco_g1_mirror_1")
m_box = float(box.root_physx_view.get_masses().sum())
print("=" * 78)
print(f"TELEOP FREEZE-PATH VERIFICATION  box={m_box:.2f} kg  freeze_enabled={term._hand_freeze_enabled}")
print(f"params: residual>{term._hand_freeze_residual} window={term._hand_freeze_window} advance<{term._hand_freeze_advance} stalls={term._hand_freeze_stalls} preload={term._hand_freeze_preload}")
print("=" * 78)

lb_ids, _ = robot.find_joints(LOWER_BODY_PATTERNS)
arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)
palm_ids, palm_names = robot.find_bodies([".*_hand_palm_link"])
l_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("left")][0])]
r_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("right")][0])]
thumb_ids, _ = robot.find_bodies(["right_hand_thumb_2_link"])
idx_ids, _ = robot.find_bodies(["right_hand_index_1_link"])
mid_ids, _ = robot.find_bodies(["right_hand_middle_1_link"])

lb_q0 = robot.data.default_joint_pos[:, lb_ids].clone()
arm_q0 = robot.data.default_joint_pos[:, arm_ids].clone()
root0 = robot.data.default_root_state[:, :7].clone()
root0[:, :3] += scene.env_origins
root0[:, 2] += 1.0
q_z180 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
root0[:, 3:7] = math_utils.quat_mul(q_z180, root0[:, 3:7])
zero6 = torch.zeros((1, 6), device=device)
lb_zero = torch.zeros_like(lb_q0)
arm_targets = arm_q0.clone()

# hand joints in the TERM's own order (left 7 + right 7)
all_hand_ids = term._all_hand_ids
hand_names_term = [robot.joint_names[i] for i in all_hand_ids]
hand_limits = robot.data.joint_pos_limits[:, all_hand_ids, :]
max_delta = float(term.cfg.hand_joint_target_max_delta)
print(f"hand joint order (term): {hand_names_term}")
print(f"hand_joint_target_max_delta={max_delta}")


def close_goal_term_order() -> torch.Tensor:
    goal = torch.zeros((1, len(all_hand_ids)), device=device)
    for i, name in enumerate(hand_names_term):
        left = name.startswith("left")
        if "thumb_0" in name:
            goal[0, i] = 0.0
        elif "thumb_1" in name:
            goal[0, i] = 1.1 if left else -1.1
        elif "thumb_2" in name:
            goal[0, i] = 1.8 if left else -1.8
        else:
            goal[0, i] = -1.8 if left else 1.8
    return goal


hand_goal_raw = [torch.zeros((1, len(all_hand_ids)), device=device)]  # start OPEN


def step_hold(n: int, on_step=None):
    for i in range(n):
        robot.write_root_pose_to_sim(root0)
        robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(lb_q0, lb_zero, joint_ids=lb_ids)
        robot.set_joint_position_target(arm_targets, joint_ids=arm_ids)
        # === the REAL teleop finger path (adaptive lead OR clamp+freeze) ===
        cur = robot.data.joint_pos[:, all_hand_ids]
        raw = hand_goal_raw[0]
        if getattr(term, "_hand_adaptive_enabled", False):
            out = term._apply_adaptive_hand_delta(raw)
        else:
            clamped = torch.clamp(raw, cur - max_delta, cur + max_delta)
            clamped = torch.max(torch.min(clamped, hand_limits[..., 1]), hand_limits[..., 0])
            out = term._apply_hand_contact_freeze(clamped, raw)
        out = torch.max(torch.min(out, hand_limits[..., 1]), hand_limits[..., 0])
        robot.set_joint_position_target(out, joint_ids=all_hand_ids)
        robot.set_joint_velocity_target(torch.zeros_like(out), joint_ids=all_hand_ids)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        if on_step is not None:
            on_step(i)


def jidx(name: str) -> int:
    return arm_names.index(name)


def pinch_mid():
    p_t = robot.data.body_pos_w[0, thumb_ids[0]]
    p_f = (robot.data.body_pos_w[0, idx_ids[0]] + robot.data.body_pos_w[0, mid_ids[0]]) / 2
    return (p_t + p_f) / 2


def set_box(pos, quat):
    pose = torch.cat([pos.view(1, 3), quat.view(1, 4)], dim=-1)
    box.write_root_pose_to_sim(pose)
    box.write_root_velocity_to_sim(zero6)


# park other boxes
quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
for name, (x, y) in (("long_box", (3.0, 3.0)), ("small_box_2", (-3.0, 3.0))):
    b = scene[name]
    pos = torch.tensor([x, y, 0.10], device=device) + scene.env_origins[0]
    b.write_root_pose_to_sim(torch.cat([pos.view(1, 3), quat_id.view(1, 4)], dim=-1))
    b.write_root_velocity_to_sim(zero6)
step_hold(200)

# raise right arm, palm up
before = float(robot.data.body_pos_w[0, r_palm, 2])
arm_targets[0, jidx("right_shoulder_pitch_joint")] += 0.4
step_hold(90)
lift_sign = 1.0 if float(robot.data.body_pos_w[0, r_palm, 2]) > before else -1.0
arm_targets = arm_q0.clone()
step_hold(90)
arm_targets[0, jidx("right_shoulder_pitch_joint")] = arm_q0[0, jidx("right_shoulder_pitch_joint")] + lift_sign * 0.5
arm_targets[0, jidx("right_elbow_joint")] = 1.1
arm_targets[0, jidx("right_wrist_roll_joint")] = arm_q0[0, jidx("right_wrist_roll_joint")] + 1.57
step_hold(300)

# place the box in the pinch pocket, biased 50% toward the THUMB: run 5 showed
# zero thumb-side contact at the nominal midpoint (F rt = 0 throughout) -- the
# thumb's closing arc never reaches there, so no opposed pinch could form
# NOTE: biasing the spawn toward the thumb (25% and 50% tried) lands inside
# the thumb collision body and explodes on spawn; at the nominal midpoint the
# thumb arc never reaches the box (F_thumb = 0 for the whole close). There is
# NO feasible opposed-pinch pose for the 5 cm cube in this hand -- this script
# therefore measures EJECTION SEVERITY, not grasp success.
place_pt = pinch_mid().clone()
set_box(place_pt, quat_id)
g_comp = torch.tensor([[[0.0, 0.0, m_box * 9.81]]], device=device)
zero_wrench = torch.zeros(1, 1, 3, device=device)
spring_on = [True]


def spring_cb(i):
    if not spring_on[0]:
        return
    err = (place_pt - box.data.root_pos_w[0]).view(1, 1, 3)
    vel = box.data.root_lin_vel_w[0].view(1, 1, 3)
    f = torch.clamp(60.0 * err - 3.0 * vel + g_comp, -4.0, 4.0)
    box.set_external_force_and_torque(f, zero_wrench)


step_hold(150, spring_cb)
print(f"\nbox placed, dist to pinch {float((box.data.root_pos_w[0] - place_pt).norm()):.3f} m")

# === free-hand calibration pass (criteria 2 & 5): full close then open with
# NO box in reach -- the adaptive lead must stay near DMAX throughout ===
if getattr(term, "_hand_adaptive_enabled", False):
    print("\n--- free-hand calibration: full close (box parked far) ---")
    saved = box.data.root_pos_w[0].clone()
    far = torch.tensor([3.0, -3.0, 0.10], device=device) + scene.env_origins[0]
    set_box(far, quat_id)
    spring_on[0] = False
    hand_goal_raw[0] = close_goal_term_order()
    t_free0 = None
    err_peak = [0.0]
    dmin_seen = [1.0]

    def cal_cb(i):
        err_peak[0] = max(err_peak[0], float(term._ad_err.max()))
        dmin_seen[0] = min(dmin_seen[0], float(term._ad_delta.mean()))

    start_q = robot.data.joint_pos[:, all_hand_ids].clone()
    for chunk in range(20):
        step_hold(30, cal_cb)
        trav = float((robot.data.joint_pos[:, all_hand_ids] - start_q).abs().mean())
        if trav > 1.30:
            t_free0 = (chunk + 1) * 30
            break
    print(f"  free close: reached full in {t_free0} steps ({(t_free0 or 600) * 0.005:.2f} s)"
          f" | err_f peak {err_peak[0]:.3f} | mean lead min {dmin_seen[0]:.3f} (DMAX 0.08)")
    hand_goal_raw[0] = torch.zeros_like(hand_goal_raw[0])
    step_hold(400)
    print(f"  reopened, travel back {float((robot.data.joint_pos[:, all_hand_ids] - start_q).abs().mean()):.3f} rad")
    set_box(place_pt, quat_id)
    spring_on[0] = True
    step_hold(150, spring_cb)
    print(f"  box re-placed, dist to pinch {float((box.data.root_pos_w[0] - place_pt).norm()):.3f} m")

# === TRIGGER PULL: raw goal jumps to full close, exactly like the VR stream ===
print("\n--- trigger pulled: raw goal = full close ---")
hand_goal_raw[0] = close_goal_term_order()
start = robot.data.joint_pos[:, all_hand_ids].clone()
box0 = box.data.root_pos_w[0].clone()

frozen_step = [None]


def log_cb(i):
    if (i + 1) % 30 == 0 or i == 0:
        travel = float((robot.data.joint_pos[:, all_hand_ids] - start).abs().mean())
        bdist = float((box.data.root_pos_w[0] - box0).norm())
        fr_l = term._hand_freeze_state["left"]["frozen"] is not None
        fr_r = term._hand_freeze_state["right"]["frozen"] is not None
        lead = float(term._ad_delta.mean()) if getattr(term, "_hand_adaptive_enabled", False) else -1.0
        forces = ""
        try:
            fm = env.scene.sensors["hand_contact"].data.net_forces_w.norm(dim=-1)
            g = term._hc_groups
            forces = (f" | F rt {float(fm[:, g['rt']].amax()):.1f} rf {float(fm[:, g['rf']].amax()):.1f}"
                      f" lt {float(fm[:, g['lt']].amax()):.1f} lf {float(fm[:, g['lf']].amax()):.1f}")
        except Exception:
            pass
        print(f"  step {i+1:4d}: travel {travel:.3f} rad | box moved {bdist:.3f} m | frozen L={fr_l} R={fr_r} | lead {lead:.3f}{forces}")
        if frozen_step[0] is None and fr_r:
            frozen_step[0] = i + 1
    if (i + 1) % 150 == 0:
        cur = robot.data.joint_pos[:, all_hand_ids]
        raw = hand_goal_raw[0]
        n_l = len(term._left_hand_ids)
        r_res = ((raw[0, n_l:] - cur[0, n_l:]) * torch.sign(raw[0, n_l:])).clamp(min=0.0)
        print(f"       right-hand residuals: {[round(v, 2) for v in r_res.tolist()]} mean {float(r_res.mean()):.3f}")


step_hold(600, lambda i: (spring_cb(i), log_cb(i)))

# release the steady spring: does the box stay or fly?
print("\n--- spring off: free hold 2 s ---")
spring_on[0] = False
box.set_external_force_and_torque(zero_wrench, zero_wrench)
step_hold(400, log_cb)
bdist = float((box.data.root_pos_w[0] - box0).norm())
bvel = float(box.data.root_lin_vel_w[0].norm())
print(f"\nRESULT: box displacement {bdist:.3f} m, speed {bvel:.2f} m/s, right-hand frozen at step {frozen_step[0]}")
if bdist > 0.5:
    print("VERDICT: BOX FLEW (reproduced the user's report)")
elif bdist > 0.15:
    print("VERDICT: box pushed away / dropped")
else:
    print("VERDICT: box stayed in hand")

env.close()
simulation_app.close()
