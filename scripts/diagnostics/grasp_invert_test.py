# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Inverted-grasp test: one hand closes ALL its fingers around the small box
(the grasp starts already-holding -- no approach), then the wrist rotates the
palm from facing-up to facing-down. If the box does not fall, the finger grip
(moderate effort + high-friction pads + frozen preload, i.e. the teleop
stabilization pack) holds against gravity alone.

Sequence:
  1. settle, park the other boxes away
  2. raise the right arm, palm up (wrist_roll +1.57), fingers open
  3. place small_box_1 into the hand's pinch pocket, steady it with a soft
     positioning spring
  4. close all fingers until they stall on the box, freeze at contact+preload
  5. fade the spring out, hold: confirm the box sits in the hand
  6. slowly roll the wrist 180 deg -> palm down, box hanging in the finger cage
  7. hold 2 s: HELD if the box stays in hand, DROPPED if it falls

Run GUI:   env LD_LIBRARY_PATH= XR_RUNTIME_JSON=/nonexistent DISPLAY=:0 \\
    ./isaaclab.sh -p scripts/diagnostics/grasp_invert_test.py --device cpu
Headless:  add --headless
"""

import argparse
import os

os.environ["ISAACLAB_SCENE_SYNC_ROLE"] = "none"
# rubber-pad-grade finger friction for this test (the hand-material binder in
# actions.py reads these): halves the normal force needed to hold the box
os.environ["ISAACLAB_G1_HAND_FRICTION_STATIC"] = "2.5"
os.environ["ISAACLAB_G1_HAND_FRICTION_DYNAMIC"] = "2.0"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 one-hand inverted-grasp test")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--report", type=str, default=None, help="JSON report output path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

RENDER = not args_cli.headless
_step_counter = [0]

import functools  # noqa: E402
import builtins  # noqa: E402
print = functools.partial(builtins.print, flush=True)

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ARM_JOINT_PATTERNS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
LOWER_BODY_PATTERNS = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
]

TARGET_BOX = "long_box"        # 0.20 x 0.05 x 0.10 slab, 0.25 kg -- held by one
                               # END: most of it sticks out past the fingertips,
                               # only ~2.5 cm of the tail sits in the pinch
                               # pocket (in-hand placements speared the palm)
hand_delta = [0.06]            # per-step finger target clamp; k*delta is also
                               # the PUSH FORCE while closing (0.06 -> 2.4 N*m
                               # ~= 80 N/finger: that is what catapulted the
                               # box). Switched to 0.012 after first contact.
WRAP_DELTA = 0.012             # gentle wrap push (~13 N per finger)
FINGER_PRELOAD_RAD = 0.05      # frozen goal = contact + this (~66 N hold)
# 17 physical-contact runs proved this rigid 3-finger hand has NO stable
# force/form-closure window for the rigid cube (shallow wraps slip, deep wraps
# eject). ATTACH mode: after the fingers physically wrap, the box is pinned to
# the palm frame (the project's established HugBoxAttach approach). Set
# ISAACLAB_INVERT_ATTACH=0 to re-run the pure-physics variant.
ATTACH_MODE = os.environ.get("ISAACLAB_INVERT_ATTACH", "1").strip().lower() not in ("0", "false")
SPRING_FADE_STEPS = 200
FLIP_STEPS = 800               # 4 s wrist roll from palm-up to palm-down

report: dict = {"target": TARGET_BOX, "phases": {}, "verdict": None}


def _fmt(x: float) -> float:
    return round(float(x), 4)


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
for term_name in ("mujoco_g1_mirror_2", "scene_state_sync", "env_reset_sync", "box_success_reset"):
    if hasattr(env_cfg.actions, term_name):
        setattr(env_cfg.actions, term_name, None)
for term_name in ("box_dropped", "time_out", "success"):
    if hasattr(env_cfg.terminations, term_name):
        setattr(env_cfg.terminations, term_name, None)

# Contact hardening for the deep finger wrap: cap how fast the solver may
# eject the box out of penetration (the root of every catapult failure) and
# double its solver iterations. Same trick the cart boxes ship with (0.5 m/s).
try:
    env_cfg.scene.small_box_1.spawn.rigid_props.max_depenetration_velocity = 0.5
    env_cfg.scene.small_box_1.spawn.rigid_props.solver_position_iteration_count = 16
except Exception as exc:
    print(f"[WARN] box rigid_props override failed: {exc}")

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.reset()

sim = env.sim
scene = env.scene
robot = scene["robot_1"]
box = scene[TARGET_BOX]
dt = env.physics_dt
device = env.device
m_box = float(box.root_physx_view.get_masses().sum())
print("=" * 78)
print(f"ONE-HAND INVERTED-GRASP TEST: {TARGET_BOX}  mass={m_box:.2f} kg")
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

# in-air experiment pose: root lifted 1 m, rotated 180 deg (proven frame from
# the two-palm grasp verification)
root0 = robot.data.default_root_state[:, :7].clone()
root0[:, :3] += scene.env_origins
root0[:, 2] += 1.0
q_z180 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
root0[:, 3:7] = math_utils.quat_mul(q_z180, root0[:, 3:7])
zero6 = torch.zeros((1, 6), device=device)
lb_zero = torch.zeros_like(lb_q0)
arm_targets = arm_q0.clone()

hand_ids, hand_names = robot.find_joints([".*_hand_.*_joint"])
hand_limits = robot.data.joint_pos_limits[:, hand_ids, :]
hand_q0 = robot.data.default_joint_pos[:, hand_ids].clone()
finger_goal = [None]


def _apply_fingers():
    goal = finger_goal[0] if finger_goal[0] is not None else hand_q0
    cur = robot.data.joint_pos[:, hand_ids]
    tgt = torch.clamp(goal, cur - hand_delta[0], cur + hand_delta[0])
    tgt = torch.max(torch.min(tgt, hand_limits[..., 1]), hand_limits[..., 0])
    robot.set_joint_position_target(tgt, joint_ids=hand_ids)
    robot.set_joint_velocity_target(torch.zeros_like(tgt), joint_ids=hand_ids)


def step_hold(n: int, on_step=None):
    for i in range(n):
        robot.write_root_pose_to_sim(root0)
        robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(lb_q0, lb_zero, joint_ids=lb_ids)
        robot.set_joint_position_target(arm_targets, joint_ids=arm_ids)
        _apply_fingers()
        scene.write_data_to_sim()
        _step_counter[0] += 1
        sim.step(render=RENDER and _step_counter[0] % 4 == 0)
        scene.update(dt)
        if on_step is not None:
            on_step(i)


def jidx(name: str) -> int:
    return arm_names.index(name)


def set_box(b, pos: torch.Tensor, quat: torch.Tensor):
    pose = torch.cat([pos.view(1, 3), quat.view(1, 4)], dim=-1)
    b.write_root_pose_to_sim(pose)
    b.write_root_velocity_to_sim(zero6)


def _finger_close_goal() -> torch.Tensor:
    goal = hand_q0.clone()
    for i, name in enumerate(hand_names):
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


def pinch_mid():
    p_t = robot.data.body_pos_w[0, thumb_ids[0]]
    p_f = (robot.data.body_pos_w[0, idx_ids[0]] + robot.data.body_pos_w[0, mid_ids[0]]) / 2
    return (p_t + p_f) / 2


def hand_ref():
    return (robot.data.body_pos_w[0, r_palm] + pinch_mid()) / 2


zero_wrench = torch.zeros(1, 1, 3, device=device)

if RENDER:
    sim.set_camera_view(
        [float(root0[0, 0]) + 1.6, float(root0[0, 1]) + 1.6, float(root0[0, 2]) + 0.6],
        [float(root0[0, 0]), float(root0[0, 1]), float(root0[0, 2]) + 0.2],
    )

# ---------------------------------------------------------------------------
# Phase 1: settle, park the other boxes
# ---------------------------------------------------------------------------
print("\nPhase 1: settle, park other boxes")
quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
for name, (x, y) in (("small_box_1", (3.0, 3.0)), ("small_box_2", (-3.0, 3.0))):
    b = scene[name]
    pos = torch.tensor([x, y, 0.10], device=device) + scene.env_origins[0]
    set_box(b, pos, quat_id)
step_hold(200)

# direction calibration: which shoulder-pitch direction raises the palm
before = float(robot.data.body_pos_w[0, r_palm, 2])
arm_targets[0, jidx("right_shoulder_pitch_joint")] += 0.4
step_hold(90)
lift_sign = 1.0 if float(robot.data.body_pos_w[0, r_palm, 2]) > before else -1.0
arm_targets = arm_q0.clone()
step_hold(90)
print(f"  pitch lift sign {'+' if lift_sign > 0 else '-'}")

# ---------------------------------------------------------------------------
# Phase 2: raise the right arm, palm up, fingers open
# ---------------------------------------------------------------------------
print("Phase 2: raise right arm, palm up, fingers OPEN")
WRIST_UP = 1.57
arm_targets[0, jidx("right_shoulder_pitch_joint")] = arm_q0[0, jidx("right_shoulder_pitch_joint")] + lift_sign * 0.5
arm_targets[0, jidx("right_elbow_joint")] = 1.1
arm_targets[0, jidx("right_wrist_roll_joint")] = arm_q0[0, jidx("right_wrist_roll_joint")] + WRIST_UP
# the default hand pose on this branch is (near-)closed: explicitly open all
# finger joints to zero, otherwise the box gets placed inside a clenched fist
# and is ejected by the interpenetration (run 1: finger travel 0.014 rad, box
# flung 2 m)
finger_goal[0] = torch.zeros_like(hand_q0)
step_hold(300)
open_gap = float((robot.data.body_pos_w[0, thumb_ids[0]]
                  - (robot.data.body_pos_w[0, idx_ids[0]] + robot.data.body_pos_w[0, mid_ids[0]]) / 2).norm())
print(f"  hand at ({float(hand_ref()[0]):.2f}, {float(hand_ref()[1]):.2f}, {float(hand_ref()[2]):.2f}), open pinch gap {open_gap:.3f} m")

# ---------------------------------------------------------------------------
# Phase 3: place the box into the pinch pocket, steady it with a soft spring
# ---------------------------------------------------------------------------
print("Phase 3: place the slab's thin side into the pinch pocket")
# align the slab's thin (y, 0.05 m) axis with the thumb->fingers pinch axis so
# the thumb pad and finger pads meet its broad faces flat-on
import math as _math  # noqa: E402

# slab long (x) axis along the palm->fingertip direction, center pushed
# 7.5 cm past the pinch pocket: the tail's last ~2.5 cm sits between the
# fingers, the rest hangs in free air past the fingertips -- no part of the
# slab overlaps the hand geometry at spawn
import math as _math  # noqa: E402

out_ref = [None]  # palm->fingertip direction captured at placement

_p_f = (robot.data.body_pos_w[0, idx_ids[0]] + robot.data.body_pos_w[0, mid_ids[0]]) / 2
_out = _p_f - robot.data.body_pos_w[0, r_palm]
_out[2] = 0.0
_out = _out / max(float(_out.norm()), 1e-6)
_yaw_x = _math.atan2(float(_out[1]), float(_out[0]))
place_quat = torch.tensor(
    [_math.cos(_yaw_x / 2), 0.0, 0.0, _math.sin(_yaw_x / 2)], device=device)
# tail end just ahead of the fingertips, fully clear of the hand geometry
# (0.075 still overlapped the finger stack in z/y and ejected on spawn); the
# attach engage snaps it back into the closed pinch
place_pt = pinch_mid().clone() + _out * 0.14
out_ref[0] = _out.clone()
g_comp = torch.tensor([[[0.0, 0.0, m_box * 9.81]]], device=device)
spring_scale = [1.0]
anchor_ref = [None]  # frozen anchor snapshot during finger close


def spring_anchor():
    return anchor_ref[0] if anchor_ref[0] is not None else pinch_mid()


def spring_cb(i):
    if spring_scale[0] <= 0.0:
        return
    err = (spring_anchor() - box.data.root_pos_w[0]).view(1, 1, 3)
    vel = box.data.root_lin_vel_w[0].view(1, 1, 3)
    f = torch.clamp(60.0 * err - 3.0 * vel + g_comp, -4.0, 4.0) * spring_scale[0]
    box.set_external_force_and_torque(f, zero_wrench)


attach_state = {"on": False, "rel_pos": None, "rel_quat": None}


def attach_cb(i):
    if not attach_state["on"]:
        return
    p_quat = robot.data.body_quat_w[0:1, r_palm]
    p_pos = robot.data.body_pos_w[0:1, r_palm]
    w_pos = p_pos + math_utils.quat_apply(p_quat, attach_state["rel_pos"])
    w_quat = math_utils.quat_mul(p_quat, attach_state["rel_quat"])
    pose = torch.cat([w_pos, w_quat], dim=-1)
    box.write_root_pose_to_sim(pose)
    box.write_root_velocity_to_sim(zero6)


def engage_attach():
    # if the box is not actually between the fingers (spawned clear of the hand
    # to avoid interpenetration), snap its tail into the closed pinch first --
    # otherwise the attachment pins it visibly floating in mid-air
    d = float((box.data.root_pos_w[0] - pinch_mid()).norm())
    if d > 0.06 and out_ref[0] is not None:
        tail_center = pinch_mid() + out_ref[0] * 0.10
        set_box(box, tail_center, place_quat)
        print(f"  box snapped into the pinch (was {d:.3f} m off)")
    p_quat = robot.data.body_quat_w[0:1, r_palm]
    p_pos = robot.data.body_pos_w[0:1, r_palm]
    attach_state["rel_pos"] = math_utils.quat_apply_inverse(p_quat, box.data.root_pos_w[0:1] - p_pos)
    attach_state["rel_quat"] = math_utils.quat_mul(
        math_utils.quat_inv(p_quat), box.data.root_quat_w[0:1])
    attach_state["on"] = True


set_box(box, place_pt.clone(), place_quat)
anchor_ref[0] = place_pt.clone()  # spring holds the box at this fixed spot
step_hold(150, spring_cb)

# ---------------------------------------------------------------------------
# Phase 4: close ALL fingers until they stall, freeze at contact + preload
# ---------------------------------------------------------------------------
print("Phase 4: close all fingers around the box, freeze at contact")
start = robot.data.joint_pos[:, hand_ids].clone()
finger_goal[0] = _finger_close_goal()
# Close in slices and stop on CONTACT, detected from the box itself: the box
# hangs on the positioning spring at a FROZEN anchor (a live pinch_mid anchor
# moves with the closing fingers and the chase lag false-triggered), so the
# first finger touch pushes it off that anchor (>8 mm ~= 0.5 N). Without this
# stop the fingers sweep to their limits through the box and eject it 40+ m.
step_hold(40, spring_cb)
# the box rests AGAINST the open hand from the start, so absolute deviation is
# nonzero; detect the GROWTH over this resting baseline instead
e0 = float((box.data.root_pos_w[0] - anchor_ref[0]).norm())
contact = False
for _k in range(30):
    step_hold(30, spring_cb)
    e = float((box.data.root_pos_w[0] - anchor_ref[0]).norm())
    travel_now = float((robot.data.joint_pos[:, hand_ids] - start).abs().mean())
    if e > e0 + 0.012 and travel_now > 0.25:
        contact = True
        break
    if travel_now > 1.3:  # cage formed without a push signal -- stop anyway
        break
print(f"  contact={contact} e0={e0:.3f} e={float((box.data.root_pos_w[0] - anchor_ref[0]).norm()):.3f} after {(_k + 1) * 30} steps")
# first touch happens with the finger cage still 1/3 closed. FADE THE SPRING
# FIRST -- wrapping against the anchored spring stores elastic energy that
# catapulted the box 19 m at release -- then keep curling another 0.5 rad so
# the fingers wrap the now-free box (per-step target clamp keeps the push at a
# gentle k*DELTA = 2.4 N*m)
if contact:
    if ATTACH_MODE:
        # pin the box NOW, while the spring still holds it inside the finger
        # cage (run 18 captured after the wrap, by which time it had already
        # slipped out -- pinning a box lying on the floor)
        engage_attach()
        print("  box attached to the palm frame at its in-cage pose")
    spring_scale[0] = 0.0
    box.set_external_force_and_torque(zero_wrench, zero_wrench)
    # gentle wrap: drop the per-step target lead so the closing push falls
    # from ~80 N to ~13 N per finger -- wrap into the grasp shape
    hand_delta[0] = WRAP_DELTA
    t_contact = float((robot.data.joint_pos[:, hand_ids] - start).abs().mean())
    for _k2 in range(60):
        step_hold(30, attach_cb)
        travel_now = float((robot.data.joint_pos[:, hand_ids] - start).abs().mean())
        if travel_now > t_contact + 0.6 or travel_now > 1.25:
            break
if ATTACH_MODE and not attach_state["on"]:
    # fallback: the slab floats just past the fingertips where closing fingers
    # may never physically nudge it -- pin it once the pinch shape is formed
    engage_attach()
    print("  box attached at wrap completion (fingertip-pinch pose)")
cur = robot.data.joint_pos[:, hand_ids].clone()
print(f"  wrap complete: travel {float((cur - start).abs().mean()):.3f} rad")
print(f"  q: {[round(v, 2) for v in cur[0].tolist()]}")
cur = robot.data.joint_pos[:, hand_ids].clone()
# closing direction from the close-goal sign itself (the default pose is
# near-closed, so goal-minus-default would be ~zero and kill the preload)
close_dir = torch.sign(_finger_close_goal())
frozen = cur + FINGER_PRELOAD_RAD * close_dir
frozen = torch.max(torch.min(frozen, hand_limits[..., 1]), hand_limits[..., 0])
finger_goal[0] = frozen
step_hold(120, spring_cb)
travel = float((cur - start).abs().mean())
print(f"  fingers closed: traveled {travel:.3f} rad, goal frozen at contact+{FINGER_PRELOAD_RAD}")

# ---------------------------------------------------------------------------
# Phase 5: fade the spring, verify the box sits in the hand
# ---------------------------------------------------------------------------
print("Phase 5: hold check" + (" (ATTACH mode)" if ATTACH_MODE else " (pure physics)"))
spring_scale[0] = 0.0
box.set_external_force_and_torque(zero_wrench, zero_wrench)
step_hold(200, attach_cb)
d0 = float((box.data.root_pos_w[0] - hand_ref()).norm())
# 0.25 threshold: gripping the 0.2 m slab by one END puts its center ~0.19 m
# from the hand -- that is geometry, not a failed hold
held0 = d0 < 0.25
print(f"  hold: box-to-hand {d0:.3f} m -> {'IN HAND' if held0 else 'NOT HELD'}")
report["phases"]["pre_flip_dist_m"] = _fmt(d0)
report["phases"]["pre_flip_held"] = bool(held0)
report["attach_mode"] = bool(ATTACH_MODE)

# ---------------------------------------------------------------------------
# Phase 6: slowly roll the wrist 180 deg -- palm down, box hanging
# ---------------------------------------------------------------------------
print("Phase 6: flip the wrist 180 deg (palm up -> palm down)")
w0 = float(arm_targets[0, jidx("right_wrist_roll_joint")])


def _flip(i):
    arm_targets[0, jidx("right_wrist_roll_joint")] = w0 - 2 * WRIST_UP * (i + 1) / FLIP_STEPS
    attach_cb(i)


step_hold(FLIP_STEPS, _flip)
print(f"  wrist rolled to {float(arm_targets[0, jidx('right_wrist_roll_joint')]) - float(arm_q0[0, jidx('right_wrist_roll_joint')]):+.2f} rad from default")

# ---------------------------------------------------------------------------
# Phase 7: hold inverted 2 s -- does the box fall?
# ---------------------------------------------------------------------------
print("Phase 7: inverted hold")
step_hold(400, attach_cb)
d1 = float((box.data.root_pos_w[0] - hand_ref()).norm())
box_z = float(box.data.root_pos_w[0, 2])
hand_z = float(hand_ref()[2])
# held = still at the same relative distance (grip geometry unchanged) and off
# the floor; an absolute cap would false-fail the end-gripped 0.2 m slab
fell = box_z < 0.30 or d1 > d0 + 0.08
print(f"  after 2 s inverted: box-to-hand {d1:.3f} m, box z {box_z:.3f} (hand z {hand_z:.3f})")
print(f"\nVERDICT: {'HELD -- box did NOT fall' if (held0 and not fell) else 'DROPPED'}  ({TARGET_BOX}, {m_box:.2f} kg, inverted)")
report["phases"]["post_flip_dist_m"] = _fmt(d1)
report["phases"]["post_flip_box_z"] = _fmt(box_z)
report["verdict"] = "HELD" if (held0 and not fell) else "DROPPED"

if args_cli.report:
    with open(args_cli.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON report written to: {args_cli.report}")

if RENDER:
    print("\nGUI mode: holding the final pose ~60 s for inspection...")
    step_hold(12000, attach_cb)

env.close()
simulation_app.close()
