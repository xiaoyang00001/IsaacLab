# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted 3-object grasp verification (0716-check branch, Windows teleop host). v5

v4 proved the winning recipe on small_box_2 (LIFTED): raise both arms, pre-curl
the fingers to half-close so the finger pads (not the extended fingertips) are
the contact surface, squeeze with a baseline-delta shoulder-roll torque stop,
fade the positioning spring, hold, lift. v4's long-box attempt built real
contact (4.5 N*m delta) but a subsequent finger-close motion knocked the box
out of the settled grip.

v5 therefore applies the SAME proven two-palm recipe to all 3 objects, with no
finger motion after contact, and a higher torque target for the heavier box.

Failure archaeology (why earlier versions lost):
  v1  absolute torque threshold saw the ~11 N*m posture torque as contact;
      0.12 rad finger preload (~300 N fingertip) exploded the box 119 m away.
  v2  palm-face offset assumed 0.148 m (true ~0.105): palms never reached the
      box; gentle preload fixed the explosion but nothing was gripping.
  v3  with fingers extended the FINGERTIPS meet the box before the palms --
      force flows through finger joints, invisible to shoulder torque, and
      deep squeezes explode. Finger travel stalled at 0.42 rad in every run
      (self-collision, not box contact).
  v4  pre-curl fix -> small_box_2 LIFTED. Long box lost to the post-contact
      finger close; cradle drop remained geometry-blind.

Run (conda env python; isaaclab.bat breaks on the space in the system python):
  d:\\miniconda3\\envs\\env_isaaclab\\python.exe scripts\\diagnostics\\grasp_verify_3objects.py ^
      --headless --device cpu --report scripts\\diagnostics\\rep_grasp3.json
"""

import argparse
import os

# Must be set before the pick_place cfg module is imported: disable all scene
# sync networking (role none => physics authority stays True, boxes dynamic).
os.environ["ISAACLAB_SCENE_SYNC_ROLE"] = "none"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 scripted 3-object grasp verification")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--report", type=str, default=None, help="JSON report output path")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# pick_place is blacklisted in isaaclab_tasks, import explicitly; pinocchio
# must be imported before AppLauncher (same as teleop_se3_agent.py).
import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# GUI mode: render every 4th physics step so the run stays watchable without
# crawling; headless keeps the pure-physics fast path.
RENDER = not args_cli.headless
_step_counter = [0]

import functools  # noqa: E402
import builtins  # noqa: E402
print = functools.partial(builtins.print, flush=True)

import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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

HAND_DELTA = 0.02        # per-step finger target clamp, same as teleop gripper drive
SPRING_FADE_STEPS = 200  # 1.0 s spring fade-out
LIFT_STEPS = 400         # 2.0 s lift ramp

report: dict = {"branch": "verify-0716", "static": {}, "objects": {}, "verdict": []}


def _fmt(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# Env construction (offline: no UDP mirror, no ZMQ sync)
# ---------------------------------------------------------------------------
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
# Keep mujoco_g1_mirror_1: its __init__ binds the hand friction material and
# initializes contact-freeze (validating those code paths), but this script
# drives sim.step() directly so the action manager never runs mid-experiment.
for term_name in ("mujoco_g1_mirror_2", "scene_state_sync",
                  "env_reset_sync", "box_success_reset"):
    if hasattr(env_cfg.actions, term_name):
        setattr(env_cfg.actions, term_name, None)
if hasattr(env_cfg.terminations, "box_dropped"):
    env_cfg.terminations.box_dropped = None
if hasattr(env_cfg.terminations, "time_out"):
    env_cfg.terminations.time_out = None

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.reset()

sim = env.sim
scene = env.scene
robot = scene["robot_1"]
dt = env.physics_dt
device = env.device

arms_act = robot.actuators["arms"]
k_arm = float(arms_act.stiffness.mean())
d_arm = float(arms_act.damping.mean())
tau_arm = (float(arms_act.effort_limit_sim.mean())
           if hasattr(arms_act, "effort_limit_sim") else float(arms_act.effort_limit.mean()))
print("=" * 78)
print(f"SCRIPTED 3-OBJECT GRASP v5 (two-palm recipe for all)  physics_dt={dt}")
print("=" * 78)
report["static"] = {"arm_stiffness": k_arm, "arm_effort_limit": tau_arm}

# ---------------------------------------------------------------------------
# Low-level control helpers
# ---------------------------------------------------------------------------
lb_ids, _ = robot.find_joints(LOWER_BODY_PATTERNS)
arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)
palm_ids, palm_names = robot.find_bodies([".*_hand_palm_link"])
l_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("left")][0])]
r_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("right")][0])]

lb_q0 = robot.data.default_joint_pos[:, lb_ids].clone()
arm_q0 = robot.data.default_joint_pos[:, arm_ids].clone()

# Root: lifted 1.0 m above spawn and rotated 180 deg -> experiments happen in
# free air, independent of the table/furniture around the spawn point.
root0 = robot.data.default_root_state[:, :7].clone()
root0[:, :3] += scene.env_origins
root0[:, 2] += 1.0
q_z180 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
root0[:, 3:7] = math_utils.quat_mul(q_z180, root0[:, 3:7])
zero6 = torch.zeros((1, 6), device=device)
lb_zero = torch.zeros_like(lb_q0)

if RENDER:
    # aim the viewport at the in-air experiment pose (robot root lifted 1 m)
    sim.set_camera_view(
        [float(root0[0, 0]) + 2.2, float(root0[0, 1]) + 2.2, float(root0[0, 2]) + 1.2],
        [float(root0[0, 0]), float(root0[0, 1]), float(root0[0, 2]) + 0.4],
    )

arm_targets = arm_q0.clone()

hand_ids, hand_names = robot.find_joints([".*_hand_.*_joint"])
hand_limits = robot.data.joint_pos_limits[:, hand_ids, :]
hand_q0 = robot.data.default_joint_pos[:, hand_ids].clone()
finger_goal = [None]


def _apply_fingers():
    goal = finger_goal[0] if finger_goal[0] is not None else hand_q0
    cur = robot.data.joint_pos[:, hand_ids]
    tgt = torch.clamp(goal, cur - HAND_DELTA, cur + HAND_DELTA)
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


def palm_pos():
    return robot.data.body_pos_w[0, l_palm].clone(), robot.data.body_pos_w[0, r_palm].clone()


def palm_gap() -> float:
    lp, rp = palm_pos()
    return float((lp - rp).norm())


def palm_mid():
    lp, rp = palm_pos()
    return (lp + rp) / 2


def set_box(b, pos: torch.Tensor, quat: torch.Tensor):
    pose = torch.cat([pos.view(1, 3), quat.view(1, 4)], dim=-1)
    b.write_root_pose_to_sim(pose)
    b.write_root_velocity_to_sim(zero6)


def park_others(active_name: str):
    """Move the non-active boxes to the ground 3 m away so a flying/dropped
    box from a previous experiment cannot disturb the current one."""
    quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    spots = {"long_box": (3.0, 3.0), "small_box_1": (3.0, -3.0), "small_box_2": (-3.0, 3.0)}
    for name, (x, y) in spots.items():
        if name == active_name:
            continue
        b = scene[name]
        pos = torch.tensor([x, y, 0.10], device=device) + scene.env_origins[0]
        set_box(b, pos, quat_id)


def est_torque(joint_name: str) -> float:
    j = robot.find_joints([joint_name])[0][0]
    q = float(robot.data.joint_pos[0, j])
    qd = float(robot.data.joint_vel[0, j])
    tgt = float(arm_targets[0, jidx(joint_name)])
    return max(-tau_arm, min(tau_arm, k_arm * (tgt - q) - d_arm * qd))


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


zero_wrench = torch.zeros(1, 1, 3, device=device)


def make_spring(box, anchor_fn, mass: float, kp: float = 80.0, kd: float = 3.0, fmax: float = 6.0):
    """Positioning spring holding the box at anchor_fn() with gravity comp."""
    g_comp = torch.tensor([[[0.0, 0.0, mass * 9.81]]], device=device)
    state = {"scale": 1.0}

    def cb(i):
        if state["scale"] <= 0.0:
            return
        err = (anchor_fn() - box.data.root_pos_w[0]).view(1, 1, 3)
        vel = box.data.root_lin_vel_w[0].view(1, 1, 3)
        f = torch.clamp(kp * err - kd * vel + g_comp, -fmax, fmax) * state["scale"]
        box.set_external_force_and_torque(f, zero_wrench)

    return cb, state


def fade_spring(box, cb, state):
    def _fade(i):
        state["scale"] = max(0.0, 1.0 - (i + 1) / SPRING_FADE_STEPS)
        cb(i)
    step_hold(SPRING_FADE_STEPS, _fade)
    state["scale"] = 0.0
    box.set_external_force_and_torque(zero_wrench, zero_wrench)


def reset_pose():
    global arm_targets
    arm_targets = arm_q0.clone()
    finger_goal[0] = None
    step_hold(200)


# ---------------------------------------------------------------------------
# Direction calibration
# ---------------------------------------------------------------------------
print("\nPhase 0: joint direction calibration")
step_hold(150)


def probe(joint: str, delta: float, metric) -> float:
    global arm_targets
    before = metric()
    arm_targets[0, jidx(joint)] += delta
    step_hold(90)
    after = metric()
    arm_targets = arm_q0.clone()
    step_hold(90)
    return after - before


d_z = probe("left_shoulder_pitch_joint", 0.4, lambda: float(robot.data.body_pos_w[0, l_palm, 2]))
lift_sign = 1.0 if d_z > 0 else -1.0
d_gap_l = probe("left_shoulder_roll_joint", 0.25, palm_gap)
inward_l = 1.0 if d_gap_l < 0 else -1.0
d_gap_r = probe("right_shoulder_roll_joint", 0.25, palm_gap)
inward_r = 1.0 if d_gap_r < 0 else -1.0
print(f"  pitch lift sign {'+' if lift_sign > 0 else '-'} | roll inward: L{'+' if inward_l > 0 else '-'} R{'+' if inward_r > 0 else '-'}")


# ---------------------------------------------------------------------------
# The proven two-palm grasp recipe (v4: LIFTED on small_box_2)
# ---------------------------------------------------------------------------
def two_palm_grasp(name: str, size_note: str, open_lo: float, open_hi: float,
                   gap_floor: float, tau_delta: float, align_yaw: bool, tag: str,
                   yaw_offset: float = 0.0) -> bool:
    print("\n" + "=" * 78)
    print(f"{tag}: {name}  two-palm grasp ({size_note})")
    print("=" * 78)
    box = scene[name]
    m = float(box.root_physx_view.get_masses().sum())
    park_others(name)
    reset_pose()

    # raised two-arm pose, palms facing each other
    for side in ("left", "right"):
        arm_targets[0, jidx(f"{side}_shoulder_pitch_joint")] = arm_q0[0, jidx(f"{side}_shoulder_pitch_joint")] + lift_sign * 0.45
        arm_targets[0, jidx(f"{side}_elbow_joint")] = 0.9
    step_hold(180)
    open_extra, tries = 0.0, 0
    while not (open_lo < palm_gap() < open_hi) and tries < 15:
        open_extra += 0.04 if palm_gap() <= open_lo else -0.04
        arm_targets[0, jidx("left_shoulder_roll_joint")] = arm_q0[0, jidx("left_shoulder_roll_joint")] - inward_l * open_extra
        arm_targets[0, jidx("right_shoulder_roll_joint")] = arm_q0[0, jidx("right_shoulder_roll_joint")] - inward_r * open_extra
        step_hold(60)
        tries += 1
    print(f"  start palm gap {palm_gap():.3f} m")

    # pre-curl fingers to half-close: pads forward, tips retracted (v3 finding:
    # extended fingertips otherwise meet the box first and squeezes explode)
    finger_goal[0] = hand_q0 + 0.5 * (_finger_close_goal() - hand_q0)
    step_hold(300)

    tl0 = est_torque("left_shoulder_roll_joint")
    tr0 = est_torque("right_shoulder_roll_joint")

    if align_yaw == "upright":
        # stand the box's long (x) axis vertical, clamp its thin y width between
        # the palms: contact is the 20x10 cm broad face, nothing protrudes into
        # the torso or fingers ("holding a standing board with both palms")
        lp, rp = palm_pos()
        ax = lp - rp
        theta = math.atan2(float(ax[1]), float(ax[0])) - math.pi / 2
        qz = torch.tensor([[math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)]], device=device)
        qy = torch.tensor([[math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]], device=device)
        quat = math_utils.quat_mul(qz, qy)[0]
    elif align_yaw:
        lp, rp = palm_pos()
        ax = lp - rp
        yaw = math.atan2(float(ax[1]), float(ax[0])) + yaw_offset
        quat = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=device)
    else:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    set_box(box, palm_mid(), quat)

    spring_cb, spring_state = make_spring(box, palm_mid, m)
    step_hold(120, spring_cb)

    # baseline-delta torque squeeze; NO finger motion after contact (v4 finding:
    # a post-contact finger close knocks the settled box out of the grip)
    tries = 0
    while tries < 60:
        dl = abs(est_torque("left_shoulder_roll_joint") - tl0)
        dr = abs(est_torque("right_shoulder_roll_joint") - tr0)
        if (dl >= tau_delta and dr >= tau_delta) or palm_gap() <= gap_floor:
            break
        arm_targets[0, jidx("left_shoulder_roll_joint")] += inward_l * 0.01
        arm_targets[0, jidx("right_shoulder_roll_joint")] += inward_r * 0.01
        step_hold(40, spring_cb)
        tries += 1
    gap_stop = palm_gap()
    dl = est_torque("left_shoulder_roll_joint") - tl0
    dr = est_torque("right_shoulder_roll_joint") - tr0
    print(f"  squeeze stop: gap {gap_stop:.3f} m | roll torque delta L{dl:+.1f} R{dr:+.1f} N*m ({tries} increments)")

    fade_spring(box, spring_cb, spring_state)
    print("  spring faded out over 1.0 s")

    # free hold, then slow lift
    z0 = float(box.data.root_pos_w[0, 2])
    step_hold(300)
    z1 = float(box.data.root_pos_w[0, 2])
    hold_drop = z0 - z1
    off = float((box.data.root_pos_w[0] - palm_mid()).norm())
    held = hold_drop < 0.05 and off < 0.15
    print(f"  free hold 1.5 s: box sank {hold_drop:.3f} m, off-anchor {off:.3f} m -> {'HELD' if held else 'DROPPED'}")

    zp0 = float(palm_mid()[2])
    zb0 = float(box.data.root_pos_w[0, 2])
    pl0 = float(arm_targets[0, jidx("left_shoulder_pitch_joint")])
    pr0 = float(arm_targets[0, jidx("right_shoulder_pitch_joint")])

    def _lift(i):
        a = (i + 1) / LIFT_STEPS
        arm_targets[0, jidx("left_shoulder_pitch_joint")] = pl0 + lift_sign * 0.45 * a
        arm_targets[0, jidx("right_shoulder_pitch_joint")] = pr0 + lift_sign * 0.45 * a

    step_hold(LIFT_STEPS, _lift)
    step_hold(150)
    rise_p = float(palm_mid()[2]) - zp0
    rise_b = float(box.data.root_pos_w[0, 2]) - zb0
    slip = rise_p - rise_b
    lifted = held and rise_p > 0.05 and rise_b > 0.05 and abs(slip) < 0.08
    print(f"  lift: hands +{rise_p:.3f} m, box {rise_b:+.3f} m, slip {slip:+.3f} m")
    print(f"  {tag} verdict: {'LIFTED' if lifted else 'FAILED (' + ('dropped at hold' if not held else 'slipped during lift') + ')'}")
    report["objects"][name] = {
        "box_mass_kg": _fmt(m), "gap_stop_m": _fmt(gap_stop),
        "squeeze_increments": tries,
        "roll_torque_delta_Nm": [_fmt(dl), _fmt(dr)],
        "hold_drop_m": _fmt(hold_drop), "off_anchor_m": _fmt(off),
        "held_after_release": bool(held),
        "hands_rise_m": _fmt(rise_p), "box_rise_m": _fmt(rise_b),
        "slip_m": _fmt(slip), "lifted": bool(lifted),
    }
    return bool(lifted)


# Torque threshold 5.0: the arm-servo drift noise plateaus at ~2.0-2.1 N*m on
# the merged-007 scene and false-triggered a 2.0 threshold with zero real
# contact; genuine pad contact shows 7+ N*m, so 5.0 separates them cleanly.
ok2 = two_palm_grasp("small_box_2", "5 cm cube, 0.08 kg", 0.28, 0.60, 0.100, 5.0, False, "G2")
ok3 = two_palm_grasp("small_box_1", "5 cm cube, 0.08 kg", 0.28, 0.60, 0.100, 5.0, False, "G3")
# long box: clamping the 0.2 m long axis end faces twists out even at 10 N*m
# (point contact); laying it horizontal across the grip jams it on the curled
# fingers. Stand it UPRIGHT and clamp the thin 5 cm width -- identical contact
# geometry to the small cubes that lifted, broad 20x10 cm faces on the pads.
ok1 = two_palm_grasp("long_box", "20x5x10 cm upright, 0.25 kg", 0.28, 0.60,
                     0.120, 5.0, "upright", "G1")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY: can the robot grasp the 3 objects?")
print("=" * 78)
report["verdict"] = [
    f"long_box (0.25 kg): {'GRASPED & LIFTED' if ok1 else 'FAILED'}",
    f"small_box_1 (0.08 kg): {'GRASPED & LIFTED' if ok3 else 'FAILED'}",
    f"small_box_2 (0.08 kg): {'GRASPED & LIFTED' if ok2 else 'FAILED'}",
]
for i, v in enumerate(report["verdict"], 1):
    print(f"  {i}. {v}")
print(f"  => {sum([ok1, ok2, ok3])}/3 objects grasped")
report["grasped_count"] = int(sum([ok1, ok2, ok3]))

if args_cli.report:
    with open(args_cli.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report written to: {args_cli.report}")
else:
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))

if RENDER:
    print("\nGUI mode: holding the final pose ~60 s for inspection (Ctrl+C to end early)...")
    step_hold(12000)

env.close()
simulation_app.close()
