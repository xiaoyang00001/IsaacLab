# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Two-palm hug-lift of the REAL cart_box2 (1.5 kg cardboard box, untouched
scene asset) in a purpose-built verification pose.

Scene design rationale (every element is a previously PROVEN ingredient):
  - the two-palm squeeze frame that lifted the 5 cm cubes in GUI runs
  - the box presents its 0.25 m width to the palms, so contact is the broad
    0.38 x 0.149 face (flat-on-flat; end-face clamps twist out, cl-series)
  - the lift is a PURE torso rise (anchored root moves up, arm targets
    frozen): palm gap cannot drift, which killed every arm-arc lift
  - thumbs stay OPEN (their tips sit ~0.11 m apart and would spear the box
    faces); the four fingers half-curl as support pads
  - box properties are the scene's own cart_box2: D05 USD, 1.5 kg, stock
    friction -- nothing about the object is modified

Run GUI:  env LD_LIBRARY_PATH= XR_RUNTIME_JSON=/nonexistent DISPLAY=:0 \\
    ./isaaclab.sh -p scripts/diagnostics/grasp_hug_cartbox.py --device cpu
Headless: add --headless
"""

import argparse
import os

os.environ["ISAACLAB_SCENE_SYNC_ROLE"] = "none"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="two-palm hug-lift of cart_box2")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--report", type=str, default=None)
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
import math  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ARM_JOINT_PATTERNS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint",
                      ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
LOWER_BODY_PATTERNS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]

TARGET = "cart_box2"        # the user's box, stock properties
BOX_W = 0.25                # clamped width (y of the D05 collider)
TAU_GRIP_DELTA_NM = 5.0     # ~12 N/side at the shoulder arm; 1.5 kg needs ~7 N/side
GAP_FLOOR_M = BOX_W + 0.04   # run 12: at +0.09 the (thinner than assumed)
                             # forearms never reached the box faces
HAND_DELTA = 0.06
SPRING_FADE_STEPS = 200
LIFT_RISE_M = 0.30
LIFT_STEPS = 500

report: dict = {"target": TARGET, "phases": {}, "verdict": None}


def _fmt(x):
    return round(float(x), 4)


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
box = scene[TARGET]
dt = env.physics_dt
device = env.device
m_box = float(box.root_physx_view.get_masses().sum())
print("=" * 78)
print(f"TWO-PALM HUG-LIFT: {TARGET} (stock asset)  mass={m_box:.2f} kg")
print("=" * 78)

lb_ids, _ = robot.find_joints(LOWER_BODY_PATTERNS)
arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)
palm_ids, palm_names = robot.find_bodies([".*_hand_palm_link"])
l_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("left")][0])]
r_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("right")][0])]
elbow_ids, elbow_names = robot.find_bodies([".*_elbow_link"])
wrist_ids, wrist_names = robot.find_bodies([".*_wrist_roll_link"])
l_elbow = elbow_ids[elbow_names.index([n for n in elbow_names if n.startswith("left")][0])]
r_elbow = elbow_ids[elbow_names.index([n for n in elbow_names if n.startswith("right")][0])]
l_wrist = wrist_ids[wrist_names.index([n for n in wrist_names if n.startswith("left")][0])]
r_wrist = wrist_ids[wrist_names.index([n for n in wrist_names if n.startswith("right")][0])]


def fore_l():
    return (robot.data.body_pos_w[0, l_elbow] + robot.data.body_pos_w[0, l_wrist]) / 2


def fore_r():
    return (robot.data.body_pos_w[0, r_elbow] + robot.data.body_pos_w[0, r_wrist]) / 2


def fore_gap():
    return float((fore_l() - fore_r()).norm())


def fore_mid():
    return (fore_l() + fore_r()) / 2

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


def step_hold(n, on_step=None):
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


def jidx(name):
    return arm_names.index(name)


def palm_pos():
    return robot.data.body_pos_w[0, l_palm].clone(), robot.data.body_pos_w[0, r_palm].clone()


def palm_gap():
    lp, rp = palm_pos()
    return float((lp - rp).norm())


def palm_mid():
    lp, rp = palm_pos()
    return (lp + rp) / 2


def set_obj(b, pos, quat):
    pose = torch.cat([pos.view(1, 3), quat.view(1, 4)], dim=-1)
    b.write_root_pose_to_sim(pose)
    b.write_root_velocity_to_sim(zero6)


k_arm = float(robot.actuators["arms"].stiffness.mean())
d_arm = float(robot.actuators["arms"].damping.mean())
tau_arm = float(getattr(robot.actuators["arms"], "effort_limit_sim", robot.actuators["arms"].effort_limit).mean())


def est_torque(name):
    j = robot.find_joints([name])[0][0]
    q = float(robot.data.joint_pos[0, j])
    qd = float(robot.data.joint_vel[0, j])
    tgt = float(arm_targets[0, jidx(name)])
    return max(-tau_arm, min(tau_arm, k_arm * (tgt - q) - d_arm * qd))


def _finger_close_goal():
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

if RENDER:
    sim.set_camera_view(
        [float(root0[0, 0]) + 2.0, float(root0[0, 1]) + 2.0, float(root0[0, 2]) + 0.8],
        [float(root0[0, 0]), float(root0[0, 1]), float(root0[0, 2]) + 0.3],
    )

# --- Phase 1: settle, park unrelated boxes far away -------------------------
print("\nPhase 1: settle")
quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
for name, (x, y) in (("long_box", (4.0, 4.0)), ("small_box_1", (4.0, -4.0)), ("small_box_2", (-4.0, 4.0))):
    b = scene[name]
    pos = torch.tensor([x, y, 0.10], device=device) + scene.env_origins[0]
    set_obj(b, pos, quat_id)
step_hold(200)

# --- Phase 2: direction calibration -----------------------------------------
print("Phase 2: joint direction calibration")


def probe(joint, delta, metric):
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
print(f"  pitch lift {'+' if lift_sign > 0 else '-'} | roll inward L{'+' if inward_l > 0 else '-'} R{'+' if inward_r > 0 else '-'}")

# --- Phase 3: two-palm pose, palms wide, four fingers half-curled ------------
# A BEAR HUG, not a palm clamp: three rounds proved the fingers always shadow
# the palms for a 0.25 m wide box (half-curled pads absorb the contact into
# invisibility, straight tips spear the box away, and this hand has no
# backward finger travel). The FOREARMS are hard, finger-free jaws -- the
# original check_hug_box.py motion, and literally what "抱起来" means.
print("Phase 3: two-palm pose (all fingers half-curled, small-cube recipe)")
for side in ("left", "right"):
    arm_targets[0, jidx(f"{side}_shoulder_pitch_joint")] = arm_q0[0, jidx(f"{side}_shoulder_pitch_joint")] + lift_sign * 0.45
    arm_targets[0, jidx(f"{side}_elbow_joint")] = 0.9
step_hold(180)
open_extra, tries = 0.0, 0
while not (0.50 < palm_gap() < 0.62) and tries < 20:
    open_extra += 0.04 if palm_gap() <= 0.50 else -0.04
    arm_targets[0, jidx("left_shoulder_roll_joint")] = arm_q0[0, jidx("left_shoulder_roll_joint")] - inward_l * open_extra
    arm_targets[0, jidx("right_shoulder_roll_joint")] = arm_q0[0, jidx("right_shoulder_roll_joint")] - inward_r * open_extra
    step_hold(60)
    tries += 1
print(f"  open palm gap {palm_gap():.3f} m (box width {BOX_W})")
finger_goal[0] = hand_q0 + 0.5 * (_finger_close_goal() - hand_q0)
step_hold(300)

# --- Phase 4: the user's box straight between the palms ----------------------
# (this placement was rock-stable from hug-run 1: dist 0.002 m, no ejection)
print("Phase 4: place cart_box2 between the palms (broad faces to the pads)")
lp0, rp0 = palm_pos()
ax = lp0 - rp0
yaw = math.atan2(float(ax[1]), float(ax[0])) + math.pi / 2  # box y (0.25 width) along the palm axis
place_quat = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=device)
place_fix = palm_mid().clone()
anchor_ref = [place_fix.clone()]
set_obj(box, place_fix, place_quat)

g_comp = torch.tensor([[[0.0, 0.0, m_box * 9.81]]], device=device)
spring_scale = [1.0]


def spring_cb(i):
    if spring_scale[0] <= 0.0:
        return
    err = (anchor_ref[0] - box.data.root_pos_w[0]).view(1, 1, 3)
    vel = box.data.root_lin_vel_w[0].view(1, 1, 3)
    # a stiff steady hand: every contact against the kinematically anchored
    # robot is tens of newtons, and a 22 N spring lost the box 5 times running
    f = torch.clamp(200.0 * err - 8.0 * vel + g_comp, -80.0, 80.0) * spring_scale[0]
    box.set_external_force_and_torque(f, zero_wrench)


step_hold(150, spring_cb)
print(f"  placed, dist to anchor {float((box.data.root_pos_w[0] - anchor_ref[0]).norm()):.3f} m")

# --- Phase 5: torque-delta squeeze (the proven recipe) -----------------------
print("Phase 5: squeeze (contact-force servo; shoulder torque is blind to "
      "contact absorbed by the soft finger joints)")
try:
    _hc = env.scene.sensors["hand_contact"]
    _hcn = _hc.body_names
    _hc_l = [i for i, n in enumerate(_hcn) if n.startswith("left")]
    _hc_r = [i for i, n in enumerate(_hcn) if n.startswith("right")]
    print(f"  contact sensor online ({len(_hcn)} bodies)")
except Exception as exc:
    _hc = None
    print(f"  [WARN] no contact sensor: {exc}")
# FIXED-INTERFERENCE close: no signals. The small-cube pad-contact gap scales
# to 0.25 + 0.156 = 0.406 for this box; stopping 1.6 cm past it gives a firm,
# gentle clamp (hug-run 1 overshot to 4.8 cm interference and ejected the box;
# every signal-based stop proved either blind or late on this geometry).
CLOSE_GAP = 0.372   # bisect: 0.390 (1.6 cm) slipped clean, 0.358 (4.8 cm) ejected
f_l = f_r = 0.0
tries = 0
while palm_gap() > CLOSE_GAP and tries < 120:
    arm_targets[0, jidx("left_shoulder_roll_joint")] += inward_l * 0.006
    arm_targets[0, jidx("right_shoulder_roll_joint")] += inward_r * 0.006
    step_hold(20, spring_cb)
    tries += 1
    if tries % 25 == 0:
        if _hc is not None:
            fm = _hc.data.net_forces_w.norm(dim=-1)
            f_l = float(fm[:, _hc_l].amax()) if _hc_l else 0.0
            f_r = float(fm[:, _hc_r].amax()) if _hc_r else 0.0
        print(f"    [{tries:3d}] palm-gap {palm_gap():.3f} | F L{f_l:.1f} R{f_r:.1f} N")
if _hc is not None:
    fm = _hc.data.net_forces_w.norm(dim=-1)
    f_l = float(fm[:, _hc_l].amax()) if _hc_l else 0.0
    f_r = float(fm[:, _hc_r].amax()) if _hc_r else 0.0
print(f"  squeeze stop: palm-gap {palm_gap():.3f} m | F L{f_l:.1f} R{f_r:.1f} N ({tries} increments)")
report["phases"]["squeeze"] = {"gap_m": _fmt(palm_gap()), "force_N": [_fmt(f_l), _fmt(f_r)]}

# --- Phase 6: fade the spring, verify the hug --------------------------------
print("Phase 6: fade spring, verify hug")


def _fade(i):
    spring_scale[0] = max(0.0, 1.0 - (i + 1) / SPRING_FADE_STEPS)
    spring_cb(i)


step_hold(SPRING_FADE_STEPS, _fade)
spring_scale[0] = 0.0
box.set_external_force_and_torque(zero_wrench, zero_wrench)
z0 = float(box.data.root_pos_w[0, 2])
step_hold(300)
sank = z0 - float(box.data.root_pos_w[0, 2])
off = float((box.data.root_pos_w[0] - palm_mid()).norm())
held = sank < 0.05 and off < 0.20
print(f"  free hug 1.5 s: box sank {sank:.3f} m, off-center {off:.3f} m -> {'HELD' if held else 'DROPPED'}")
report["phases"]["hug"] = {"sank_m": _fmt(sank), "off_m": _fmt(off), "held": bool(held)}

# --- Phase 7: lift by raising the WHOLE torso (arms frozen, gap untouched) ---
print("Phase 7: torso-rise lift")
zb0 = float(box.data.root_pos_w[0, 2])
zp0 = float(palm_mid()[2])
z_start = float(root0[0, 2])


def _rise(i):
    root0[:, 2] = z_start + LIFT_RISE_M * (i + 1) / LIFT_STEPS


step_hold(LIFT_STEPS, _rise)
step_hold(200)
rise_p = float(palm_mid()[2]) - zp0
rise_b = float(box.data.root_pos_w[0, 2]) - zb0
slip = rise_p - rise_b
lifted = held and rise_b > 0.20 and abs(slip) < 0.08
print(f"  lift: hands +{rise_p:.3f} m, box {rise_b:+.3f} m, slip {slip:+.3f} m")
print(f"\nVERDICT: {'HUG-LIFTED -- the 1.5 kg box was carried up in both palms' if lifted else 'FAILED'}")
report["phases"]["lift"] = {"hands_m": _fmt(rise_p), "box_m": _fmt(rise_b), "slip_m": _fmt(slip)}
report["verdict"] = "HUG-LIFTED" if lifted else "FAILED"

if args_cli.report:
    with open(args_cli.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON report written to: {args_cli.report}")

if RENDER:
    print("\nGUI mode: holding the final pose ~60 s for inspection...")
    step_hold(12000)

env.close()
simulation_app.close()
