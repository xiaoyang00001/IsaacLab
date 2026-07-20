# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Real-scene grasp: walk-up (teleport) to the pushcart and grab the TOP cardboard
box (cart_box2, 1.5 kg) off the stack -- no box teleporting, no helper spring.

Sequence:
  1. settle the scene, park robot_2 far away (it has no drive and would topple
     onto the cart otherwise)
  2. joint direction calibration (arms swing in free space; cart is 2 m away)
  3. raise both arms into the proven two-palm pose, fingers pre-curled half-shut
  4. servo shoulder pitch until the palm midpoint matches the box height
  5. translate the (anchored) robot root so the palm midpoint straddles the box
  6. squeeze with the baseline-delta torque stop (8 N*m for the 1.5 kg box)
  7. lift and check that the box rides up off the stack

Run GUI:   env LD_LIBRARY_PATH= XR_RUNTIME_JSON=/nonexistent DISPLAY=:0 \\
    ./isaaclab.sh -p scripts/diagnostics/grasp_cart_box.py --device cpu
Headless:  add --headless
"""

import argparse
import os

os.environ["ISAACLAB_SCENE_SYNC_ROLE"] = "none"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 grabs the top pushcart box")
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
# waist_pitch is deliberately NOT state-locked: it is driven as a position
# target so the robot can lean forward to reach the low box (hanging-arm palm
# floor is ~0.78 m; the box top sits at ~0.56 m -- the missing 0.2 m comes from
# the waist lean, exactly like a human picking from a cart).
LOWER_BODY_PATTERNS = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
]

TARGET_BOX = "cart_box2"      # top box on the cardboard pushcart stack
TAU_GRIP_DELTA_NM = 10.0      # 1.5 kg box at low-arm leverage: ~18 N/side
GAP_FLOOR_M = 0.385           # box y-width 0.25 (measured collider) + pad
                              # offset 0.155 - 2 cm press; torque stop primary
GRIP_Z_OFFSET = -0.03         # palm-mid slightly BELOW box center: the leaned
                              # pose tilts the palm faces down, so their lower
                              # edges make first contact -- aiming low puts that
                              # contact on the box's mid side (horizontal pinch)
                              # instead of its top edge (which flipped the box)
HAND_DELTA = 0.02
LIFT_STEPS = 400

report: dict = {"target": TARGET_BOX, "phases": {}, "verdict": None}


def _fmt(x: float) -> float:
    return round(float(x), 4)


env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
for term_name in ("mujoco_g1_mirror_2", "scene_state_sync", "env_reset_sync", "box_success_reset"):
    if hasattr(env_cfg.actions, term_name):
        setattr(env_cfg.actions, term_name, None)
if hasattr(env_cfg.terminations, "box_dropped"):
    env_cfg.terminations.box_dropped = None
if hasattr(env_cfg.terminations, "time_out"):
    env_cfg.terminations.time_out = None
if hasattr(env_cfg.terminations, "success"):
    env_cfg.terminations.success = None

env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.reset()

sim = env.sim
scene = env.scene
robot = scene["robot_1"]
box = scene[TARGET_BOX]
dt = env.physics_dt
device = env.device
box_mass = float(box.root_physx_view.get_masses().sum())
print("=" * 78)
print(f"GRAB THE TOP PUSHCART BOX: {TARGET_BOX}  mass={box_mass:.2f} kg  physics_dt={dt}")
print("=" * 78)

lb_ids, _ = robot.find_joints(LOWER_BODY_PATTERNS)
arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)
palm_ids, palm_names = robot.find_bodies([".*_hand_palm_link"])
l_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("left")][0])]
r_palm = palm_ids[palm_names.index([n for n in palm_names if n.startswith("right")][0])]

lb_q0 = robot.data.default_joint_pos[:, lb_ids].clone()
arm_q0 = robot.data.default_joint_pos[:, arm_ids].clone()

# Root anchored at the SPAWN pose (standing on the ground, facing the cart 2 m
# ahead); phase 5 translates this anchor to the grasp station.
root0 = robot.data.default_root_state[:, :7].clone()
root0[:, :3] += scene.env_origins
zero6 = torch.zeros((1, 6), device=device)
lb_zero = torch.zeros_like(lb_q0)
arm_targets = arm_q0.clone()

hand_ids, hand_names = robot.find_joints([".*_hand_.*_joint"])
hand_limits = robot.data.joint_pos_limits[:, hand_ids, :]
hand_q0 = robot.data.default_joint_pos[:, hand_ids].clone()
finger_goal = [None]

wp_ids, _ = robot.find_joints(["waist_pitch_joint"])
waist_q0 = float(robot.data.default_joint_pos[0, wp_ids[0]])
waist_target = [waist_q0]


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
        robot.set_joint_position_target(
            torch.tensor([[waist_target[0]]], device=device), joint_ids=wp_ids)
        robot.set_joint_velocity_target(
            torch.zeros((1, 1), device=device), joint_ids=wp_ids)
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


k_arm = float(robot.actuators["arms"].stiffness.mean())
d_arm = float(robot.actuators["arms"].damping.mean())
tau_arm = float(getattr(robot.actuators["arms"], "effort_limit_sim", robot.actuators["arms"].effort_limit).mean())


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


# ---------------------------------------------------------------------------
# Phase 1: settle; park robot_2 (undriven ragdoll) far from the cart
# ---------------------------------------------------------------------------
print("\nPhase 1: settle scene, park robot_2 clear of the cart")
if RENDER:
    # frame the cardboard pushcart grasp station
    sim.set_camera_view([-5.4, 17.3, 1.9], [-6.5, 19.4, 0.7])
try:
    r2 = scene["robot_2"]
    r2_pose = r2.data.default_root_state[:, :7].clone()
    r2_pose[:, 0] = 5.0
    r2_pose[:, 1] = 25.0
    r2_pose[:, :3] += scene.env_origins
    r2.write_root_pose_to_sim(r2_pose)
    r2.write_root_velocity_to_sim(zero6)
except KeyError:
    pass
step_hold(300)
box_pos0 = box.data.root_pos_w[0].clone()
print(f"  {TARGET_BOX} settled at ({box_pos0[0]:.2f}, {box_pos0[1]:.2f}, {box_pos0[2]:.3f})")
report["phases"]["box_settled_z"] = _fmt(box_pos0[2])

# ---------------------------------------------------------------------------
# Phase 2: direction calibration (free space -- cart is 2 m ahead)
# ---------------------------------------------------------------------------
print("Phase 2: joint direction calibration")


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

# waist lean direction: which waist_pitch direction lowers the palms
waist_target[0] = waist_q0 + 0.25
step_hold(90)
d_z_w = float(palm_mid()[2])
waist_target[0] = waist_q0
step_hold(90)
lean_down = 1.0 if d_z_w < float(palm_mid()[2]) else -1.0

# wrist pitch direction: which way tilts the fingertips UP (palm-face up-pitch).
# Needed to counter-rotate the palms after the forward swing + waist lean,
# which otherwise point the palm faces diagonally DOWN onto the box top edge.
l_tip_ids, _ = robot.find_bodies(["left_hand_index_1_link"])
r_tip_ids, _ = robot.find_bodies(["right_hand_index_1_link"])
wrist_up_l = wrist_up_r = None
try:
    d_tip = probe("left_wrist_pitch_joint", 0.4, lambda: float(robot.data.body_pos_w[0, l_tip_ids[0], 2]))
    wrist_up_l = 1.0 if d_tip > 0 else -1.0
    d_tip = probe("right_wrist_pitch_joint", 0.4, lambda: float(robot.data.body_pos_w[0, r_tip_ids[0], 2]))
    wrist_up_r = 1.0 if d_tip > 0 else -1.0
except ValueError:
    print("  [WARN] wrist_pitch_joint not found; palm counter-rotation disabled")
print(f"  pitch lift sign {'+' if lift_sign > 0 else '-'} | roll inward: L{'+' if inward_l > 0 else '-'} R{'+' if inward_r > 0 else '-'} | waist lean-down {'+' if lean_down > 0 else '-'} | wrist-up L{wrist_up_l} R{wrist_up_r}")

# ---------------------------------------------------------------------------
# Phase 2b: measure the cart at runtime (its USD references a cloud asset, so
# offline probing was blind), pick the approach side OPPOSITE the handle, and
# jump to a pre-station 0.95 m out, facing the box.
# ---------------------------------------------------------------------------
print("Phase 2b: cart geometry probe + approach side")
import math as _math  # noqa: E402
import omni.usd as _omni_usd  # noqa: E402
from pxr import Usd as _Usd, UsdGeom as _UsdGeom  # noqa: E402

_stage = _omni_usd.get_context().get_stage()
_cache = _UsdGeom.BBoxCache(_Usd.TimeCode.Default(), ["default", "render"])
_cart = _stage.GetPrimAtPath("/World/envs/env_0/Pushcart")
handle_dx = None
if _cart and _cart.IsValid():
    _b = _cache.ComputeWorldBound(_cart).ComputeAlignedRange()
    if not _b.IsEmpty():
        _mn, _mx = _b.GetMin(), _b.GetMax()
        print(f"  cart bbox x[{_mn[0]:.2f},{_mx[0]:.2f}] y[{_mn[1]:.2f},{_mx[1]:.2f}] z[{_mn[2]:.2f},{_mx[2]:.2f}]")
        _hx = []
        for _p in _Usd.PrimRange(_cart):
            if not _p.IsA(_UsdGeom.Boundable):
                continue
            _bb = _cache.ComputeWorldBound(_p).ComputeAlignedRange()
            if _bb.IsEmpty() or _bb.GetMax()[2] < 0.55:
                continue
            _cx = (_bb.GetMin()[0] + _bb.GetMax()[0]) / 2
            _hx.append(_cx)
            print(f"    tall part {str(_p.GetPath())[-45:]}: x_center {_cx:+.2f}, z_max {_bb.GetMax()[2]:.2f}")
        if _hx:
            handle_dx = sum(_hx) / len(_hx) - float(box.data.root_pos_w[0, 0])
box_now = box.data.root_pos_w[0].clone()
approach_side = 1.0  # default: stand on +X (the spawn side)
if handle_dx is not None and handle_dx > 0.05:
    approach_side = -1.0  # handle sits on +X -> come in from -X instead
print(f"  handle dx: {f'{handle_dx:+.2f}' if handle_dx is not None else 'n/a'} -> stand on {'+X' if approach_side > 0 else '-X'} side")
yaw = _math.pi if approach_side > 0 else 0.0
root0[:, 0] = box_now[0] + approach_side * 0.95
root0[:, 1] = box_now[1]
root0[:, 3] = _math.cos(yaw / 2)
root0[:, 4] = 0.0
root0[:, 5] = 0.0
root0[:, 6] = _math.sin(yaw / 2)
step_hold(150)
report["phases"]["approach_side"] = "+X" if approach_side > 0 else "-X"

# ---------------------------------------------------------------------------
# Phase 3: arms up, palms wide, fingers half-curled
# ---------------------------------------------------------------------------
# The box top sits at ~0.55 m -- exactly the natural hanging-arm palm height.
# Do NOT raise the arms first (the raised-arm pose only reaches down to ~1.2 m
# and the pitch->height map turns non-monotonic outside that region, which sent
# the previous servo into the joint limits). Start from hanging arms with a
# slight forward swing, then micro-servo pitch within a bounded window.
print("Phase 3: low two-palm pose (arms swung well forward)")
# Swing 0.5 rad: reaches the palms ~0.17 m further forward so the torso/legs
# stand clear of the cart edge (user-observed failure: the legs rammed the cart
# during creep-in and catapulted the box). The extra palm height this costs is
# repaid by a deeper crouch.
PITCH_SWING = 0.50
for side in ("left", "right"):
    arm_targets[0, jidx(f"{side}_shoulder_pitch_joint")] = arm_q0[0, jidx(f"{side}_shoulder_pitch_joint")] + lift_sign * PITCH_SWING
    # keep the elbow nearly straight: a bent elbow raises the palm floor above
    # the 0.55 m box height
    arm_targets[0, jidx(f"{side}_elbow_joint")] = 0.15
step_hold(180)


def open_palms(lo: float = 0.50, hi: float = 0.62):
    open_extra, tries = 0.0, 0
    while not (lo < palm_gap() < hi) and tries < 20:
        open_extra += 0.04 if palm_gap() <= lo else -0.04
        arm_targets[0, jidx("left_shoulder_roll_joint")] = arm_q0[0, jidx("left_shoulder_roll_joint")] - inward_l * open_extra
        arm_targets[0, jidx("right_shoulder_roll_joint")] = arm_q0[0, jidx("right_shoulder_roll_joint")] - inward_r * open_extra
        step_hold(60)
        tries += 1


open_palms()
print(f"  open palm gap {palm_gap():.3f} m, palm mid z {float(palm_mid()[2]):.3f} m")
# Approach hand shape: half-curl the FOUR fingers only. Half-curled thumbs jut
# into the palm-gap midline (tip spacing ~0.11 m vs the 0.25 m box) and punched
# the box face over during the creep-in -- they stay open until the squeeze.
half_close = hand_q0 + 0.5 * (_finger_close_goal() - hand_q0)
thumb_cols = torch.tensor(
    [1.0 if "thumb" in n else 0.0 for n in hand_names], device=device).view(1, -1)
finger_goal[0] = half_close * (1 - thumb_cols) + hand_q0 * thumb_cols
step_hold(300)

# ---------------------------------------------------------------------------
# Phase 4: bounded micro-servo of palm height (pitch window +-0.9 rad around
# default, small steps, stop when the error stops improving)
# ---------------------------------------------------------------------------
target_z = float(box.data.root_pos_w[0, 2]) + GRIP_Z_OFFSET
print(f"Phase 4: waist-lean + crouch servo, palm height target {target_z:.3f} m")
# stage A: lean the waist forward until the palms reach box height (the
# hanging-arm palm floor is ~0.2 m above the box; the lean supplies part)
for it in range(24):
    err = float(palm_mid()[2]) - target_z
    if err < 0.02:
        break
    nxt = waist_target[0] + lean_down * min(0.05, err * 0.5)
    if abs(nxt - waist_q0) > 0.55:
        break
    waist_target[0] = nxt
    step_hold(60)
print(f"  after lean: palm mid z {float(palm_mid()[2]):.3f} m, waist {waist_target[0] - waist_q0:+.2f} rad")
# stage A2: crouch -- sink the anchored root to cover the remaining reach gap
# (the legs are state-locked so a real knee bend is unavailable; the root sink
# stands in for it, capped at 0.25 m)
crouch = 0.0
for it in range(20):
    err = float(palm_mid()[2]) - target_z
    if err < 0.02 or crouch >= 0.45:
        break
    dz = min(0.03, err, 0.45 - crouch)
    root0[:, 2] -= dz
    crouch += dz
    step_hold(40)
print(f"  after crouch: palm mid z {float(palm_mid()[2]):.3f} m, root sink {crouch:.2f} m")
report["phases"]["crouch_m"] = _fmt(crouch)
# stage B: small bounded pitch trim -- tight window around the forward swing so
# the trim cannot un-reach the arms (leg clearance depends on that reach)
pitch_lo = float(arm_q0[0, jidx("left_shoulder_pitch_joint")]) + lift_sign * PITCH_SWING - 0.15
pitch_hi = float(arm_q0[0, jidx("left_shoulder_pitch_joint")]) + lift_sign * PITCH_SWING + 0.15
pitch_lo, pitch_hi = min(pitch_lo, pitch_hi), max(pitch_lo, pitch_hi)
best_err = abs(target_z - float(palm_mid()[2]))
stall = 0
for it in range(15):
    err = target_z - float(palm_mid()[2])
    if abs(err) < 0.02:
        break
    step = max(-0.05, min(0.05, err * 0.8))
    for side in ("left", "right"):
        j = jidx(f"{side}_shoulder_pitch_joint")
        arm_targets[0, j] = max(pitch_lo, min(pitch_hi, float(arm_targets[0, j]) + lift_sign * step))
    step_hold(50)
    new_err = abs(target_z - float(palm_mid()[2]))
    if new_err >= best_err - 0.005:
        stall += 1
        if stall >= 3:
            break
    else:
        best_err = new_err
        stall = 0
print(f"  palm mid z {float(palm_mid()[2]):.3f} m after trim")
# NOTE: wrist counter-rotation was tried and reverted -- it points the
# half-curled fingers forward and they speared the box during the creep-in.
# The down-tilted palms are handled by aiming the grip BELOW box center instead
# (GRIP_Z_OFFSET), keeping the clean fingers-down approach shape of run 8.
open_palms()
print(f"  re-opened palm gap {palm_gap():.3f} m")
report["phases"]["palm_z_after_servo"] = _fmt(float(palm_mid()[2]))
report["phases"]["waist_lean_rad"] = _fmt(waist_target[0] - waist_q0)

# ---------------------------------------------------------------------------
# Phase 5: translate the root anchor so the palms straddle the box
# ---------------------------------------------------------------------------
print("Phase 5: creep in from the pre-station")
# creep in 5 cm steps so the open palms slide in around the
# box instead of materializing inside it. Aim at a SNAPSHOT of the box position
# (v3 chased the live position 1.4 m across the floor after knocking it off)
# and abort early if the box gets knocked off its stack.
goal_xy = box.data.root_pos_w[0][:2].clone()
box_z_ref = float(box.data.root_pos_w[0, 2])
quat_ref = box.data.root_quat_w[0].clone()
knocked = False
for k in range(16):
    qdot = abs(float((box.data.root_quat_w[0] * quat_ref).sum()))
    tilt_deg = 2.0 * _math.degrees(_math.acos(min(1.0, qdot)))
    if float(box.data.root_pos_w[0, 2]) < box_z_ref - 0.10 or tilt_deg > 25.0:
        knocked = True
        print(f"  ABORT at creep step {k}: box disturbed (dz {float(box.data.root_pos_w[0, 2]) - box_z_ref:+.3f} m, tilt {tilt_deg:.0f} deg)")
        break
    remain_vec = goal_xy - palm_mid()[:2]
    remain = float(remain_vec.norm())
    if remain < 0.03:
        break
    stepd = min(0.05, remain)
    d = remain_vec / max(remain, 1e-6)
    root0[:, 0] += float(d[0] * stepd)
    root0[:, 1] += float(d[1] * stepd)
    step_hold(50)
step_hold(100)
miss = float((box.data.root_pos_w[0][:2] - palm_mid()[:2]).norm())
print(f"  station reached: palm-mid xy err {miss:.3f} m, gap {palm_gap():.3f} m, box z {float(box.data.root_pos_w[0, 2]):.3f}")
report["phases"]["station_xy_err"] = _fmt(miss)

# ---------------------------------------------------------------------------
# Phase 6: torque-delta squeeze (no helper spring -- the box sits on its stack)
# ---------------------------------------------------------------------------
print("Phase 6: squeeze (contact-force + tilt closed loop)")
# Closed-loop clamp using the hand ContactSensor and the box attitude:
#   - growing tilt = the contact line sits above the box's belly and the
#     down-slanted palms are prying it over -> crouch 1 cm more to bring the
#     contact lower on the box face, then resume squeezing
#   - both palms carrying force (opposed contact, forces cancel) = grip formed
step_hold(150)
try:
    _hc = env.scene.sensors["hand_contact"]
    _hc_names = _hc.body_names
    _hc_l = [i for i, n in enumerate(_hc_names) if n.startswith("left")]
    _hc_r = [i for i, n in enumerate(_hc_names) if n.startswith("right")]
    print(f"  contact sensor online: {len(_hc_names)} hand bodies")
except Exception as exc:
    _hc = None
    print(f"  [WARN] no hand contact sensor ({exc}); falling back to box-response only")
box_y0 = float(box.data.root_pos_w[0, 1])
box_quat0 = box.data.root_quat_w[0].clone()
tries = 0
grip_confirm = 0
crouch_extra = 0.0
tilt = 0.0
f_l = f_r = 0.0
while tries < 400:
    qd_dot = abs(float((box.data.root_quat_w[0] * box_quat0).sum()))
    tilt = 2.0 * _math.degrees(_math.acos(min(1.0, qd_dot)))
    if _hc is not None:
        fm = _hc.data.net_forces_w.norm(dim=-1)
        f_l = float(fm[:, _hc_l].amax()) if _hc_l else 0.0
        f_r = float(fm[:, _hc_r].amax()) if _hc_r else 0.0
    if tilt > 25.0:
        print(f"  ABORT: box tipped during squeeze (tilt {tilt:.0f} deg)")
        break
    if tilt > 2.5 and crouch_extra < 0.06:
        # prying detected: lower the contact line on the box, don't squeeze
        root0[:, 2] -= 0.01
        crouch_extra += 0.01
        step_hold(60)
        print(f"  tilt {tilt:.1f} deg -> crouch +{crouch_extra * 100:.0f} cm (contact line too high)")
        continue
    if min(f_l, f_r) > 2.0 and tilt < 3.0:
        grip_confirm += 1
        if grip_confirm >= 5:
            print(f"  OPPOSED GRIP formed: F L{f_l:.1f} R{f_r:.1f} N, tilt {tilt:.1f} deg")
            break
    else:
        grip_confirm = 0
    if palm_gap() <= GAP_FLOOR_M:
        print("  gap floor reached")
        break
    # asymmetric advance (same physics as the finger compliance): a palm that
    # already carries force STOPS advancing -- it waits for the other side, so
    # the first touch cannot pry the box and forces stay in the gentle range
    # instead of ramping to 345 N as the stiff arms wound up in run 1
    step_l = 0.003 if f_l < 3.0 else 0.0
    step_r = 0.003 if f_r < 3.0 else 0.0
    if step_l == 0.0 and step_r == 0.0:
        step_l = step_r = 0.0008  # both touching, creep to grip force together
    arm_targets[0, jidx("left_shoulder_roll_joint")] += inward_l * step_l
    arm_targets[0, jidx("right_shoulder_roll_joint")] += inward_r * step_r
    step_hold(15)
    tries += 1
    if tries % 25 == 0:
        print(f"    [{tries:3d}] gap {palm_gap():.3f} | F L{f_l:.1f} R{f_r:.1f} N | tilt {tilt:.1f} deg")
gap_stop = palm_gap()
print(f"  squeeze stop: gap {gap_stop:.3f} m | F L{f_l:.1f} R{f_r:.1f} N | tilt {tilt:.1f} deg | extra crouch {crouch_extra * 100:.0f} cm ({tries} increments)")
report["phases"]["squeeze"] = {"gap_stop_m": _fmt(gap_stop), "increments": tries,
                              "force_LR_N": [_fmt(f_l), _fmt(f_r)],
                              "box_tilt_deg": _fmt(tilt), "extra_crouch_m": _fmt(crouch_extra)}
# grip established -- now curl the thumbs too, wrapping the hold
finger_goal[0] = half_close
step_hold(150)
print("  thumbs curled in to wrap the grip")

# ---------------------------------------------------------------------------
# Phase 7: lift the box off the stack
# ---------------------------------------------------------------------------
print("Phase 7: lift")
# physical constant-force grip: clamp the shoulder-roll effort limit to
# 12 N*m for the lift. The pitch arc geometrically compresses the palm gap
# and a position-servo cannot react fast enough (runs 4/5: force spiked past
# 25 N within ~10 steps and shot the box 1.3 m upward). With the torque cap
# the roll joints yield physically -- grip force is pinned at ~30 N with zero
# latency (posture torque budget here is only ~2 N*m, so 12 leaves ~25 N grip).
try:
    _rl_ids = [robot.find_joints(["left_shoulder_roll_joint"])[0][0],
               robot.find_joints(["right_shoulder_roll_joint"])[0][0]]
    robot.write_joint_effort_limit_to_sim(
        torch.tensor([[12.0, 12.0]], device=device), joint_ids=_rl_ids)
    print("  shoulder-roll effort limit clamped to 12 N*m for the lift")
except Exception as exc:
    print(f"  [WARN] effort-limit clamp unavailable ({exc}); relying on force servo only")
zb0 = float(box.data.root_pos_w[0, 2])
zp0 = float(palm_mid()[2])
# LIFT WITH THE LEGS AND BACK, NOT THE ARMS: raising via shoulder pitch sweeps
# the palms along an arc that geometrically compresses the gap and shoots the
# box out (runs 2-6: no arm-side servo or torque cap could react fast enough).
# Instead the arm targets stay FROZEN -- palm gap and grip force are untouched
# -- and the whole torso rises: root un-crouches and the waist straightens.
z_lift_start = float(root0[0, 2])
waist_lift_start = waist_target[0]
waist_return = 0.35 * (waist_lift_start - waist_q0)  # straighten 35% of the lean


def _lift(i):
    a = (i + 1) / LIFT_STEPS
    if a <= 0.5:
        # stage 1: PURE vertical rise, waist & arms frozen -- the box is still
        # jammed against the stack below; straightening the waist here pulls
        # the palms backward off the box (run 7: grip lost in the first 50
        # steps). A pure translation keeps palms and box relatively static.
        root0[:, 2] = z_lift_start + 0.24 * a
    else:
        # stage 2: box is clear of the stack -- keep rising and straighten up
        root0[:, 2] = z_lift_start + 0.12 + 0.36 * (a - 0.5)
        waist_target[0] = waist_lift_start - waist_return * (a - 0.5) * 2.0
    if _hc is not None and (i + 1) % 50 == 0:
        fm2 = _hc.data.net_forces_w.norm(dim=-1)
        fl2 = float(fm2[:, _hc_l].amax()) if _hc_l else 0.0
        fr2 = float(fm2[:, _hc_r].amax()) if _hc_r else 0.0
        print(f"    lift[{i+1:3d}] box_z {float(box.data.root_pos_w[0, 2]):.3f} | gap {palm_gap():.3f} | F L{fl2:.1f} R{fr2:.1f} N")


step_hold(LIFT_STEPS, _lift)
step_hold(200)
rise_p = float(palm_mid()[2]) - zp0
rise_b = float(box.data.root_pos_w[0, 2]) - zb0
slip = rise_p - rise_b
off = float((box.data.root_pos_w[0] - palm_mid()).norm())
lifted = rise_b > 0.08 and off < 0.30 and abs(slip) < 0.10
print(f"  lift: hands +{rise_p:.3f} m, box {rise_b:+.3f} m, slip {slip:+.3f} m, off-palm {off:.3f} m")
print(f"\nVERDICT: {'BOX GRABBED OFF THE CART' if lifted else 'FAILED'}  ({TARGET_BOX}, {box_mass:.2f} kg)")
report["phases"]["lift"] = {"hands_rise_m": _fmt(rise_p), "box_rise_m": _fmt(rise_b),
                            "slip_m": _fmt(slip), "off_palm_m": _fmt(off)}
report["verdict"] = "LIFTED" if lifted else "FAILED"

if args_cli.report:
    with open(args_cli.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON report written to: {args_cli.report}")

if RENDER:
    print("\nGUI mode: holding the final pose ~60 s for inspection...")
    step_hold(12000)

env.close()
simulation_app.close()
