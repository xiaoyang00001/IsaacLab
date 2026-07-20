# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""诊断脚本：G1 双臂「抱箱子」抱不起来的根因排查。

针对 Isaac-PickPlace-Locomanipulation-G1-Abs-v0 环境，量化验证三个候选根因：
  1. 前臂/上臂碰撞体缺失 —— 箱子直接穿透手臂（静态审计 + 前臂承托实验）
  2. 摩擦不足 —— 手掌夹住后抬升时打滑（夹持抬升实验）
  3. 手臂驱动无力 —— PD 力矩饱和（力矩余量估算）

实验全程绕过 action manager（不依赖 UDP/deploy 数据源），直接用底层 API
复现 mirror 模式的根/下肢硬写 + 手臂 PD 目标驱动，与实跑链路的物理行为一致。

用法（无头）：
    ./isaaclab.sh -p scripts/diagnostics/check_hug_box.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 双臂抱箱失败根因诊断")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--report", type=str, default=None, help="JSON 报告输出路径（默认打印到控制台）")
parser.add_argument("--skip_dynamic", action="store_true", help="只做静态审计，跳过动态实验")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# pick_place 任务在 isaaclab_tasks 的 _BLACKLIST_PKGS 里，须显式导入；
# pinocchio 要在 AppLauncher 之前 import，强制用 IsaacLab 安装的版本（与 teleop_se3_agent.py 一致）。
import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 以下 import 必须在 AppLauncher 之后
import gymnasium as gym  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import torch  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdPhysics, UsdShade  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401, E402  (黑名单包，显式注册)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
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
ARM_LINK_KEYWORDS = ("shoulder", "elbow", "wrist", "hand", "palm")

report: dict = {"static": {}, "dynamic": {}, "verdict": []}


def _fmt(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# 环境构建
# ---------------------------------------------------------------------------
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
# 诊断全程绕过 action manager 直接驱动底层 API,不需要 UDP 数据源;
# 禁用 mirror 接收,避免与同机正在运行的 teleop 实例抢 5557/5558 的包。
env_cfg.actions.mujoco_g1_mirror_1.enabled = False
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
env.reset()

sim = env.sim
scene = env.scene
robot = scene["robot_1"]
box = scene["long_box"]
dt = env.physics_dt
device = env.device

# ---------------------------------------------------------------------------
# 阶段 A：静态审计
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("阶段 A：静态审计（实际加载到 PhysX 的资产状态）")
print("=" * 78)

stage = omni.usd.get_context().get_stage()
robot_root = robot.cfg.prim_path.replace("env_.*", "env_0")

# A1. 每个 link 的碰撞体清单
link_collisions: dict[str, list[str]] = {}
body_name_set = set(robot.body_names)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not path.startswith(robot_root + "/"):
        continue
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        # 归属 link = 路径里第一个匹配 body 名的段
        owner = next((seg for seg in path[len(robot_root) + 1:].split("/") if seg in body_name_set), "?")
        link_collisions.setdefault(owner, []).append(path.rsplit("/", 1)[-1])

arm_links = [n for n in robot.body_names if any(k in n for k in ARM_LINK_KEYWORDS)]
arm_with_col = sorted(n for n in arm_links if n in link_collisions)
arm_without_col = sorted(n for n in arm_links if n not in link_collisions)

print(f"\n[A1] 双臂 link 碰撞体分布（共 {len(arm_links)} 个臂/手 link）")
print(f"  有碰撞体 ({len(arm_with_col)}): {arm_with_col}")
print(f"  ❌ 无碰撞体 ({len(arm_without_col)}): {arm_without_col}")
report["static"]["arm_links_with_collision"] = arm_with_col
report["static"]["arm_links_without_collision"] = arm_without_col

# A2. 碰撞体的物理材质（摩擦）
def _collision_friction(prim_path_prefix: str) -> list[tuple[str, float | None, float | None]]:
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prim_path_prefix) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        static_f = dynamic_f = None
        try:
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            if mat and mat.GetPrim().IsValid():
                mat_api = UsdPhysics.MaterialAPI(mat.GetPrim())
                static_f = mat_api.GetStaticFrictionAttr().Get()
                dynamic_f = mat_api.GetDynamicFrictionAttr().Get()
        except Exception:
            pass
        out.append((path, static_f, dynamic_f))
    return out

hand_cols = [
    (p, s, d) for (p, s, d) in _collision_friction(robot_root)
    if any(k in p for k in ("palm", "hand", "wrist"))
]
unbound = [p for (p, s, d) in hand_cols if s is None]
bound = [(p, s, d) for (p, s, d) in hand_cols if s is not None]
default_mat = env_cfg.sim.physics_material
print(f"\n[A2] 手部碰撞体物理材质（共 {len(hand_cols)} 个碰撞 prim）")
print(f"  显式绑定材质: {len(bound)} 个")
for p, s, d in bound[:4]:
    print(f"    {p.rsplit('/', 2)[-2]}: static={s} dynamic={d}")
print(f"  未绑定材质(用 sim 默认): {len(unbound)} 个 → 默认 static={default_mat.static_friction} dynamic={default_mat.dynamic_friction}")
box_mat = box.cfg.spawn.physics_material
print(f"  箱子材质: static={box_mat.static_friction} dynamic={box_mat.dynamic_friction}")
report["static"]["hand_collision_prims"] = len(hand_cols)
report["static"]["hand_prims_unbound_material"] = len(unbound)
report["static"]["sim_default_friction"] = [default_mat.static_friction, default_mat.dynamic_friction]
report["static"]["box_friction"] = [box_mat.static_friction, box_mat.dynamic_friction]

# A3. 手臂执行器
arms_act = robot.actuators["arms"]
k_arm = float(arms_act.stiffness.mean())
d_arm = float(arms_act.damping.mean())
tau_lim = float(arms_act.effort_limit_sim.mean()) if hasattr(arms_act, "effort_limit_sim") else float(arms_act.effort_limit.mean())
print(f"\n[A3] 手臂执行器: stiffness={k_arm:.0f} damping={d_arm:.0f} effort_limit={tau_lim:.0f} N·m")
report["static"]["arm_actuator"] = {"stiffness": k_arm, "damping": d_arm, "effort_limit": tau_lim}

# A4. 镜像动作项的硬写白名单（确认手臂是 PD 驱动而非硬写）
try:
    term = env.action_manager.get_term("mujoco_g1_mirror_1")
    tcfg = term.cfg
    print(f"\n[A4] 镜像动作项: write_body_joint_state={tcfg.write_body_joint_state}")
    print(f"  硬写白名单(下肢/腰): {tcfg.body_state_write_joint_names}")
    print(f"  手臂=target-only PD 驱动, 目标限速 max_delta={tcfg.body_joint_target_max_delta} rad/步")
    report["static"]["mirror_term"] = {
        "write_body_joint_state": bool(tcfg.write_body_joint_state),
        "body_state_write_joint_names": list(tcfg.body_state_write_joint_names),
        "body_joint_target_max_delta": float(tcfg.body_joint_target_max_delta),
    }
except Exception as exc:  # noqa: BLE001
    print(f"\n[A4] 镜像动作项读取失败: {exc}")

# A5. 箱子
box_mass = float(box.root_physx_view.get_masses().sum())
print(f"\n[A5] 长箱: 尺寸 0.20×0.05×0.10 m, 质量 {box_mass:.3f} kg, 抬起需承载 {box_mass * 9.81:.2f} N")
report["static"]["box_mass_kg"] = _fmt(box_mass)

if args_cli.skip_dynamic:
    print(json.dumps(report, ensure_ascii=False, indent=2))
    env.close()
    simulation_app.close()
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 底层控制工具（绕过 action manager，复现 mirror 模式物理行为）
# ---------------------------------------------------------------------------
lb_ids, _ = robot.find_joints(LOWER_BODY_PATTERNS)
arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)
palm_ids, palm_names = robot.find_bodies([".*_hand_palm_link"])
elbow_ids, _ = robot.find_bodies([".*_elbow_link"])
wrist_ids, _ = robot.find_bodies([".*_wrist_roll_link"])
l_palm, r_palm = (palm_ids[i] for i in (palm_names.index([n for n in palm_names if n.startswith("left")][0]),
                                        palm_names.index([n for n in palm_names if n.startswith("right")][0])))

lb_q0 = robot.data.default_joint_pos[:, lb_ids].clone()
arm_q0 = robot.data.default_joint_pos[:, arm_ids].clone()

# 根姿态：转 180° 背对桌子，腾出前方空间做实验
root0 = robot.data.default_root_state[:, :7].clone()
root0[:, :3] += scene.env_origins
q_z180 = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
root0[:, 3:7] = math_utils.quat_mul(q_z180, root0[:, 3:7])
zero6 = torch.zeros((1, 6), device=device)
lb_zero = torch.zeros_like(lb_q0)

arm_targets = arm_q0.clone()  # 当前手臂 PD 目标（全局状态，逐步更新）


def step_hold(n: int, on_step=None):
    """跑 n 个物理步：根+下肢硬写（mirror 模式），手臂 PD 跟目标。"""
    for i in range(n):
        robot.write_root_pose_to_sim(root0)
        robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(lb_q0, lb_zero, joint_ids=lb_ids)
        robot.set_joint_position_target(arm_targets, joint_ids=arm_ids)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        if on_step is not None:
            on_step(i)


def jidx(name: str) -> int:
    """手臂目标向量里某关节的下标。"""
    return arm_names.index(name)


def palm_pos():
    return robot.data.body_pos_w[0, l_palm].clone(), robot.data.body_pos_w[0, r_palm].clone()


def palm_gap() -> float:
    lp, rp = palm_pos()
    return float((lp - rp).norm())


def set_box(pos: torch.Tensor, quat: torch.Tensor):
    pose = torch.cat([pos.view(1, 3), quat.view(1, 4)], dim=-1)
    box.write_root_pose_to_sim(pose)
    box.write_root_velocity_to_sim(zero6)


def est_torque(joint_name: str) -> float:
    """隐式执行器 PD 力矩估算：k*(target-q) - d*qd，截断到 effort limit。"""
    j = robot.find_joints([joint_name])[0][0]
    q = float(robot.data.joint_pos[0, j])
    qd = float(robot.data.joint_vel[0, j])
    tgt = float(arm_targets[0, jidx(joint_name)])
    return max(-tau_lim, min(tau_lim, k_arm * (tgt - q) - d_arm * qd))


def point_seg_dist(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
    ab = b - a
    t = torch.clamp(torch.dot(p - a, ab) / torch.dot(ab, ab).clamp(min=1e-9), 0.0, 1.0)
    return float((p - (a + t * ab)).norm())


print("\n" + "=" * 78)
print("阶段 B：动态实验（根/下肢硬写复现 mirror 模式，手臂 PD 驱动）")
print("=" * 78)

# 沉降
step_hold(120)

# ---------------------------------------------------------------------------
# B0. 方向标定：shoulder_pitch 哪个方向抬手、shoulder_roll 哪个方向收拢
# ---------------------------------------------------------------------------
def probe(joint: str, delta: float, metric) -> float:
    """给单关节加 delta 目标，返回 metric 变化量，然后恢复默认姿态。"""
    global arm_targets
    before = metric()
    arm_targets[0, jidx(joint)] += delta
    step_hold(90)
    after = metric()
    arm_targets = arm_q0.clone()
    step_hold(90)
    return after - before

lp0, _ = palm_pos()
d_z = probe("left_shoulder_pitch_joint", 0.4, lambda: float(robot.data.body_pos_w[0, l_palm, 2]))
lift_sign = 1.0 if d_z > 0 else -1.0
d_gap_l = probe("left_shoulder_roll_joint", 0.25, palm_gap)
inward_l = 1.0 if d_gap_l < 0 else -1.0
d_gap_r = probe("right_shoulder_roll_joint", 0.25, palm_gap)
inward_r = 1.0 if d_gap_r < 0 else -1.0
print(f"\n[B0] 方向标定: pitch {'+' if lift_sign > 0 else '-'}方向抬手 | "
      f"roll 收拢方向 左{'+' if inward_l > 0 else '-'} 右{'+' if inward_r > 0 else '-'}")

# ---------------------------------------------------------------------------
# B1. 前臂承托实验：箱子放在双前臂上，看是否穿透坠落
# ---------------------------------------------------------------------------
print("\n[B1] 前臂穿透实验（箱子放单侧前臂正上方 5.5cm，长轴顺前臂方向）")
# 抬臂前伸，让前臂离开躯干
for side in ("left", "right"):
    arm_targets[0, jidx(f"{side}_shoulder_pitch_joint")] = arm_q0[0, jidx(f"{side}_shoulder_pitch_joint")] + lift_sign * 0.55
    arm_targets[0, jidx(f"{side}_elbow_joint")] = 1.2
step_hold(200)

a_f = robot.data.body_pos_w[0, elbow_ids[0]].clone()   # 左肘
b_f = robot.data.body_pos_w[0, wrist_ids[0]].clone()   # 左腕(wrist_roll)
fore_dir = (b_f - a_f) / (b_f - a_f).norm()
yaw = math.atan2(float(fore_dir[1]), float(fore_dir[0]))
box_quat = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=device)
drop_pos = (a_f + b_f) / 2 + torch.tensor([0.0, 0.0, 0.055], device=device)
set_box(drop_pos, box_quat)

traj_z, min_seg_dist = [], [1e9]

def _b1_track(i):
    traj_z.append(float(box.data.root_pos_w[0, 2]))
    p = box.data.root_pos_w[0]
    a1, b1 = robot.data.body_pos_w[0, elbow_ids[0]], robot.data.body_pos_w[0, wrist_ids[0]]
    min_seg_dist[0] = min(min_seg_dist[0], point_seg_dist(p, a1, b1))

step_hold(300, _b1_track)

z_drop = float(drop_pos[2]) - traj_z[-1]
rested = z_drop < 0.08
fell_through = z_drop > 0.30 and min_seg_dist[0] < 0.05
print(f"  放置高度 z={float(drop_pos[2]):.3f} → 最终 z={traj_z[-1]:.3f}（下坠 {z_drop:.3f} m）")
print(f"  坠落过程中箱心与前臂轴线最小距离: {min_seg_dist[0]:.3f} m（< 0.05 即已进入前臂体内）")
if rested:
    print("  结论: ✅ 前臂接住了箱子（前臂有碰撞体）")
elif fell_through:
    print("  结论: ❌ 箱子从前臂正上方穿透坠落 —— 前臂无碰撞体实锤")
else:
    print("  结论: ⚠️ 箱子滑落且未穿过前臂轴线（几何未对准，本次不判定）")
report["dynamic"]["cradle_test"] = {
    "drop_m": _fmt(z_drop),
    "min_dist_to_forearm_axis_m": _fmt(min_seg_dist[0]),
    "rested_on_forearm": bool(rested),
    "fell_through": bool(fell_through),
}

# 复位手臂
arm_targets = arm_q0.clone()
step_hold(150)

# ---------------------------------------------------------------------------
# B2. 手掌夹持抬升实验：手掌有碰撞体，验证纯掌夹能否抱起
# ---------------------------------------------------------------------------
print("\n[B2] 手掌夹持抬升实验（扶住箱子 → 收拢手掌夹紧 → 抬升）")
# 前伸并把掌间距调到 0.27~0.35 的窗口（比箱长 0.20 略宽）
for side, inw in (("left", inward_l), ("right", inward_r)):
    arm_targets[0, jidx(f"{side}_shoulder_pitch_joint")] = arm_q0[0, jidx(f"{side}_shoulder_pitch_joint")] + lift_sign * 0.45
    arm_targets[0, jidx(f"{side}_elbow_joint")] = 0.9
step_hold(180)
open_extra, tries = 0.0, 0
while not (0.27 < palm_gap() < 0.35) and tries < 12:
    open_extra += 0.05 if palm_gap() <= 0.27 else -0.05
    arm_targets[0, jidx("left_shoulder_roll_joint")] = arm_q0[0, jidx("left_shoulder_roll_joint")] - inward_l * open_extra
    arm_targets[0, jidx("right_shoulder_roll_joint")] = arm_q0[0, jidx("right_shoulder_roll_joint")] - inward_r * open_extra
    step_hold(60)
    tries += 1

gap_start = palm_gap()
print(f"  初始掌间距 {gap_start:.3f} m（箱长 0.200，长轴对齐掌轴）")

# 箱子用「重力补偿 + 弱弹簧」悬浮在掌中点：可以被手掌推动（不同于瞬移的无限刚度），
# 掌间距失速且夹持力矩上来 = 接触建立，撤掉弹簧把重力交给夹持力。
lp, rp = palm_pos()
ax = lp - rp
yaw = math.atan2(float(ax[1]), float(ax[0]))
box_quat_palm = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=device)
set_box((lp + rp) / 2, box_quat_palm)

roll_l0 = float(arm_targets[0, jidx("left_shoulder_roll_joint")])
roll_r0 = float(arm_targets[0, jidx("right_shoulder_roll_joint")])
SQUEEZE_STEPS = 500
gap_curve: list[float] = []
spring_on = [True]
contact_gap = [None]
zero_wrench = torch.zeros(1, 1, 3, device=device)
g_comp = torch.tensor([[[0.0, 0.0, box_mass * 9.81]]], device=device)

def _spring_force():
    lp2, rp2 = palm_pos()
    err = ((lp2 + rp2) / 2 - box.data.root_pos_w[0]).view(1, 1, 3)
    vel = box.data.root_lin_vel_w[0].view(1, 1, 3)
    return torch.clamp(60.0 * err - 3.0 * vel + g_comp, -12.0, 12.0)

def _squeeze(i):
    a = (i + 1) / SQUEEZE_STEPS
    arm_targets[0, jidx("left_shoulder_roll_joint")] = roll_l0 + inward_l * 0.55 * a
    arm_targets[0, jidx("right_shoulder_roll_joint")] = roll_r0 + inward_r * 0.55 * a
    g = palm_gap()
    if i % 25 == 0:
        gap_curve.append(round(g, 4))
    if spring_on[0]:
        box.set_external_force_and_torque(_spring_force(), zero_wrench)
        if (
            i > 120
            and len(gap_curve) >= 3
            and abs(gap_curve[-1] - gap_curve[-3]) < 0.004
            and abs(est_torque("left_shoulder_roll_joint")) > 20
        ):
            spring_on[0] = False
            contact_gap[0] = g
            box.set_external_force_and_torque(zero_wrench, zero_wrench)
            print(f"    接触建立于掌间距 {g:.3f} m（步 {i}），撤掉扶持弹簧，重力交给夹持")

step_hold(SQUEEZE_STEPS, _squeeze)
if spring_on[0]:
    spring_on[0] = False
    box.set_external_force_and_torque(zero_wrench, zero_wrench)
    print("    ⚠️ 收拢全程未检测到接触失速，撤弹簧观察")
step_hold(200)

gap_hold = palm_gap()
lp, rp = palm_pos()
mid = (lp + rp) / 2
box_offcenter = float((box.data.root_pos_w[0] - mid).norm())
tau_l = est_torque("left_shoulder_roll_joint")
tau_r = est_torque("right_shoulder_roll_joint")
jl = robot.find_joints(["left_shoulder_roll_joint"])[0][0]
jr = robot.find_joints(["right_shoulder_roll_joint"])[0][0]
print(f"  掌间距曲线(每0.125s): {gap_curve}")
print(f"  roll 实际/目标: 左 {float(robot.data.joint_pos[0, jl]):.3f}/{float(arm_targets[0, jidx('left_shoulder_roll_joint')]):.3f}"
      f"  右 {float(robot.data.joint_pos[0, jr]):.3f}/{float(arm_targets[0, jidx('right_shoulder_roll_joint')]):.3f}")
print(f"  收紧后掌间距 {gap_hold:.3f} m（箱长 0.200）| 箱心偏离掌中点 {box_offcenter:.3f} m")
print(f"  roll 力矩估算 左{tau_l:.1f} 右{tau_r:.1f} / 上限 {tau_lim:.0f} N·m")
grip_established = contact_gap[0] is not None and box_offcenter < 0.12 and gap_hold > 0.17
if not grip_established:
    print(f"  ⚠️ 夹持未建立（{'未检测到接触' if contact_gap[0] is None else ('箱子脱出掌间' if box_offcenter >= 0.12 else '掌间距穿过箱体')}）")

# 抬升：双肩 pitch 再抬 0.5 rad
z_palm_0 = float((palm_pos()[0] + palm_pos()[1])[2]) / 2
z_box_0 = float(box.data.root_pos_w[0, 2])
LIFT_STEPS = 300
p_l0 = float(arm_targets[0, jidx("left_shoulder_pitch_joint")])
p_r0 = float(arm_targets[0, jidx("right_shoulder_pitch_joint")])

def _lift(i):
    a = (i + 1) / LIFT_STEPS
    arm_targets[0, jidx("left_shoulder_pitch_joint")] = p_l0 + lift_sign * 0.5 * a
    arm_targets[0, jidx("right_shoulder_pitch_joint")] = p_r0 + lift_sign * 0.5 * a

step_hold(LIFT_STEPS, _lift)
step_hold(150)

z_palm_1 = float((palm_pos()[0] + palm_pos()[1])[2]) / 2
z_box_1 = float(box.data.root_pos_w[0, 2])
palm_rise, box_rise = z_palm_1 - z_palm_0, z_box_1 - z_box_0
slip = palm_rise - box_rise
print(f"  抬升: 手掌 +{palm_rise:.3f} m, 箱子 {box_rise:+.3f} m, 滑移 {slip:.3f} m")
lifted = grip_established and palm_rise > 0.05 and slip < 0.05 and box_rise > 0.05
print(f"  结论: {'✅ 纯手掌夹持可以抱起' if lifted else '❌ 手掌夹持未能抱起（' + ('夹持未建立' if not grip_established else '抬升打滑/脱落') + '）'}")
report["dynamic"]["palm_grip_test"] = {
    "gap_start_m": _fmt(gap_start),
    "gap_curve_m": gap_curve,
    "contact_gap_m": _fmt(contact_gap[0]) if contact_gap[0] is not None else None,
    "gap_hold_m": _fmt(gap_hold),
    "box_offcenter_m": _fmt(box_offcenter),
    "roll_torque_est_Nm": [_fmt(tau_l), _fmt(tau_r)],
    "grip_established": bool(grip_established),
    "palm_rise_m": _fmt(palm_rise),
    "box_rise_m": _fmt(box_rise),
    "slip_m": _fmt(slip),
    "lifted": bool(lifted),
}

# ---------------------------------------------------------------------------
# B3. 指扣抓取实验：轻掌夹（掌面刚贴箱端）+ 手指收拢扣箱 + 抬升
#     手指驱动严格复现实跑 mirror 链路：目标限幅 ±HAND_DELTA、速度目标清零
#     （对应 ISAACLAB_G1_HAND_JOINT_TARGET_MAX_DELTA / ZERO_TARGET_ONLY_HAND_VELOCITY=1）
# ---------------------------------------------------------------------------
print("\n[B3] 指扣抓取实验（轻掌夹到掌面贴箱 → 手指收拢扣箱 → 抬升）")

# 掌面比 palm link 原点外扩约 0.07 m/侧（B2 复盘标定：palm_gap=0.348 时掌面刚贴 0.20 箱端）
PALM_FACE_TOTAL_OFFSET = 0.148
PINCH_GAP = 0.200 + PALM_FACE_TOTAL_OFFSET - 0.003  # 掌面各压入约 1.5 mm
HAND_DELTA = 0.02

hand_ids, hand_names = robot.find_joints([".*_hand_.*_joint"])
hand_limits = robot.data.joint_pos_limits[:, hand_ids, :]
hand_q0 = robot.data.default_joint_pos[:, hand_ids].clone()
hands_act = robot.actuators["hands"]
k_hand = float(hands_act.stiffness.mean())
tau_hand = (
    float(hands_act.effort_limit_sim.mean())
    if hasattr(hands_act, "effort_limit_sim")
    else float(hands_act.effort_limit.mean())
)
print(f"  手指执行器: stiffness={k_hand:.0f} effort_limit={tau_hand:.1f} N·m, "
      f"目标限幅 ±{HAND_DELTA} rad → 稳态接触力矩上限 ≈ {k_hand * HAND_DELTA:.2f} N·m")


def _finger_close_goal() -> torch.Tensor:
    """index/middle 全收 + 拇指扣合，符号与 _compose_hand_target 一致。"""
    goal = hand_q0.clone()
    for i, name in enumerate(hand_names):
        left = name.startswith("left")
        if "thumb_0" in name:
            goal[0, i] = 0.0
        elif "thumb_1" in name:
            goal[0, i] = 1.1 if left else -1.1
        elif "thumb_2" in name:
            goal[0, i] = 1.8 if left else -1.8
        else:  # index_* / middle_*
            goal[0, i] = -1.8 if left else 1.8
    return goal


finger_goal = [None]  # None = 维持默认张开


def _apply_fingers():
    goal = finger_goal[0] if finger_goal[0] is not None else hand_q0
    cur = robot.data.joint_pos[:, hand_ids]
    tgt = torch.clamp(goal, cur - HAND_DELTA, cur + HAND_DELTA)
    tgt = torch.max(torch.min(tgt, hand_limits[..., 1]), hand_limits[..., 0])
    robot.set_joint_position_target(tgt, joint_ids=hand_ids)
    robot.set_joint_velocity_target(torch.zeros_like(tgt), joint_ids=hand_ids)


def step_hold3(n: int, on_step=None):
    """同 step_hold，但每步同时下发实跑链路语义的手指目标。"""
    for i in range(n):
        robot.write_root_pose_to_sim(root0)
        robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(lb_q0, lb_zero, joint_ids=lb_ids)
        robot.set_joint_position_target(arm_targets, joint_ids=arm_ids)
        _apply_fingers()
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        if on_step is not None:
            on_step(i)


# 复位手臂/手指
arm_targets = arm_q0.clone()
finger_goal[0] = None
step_hold3(150)

# 前伸，起手掌间距调到掌面距箱端 2~5 cm（palm_gap 0.38~0.42）
for side in ("left", "right"):
    arm_targets[0, jidx(f"{side}_shoulder_pitch_joint")] = arm_q0[0, jidx(f"{side}_shoulder_pitch_joint")] + lift_sign * 0.45
    arm_targets[0, jidx(f"{side}_elbow_joint")] = 0.9
step_hold3(180)
open3, tries = 0.0, 0
while not (0.38 < palm_gap() < 0.42) and tries < 15:
    open3 += 0.04 if palm_gap() <= 0.38 else -0.04
    arm_targets[0, jidx("left_shoulder_roll_joint")] = arm_q0[0, jidx("left_shoulder_roll_joint")] - inward_l * open3
    arm_targets[0, jidx("right_shoulder_roll_joint")] = arm_q0[0, jidx("right_shoulder_roll_joint")] - inward_r * open3
    step_hold3(60)
    tries += 1
gap3_start = palm_gap()
print(f"  起手掌间距 {gap3_start:.3f} m（掌面间距 ≈ {gap3_start - PALM_FACE_TOTAL_OFFSET:.3f}，箱长 0.200）")

# 箱子悬浮在掌中点（重力补偿弱弹簧扶持）
lp, rp = palm_pos()
ax3 = lp - rp
yaw3 = math.atan2(float(ax3[1]), float(ax3[0]))
box_quat3 = torch.tensor([math.cos(yaw3 / 2), 0.0, 0.0, math.sin(yaw3 / 2)], device=device)
set_box((lp + rp) / 2, box_quat3)
spring3 = [True]


def _b3_spring(i):
    if spring3[0]:
        box.set_external_force_and_torque(_spring_force(), zero_wrench)


step_hold3(120, _b3_spring)

# 收拢到掌面贴箱。不设力矩哨兵：指尖伸在掌面前方、会先接触箱子并把
# 推挤反力传回肩部（实测一轮即 ~8 N·m），这是软手指被顶弯的正常过程，
# 掌面必须继续压到位让指尖卷到箱体侧面形成环扣。弹簧(±12 N)扶稳箱子。
tries = 0
while palm_gap() > PINCH_GAP and tries < 25:
    arm_targets[0, jidx("left_shoulder_roll_joint")] += inward_l * 0.015
    arm_targets[0, jidx("right_shoulder_roll_joint")] += inward_r * 0.015
    step_hold3(40, _b3_spring)
    tries += 1
gap3_pinch = palm_gap()
tau3_l = est_torque("left_shoulder_roll_joint")
tau3_r = est_torque("right_shoulder_roll_joint")
print(f"  轻夹后掌间距 {gap3_pinch:.3f} m（掌面间距 ≈ {gap3_pinch - PALM_FACE_TOTAL_OFFSET:.3f}）"
      f" | roll 力矩 左{tau3_l:.1f} 右{tau3_r:.1f} N·m")

# 手指收拢扣箱（0.02 限幅 + velocity_limit 2.5 rad/s，1.8 rad 行程约 1 s，给 2.5 s）
finger_goal[0] = _finger_close_goal()
step_hold3(500, _b3_spring)
fin_err = float((robot.data.joint_pos[:, hand_ids] - _finger_close_goal()).abs().mean())
print(f"  手指收拢完成，平均残余误差 {fin_err:.3f} rad（被箱体挡住的指节会有较大残差=接触力）")

# 撤弹簧，重力交给抓取
spring3[0] = False
box.set_external_force_and_torque(zero_wrench, zero_wrench)
z3_hold0 = float(box.data.root_pos_w[0, 2])
step_hold3(300)
z3_hold1 = float(box.data.root_pos_w[0, 2])
hold_drop3 = z3_hold0 - z3_hold1
lp, rp = palm_pos()
off3 = float((box.data.root_pos_w[0] - (lp + rp) / 2).norm())
held3 = hold_drop3 < 0.05 and off3 < 0.15
print(f"  撤簧静置 1.5 s: 箱子下沉 {hold_drop3:.3f} m, 偏离掌中点 {off3:.3f} m → {'✅ 稳持' if held3 else '❌ 脱落'}")

# 抬升
z3_palm0 = float((palm_pos()[0] + palm_pos()[1])[2]) / 2
z3_box0 = float(box.data.root_pos_w[0, 2])
pl3 = float(arm_targets[0, jidx("left_shoulder_pitch_joint")])
pr3 = float(arm_targets[0, jidx("right_shoulder_pitch_joint")])


def _b3_lift(i):
    a = (i + 1) / LIFT_STEPS
    arm_targets[0, jidx("left_shoulder_pitch_joint")] = pl3 + lift_sign * 0.5 * a
    arm_targets[0, jidx("right_shoulder_pitch_joint")] = pr3 + lift_sign * 0.5 * a


step_hold3(LIFT_STEPS, _b3_lift)
step_hold3(150)
z3_palm1 = float((palm_pos()[0] + palm_pos()[1])[2]) / 2
z3_box1 = float(box.data.root_pos_w[0, 2])
rise3_p, rise3_b = z3_palm1 - z3_palm0, z3_box1 - z3_box0
slip3 = rise3_p - rise3_b
lifted3 = held3 and rise3_p > 0.05 and rise3_b > 0.05 and slip3 < 0.05
print(f"  抬升: 手掌 +{rise3_p:.3f} m, 箱子 {rise3_b:+.3f} m, 滑移 {slip3:.3f} m")
print(f"  结论: {'✅ 轻掌夹+指扣可以抱起长箱' if lifted3 else '❌ 指扣抓取失败（' + ('静置即脱落' if not held3 else '抬升打滑/脱落') + '）'}")
report["dynamic"]["finger_pinch_test"] = {
    "gap_start_m": _fmt(gap3_start),
    "gap_pinch_m": _fmt(gap3_pinch),
    "palm_face_gap_est_m": _fmt(gap3_pinch - PALM_FACE_TOTAL_OFFSET),
    "roll_torque_at_pinch_Nm": [_fmt(tau3_l), _fmt(tau3_r)],
    "finger_goal_residual_rad": _fmt(fin_err),
    "hold_drop_m": _fmt(hold_drop3),
    "box_offcenter_m": _fmt(off3),
    "held_after_spring_off": bool(held3),
    "palm_rise_m": _fmt(rise3_p),
    "box_rise_m": _fmt(rise3_b),
    "slip_m": _fmt(slip3),
    "lifted": bool(lifted3),
}

# ---------------------------------------------------------------------------
# B4. 单手指托对照：小箱落在收拢的右手手指上——排除对夹几何/对称性因素，
#     单测软手指(稳态 0.3 N·m/指节)能否承托 0.12 kg。
# ---------------------------------------------------------------------------
print("\n[B4] 单手指托对照（小箱 0.12 kg 落在右手收拢的手指上）")
small = scene["small_box_1"]
small_mass = float(small.root_physx_view.get_masses().sum())

arm_targets = arm_q0.clone()
finger_goal[0] = None
step_hold3(150)
# 右臂前伸、掌心大致朝上（wrist_roll 转 90°），手指收拢成托架
arm_targets[0, jidx("right_shoulder_pitch_joint")] = arm_q0[0, jidx("right_shoulder_pitch_joint")] + lift_sign * 0.5
arm_targets[0, jidx("right_elbow_joint")] = 1.1
arm_targets[0, jidx("right_wrist_roll_joint")] = arm_q0[0, jidx("right_wrist_roll_joint")] + 1.57
step_hold3(250)
finger_goal[0] = _finger_close_goal()
step_hold3(400)

# 小箱放右手指环正上方 6 cm 自由落下
r_idx_ids, _ = robot.find_bodies(["right_hand_index_1_link"])
r_mid_ids, _ = robot.find_bodies(["right_hand_middle_1_link"])
r_palm_pos = robot.data.body_pos_w[0, r_palm].clone()
tip_pos = (
    robot.data.body_pos_w[0, r_idx_ids[0]] + robot.data.body_pos_w[0, r_mid_ids[0]]
) / 2
drop4 = (r_palm_pos + tip_pos) / 2
drop4[2] = float(tip_pos[2]) + 0.06
small.write_root_pose_to_sim(torch.cat([drop4.view(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)], dim=-1))
small.write_root_velocity_to_sim(zero6)

z4: list[float] = []
step_hold3(400, lambda i: z4.append(float(small.data.root_pos_w[0, 2])))
drop4_total = float(drop4[2]) - z4[-1]
dist4 = float((small.data.root_pos_w[0] - (r_palm_pos + tip_pos) / 2).norm())
caught4 = drop4_total < 0.12 and dist4 < 0.15
print(f"  放置 z={float(drop4[2]):.3f} → 最终 z={z4[-1]:.3f}（下坠 {drop4_total:.3f} m，距手 {dist4:.3f} m）")
print(f"  结论: {'✅ 软手指能承托小箱——参数力预算成立' if caught4 else '❌ 手指没接住——需区分滑出还是压塌'}")
report["dynamic"]["single_hand_cradle_test"] = {
    "box_mass_kg": _fmt(small_mass),
    "drop_m": _fmt(drop4_total),
    "dist_to_hand_m": _fmt(dist4),
    "caught": bool(caught4),
}

# ---------------------------------------------------------------------------
# 汇总判定
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("诊断汇总")
print("=" * 78)
if arm_without_col:
    report["verdict"].append(
        f"前臂/上臂共 {len(arm_without_col)} 个 link 无碰撞体（{', '.join(arm_without_col)}）——"
        "『抱』的主要受力面在物理上不存在，箱子会穿透手臂。"
    )
if report["dynamic"].get("cradle_test", {}).get("fell_through"):
    report["verdict"].append("前臂承托实验：箱子穿透前臂自由坠落，动态实锤碰撞体缺失。")
if report["dynamic"].get("palm_grip_test", {}).get("lifted"):
    report["verdict"].append("手掌两侧夹持端面可以抱起长箱——手掌/手指碰撞体和摩擦足够，可作为当前资产下的可行抓法。")
elif "palm_grip_test" in report["dynamic"]:
    report["verdict"].append("纯手掌夹持也未能抱起，需进一步看力矩/摩擦数据。")
fp = report["dynamic"].get("finger_pinch_test", {})
if fp.get("lifted"):
    report["verdict"].append(
        "轻掌夹+手指扣抓可以抱起 0.5 kg 长箱——当前软手指参数下的可行抓法，"
        "遥操应采用『掌面贴箱端 + 扣指』而非深压对夹。"
    )
elif fp:
    report["verdict"].append(
        f"指扣抓取也失败（{'撤簧即脱落' if not fp.get('held_after_spring_off') else '抬升打滑/脱落'}）"
        "——软手指力矩或接触几何仍不满足，需看 finger_pinch_test 数据定位。"
    )
for i, v in enumerate(report["verdict"], 1):
    print(f"  {i}. {v}")
print("\n⚠️ 本结论基于 Linux 端加载的 g1_43dof.usd；实跑机器如资产版本不同需另行核对（对比文件字节数即可）。")

if args_cli.report:
    with open(args_cli.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告已写入: {args_cli.report}")
else:
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))

env.close()
simulation_app.close()
