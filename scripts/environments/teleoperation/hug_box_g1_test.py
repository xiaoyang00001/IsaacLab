# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
PickPlace G1 抱箱演示：证明 test_box 可以被双臂抱住。

启动方式：
  ./isaaclab.sh -p scripts/environments/teleoperation/hug_box_g1_test.py \\
      --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --device cuda:0

或从 start_teleop_g1.sh 加 --hug-test 参数。

流程：
  ① 方向标定：给肩俯仰/肩滚转/肘各一个小目标增量，观测手掌实际位移，
     自动求出"前伸/内合/上抬"方向与增益（不依赖关节符号约定）
  ② 张臂预备：双臂前伸、屈肘，滚转收到手掌间距 ≈ 箱宽 + 2cm
  ③ 放箱合抱：箱子瞬移到两掌中点（短边对准两掌连线），滚转过冲挤压夹紧
  ④ 悬空保持 3s：箱子离地 ~1m，不掉 = 抱住 ✅
  ⑤ 腰部提箱：腰俯仰带动上身整体抬升，箱子应随之升高
  ⑥ 松手对照：张开双臂，箱子落地 → 反证此前是抱持力在起作用

说明：本环境动作项全部为网络镜像（ZMQ/UDP），脚本绕过 action manager
直接写关节 PD 目标并步进仿真；默认每步锚定机器人根部（等效固定基座），
只验证"双臂 + 摩擦能否抱住箱子"，不考验下肢平衡。
"""

import argparse

# Linux 需在 AppLauncher 之前导入 pinocchio 避免符号冲突；Windows 环境无此包时跳过
try:
    import pinocchio  # noqa: F401
except ImportError:
    pass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0", help="任务名称")
parser.add_argument("--robot", type=str, default=None, help="机器人实体名（默认自动：robot 或 robot_1）")
parser.add_argument("--box", type=str, default="test_box", help="被抱箱子实体名")
parser.add_argument("--free-root", action="store_true", help="不锚定根部（默认每步锚定，等效固定基座）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # 强制图形界面
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

# ── 输出辅助 ──────────────────────────────────────────────

PURPLE = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def banner(msg: str):
    n = len(msg) + 4
    print(f"\n{PURPLE}{'=' * n}{RESET}")
    print(f"{PURPLE}│ {BOLD}{msg}{RESET}{PURPLE} │{RESET}")
    print(f"{PURPLE}{'=' * n}{RESET}\n")


def info(msg: str):
    print(f"  {CYAN}{msg}{RESET}", flush=True)


def ok(msg: str):
    print(f"  {GREEN}✅ {msg}{RESET}", flush=True)


def warn(msg: str):
    print(f"  {YELLOW}⚠️  {msg}{RESET}", flush=True)


# ── 主流程 ────────────────────────────────────────────────

def main():
    banner("PickPlace G1 抱箱演示")
    print("  场景加载中，请稍候...\n")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    robot_key = args.robot or ("robot" if "robot" in env.scene.keys() else "robot_1")
    robot = env.scene[robot_key]
    box = env.scene[args.box]
    info(f"机器人实体: {robot_key}  箱子实体: {args.box}")

    # 订阅端箱子是 kinematic（跟随远端），物理上抱不住，直接拒绝
    box_spawn = getattr(env_cfg.scene, args.box, None)
    rigid_props = getattr(getattr(box_spawn, "spawn", None), "rigid_props", None)
    if rigid_props is not None and getattr(rigid_props, "kinematic_enabled", False):
        warn("该箱子是 kinematic 刚体（ZMQ 订阅端跟随模式），无法做物理抱持测试")
        warn("请在 ISAACLAB_LOCAL_ROBOT_ID=1（publisher 角色）下运行")
        env.close()
        simulation_app.close()
        return

    box_size = getattr(getattr(box_spawn, "spawn", None), "size", (0.32, 0.22, 0.24))
    box_w = box_size[1]  # 抱持方向用短边（y 边）
    dt = env.physics_dt

    jnames = robot.data.joint_names
    bnames = robot.data.body_names

    def jidx(name: str) -> int:
        return jnames.index(name)

    def body_idx(candidates: list[str]) -> int:
        for c in candidates:
            if c in bnames:
                return bnames.index(c)
        raise KeyError(f"找不到 link，候选 {candidates}，实际 body_names={bnames}")

    palm_l = body_idx(["left_hand_palm_link", "left_wrist_yaw_link"])
    palm_r = body_idx(["right_hand_palm_link", "right_wrist_yaw_link"])

    targets = robot.data.default_joint_pos.clone()
    root_pose0 = robot.data.root_state_w[:, :7].clone()
    zero_vel6 = torch.zeros((1, 6), device=env.device)

    def sim_steps(n: int):
        for i in range(n):
            if not args.free_root:
                robot.write_root_pose_to_sim(root_pose0)
                robot.write_root_velocity_to_sim(zero_vel6)
            robot.set_joint_position_target(targets)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt)
            if i % 4 == 0:
                env.sim.render()

    def palm_pos():
        return (
            robot.data.body_link_pos_w[0, palm_l].clone(),
            robot.data.body_link_pos_w[0, palm_r].clone(),
        )

    # 相机对准机器人正面（机器人朝向 = 根四元数旋转后的 +X）
    root_p = root_pose0[0, :3]
    fwd = quat_apply(root_pose0[:, 3:7], torch.tensor([[1.0, 0.0, 0.0]], device=env.device))[0]
    try:
        eye = (root_p + fwd * 2.2 + torch.tensor([0.6, 0.0, 0.6], device=env.device)).tolist()
        target = (root_p + torch.tensor([0.0, 0.0, 0.2], device=env.device)).tolist()
        env.sim.set_camera_view(eye, target)
    except Exception:
        pass

    info("等待场景稳定...")
    sim_steps(80)
    if args.free_root:
        warn("--free-root 模式：根部不锚定，机器人可能站不稳")

    # ===============================================================
    # ① 方向标定：小增量 → 手掌位移，求方向与增益
    # ===============================================================
    banner("① 关节方向标定")

    PROBE = 0.25  # rad

    def probe(joint: str, palm: int) -> torch.Tensor:
        """给 joint 目标 +PROBE，返回手掌位移向量（随后恢复）。"""
        j = jidx(joint)
        base = targets[0, j].item()
        p0 = robot.data.body_link_pos_w[0, palm].clone()
        targets[0, j] = base + PROBE
        sim_steps(50)
        dp = robot.data.body_link_pos_w[0, palm] - p0
        targets[0, j] = base
        sim_steps(50)
        return dp

    calib = {}
    for side, palm in (("left", palm_l), ("right", palm_r)):
        # 肩俯仰：取让手掌沿机器人前向移动的方向
        dp = probe(f"{side}_shoulder_pitch_joint", palm)
        d_fwd = torch.dot(dp, fwd).item()
        calib[f"{side}_pitch_fwd"] = 1.0 if d_fwd > 0 else -1.0
        # 肘：取让手掌"上抬 + 前伸"的方向
        dp = probe(f"{side}_elbow_joint", palm)
        d_up = dp[2].item() + torch.dot(dp, fwd).item()
        calib[f"{side}_elbow_bend"] = 1.0 if d_up > 0 else -1.0
        # 肩滚转：取让两掌间距缩小的方向，并记录增益 m/rad
        pl0, pr0 = palm_pos()
        sep0 = torch.norm(pl0 - pr0).item()
        dp = probe(f"{side}_shoulder_roll_joint", palm)
        # dp 在两掌连线上的投影：左掌沿 L→R 为内合，右掌反之
        lat = (pr0 - pl0) / max(sep0, 1e-6)
        d_lat = torch.dot(dp, lat).item()  # 左掌沿 L→R 方向为内合
        inward = d_lat if palm == palm_l else -d_lat
        calib[f"{side}_roll_in"] = 1.0 if inward > 0 else -1.0
        calib[f"{side}_roll_gain"] = abs(d_lat) / PROBE  # m/rad
        info(
            f"{side:>5s} 臂: pitch前伸 {calib[f'{side}_pitch_fwd']:+.0f}  "
            f"elbow屈 {calib[f'{side}_elbow_bend']:+.0f}  "
            f"roll内合 {calib[f'{side}_roll_in']:+.0f} (增益 {calib[f'{side}_roll_gain']:.2f} m/rad)"
        )

    # 腰俯仰：取让手掌升高的方向（提箱阶段用腰带动上身，不改变双臂夹持几何）
    if "waist_pitch_joint" in jnames:
        dp = probe("waist_pitch_joint", palm_l)
        calib["waist_lift"] = 1.0 if dp[2].item() > 0 else -1.0
        info(f"waist_pitch 抬升方向 {calib['waist_lift']:+.0f}")

    # ===============================================================
    # ② 张臂预备
    # ===============================================================
    banner("② 张臂预备：前伸屈肘，收到箱宽 + 2cm")

    for side in ("left", "right"):
        targets[0, jidx(f"{side}_shoulder_pitch_joint")] += calib[f"{side}_pitch_fwd"] * 0.55
        targets[0, jidx(f"{side}_elbow_joint")] += calib[f"{side}_elbow_bend"] * 0.55
    sim_steps(150)

    # 用标定增益 P 控制收臂，让两掌间距 → 箱宽 + 2cm
    sep_goal = box_w + 0.02
    for it in range(8):
        pl, pr = palm_pos()
        sep = torch.norm(pl - pr).item()
        err = sep - sep_goal
        if abs(err) < 0.015:
            break
        for side in ("left", "right"):
            gain = max(calib[f"{side}_roll_gain"], 0.05)
            delta = min(max(err / (2.0 * gain), -0.25), 0.25)
            targets[0, jidx(f"{side}_shoulder_roll_joint")] += calib[f"{side}_roll_in"] * delta
        sim_steps(50)
        info(f"  收臂第 {it + 1} 轮: 两掌间距 {sep:.3f} → 目标 {sep_goal:.3f}")

    pl, pr = palm_pos()
    sep = torch.norm(pl - pr).item()
    info(f"预备完成：两掌间距 {sep:.3f} m（箱宽 {box_w:.2f} m）")

    # ===============================================================
    # ③ 放箱合抱
    # ===============================================================
    banner("③ 放箱合抱：箱子瞬移到两掌之间，滚转挤压夹紧")

    mid = (pl + pr) / 2.0
    lat = (pr - pl) / max(sep, 1e-6)
    # 箱子局部 y 轴（短边）对准两掌连线：yaw = atan2(lat_y, lat_x) - 90°
    a = torch.atan2(lat[1], lat[0]).item() - math.pi / 2.0
    box_quat = torch.tensor([[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]], device=env.device)
    box_pose = torch.cat([mid.unsqueeze(0), box_quat], dim=1)
    box.write_root_pose_to_sim(box_pose)
    box.write_root_velocity_to_sim(zero_vel6)
    hold_z0 = mid[2].item()
    info(f"箱子就位：({mid[0].item():.3f}, {mid[1].item():.3f}, {mid[2].item():.3f})，离地 {hold_z0:.2f} m")

    # 滚转过冲 0.25 rad → 双臂持续挤压（PD 顶住箱面，摩擦 μ=1.2 提供竖向支撑）
    for side in ("left", "right"):
        targets[0, jidx(f"{side}_shoulder_roll_joint")] += calib[f"{side}_roll_in"] * 0.25
    sim_steps(120)

    bz = box.data.root_pos_w[0, 2].item()
    if bz < hold_z0 - 0.3:
        warn(f"合抱阶段箱子滑落（z={bz:.3f}），加大挤压再试一次")
        box.write_root_pose_to_sim(box_pose)
        box.write_root_velocity_to_sim(zero_vel6)
        for side in ("left", "right"):
            targets[0, jidx(f"{side}_shoulder_roll_joint")] += calib[f"{side}_roll_in"] * 0.15
        sim_steps(120)

    # ===============================================================
    # ④ 悬空保持 3 秒
    # ===============================================================
    banner("④ 悬空保持 3s：箱子不掉 = 抱住")

    z_start = box.data.root_pos_w[0, 2].item()
    for k in range(6):
        sim_steps(100)  # 0.5 s
        bp = box.data.root_pos_w[0]
        info(f"  t={0.5 * (k + 1):.1f}s: 箱子 z={bp[2].item():.4f}（初始 {z_start:.4f}）")

    z_end = box.data.root_pos_w[0, 2].item()
    drop = z_start - z_end
    held = drop < 0.10 and z_end > 0.5
    if held:
        ok(f"抱住成功：3s 仅下沉 {drop * 100:.1f} cm，箱子悬空 {z_end:.2f} m 未掉落")
    else:
        warn(f"未抱住：箱子从 {z_start:.3f} 降到 {z_end:.3f}（下落 {drop:.3f} m）")

    # ===============================================================
    # ⑤ 抬臂提箱
    # ===============================================================
    lifted_ok = False
    if held and "waist_lift" in calib:
        banner("⑤ 腰部提箱：腰俯仰带动上身整体抬升（双臂夹持几何不变）")
        # 肩关节动会改变夹持几何导致滑落（实测），改用腰：臂相对躯干不动，夹持力方向不变
        for side in ("left", "right"):
            targets[0, jidx(f"{side}_shoulder_roll_joint")] += calib[f"{side}_roll_in"] * 0.10
        sim_steps(60)
        for _ in range(10):
            targets[0, jidx("waist_pitch_joint")] += calib["waist_lift"] * 0.02
            sim_steps(25)
        z_lift = box.data.root_pos_w[0, 2].item()
        lifted_ok = z_lift > z_end + 0.02
        if lifted_ok:
            ok(f"提箱成功：箱子随上身升高 {(z_lift - z_end) * 100:.1f} cm（z={z_lift:.3f}）")
        else:
            warn(f"箱子未随上身升高（z={z_lift:.3f}），可能在臂间滑动")

    # ===============================================================
    # ⑥ 松手对照
    # ===============================================================
    banner("⑥ 松手对照：腰回正 + 张开双臂 → 箱子应落地")
    z_before_release = box.data.root_pos_w[0, 2].item()
    # 先把腰回正（后仰姿态会让箱子靠在胸口不落），再张臂
    if "waist_lift" in calib:
        for _ in range(10):
            targets[0, jidx("waist_pitch_joint")] -= calib["waist_lift"] * 0.02
            sim_steps(15)
    for side in ("left", "right"):
        targets[0, jidx(f"{side}_shoulder_roll_joint")] -= calib[f"{side}_roll_in"] * 0.9
    sim_steps(300)
    z_rel = box.data.root_pos_w[0, 2].item()
    if z_before_release > 0.5 and z_rel < 0.4:
        ok(f"松手后箱子从 {z_before_release:.2f} m 落地（z={z_rel:.3f}）→ 反证此前是双臂抱持力托住箱子")
    elif z_before_release <= 0.5:
        info(f"箱子在⑤阶段已滑落（松手前 z={z_before_release:.3f}），对照略过")
    else:
        info(f"松手后箱子 z={z_rel:.3f}")

    banner("测试完成")
    if held:
        ok("结论：箱子可以被 G1 双臂抱住（挤压 + 摩擦支撑，悬空 3s 不掉）")
    else:
        warn("结论：本次未抱住，可调大挤压量（③ 中 0.25 rad）或检查手臂碰撞体")
    info("窗口保持 10s 后退出，可自由旋转视角观察...")

    sim_steps(2000)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
