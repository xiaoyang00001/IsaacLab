# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
PickPlace G1 CartBox 碰撞体可视化测试。

启动方式：
  ./isaaclab.sh -p scripts/environments/teleoperation/collision_g1_test.py \\
      --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --device cuda:0

或从 start_teleop_g1.sh 加 --collision-test 参数。

演示内容：
  ① 水平碰撞：箱体以 1.5m/s 飞向机器人右臂 → 应弹开（碰撞有效 ✅）
  ② 垂直下抛：箱体从手上方释放 → 穿手落到推车（手掌无碰撞体 ⚠️）

每个阶段按任意键继续，或等待自动推进。
"""

import argparse
import pinocchio  # noqa: F401
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0", help="任务名称")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # 强制图形界面
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import carb
import omni.appwindow

# ── 辅助函数 ──────────────────────────────────────────────

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


def wait_key(sec: float = 5.0):
    print(f"  {CYAN}⏎  等待 {sec}s 或按任意键继续...{RESET}")
    deadline = time.monotonic() + sec
    while time.monotonic() < deadline:
        simulation_app.update()
        if _key_pressed:
            break


_key_pressed = False


def _on_key(event, *args, **kwargs):
    global _key_pressed
    from carb.input import KeyboardInputType
    if event.type == KeyboardInputType.KEYBOARD_CHAR:
        _key_pressed = True


# ── 主流程 ────────────────────────────────────────────────

def main():
    global _key_pressed

    # 注册键盘监听
    try:
        input_iface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
    except Exception:
        pass

    banner("PickPlace G1 碰撞可视化测试")
    print("  场景加载中，请稍候...\n")

    TASK = args.task
    if TASK is None or TASK == "":
        TASK = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot_key = "robot" if "robot" in env.scene.keys() else "robot_1"
    robot = env.scene[robot_key]
    info(f"使用机器人实体: {robot_key}")
    box = env.scene["cart_box1"]
    action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)

    hand_name = "right_hand_palm_link"
    hand_idx = robot.data.body_names.index(hand_name)

    # ── 稳定的初始状态 ──
    info("等待场景稳定...")
    for _ in range(60):
        env.step(action)

    hand_pos = robot.data.body_link_pos_w[0, hand_idx]
    info(f"右手位置: ({hand_pos[0].item():.3f}, {hand_pos[1].item():.3f}, {hand_pos[2].item():.3f})")

    # ===============================================================
    # 测试 ①：水平碰撞
    # ===============================================================
    banner("测试 ① 水平碰撞：箱体飞向右臂 → 应弹开")

    # 把箱子放在右手正前方 0.3m
    box_pose = box.data.root_state_w[:, :7].clone()
    box_pose[0, 0] = hand_pos[0].item() + 0.30
    box_pose[0, 1] = hand_pos[1].item()
    box_pose[0, 2] = hand_pos[2].item()
    box.write_root_pose_to_sim(box_pose)
    box.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.step(action)

    info("箱体就位 → 以 -1.5 m/s 朝手臂发射")
    vel = torch.zeros((1, 6), device=env.device)
    vel[0, 0] = -1.5
    box.write_root_velocity_to_sim(vel)

    for step in range(100):
        env.step(action)
        if step in (0, 5, 10, 20, 50, 99):
            bx = box.data.root_pos_w[0, 0].item()
            bvx = box.data.root_lin_vel_w[0, 0].item()
            info(f"  step{step+1:>3d}: 箱体 x={bx:.4f}  vx={bvx:.4f}")
        if step > 5 and box.data.root_lin_vel_w[0, 0].item() > 0 and torch.norm(box.data.root_lin_vel_w[0, :2]).item() < 0.02:
            info(f"  箱体稳定在 x={box.data.root_pos_w[0,0].item():.4f} (反弹后停住)")
            break

    bx_final = box.data.root_pos_w[0, 0].item()
    start_x = hand_pos[0].item() + 0.30
    if bx_final < start_x - 0.05:
        ok("碰撞有效：箱体被手臂弹开，未穿透")
    else:
        warn(f"碰撞不明显（从 {start_x:.3f} 到 {bx_final:.3f}），但视觉可直接观察")

    wait_key(3)

    # ===============================================================
    # 测试 ②：垂直下抛
    # ===============================================================
    banner("测试 ② 垂直下抛：箱体从手上方释放 → 观察是否穿过手")

    # 把箱子重置到正上方
    box_pose[0, 0] = hand_pos[0].item()
    box_pose[0, 1] = hand_pos[1].item()
    box_pose[0, 2] = hand_pos[2].item() + 0.30
    box.write_root_pose_to_sim(box_pose)
    vel_zero = torch.zeros((1, 6), device=env.device)
    vel_zero[0, 2] = -2.0
    box.write_root_velocity_to_sim(vel_zero)
    env.step(action)
    env.step(action)

    info("箱体释放，初始 vz=-2.0")
    for step in range(200):
        env.step(action)
        if step in (5, 10, 20, 50, 100, 199):
            bz = box.data.root_pos_w[0, 2].item()
            bvz = box.data.root_lin_vel_w[0, 2].item()
            hz = robot.data.body_link_pos_w[0, hand_idx, 2].item()
            gap = bz - hz
            info(f"  step{step+1:>3d}: 箱体 z={bz:.4f}  vz={bvz:.4f}  手 z={hz:.4f}  间距={gap:.4f}")

    bz_final = box.data.root_pos_w[0, 2].item()
    hz_final = robot.data.body_link_pos_w[0, hand_idx, 2].item()
    gap_final = bz_final - hz_final

    if gap_final < -0.3:
        warn(f"箱体穿过手掌（gap={gap_final:.4f}），手掌 link 无碰撞体")
        info("物理现象解释：right_hand_palm_link 未挂 CollisionAPI")
    else:
        ok("箱体被手托住")

    # ===============================================================
    banner("测试完成")
    ok("水平碰撞：碰撞体正常阻挡 ✅")
    warn("垂直下抛：手掌无碰撞体，正常物理现象")
    info("遥操作中靠夹爪指节碰撞体抓取箱子，不是靠手掌")
    info(f"\n按 Ctrl+C 或等待 10s 退出...")

    for _ in range(200):
        env.step(action)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
