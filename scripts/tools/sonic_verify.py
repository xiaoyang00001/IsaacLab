# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal launcher to verify SONICWholeBodyAction pipeline on sonic_robot.

`pick_place` 在 isaaclab_tasks 的 _BLACKLIST_PKGS 里，自动注册会跳过它，需要手动 import 触发
gym.register。该脚本基于 zero_agent.py + 手动 import + SONIC 进度日志。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SONIC pipeline verification (zero action driver).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0", help="Task name."
)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N env steps (0 = run forever).")
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

from isaaclab_tasks.utils import parse_env_cfg


def main():
    print(f"[sonic_verify] task={args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[sonic_verify] env created; action_space={env.action_space}", flush=True)

    env.reset()
    print("[sonic_verify] reset done; entering step loop (press Ctrl+C to stop)", flush=True)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
        step += 1
        if step % 100 == 0:
            print(f"[sonic_verify] step={step}", flush=True)
        if args_cli.max_steps and step >= args_cli.max_steps:
            print(f"[sonic_verify] reached max_steps={args_cli.max_steps}, exiting", flush=True)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
