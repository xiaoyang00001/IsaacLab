# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环抖动量化验证运行器（IsaacLab 端，in-process）。

复现「MuJoCo 里正常、Isaac 里上身抖」的闭环场景并逐 env 步落盘原始数据，
供 sonic_jitter_report.py 计算抖动指标做 A/B 对比：

    锁根跟随（deploy 目标流入，root 锚定）
        → 程序化 unlock（等价 teleop 的 U 键，headless 下唯一途径）
        → 自由根闭环（软 PD + policy 平衡，抖动主战场）

三端编排（proxy/deploy/BVH 流）由 run_sonic_jitter_closed_loop.sh 负责；
本脚本只跑 IsaacLab 侧并保证 50Hz 实时节拍（deploy 步态相位按墙钟走，
env_hz 偏离实时则闭环数据无效——见 KB《SONIC闭环日志分析SOP》§2）。

`pick_place` 在 isaaclab_tasks 的 _BLACKLIST_PKGS 里，需手动 import 触发注册
（与 sonic_verify.py 同款）。
"""

import argparse
import json
import os
import time

# SONIC 闭环旗标必须在 pick_place 包 import 前生效（模块级读取）。
# setdefault：编排脚本可整体覆盖，单跑本脚本也能起对配置。
_SONIC_ENV_DEFAULTS = {
    "SONIC_G1_PHYSICS_MODE": "1",
    "SONIC_G1_VISUAL_SERVO_MODE": "0",
    "SONIC_G1_SELF_COLLISIONS": "0",
    "SONIC_DEPLOY_STABILIZE_ROOT": "1",
    "SONIC_DEPLOY_TARGET_RATE_LIMIT": "0.04",
    "SONIC_DEPLOY_TRANSPORT": "zmq",
    "SONIC_DEPLOY_ENDPOINT": "tcp://127.0.0.1:5557",
    "SONIC_DEPLOY_TOPIC": "g1_debug",
    "SONIC_DEPLOY_TARGET_FIELD": "last_action",
    "SONIC_DEPLOY_REFERENCE_TARGET_FIELD": "body_q_target",
    "SONIC_PUBLISH_STATE_ZMQ": "1",
    "SONIC_STATE_ZMQ_BIND": "tcp://*:5560",
    "SONIC_STATE_ZMQ_TOPIC": "sonic_state",
    # headless 严禁静默挂上 SteamVR runtime（env_hz 50→4.7 判例）
    "XR_RUNTIME_JSON": "/nonexistent",
}
for _key, _value in _SONIC_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SONIC closed-loop jitter measurement runner.")
parser.add_argument("--task", type=str, default="Isaac-SonicSolo-Locomanipulation-G1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--out", type=str, required=True, help="Output .npz path for per-step recordings.")
parser.add_argument("--locked_seconds", type=float, default=15.0, help="Recording length while root is locked.")
parser.add_argument("--free_seconds", type=float, default=30.0, help="Recording length after root unlock.")
parser.add_argument("--no_unlock", action="store_true", help="Skip the unlock phase (locked-only run).")
parser.add_argument(
    "--wait_packets_s", type=float, default=180.0,
    help="Max wall seconds to wait for deploy targets before giving up (exit code 3).",
)
parser.add_argument(
    "--warmup_packets", type=int, default=25,
    help="Deploy packets to consume before the locked recording phase starts (~0.5s at 50Hz).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import numpy as np
import sys
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401  (blacklist 包手动注册)
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def _log(message: str) -> None:
    print(f"[JitterVerify] {message}", flush=True)


class StepRecorder:
    """逐 env 步累积原始序列，最后一次性 np.stack 落盘。"""

    def __init__(self):
        self.wall_t: list[float] = []
        self.phase: list[int] = []  # 0=locked, 1=free
        self.q: list[np.ndarray] = []
        self.dq: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.step_delta: list[float] = []
        self.root_pos: list[np.ndarray] = []
        self.root_quat: list[np.ndarray] = []
        self.tilt_deg: list[float] = []
        self.packets: list[int] = []

    def record(self, term, asset, joint_ids, env_origin_z: float, phase: int) -> None:
        data = asset.data
        self.wall_t.append(time.monotonic())
        self.phase.append(phase)
        self.q.append(data.joint_pos[0, joint_ids].detach().cpu().numpy().copy())
        self.dq.append(data.joint_vel[0, joint_ids].detach().cpu().numpy().copy())
        self.target.append(term.processed_actions[0].detach().cpu().numpy().copy())
        self.step_delta.append(float(term._last_target_step_delta_absmax[0].item()))
        root_pos = data.root_pos_w[0].detach().cpu().numpy().copy()
        root_pos[2] -= env_origin_z
        self.root_pos.append(root_pos)
        self.root_quat.append(data.root_quat_w[0].detach().cpu().numpy().copy())
        gb = data.projected_gravity_b[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(gb[2].item())))))
        self.tilt_deg.append(tilt)
        self.packets.append(int(term._packet_count))

    def save(self, path: str, joint_names: list[str], meta: dict) -> None:
        np.savez_compressed(
            path,
            wall_t=np.asarray(self.wall_t, dtype=np.float64),
            phase=np.asarray(self.phase, dtype=np.int8),
            q=np.stack(self.q).astype(np.float32),
            dq=np.stack(self.dq).astype(np.float32),
            target=np.stack(self.target).astype(np.float32),
            step_delta=np.asarray(self.step_delta, dtype=np.float32),
            root_pos=np.stack(self.root_pos).astype(np.float32),
            root_quat=np.stack(self.root_quat).astype(np.float32),
            tilt_deg=np.asarray(self.tilt_deg, dtype=np.float32),
            packets=np.asarray(self.packets, dtype=np.int64),
            joint_names=np.asarray(joint_names),
            meta=np.asarray(json.dumps(meta)),
        )
        _log(f"recording saved: {path} steps={len(self.wall_t)}")


def main() -> int:
    _log(f"task={args_cli.task} out={args_cli.out}")
    _log(
        "flags "
        + " ".join(f"{key}={os.environ.get(key)}" for key in sorted(_SONIC_ENV_DEFAULTS))
    )
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    unwrapped = env.unwrapped
    term = unwrapped.action_manager.get_term("sonic_wholebody")
    asset = unwrapped.scene[term.cfg.asset_name]
    joint_ids = term._joint_ids
    joint_names = list(term._joint_names)
    env_origin_z = float(unwrapped.scene.env_origins[0, 2].item())
    step_dt = float(unwrapped.step_dt)
    _log(
        f"env ready; step_dt={step_dt:.4f}s joints={len(joint_ids)} "
        f"settle_steps={int(term.cfg.startup_settle_steps)} "
        f"unlock_blend_steps={int(term.cfg.unlock_blend_steps)} "
        f"rate_limit={float(term.cfg.target_rate_limit_rad_per_step):.4f}"
    )

    zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    next_tick = time.monotonic()
    wall_dts: list[float] = []
    last_step_wall = time.monotonic()

    def paced_step() -> None:
        nonlocal next_tick, last_step_wall
        next_tick += step_dt
        now = time.monotonic()
        sleep_s = next_tick - now
        if sleep_s > 0:
            time.sleep(sleep_s)
        elif sleep_s < -1.0:
            # 长停顿（场景加载等）后重新对表，不追帧
            next_tick = time.monotonic()
        with torch.inference_mode():
            env.step(zero_actions)
        now = time.monotonic()
        wall_dts.append(now - last_step_wall)
        last_step_wall = now

    # ---- 阶段 0：等 deploy 目标流入（期间照常推进：settle 走完、5560 状态持续发布）----
    _log("waiting for deploy packets (start proxy/deploy now; press ']' in deploy)")
    wait_deadline = time.monotonic() + float(args_cli.wait_packets_s)
    settle_steps = int(term.cfg.startup_settle_steps)
    while True:
        paced_step()
        settle_done = int(term._settle_step_counter) >= settle_steps
        if settle_done and int(term._packet_count) >= int(args_cli.warmup_packets):
            break
        if time.monotonic() > wait_deadline:
            _log(
                f"ERROR no deploy targets within {args_cli.wait_packets_s:.0f}s "
                f"(packets={int(term._packet_count)}); aborting"
            )
            env.close()
            return 3
    _log(f"deploy targets flowing (packets={int(term._packet_count)}); start locked recording")

    recorder = StepRecorder()

    # ---- 阶段 1：锁根跟随 ----
    locked_steps = max(1, int(round(args_cli.locked_seconds / step_dt)))
    for _ in range(locked_steps):
        paced_step()
        recorder.record(term, asset, joint_ids, env_origin_z, phase=0)

    unlocked = False
    if not args_cli.no_unlock:
        # ---- 阶段 2：程序化解锁 → 自由根闭环 ----
        _log("unlocking root pose (programmatic U)")
        term.unlock_root_pose()
        unlocked = True
        free_steps = max(1, int(round(args_cli.free_seconds / step_dt)))
        for _ in range(free_steps):
            paced_step()
            recorder.record(term, asset, joint_ids, env_origin_z, phase=1)

    # ---- 汇总 ----
    wall = np.asarray(wall_dts[10:], dtype=np.float64)  # 掐头：首步含 JIT/加载
    env_hz = 1.0 / np.clip(wall, 1e-6, None)
    meta = {
        "task": args_cli.task,
        "locked_seconds": float(args_cli.locked_seconds),
        "free_seconds": float(args_cli.free_seconds) if unlocked else 0.0,
        "unlocked": unlocked,
        "step_dt": step_dt,
        "settle_steps": settle_steps,
        "unlock_blend_steps": int(term.cfg.unlock_blend_steps),
        "target_rate_limit": float(term.cfg.target_rate_limit_rad_per_step),
        "post_unlock_cap": float(getattr(term.cfg, "post_unlock_rate_limit_max_delta", -1.0)),
        "post_unlock_growth": float(getattr(term.cfg, "post_unlock_rate_limit_growth_steps", -1.0)),
        "env_hz_mean": float(env_hz.mean()),
        "env_hz_p5": float(np.percentile(env_hz, 5)),
        "packets_total": int(term._packet_count),
        "sonic_env": {key: os.environ.get(key, "") for key in sorted(_SONIC_ENV_DEFAULTS)},
    }
    _log(
        f"done; env_hz mean={meta['env_hz_mean']:.1f} p5={meta['env_hz_p5']:.1f} "
        f"packets={meta['packets_total']} unlocked={unlocked}"
    )
    recorder.save(args_cli.out, joint_names, meta)
    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
