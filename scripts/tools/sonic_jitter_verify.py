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
import hashlib
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
    # 评测默认空场景（纯 G1+地面+天光）：SonicSolo 的抱箱台/纸箱在场时，自由根段
    # 漂移/行走 10m+ 会撞上台座摔倒，污染稳定性判据（实测含台座 3/1/0 摔，
    # 空场景 0 摔、healthy 0.94→0.99）。场景侧默认仍是「开」，只有本评测运行器
    # 关掉；想在评测里保留演示物体显式传 SONIC_SOLO_DEMO_OBJECT=1 即可。
    "SONIC_SOLO_DEMO_OBJECT": "0",
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
parser.add_argument(
    "--seed",
    type=int,
    default=20260716,
    help="Isaac environment/reset RNG seed. Matrix runs pair this value within each block.",
)
parser.add_argument(
    "--stop_after_fall_grace_s",
    "--stop-after-fall-grace-s",
    dest="stop_after_fall_grace_s",
    type=float,
    default=0.0,
    help="After a definitive fall, record this grace period then end early; 0 records the full horizon.",
)
parser.add_argument(
    "--unlock_source_index",
    "--unlock-source-index",
    dest="unlock_source_index",
    type=int,
    default=-1,
    help="Keep root locked until the received deploy source counter reaches this value; -1 disables.",
)
parser.add_argument(
    "--unlock_source_wait_s",
    "--unlock-source-wait-s",
    dest="unlock_source_wait_s",
    type=float,
    default=5.0,
    help="Maximum extra locked-root wait for --unlock-source-index.",
)
parser.add_argument("--no_unlock", action="store_true", help="Skip the unlock phase (locked-only run).")
parser.add_argument(
    "--wait_packets_s", type=float, default=180.0,
    help="Max wall seconds to wait for deploy targets before giving up (exit code 3).",
)
parser.add_argument(
    "--warmup_packets", type=int, default=25,
    help="Valid deploy targets to consume before locked recording starts (~0.5s at 50Hz).",
)
parser.add_argument(
    "--max_target_stale_s",
    "--max-target-stale-s",
    dest="max_target_stale_s",
    type=float,
    default=0.10,
    help="Fresh-target threshold used to compute stale-step fraction; <=0 disables this soft gate.",
)
parser.add_argument(
    "--max_stale_fraction",
    "--max-stale-fraction",
    dest="max_stale_fraction",
    type=float,
    default=0.02,
    help="Maximum fraction of recorded steps above --max-target-stale-s.",
)
parser.add_argument(
    "--hard_target_stale_s",
    "--hard-target-stale-s",
    dest="hard_target_stale_s",
    type=float,
    default=0.50,
    help="Abort a partial run only after this sustained target outage; <=0 disables hard abort.",
)
parser.add_argument(
    "--min_valid_coverage",
    "--min-valid-coverage",
    dest="min_valid_coverage",
    type=float,
    default=0.80,
    help="Minimum fraction of recorded env steps with a fresh valid target update; <=0 disables this gate.",
)
parser.add_argument(
    "--max_invalid_target_fraction",
    "--max-invalid-target-fraction",
    dest="max_invalid_target_fraction",
    type=float,
    default=0.01,
    help="Maximum invalid payload fraction among classified target packets.",
)
parser.add_argument(
    "--disable_target_gates",
    "--disable-target-gates",
    dest="disable_target_gates",
    action="store_true",
    help="Disable stale/coverage result gates (waiting for an initial valid target is still required).",
)
parser.add_argument(
    "--run_manifest",
    type=str,
    default="",
    help="Optional run manifest JSON object; loaded verbatim into meta['run_manifest'].",
)
parser.add_argument(
    "--runtime_manifest",
    type=str,
    default="",
    help=(
        "Optional runtime component sidecar. It is created after proxy/deploy startup "
        "and loaded when the NPZ metadata is finalized."
    ),
)
parser.add_argument(
    "--status_file",
    "--status-file",
    dest="status_file",
    type=str,
    default="",
    help="Optional atomic JSON sidecar carrying the authoritative runner exit code.",
)
parser.add_argument(
    "--hold_seconds", type=float, default=0.0,
    help="After measurement, keep stepping in realtime for N seconds without recording "
    "(GUI observation; close the window or Ctrl+C to end early).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not 0.0 <= float(args_cli.max_stale_fraction) <= 1.0:
    parser.error("--max-stale-fraction must be within [0, 1]")
if not 0.0 <= float(args_cli.max_invalid_target_fraction) <= 1.0:
    parser.error("--max-invalid-target-fraction must be within [0, 1]")
if float(args_cli.stop_after_fall_grace_s) < 0.0:
    parser.error("--stop-after-fall-grace-s must be >= 0")
if float(args_cli.unlock_source_wait_s) <= 0.0:
    parser.error("--unlock-source-wait-s must be > 0")
if (
    float(args_cli.hard_target_stale_s) > 0.0
    and float(args_cli.max_target_stale_s) > 0.0
    and float(args_cli.hard_target_stale_s) < float(args_cli.max_target_stale_s)
):
    parser.error("--hard-target-stale-s must be >= --max-target-stale-s")

_RUN_MANIFEST: dict = {}
_RUN_MANIFEST_PATH = ""
if args_cli.run_manifest:
    _RUN_MANIFEST_PATH = os.path.abspath(os.path.expanduser(args_cli.run_manifest))
    try:
        with open(_RUN_MANIFEST_PATH, encoding="utf-8") as manifest_file:
            _RUN_MANIFEST = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"--run_manifest could not be loaded from {_RUN_MANIFEST_PATH!r}: {exc}")
    if not isinstance(_RUN_MANIFEST, dict):
        parser.error(
            f"--run_manifest must contain a JSON object, got {type(_RUN_MANIFEST).__name__}"
        )
_RUNTIME_MANIFEST_PATH = (
    os.path.abspath(os.path.expanduser(args_cli.runtime_manifest))
    if args_cli.runtime_manifest
    else ""
)


def _load_runtime_manifest() -> dict:
    if not _RUNTIME_MANIFEST_PATH:
        return {}
    try:
        with open(_RUNTIME_MANIFEST_PATH, encoding="utf-8") as runtime_file:
            payload = json.load(runtime_file)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1,
            "valid": False,
            "reasons": [f"runtime_manifest_unreadable={exc}"],
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": 1,
            "valid": False,
            "reasons": [
                f"runtime_manifest_not_object={type(payload).__name__}"
            ],
        }
    return payload

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
import isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions as sonic_actions_module
from isaaclab_tasks.manager_based.locomanipulation.pick_place import (
    locomanipulation_g1_env_cfg as sonic_env_cfg_module,
)

from isaaclab_tasks.utils import parse_env_cfg


def _log(message: str) -> None:
    print(f"[JitterVerify] {message}", flush=True)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StepRecorder:
    """逐 env 步累积原始序列，最后一次性 np.stack 落盘。"""

    def __init__(self):
        self.wall_t: list[float] = []
        self.wall_time_unix_s: list[float] = []
        self.control_state: list[int] = []  # 0=locked, 1=handover, 2=true-free, 3=recovery
        self.phase: list[int] = []  # Compatibility: 1 only for true-free control_state=2.
        self.q: list[np.ndarray] = []
        self.dq: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.reference: list[np.ndarray] = []
        self.reference_valid: list[bool] = []
        self.step_delta: list[float] = []
        self.root_pos: list[np.ndarray] = []
        self.root_quat: list[np.ndarray] = []
        self.tilt_deg: list[float] = []
        self.packet_count: list[int] = []
        self.valid_target_count: list[int] = []
        self.invalid_target_count: list[int] = []
        self.target_age_s: list[float] = []
        self.recovery_count: list[int] = []
        self.source_index: list[int] = []
        self.source_timestamp: list[float] = []

    def record(self, term, asset, joint_ids, env_origin_z: float) -> None:
        data = asset.data
        self.wall_t.append(time.monotonic())
        self.wall_time_unix_s.append(time.time())
        control_state = int(term.control_state)
        self.control_state.append(control_state)
        self.phase.append(1 if control_state == 2 else 0)
        self.q.append(data.joint_pos[0, joint_ids].detach().cpu().numpy().copy())
        self.dq.append(data.joint_vel[0, joint_ids].detach().cpu().numpy().copy())
        self.target.append(term.processed_actions[0].detach().cpu().numpy().copy())
        reference = term._last_payload_reference_target
        if reference is None:
            self.reference.append(np.full(len(joint_ids), np.nan, dtype=np.float32))
            self.reference_valid.append(False)
        else:
            self.reference.append(reference[0].detach().cpu().numpy().astype(np.float32, copy=True))
            self.reference_valid.append(True)
        self.step_delta.append(float(term._last_target_step_delta_absmax[0].item()))
        root_pos = data.root_pos_w[0].detach().cpu().numpy().copy()
        root_pos[2] -= env_origin_z
        self.root_pos.append(root_pos)
        self.root_quat.append(data.root_quat_w[0].detach().cpu().numpy().copy())
        gb = data.projected_gravity_b[0]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(gb[2].item())))))
        self.tilt_deg.append(tilt)
        self.packet_count.append(int(term._packet_count))
        self.valid_target_count.append(int(term._valid_target_count))
        self.invalid_target_count.append(int(term._invalid_target_count))
        self.target_age_s.append(float(term.target_age_s))
        self.recovery_count.append(int(term._recovery_count))
        self.source_index.append(int(term._last_source_index))
        self.source_timestamp.append(float(term._last_source_timestamp))

    def save(self, path: str, joint_names: list[str], meta: dict) -> None:
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)
        packet_count = np.asarray(self.packet_count, dtype=np.int64)
        np.savez_compressed(
            path,
            wall_t=np.asarray(self.wall_t, dtype=np.float64),
            wall_time_unix_s=np.asarray(self.wall_time_unix_s, dtype=np.float64),
            control_state=np.asarray(self.control_state, dtype=np.int8),
            phase=np.asarray(self.phase, dtype=np.int8),
            q=np.stack(self.q).astype(np.float32),
            dq=np.stack(self.dq).astype(np.float32),
            target=np.stack(self.target).astype(np.float32),
            reference=np.stack(self.reference).astype(np.float32),
            reference_valid=np.asarray(self.reference_valid, dtype=np.bool_),
            step_delta=np.asarray(self.step_delta, dtype=np.float32),
            root_pos=np.stack(self.root_pos).astype(np.float32),
            root_quat=np.stack(self.root_quat).astype(np.float32),
            tilt_deg=np.asarray(self.tilt_deg, dtype=np.float32),
            packet_count=packet_count,
            packets=packet_count,  # Backward-compatible alias.
            valid_target_count=np.asarray(self.valid_target_count, dtype=np.int64),
            invalid_target_count=np.asarray(self.invalid_target_count, dtype=np.int64),
            target_age_s=np.asarray(self.target_age_s, dtype=np.float64),
            recovery_count=np.asarray(self.recovery_count, dtype=np.int64),
            source_index=np.asarray(self.source_index, dtype=np.int64),
            source_timestamp=np.asarray(self.source_timestamp, dtype=np.float64),
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
    gr00t_43dof_realpath = os.path.realpath(
        str(sonic_env_cfg_module.G1_43DOF_GR00T_CFG.spawn.usd_path)
    )
    requested_gr00t_43dof = os.environ.get("SONIC_GR00T_43DOF_USD")
    if requested_gr00t_43dof and gr00t_43dof_realpath != os.path.realpath(
        os.path.expanduser(requested_gr00t_43dof)
    ):
        raise RuntimeError(
            "configured GR00T 43-DoF import asset does not match "
            "SONIC_GR00T_43DOF_USD: "
            f"{gr00t_43dof_realpath!r} != {requested_gr00t_43dof!r}"
        )
    gr00t_43dof_import_asset = {
        "realpath": gr00t_43dof_realpath,
        "sha256": _sha256_file(gr00t_43dof_realpath),
    }
    _log(
        "gr00t_43dof_import_asset="
        f"{gr00t_43dof_import_asset['realpath']} "
        f"sha256={gr00t_43dof_import_asset['sha256']}"
    )
    env_cfg.seed = int(args_cli.seed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=int(args_cli.seed))

    unwrapped = env.unwrapped
    term = unwrapped.action_manager.get_term("sonic_wholebody")
    asset = unwrapped.scene[term.cfg.asset_name]
    scene_robot_cfg = getattr(unwrapped.cfg.scene, term.cfg.asset_name)
    scene_robot_usd_path = getattr(scene_robot_cfg.spawn, "usd_path", None)
    if not isinstance(scene_robot_usd_path, str) or not scene_robot_usd_path:
        raise RuntimeError(
            f"scene asset {term.cfg.asset_name!r} has no auditable USD path"
        )
    scene_robot_asset = {
        "asset_name": str(term.cfg.asset_name),
        "prim_path": str(scene_robot_cfg.prim_path),
        "usd_path": scene_robot_usd_path,
    }
    _log(
        "scene_robot_asset="
        f"{scene_robot_asset['asset_name']} "
        f"prim={scene_robot_asset['prim_path']} "
        f"usd={scene_robot_asset['usd_path']}"
    )
    joint_ids = term._joint_ids
    joint_names = list(term._joint_names)
    env_origin_z = float(unwrapped.scene.env_origins[0, 2].item())
    step_dt = float(unwrapped.step_dt)

    if unwrapped.sim.has_gui():
        # 观察模式：把 GUI 相机摆到机器人斜前方（出生朝向 +X），对准躯干。
        try:
            root = asset.data.root_pos_w[0].detach().cpu().tolist()
            eye = (root[0] + 2.6, root[1] - 2.0, root[2] + 0.9)
            target = (root[0], root[1], root[2] + 0.15)
            unwrapped.sim.set_camera_view(eye, target)
            _log(f"GUI camera placed eye=({eye[0]:+.2f},{eye[1]:+.2f},{eye[2]:+.2f})")
        except Exception as exc:
            _log(f"GUI camera placement skipped: {exc}")
    _log(
        f"env ready; step_dt={step_dt:.4f}s joints={len(joint_ids)} "
        f"seed={int(args_cli.seed)} actual_seed={unwrapped.cfg.seed} "
        f"settle_steps={int(term.cfg.startup_settle_steps)} "
        f"unlock_blend_steps={int(term.cfg.unlock_blend_steps)} "
        f"rate_limit={float(term.cfg.target_rate_limit_rad_per_step):.4f}"
    )

    zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    next_tick = time.monotonic()
    wall_dts: list[float] = []
    last_step_wall = time.monotonic()

    # ---- 锁相/测量节拍（SONIC_ENV_PHASE_LOCK，sim2sim 桥 3.4Hz 环路延迟共振治理）----
    # off：原始固定墙钟栅格（基线，默认）。
    # measure：栅格不变，只把整段睡眠切成 ≤1ms 片轮询目标 socket，记录 deploy
    #   目标包到达时刻（验证相位抽签+拍频、env.step slack、每拍包数三个前提）。
    # lock：睡到新目标包到达+δ 再步进，栅格重锚到包到达时刻——把 deploy→Isaac
    #   下半环相位抽签(0~20ms)钉成 ~δ 常数；超时(max_slip)回退原栅格并计 miss，
    #   任何时刻不劣于基线。仍是严格 1.0× 实时（deploy tick 本身就是 50Hz 墙钟）。
    phase_lock_mode = os.environ.get("SONIC_ENV_PHASE_LOCK", "0").strip().lower()
    if phase_lock_mode in ("", "0", "false", "no", "off"):
        phase_lock_mode = "off"
    elif phase_lock_mode in ("1", "true", "yes", "on", "lock"):
        phase_lock_mode = "lock"
    elif phase_lock_mode != "measure":
        raise SystemExit(f"invalid SONIC_ENV_PHASE_LOCK={phase_lock_mode!r} (expect 0/1/measure)")
    _PHASE_POLL_SLICE_S = 0.001
    _PHASE_LOCK_DELTA_S = float(os.environ.get("SONIC_ENV_PHASE_LOCK_DELTA_S", "0.002"))
    _PHASE_LOCK_MAX_SLIP_S = float(os.environ.get("SONIC_ENV_PHASE_LOCK_MAX_SLIP_S", "0.005"))
    phase_step_count = 0
    phase_arrival_step: list[int] = []
    phase_arrival_t: list[float] = []
    phase_step_compute_s: list[float] = []
    phase_lock_wait_s: list[float] = []
    phase_lock_hits = 0
    phase_lock_miss = 0
    if phase_lock_mode != "off":
        _log(
            f"env phase-lock mode={phase_lock_mode} "
            f"delta={_PHASE_LOCK_DELTA_S * 1000:.1f}ms max_slip={_PHASE_LOCK_MAX_SLIP_S * 1000:.1f}ms"
        )

    def paced_step() -> None:
        nonlocal next_tick, last_step_wall, phase_step_count, phase_lock_hits, phase_lock_miss
        next_tick += step_dt
        if phase_lock_mode == "off":
            now = time.monotonic()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -1.0:
                # 长停顿（场景加载等）后重新对表，不追帧
                next_tick = time.monotonic()
        elif phase_lock_mode == "measure":
            # 栅格与基线完全一致；睡眠切片轮询仅提高到达时刻观测精度（~1ms）。
            while True:
                arrival = term.poll_fresh_target()
                if arrival is not None:
                    phase_arrival_step.append(phase_step_count)
                    phase_arrival_t.append(arrival)
                now = time.monotonic()
                remaining = next_tick - now
                if remaining <= 0.0:
                    if remaining < -1.0:
                        next_tick = now
                    break
                time.sleep(min(_PHASE_POLL_SLICE_S, remaining))
        else:  # lock
            wait_start = time.monotonic()
            deadline = next_tick + _PHASE_LOCK_MAX_SLIP_S
            while True:
                arrival = term.poll_fresh_target()
                now = time.monotonic()
                if arrival is not None:
                    phase_arrival_step.append(phase_step_count)
                    phase_arrival_t.append(arrival)
                    # 重锚：本步起点钉在包到达+δ，消灭消费时刻的相位抽签。
                    next_tick = arrival + _PHASE_LOCK_DELTA_S
                    if next_tick > now:
                        time.sleep(next_tick - now)
                    phase_lock_hits += 1
                    phase_lock_wait_s.append(time.monotonic() - wait_start)
                    break
                if now >= deadline:
                    # 超时兜底：按原栅格推进（+max_slip 恒定偏移，不累积）。
                    phase_lock_miss += 1
                    if next_tick - now < -1.0:
                        next_tick = now
                    break
                time.sleep(min(_PHASE_POLL_SLICE_S, max(deadline - now, 0.0)))
        compute_start = time.monotonic()
        with torch.inference_mode():
            env.step(zero_actions)
        now = time.monotonic()
        phase_step_compute_s.append(now - compute_start)
        wall_dts.append(now - last_step_wall)
        last_step_wall = now
        phase_step_count += 1
        if phase_lock_mode != "off" and phase_step_count % 500 == 0:
            recent = np.asarray(phase_arrival_t[-250:], dtype=np.float64)
            if recent.size:
                offsets_ms = np.mod(recent, step_dt) * 1000.0
                offset_text = (
                    f"arrival_mod20ms recent min/med/max="
                    f"{offsets_ms.min():.1f}/{np.median(offsets_ms):.1f}/{offsets_ms.max():.1f}ms"
                )
            else:
                offset_text = "arrival_mod20ms recent n=0"
            compute_ms = np.asarray(phase_step_compute_s[-500:], dtype=np.float64) * 1000.0
            _log(
                f"phase[{phase_lock_mode}] step={phase_step_count} {offset_text} "
                f"compute p50/p95={np.percentile(compute_ms, 50):.1f}/{np.percentile(compute_ms, 95):.1f}ms "
                f"hits={phase_lock_hits} miss={phase_lock_miss}"
            )

    def save_phase_telemetry() -> None:
        if phase_lock_mode == "off":
            return
        base, _ = os.path.splitext(os.path.abspath(args_cli.out))
        phase_path = base + ".phase.npz"
        np.savez_compressed(
            phase_path,
            mode=np.asarray(phase_lock_mode),
            step_dt=np.float64(step_dt),
            lock_delta_s=np.float64(_PHASE_LOCK_DELTA_S),
            lock_max_slip_s=np.float64(_PHASE_LOCK_MAX_SLIP_S),
            arrival_step=np.asarray(phase_arrival_step, dtype=np.int64),
            arrival_t=np.asarray(phase_arrival_t, dtype=np.float64),
            step_compute_s=np.asarray(phase_step_compute_s, dtype=np.float64),
            lock_wait_s=np.asarray(phase_lock_wait_s, dtype=np.float64),
            lock_hits=np.int64(phase_lock_hits),
            lock_miss=np.int64(phase_lock_miss),
        )
        _log(f"phase telemetry saved: {phase_path}")

    target_gates_enabled = not bool(args_cli.disable_target_gates)
    _log(
        "target gates "
        f"enabled={target_gates_enabled} "
        f"fresh_threshold={float(args_cli.max_target_stale_s):.3f}s "
        f"max_stale_fraction={float(args_cli.max_stale_fraction):.3f} "
        f"hard_stale={float(args_cli.hard_target_stale_s):.3f}s "
        f"min_valid_coverage={float(args_cli.min_valid_coverage):.3f} "
        f"max_invalid_fraction={float(args_cli.max_invalid_target_fraction):.3f}"
    )

    free_steps_recorded = 0
    termination_reason = "not_started"
    unlock_source_index_actual = -1

    def build_meta(*, status: str, unlocked: bool, target_gate: dict) -> dict:
        wall_samples = wall_dts[10:] if len(wall_dts) > 10 else wall_dts
        if wall_samples:
            env_hz = 1.0 / np.clip(np.asarray(wall_samples, dtype=np.float64), 1.0e-6, None)
            env_hz_mean = float(env_hz.mean())
            env_hz_p5 = float(np.percentile(env_hz, 5))
        else:
            env_hz_mean = math.nan
            env_hz_p5 = math.nan
        return {
            "schema_version": 3,
            "status": status,
            "task": args_cli.task,
            "seed_requested": int(args_cli.seed),
            "seed_actual": int(unwrapped.cfg.seed),
            "locked_seconds": float(args_cli.locked_seconds),
            "free_seconds": float(args_cli.free_seconds) if unlocked else 0.0,
            "planned_free_seconds": float(args_cli.free_seconds) if unlocked else 0.0,
            "recorded_free_steps": int(free_steps_recorded),
            "recorded_free_seconds": float(free_steps_recorded * step_dt),
            "termination_reason": termination_reason,
            "stop_after_fall_grace_s": float(args_cli.stop_after_fall_grace_s),
            "unlock_source_index_target": int(args_cli.unlock_source_index),
            "unlock_source_index_actual": int(unlock_source_index_actual),
            "fall_detection": {"tilt_deg_gt": 45.0, "root_z_m_lt": 0.35},
            "unlocked": unlocked,
            "step_dt": step_dt,
            "settle_steps": int(term.cfg.startup_settle_steps),
            "unlock_blend_steps": int(term.cfg.unlock_blend_steps),
            "target_rate_limit": float(term.cfg.target_rate_limit_rad_per_step),
            "post_unlock_cap": float(getattr(term.cfg, "post_unlock_rate_limit_max_delta", -1.0)),
            "post_unlock_growth": float(getattr(term.cfg, "post_unlock_rate_limit_growth_steps", -1.0)),
            "env_hz_mean": env_hz_mean,
            "env_hz_p5": env_hz_p5,
            "packets_total": int(term._packet_count),
            "valid_targets_total": int(term._valid_target_count),
            "invalid_targets_total": int(term._invalid_target_count),
            "recoveries_total": int(term._recovery_count),
            "source_index_field": str(term._last_source_index_field),
            "source_timestamp_field": str(term._last_source_timestamp_field),
            "control_state_encoding": {
                "locked": 0,
                "handover_or_blend": 1,
                "true_free": 2,
                "recovery": 3,
            },
            "target_gate": target_gate,
            "phase_lock": {
                "mode": phase_lock_mode,
                "delta_s": _PHASE_LOCK_DELTA_S,
                "max_slip_s": _PHASE_LOCK_MAX_SLIP_S,
                "hits": int(phase_lock_hits),
                "misses": int(phase_lock_miss),
                "arrivals_recorded": len(phase_arrival_t),
            },
            "run_manifest_path": _RUN_MANIFEST_PATH,
            "run_manifest": _RUN_MANIFEST,
            "runtime_manifest_path": _RUNTIME_MANIFEST_PATH,
            "runtime_components": _load_runtime_manifest(),
            "gr00t_43dof_import_asset": gr00t_43dof_import_asset,
            "scene_robot_asset": scene_robot_asset,
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "isaaclab_tasks_file": os.path.realpath(str(isaaclab_tasks.__file__)),
            "actions_module_file": os.path.realpath(str(sonic_actions_module.__file__)),
            "sonic_env": {
                key: value for key, value in sorted(os.environ.items()) if key.startswith("SONIC_")
            },
        }

    # ---- 阶段 0：等 deploy 目标流入（期间照常推进：settle 走完、5560 状态持续发布）----
    _log("waiting for valid deploy targets (start proxy/deploy now; press ']' in deploy)")
    wait_deadline = time.monotonic() + float(args_cli.wait_packets_s)
    settle_steps = int(term.cfg.startup_settle_steps)
    wait_recorder = StepRecorder()
    wait_failure = ""
    while True:
        if not simulation_app.is_running():
            wait_failure = "simulation stopped while waiting for a valid deploy target"
            break
        paced_step()
        wait_recorder.record(term, asset, joint_ids, env_origin_z)
        settle_done = int(term._settle_step_counter) >= settle_steps
        if settle_done and int(term._valid_target_count) >= int(args_cli.warmup_packets):
            break
        if time.monotonic() > wait_deadline:
            wait_failure = (
                f"no valid deploy targets within {args_cli.wait_packets_s:.0f}s "
                f"(packets={int(term._packet_count)} valid={int(term._valid_target_count)} "
                f"invalid={int(term._invalid_target_count)})"
            )
            break

    if wait_failure:
        _log(f"ERROR {wait_failure}; saving diagnostic recording")
        wait_valid_counts = np.asarray(wait_recorder.valid_target_count, dtype=np.int64)
        if wait_valid_counts.size:
            wait_previous_counts = np.concatenate(
                [np.zeros(1, dtype=np.int64), wait_valid_counts[:-1]]
            )
            wait_valid_coverage = float(np.mean(wait_valid_counts > wait_previous_counts))
        else:
            wait_valid_coverage = 0.0
        wait_classified = int(term._valid_target_count + term._invalid_target_count)
        target_gate = {
            "enabled": target_gates_enabled,
            "passed": False,
            "failures": [wait_failure],
            "max_target_stale_s": float(args_cli.max_target_stale_s),
            "max_stale_fraction": float(args_cli.max_stale_fraction),
            "hard_target_stale_s": float(args_cli.hard_target_stale_s),
            "min_valid_coverage": float(args_cli.min_valid_coverage),
            "max_invalid_target_fraction": float(args_cli.max_invalid_target_fraction),
            "valid_updates": int(term._valid_target_count),
            "invalid_updates": int(term._invalid_target_count),
            "valid_coverage": wait_valid_coverage,
            "payload_valid_ratio": (
                float(term._valid_target_count)
                / max(float(wait_classified), 1.0)
            ),
            "max_observed_target_age_s": (
                float(np.max(wait_recorder.target_age_s)) if wait_recorder.target_age_s else math.inf
            ),
        }
        if wait_recorder.wall_t:
            meta = build_meta(status="invalid_no_target", unlocked=False, target_gate=target_gate)
            wait_recorder.save(args_cli.out, joint_names, meta)
        save_phase_telemetry()
        env.close()
        return 3

    _log(
        f"deploy targets flowing (packets={int(term._packet_count)} "
        f"valid={int(term._valid_target_count)} invalid={int(term._invalid_target_count)}); "
        "start locked recording"
    )

    recorder = StepRecorder()
    count_baseline = {
        "packets": int(term._packet_count),
        "valid": int(term._valid_target_count),
        "invalid": int(term._invalid_target_count),
        "recoveries": int(term._recovery_count),
    }
    abort_reason = ""
    runtime_failure = ""

    def record_and_check_target_health() -> None:
        nonlocal abort_reason
        recorder.record(term, asset, joint_ids, env_origin_z)
        if (
            target_gates_enabled
            and float(args_cli.hard_target_stale_s) > 0.0
            and recorder.target_age_s[-1] > float(args_cli.hard_target_stale_s)
        ):
            abort_reason = (
                f"target stale for {recorder.target_age_s[-1]:.3f}s "
                f"(hard limit {float(args_cli.hard_target_stale_s):.3f}s)"
            )

    # ---- 阶段 1：锁根跟随 ----
    locked_steps = max(1, int(round(args_cli.locked_seconds / step_dt)))
    for _ in range(locked_steps):
        if not simulation_app.is_running():
            runtime_failure = "simulation stopped during locked measurement"
            break
        paced_step()
        record_and_check_target_health()
        if abort_reason:
            _log(f"ERROR {abort_reason}; stopping measurement and saving partial recording")
            break

    if (
        not abort_reason
        and not runtime_failure
        and not args_cli.no_unlock
        and int(args_cli.unlock_source_index) >= 0
        and simulation_app.is_running()
    ):
        target_source_index = int(args_cli.unlock_source_index)
        alignment_deadline = time.monotonic() + float(args_cli.unlock_source_wait_s)
        _log(
            f"aligning unlock under locked root: source_index>={target_source_index}"
        )
        while int(term._last_source_index) < target_source_index:
            if not simulation_app.is_running():
                runtime_failure = "simulation stopped while aligning unlock source index"
                break
            if time.monotonic() > alignment_deadline:
                runtime_failure = (
                    f"unlock source index did not reach {target_source_index} within "
                    f"{float(args_cli.unlock_source_wait_s):.1f}s "
                    f"(actual={int(term._last_source_index)})"
                )
                break
            paced_step()
            record_and_check_target_health()
            if abort_reason:
                termination_reason = "target_health_abort"
                break
        unlock_source_index_actual = int(term._last_source_index)
        if not runtime_failure and not abort_reason:
            _log(f"unlock source aligned at index={unlock_source_index_actual}")

    unlocked = False
    if not abort_reason and not runtime_failure and not args_cli.no_unlock and simulation_app.is_running():
        # ---- 阶段 2：程序化解锁 → 自由根闭环 ----
        _log("unlocking root pose (programmatic U)")
        term.unlock_root_pose()
        unlocked = True
        termination_reason = "planned_duration_complete"
        free_steps = max(1, int(round(args_cli.free_seconds / step_dt)))
        first_fall_wall_t: float | None = None
        for _ in range(free_steps):
            if not simulation_app.is_running():
                runtime_failure = "simulation stopped during free-root measurement"
                termination_reason = "simulation_stopped"
                break
            paced_step()
            record_and_check_target_health()
            free_steps_recorded += 1
            if abort_reason:
                termination_reason = "target_health_abort"
                _log(f"ERROR {abort_reason}; stopping measurement and saving partial recording")
                break
            fallen_now = (
                recorder.tilt_deg[-1] > 45.0
                or recorder.root_pos[-1][2] < 0.35
            )
            if fallen_now and first_fall_wall_t is None:
                first_fall_wall_t = recorder.wall_t[-1]
                _log(
                    f"definitive fall observed; grace={float(args_cli.stop_after_fall_grace_s):.3f}s"
                )
            if (
                first_fall_wall_t is not None
                and float(args_cli.stop_after_fall_grace_s) > 0.0
                and recorder.wall_t[-1] - first_fall_wall_t
                >= float(args_cli.stop_after_fall_grace_s)
            ):
                termination_reason = "fall_observed"
                _log("fall grace recorded; ending this trial early")
                break

    if (
        not abort_reason
        and not runtime_failure
        and args_cli.hold_seconds > 0
        and simulation_app.is_running()
    ):
        # ---- 阶段 3（可选，GUI 观察）：继续实时推进但不记录 ----
        _log(f"measurement done; holding {args_cli.hold_seconds:.0f}s for observation (Ctrl+C to end)")
        hold_steps = int(round(args_cli.hold_seconds / step_dt))
        try:
            for _ in range(hold_steps):
                if not simulation_app.is_running():
                    break
                paced_step()
        except KeyboardInterrupt:
            _log("observation interrupted by user")

    # ---- 汇总 ----
    valid_updates = int(term._valid_target_count) - count_baseline["valid"]
    invalid_updates = int(term._invalid_target_count) - count_baseline["invalid"]
    packet_updates = int(term._packet_count) - count_baseline["packets"]
    recovery_updates = int(term._recovery_count) - count_baseline["recoveries"]
    classified_updates = valid_updates + invalid_updates
    payload_valid_ratio = (
        float(valid_updates) / float(classified_updates) if classified_updates > 0 else 0.0
    )
    recorded_valid_counts = np.asarray(recorder.valid_target_count, dtype=np.int64)
    if recorded_valid_counts.size:
        previous_valid_counts = np.concatenate(
            [np.asarray([count_baseline["valid"]], dtype=np.int64), recorded_valid_counts[:-1]]
        )
        valid_update_mask = recorded_valid_counts > previous_valid_counts
        valid_update_steps = int(np.count_nonzero(valid_update_mask))
        valid_coverage = float(np.mean(valid_update_mask))
    else:
        valid_update_steps = 0
        valid_coverage = 0.0
    observed_ages = np.asarray(recorder.target_age_s, dtype=np.float64)
    max_observed_age = float(np.max(observed_ages)) if observed_ages.size else math.inf
    stale_fraction = (
        float(np.mean(observed_ages > float(args_cli.max_target_stale_s)))
        if observed_ages.size and float(args_cli.max_target_stale_s) > 0.0
        else 0.0
    )
    failures: list[str] = []
    if runtime_failure:
        failures.append(runtime_failure)
    if abort_reason:
        failures.append(abort_reason)
    if target_gates_enabled:
        if valid_updates <= 0:
            failures.append("no valid target update during the measurement window")
        if (
            float(args_cli.min_valid_coverage) > 0.0
            and valid_coverage < float(args_cli.min_valid_coverage)
        ):
            failures.append(
                f"valid target coverage {valid_coverage:.3f} is below "
                f"{float(args_cli.min_valid_coverage):.3f}"
            )
        invalid_target_fraction = 1.0 - payload_valid_ratio
        if (
            classified_updates > 0
            and invalid_target_fraction > float(args_cli.max_invalid_target_fraction)
        ):
            failures.append(
                f"invalid target fraction {invalid_target_fraction:.3f} exceeds "
                f"{float(args_cli.max_invalid_target_fraction):.3f}"
            )
        if (
            float(args_cli.max_target_stale_s) > 0.0
            and stale_fraction > float(args_cli.max_stale_fraction)
        ):
            failures.append(
                f"stale target fraction {stale_fraction:.3f} exceeds "
                f"{float(args_cli.max_stale_fraction):.3f} "
                f"(fresh threshold {float(args_cli.max_target_stale_s):.3f}s)"
            )
    target_gate = {
        "enabled": target_gates_enabled,
        "passed": not failures,
        "failures": failures,
        "max_target_stale_s": float(args_cli.max_target_stale_s),
        "max_stale_fraction": float(args_cli.max_stale_fraction),
        "hard_target_stale_s": float(args_cli.hard_target_stale_s),
        "min_valid_coverage": float(args_cli.min_valid_coverage),
        "max_invalid_target_fraction": float(args_cli.max_invalid_target_fraction),
        "packet_updates": packet_updates,
        "valid_updates": valid_updates,
        "invalid_updates": invalid_updates,
        "valid_update_steps": valid_update_steps,
        "valid_coverage": valid_coverage,
        "payload_valid_ratio": payload_valid_ratio,
        "max_observed_target_age_s": max_observed_age,
        "stale_step_fraction": stale_fraction,
        "recovery_updates": recovery_updates,
    }
    status = "ok" if not failures else "invalid_target_gate"
    meta = build_meta(status=status, unlocked=unlocked, target_gate=target_gate)
    _log(
        f"done; env_hz mean={meta['env_hz_mean']:.1f} p5={meta['env_hz_p5']:.1f} "
        f"packets={meta['packets_total']} valid_coverage={valid_coverage:.3f} "
        f"max_target_age={max_observed_age:.3f}s recoveries={recovery_updates} "
        f"gate_passed={target_gate['passed']} unlocked={unlocked}"
    )
    if recorder.wall_t:
        recorder.save(args_cli.out, joint_names, meta)
    else:
        _log("ERROR measurement produced no frames; no NPZ could be saved")
        failures.append("measurement produced no frames")
    save_phase_telemetry()
    env.close()
    return 0 if not failures else 5


def _write_status_file(*, exit_code: int, completed: bool, error: str = "") -> None:
    if not args_cli.status_file:
        return
    path = os.path.abspath(os.path.expanduser(args_cli.status_file))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    payload = {
        "schema_version": 1,
        "completed": bool(completed),
        "exit_code": int(exit_code),
        "error": str(error),
        "finished_unix_s": time.time(),
        "out": os.path.abspath(os.path.expanduser(args_cli.out)),
    }
    try:
        with open(temporary, "w", encoding="utf-8") as status_file:
            json.dump(payload, status_file, ensure_ascii=False, indent=2, sort_keys=True)
            status_file.write("\n")
            status_file.flush()
            os.fsync(status_file.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        _log(f"ERROR could not write status sidecar {path}: {exc}")
        try:
            os.unlink(temporary)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        exit_code = main()
        _write_status_file(exit_code=exit_code, completed=True)
    except BaseException as exc:
        exception_code = (
            int(exc.code)
            if isinstance(exc, SystemExit) and isinstance(exc.code, int)
            else 130 if isinstance(exc, KeyboardInterrupt) else 1
        )
        _write_status_file(
            exit_code=exception_code,
            completed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        simulation_app.close()
        raise
    simulation_app.close()
    sys.exit(exit_code)
