# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC deploy g1_debug 流录制器（sim 后端无关，MuJoCo/Isaac 对拍取数用）。

订阅 deploy 的 ZMQ debug PUB（单帧 [topic字节][msgpack字节]，50Hz/控制tick），
把 policy 输出目标（last_action，含 action_scale+default_angles，绝对关节角）与
lowstate 实测关节角（body_q/body_dq，MuJoCo 序）逐包落盘为与
sonic_jitter_verify.py 同键名的 npz —— sonic_jitter_by_group.py /
sonic_jitter_report.py 可直接吃。

可选 --base_pose_udp_port：MuJoCo 侧 base_sim.py 的浮动基座地面真值
（JSON: sim_time_s/base_pos/base_quat_wxyz/fall，SONIC_SIM_BASE_POSE_PORT 开启），
用于漂移/tilt 对比（g1_debug 里没有基座平移）。

依赖 pyzmq + msgpack + numpy：用 sony 仓库 .venv_teleop/bin/python 跑最稳。

用法:
    python sonic_debug_stream_recorder.py --endpoint tcp://127.0.0.1:5657 \
        --out /tmp/sonic_jitter/mj_stand.npz --seconds 120 --base_pose_udp_port 5658
"""

import argparse
import json
import math
import socket
import time

import msgpack
import numpy as np
import zmq

# deploy g1_debug 的 body_q/last_action 均为 MuJoCo 序（重映射发生在 deploy C++ 内）。
# 名表照抄 IsaacLab actions.py SONIC_G1_29DOF_MUJOCO_JOINT_ORDER。
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def tilt_deg_from_quat_wxyz(w: float, x: float, y: float, z: float) -> float:
    """机体 z 轴与世界 z 轴夹角（度）。R[2][2] = 1 - 2(x^2 + y^2)。"""
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, r22))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5657")
    parser.add_argument("--topic", default="g1_debug")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--base_pose_udp_port", type=int, default=0, help="0=不录基座真值")
    parser.add_argument(
        "--wait_first_packet_s", type=float, default=60.0,
        help="等首包的超时（deploy 未进 CONTROL 时 g1_debug 不发包）",
    )
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVTIMEO, 200)
    sub.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sub.connect(args.endpoint)
    topic_len = len(args.topic.encode())

    udp = None
    if args.base_pose_udp_port:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("127.0.0.1", args.base_pose_udp_port))
        udp.setblocking(False)

    def log(msg: str) -> None:
        print(f"[StreamRecorder] {msg}", flush=True)

    def drain_base_pose(last):
        if udp is None:
            return last
        while True:
            try:
                raw, _ = udp.recvfrom(65536)
            except BlockingIOError:
                return last
            try:
                last = json.loads(raw.decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

    wall_t, q, dq, target, packets = [], [], [], [], []
    reference = []  # body_q_target（参考动作目标，跟随延迟测量用；包内缺失则整体不落盘）
    root_pos, root_quat, tilt, fall = [], [], [], []
    base = None
    n_bad = 0

    log(f"subscribing {args.endpoint} topic={args.topic}; waiting first packet")
    deadline_first = time.monotonic() + args.wait_first_packet_s
    started = None
    while True:
        try:
            raw = sub.recv()
        except zmq.Again:
            if started is None and time.monotonic() > deadline_first:
                log("ERROR: no g1_debug packet (deploy not in CONTROL?)")
                return 3
            continue
        payload = raw[topic_len:]
        try:
            m = msgpack.unpackb(payload, raw=False)
        except Exception:
            n_bad += 1
            continue
        la, bq = m.get("last_action"), m.get("body_q")
        if la is None or bq is None or len(la) != 29 or len(bq) != 29:
            n_bad += 1
            continue
        if started is None:
            started = time.monotonic()
            log(f"first packet; recording {args.seconds:.0f}s")
        base = drain_base_pose(base)
        wall_t.append(time.monotonic())
        q.append(np.asarray(bq, dtype=np.float32))
        dq.append(np.asarray(m.get("body_dq", [0.0] * 29), dtype=np.float32))
        target.append(np.asarray(la, dtype=np.float32))
        packets.append(int(m.get("index", len(packets))))
        ref = m.get("body_q_target")
        if ref is not None and len(ref) == 29:
            reference.append(np.asarray(ref, dtype=np.float32))
        if base is not None:
            root_pos.append(np.asarray(base["base_pos"], dtype=np.float32))
            bw, bx, by, bz = base["base_quat_wxyz"]
            root_quat.append(np.asarray([bw, bx, by, bz], dtype=np.float32))
            tilt.append(tilt_deg_from_quat_wxyz(bw, bx, by, bz))
            fall.append(bool(base.get("fall", False)))
        if time.monotonic() - started >= args.seconds:
            break

    n = len(q)
    if n < 50:
        log(f"ERROR: only {n} packets recorded")
        return 4
    tgt = np.stack(target)
    step_delta = np.abs(np.diff(tgt, axis=0)).max(axis=1)
    step_delta = np.concatenate([[0.0], step_delta]).astype(np.float32)
    out = {
        "wall_t": np.asarray(wall_t, dtype=np.float64),
        "phase": np.ones(n, dtype=np.int8),
        "q": np.stack(q),
        "dq": np.stack(dq),
        "target": tgt,
        "step_delta": step_delta,
        "packets": np.asarray(packets, dtype=np.int64),
        "joint_names": np.asarray(MUJOCO_JOINT_NAMES),
        "meta": np.asarray(json.dumps({
            "source": "sonic_debug_stream_recorder",
            "endpoint": args.endpoint,
            "seconds": float(args.seconds),
            "n_bad_packets": n_bad,
            "has_base_pose": bool(root_pos),
            "joint_order": "mujoco",
        })),
    }
    if len(reference) == n:
        out["reference"] = np.stack(reference)
    if root_pos:
        out["root_pos"] = np.stack(root_pos)
        out["root_quat"] = np.stack(root_quat)
        out["tilt_deg"] = np.asarray(tilt, dtype=np.float32)
        out["fall"] = np.asarray(fall, dtype=np.bool_)
    np.savez_compressed(args.out, **out)
    hz = (n - 1) / max(1e-6, wall_t[-1] - wall_t[0])
    log(f"saved {args.out} packets={n} rate={hz:.1f}Hz bad={n_bad} base_pose={len(root_pos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
