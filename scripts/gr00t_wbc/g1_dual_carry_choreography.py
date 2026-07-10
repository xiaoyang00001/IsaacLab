# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""程序化双 G1 机器人搬运编排：行走 → 相向转身 → 夹箱 → 抬升 → 侧步搬运。

本脚本不依赖 Isaac Sim。它按 ``MuJoCoG1MirrorAction`` 的 UDP 协议本地生成
两台机器人的根轨迹 + 29 DoF 关节轨迹并发布到镜像端口，替代远端 SONIC/MuJoCo
发送端。Isaac Lab 侧启动命令与 XR 遥操完全一致（transport=udp 由
``g1_udp_network.env`` 默认给出）::

    # 终端 1（Isaac Lab，命令不变）
    .\\isaaclab.bat -p scripts\\environments\\teleoperation\\teleop_se3_agent.py --xr ^
        --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 ^
        --teleop_device motion_controllers --enable_pinocchio

    # 终端 2（本编排脚本，任何带 numpy+msgpack 的 python 均可）
    D:\\miniconda3\\envs\\env_isaaclab\\python.exe scripts\\gr00t_wbc\\g1_dual_carry_choreography.py

搬运物是场景里的 ``carry_crate``（1.0×0.22×0.24 m 木箱，架在两台机器人
前方中点的 CarryStand 高台上）。几何常量与
``locomanipulation_g1_env_cfg.py`` 中的道具摆放一一对应，改动一侧必须同步另一侧。

手臂关键帧经 pinocchio FK 校准（g1_29dof.urdf，pelvis 系）：
    reach   sp=-0.35 sr=±0.15 el=0.40 → 掌心 [0.31, ±0.17, +0.10]
    squeeze sp=-0.35 sr=∓0.06 el=0.40 → 掌心 y=±0.12，压入箱侧 ~1 cm
    lift    sp=-0.65 sr=∓0.06 el=0.55 → 掌心抬高 ~5 cm，箱底离台

夹持原理：根/腿是运动学镜像（两机器人间距被脚本刚性锁定），手臂走 PD
（pd_drive_joint_names），roll 内收目标的跟踪误差转成持续夹持力，摩擦托住箱子。

复位：仿真端按 R 键将箱子放回高台；重启本脚本会把机器人瞬移回出生点重演。
"""

from __future__ import annotations

import argparse
import math
import socket
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import msgpack
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 msgpack：pip install msgpack") from exc

# ---------------------------------------------------------------------------
# 与 MuJoCoG1MirrorAction 一致的 29 DoF IsaacLab 关节顺序
# ---------------------------------------------------------------------------
ISAACLAB_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
JOINT_INDEX = {name: i for i, name in enumerate(ISAACLAB_29DOF_JOINT_NAMES)}

# ---------------------------------------------------------------------------
# 场景几何（与 locomanipulation_g1_env_cfg.py 对齐）
# ---------------------------------------------------------------------------
ROBOT_SPAWN = {
    1: (-3.8, 19.008, 0.78),
    2: (-2.3, 19.008, 0.78),
}
SPAWN_YAW_DEG = 90.0  # 出生朝向 +Y
CRATE_CENTER_XY = (-3.05, 20.10)  # carry_crate / CarryStand 的世界 xy
FACE_YAW_DEG = {1: 0.0, 2: 180.0}  # 转身后：R1 面向 +X，R2 面向 -X

# ---------------------------------------------------------------------------
# 站立基准位（与 ArticulationCfg.init_state 一致，避免接管瞬间跳变）
# ---------------------------------------------------------------------------
STAND_POSE = {
    "left_hip_pitch_joint": -0.10,
    "right_hip_pitch_joint": -0.10,
    "left_knee_joint": 0.30,
    "right_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.20,
    "right_ankle_pitch_joint": -0.20,
}


@dataclass(frozen=True)
class ArmPose:
    """对称双臂关键帧。roll 取左臂符号，右臂自动镜像。"""

    shoulder_pitch: float
    shoulder_roll_left: float
    elbow: float
    shoulder_yaw: float = 0.0
    wrist: float = 0.0

    def joints(self) -> dict[str, float]:
        return {
            "left_shoulder_pitch_joint": self.shoulder_pitch,
            "right_shoulder_pitch_joint": self.shoulder_pitch,
            "left_shoulder_roll_joint": self.shoulder_roll_left,
            "right_shoulder_roll_joint": -self.shoulder_roll_left,
            "left_shoulder_yaw_joint": self.shoulder_yaw,
            "right_shoulder_yaw_joint": -self.shoulder_yaw,
            "left_elbow_joint": self.elbow,
            "right_elbow_joint": self.elbow,
            "left_wrist_roll_joint": self.wrist,
            "right_wrist_roll_joint": -self.wrist,
        }


ARM_READY = ArmPose(shoulder_pitch=0.2, shoulder_roll_left=0.2, elbow=0.6)
ARM_REACH = ArmPose(shoulder_pitch=-0.35, shoulder_roll_left=0.15, elbow=0.40)
ARM_SQUEEZE = ArmPose(shoulder_pitch=-0.35, shoulder_roll_left=-0.06, elbow=0.40)
ARM_LIFT = ArmPose(shoulder_pitch=-0.65, shoulder_roll_left=-0.06, elbow=0.55)

# ---------------------------------------------------------------------------
# 步态参数
# ---------------------------------------------------------------------------
WALK_FREQ_HZ = 1.5  # 单腿摆动频率
WALK_HIP_AMP = 0.25
WALK_KNEE_AMP = 0.25
WALK_ANKLE_AMP = 0.15
WALK_ARM_SWING = 0.12
MARCH_KNEE_AMP = 0.18
MARCH_HIP_AMP = 0.10
SIDE_FREQ_HZ = 1.1
SIDE_KNEE_AMP = 0.12
SIDE_HIP_ROLL_AMP = 0.05
GAIT_RAMP_S = 0.4  # 相位边界步态幅度渐入渐出
BOB_AMP = 0.012  # 行走时根部轻微起伏


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def pos_part(x: float) -> float:
    return x if x > 0.0 else 0.0


def yaw_quat_wxyz(yaw_rad: float) -> list[float]:
    return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]


# ---------------------------------------------------------------------------
# 相位时间线
# ---------------------------------------------------------------------------
@dataclass
class Phase:
    name: str
    duration: float
    gait: str  # "stand" | "walk" | "march" | "side"
    arm_from: ArmPose
    arm_to: ArmPose
    move_y: float = 0.0  # 本相位世界 +Y 位移
    turn_to_face: bool = False  # 本相位内 yaw 从出生朝向插值到面向箱子


def build_timeline(args: argparse.Namespace) -> list[Phase]:
    advance_dy = CRATE_CENTER_XY[1] - ROBOT_SPAWN[1][1]
    return [
        Phase("settle", args.settle_s, "stand", ARM_READY, ARM_READY),
        Phase("advance", advance_dy / args.walk_speed, "walk", ARM_READY, ARM_READY, move_y=advance_dy),
        Phase("turn", args.turn_s, "march", ARM_READY, ARM_READY, turn_to_face=True),
        Phase("reach", args.reach_s, "stand", ARM_READY, ARM_REACH),
        Phase("squeeze", args.squeeze_s, "stand", ARM_REACH, ARM_SQUEEZE),
        Phase("lift", args.lift_s, "stand", ARM_SQUEEZE, ARM_LIFT),
        Phase("carry", args.carry_distance / args.carry_speed, "side", ARM_LIFT, ARM_LIFT, move_y=args.carry_distance),
        Phase("hold", math.inf, "stand", ARM_LIFT, ARM_LIFT),
    ]


# ---------------------------------------------------------------------------
# 步态生成：返回关节名 → 相对站立位/摆臂中心的角度
# ---------------------------------------------------------------------------
def gait_overlay(mode: str, t_phase: float, duration: float, arm_pose: ArmPose) -> dict[str, float]:
    joints: dict[str, float] = dict(STAND_POSE)
    joints.update(arm_pose.joints())
    if mode == "stand":
        return joints

    # 相位边界渐入渐出，避免步态幅度跳变
    env_in = smoothstep(t_phase / GAIT_RAMP_S)
    env_out = smoothstep((duration - t_phase) / GAIT_RAMP_S) if math.isfinite(duration) else 1.0
    env = min(env_in, env_out)

    if mode == "walk":
        omega = 2.0 * math.pi * WALK_FREQ_HZ
        phi_l = omega * t_phase
        phi_r = phi_l + math.pi
        for side, phi, phi_opp in (("left", phi_l, phi_r), ("right", phi_r, phi_l)):
            swing = math.sin(phi)
            lift = pos_part(math.sin(phi + 0.5))
            joints[f"{side}_hip_pitch_joint"] = STAND_POSE[f"{side}_hip_pitch_joint"] - env * WALK_HIP_AMP * swing
            joints[f"{side}_knee_joint"] = STAND_POSE[f"{side}_knee_joint"] + env * WALK_KNEE_AMP * lift
            joints[f"{side}_ankle_pitch_joint"] = (
                STAND_POSE[f"{side}_ankle_pitch_joint"]
                + env * (WALK_ANKLE_AMP * swing - 0.5 * WALK_KNEE_AMP * lift)
            )
            # 摆臂与对侧腿同步
            joints[f"{side}_shoulder_pitch_joint"] += env * WALK_ARM_SWING * math.sin(phi_opp)
    elif mode == "march":
        omega = 2.0 * math.pi * WALK_FREQ_HZ
        for side, offset in (("left", 0.0), ("right", math.pi)):
            lift = pos_part(math.sin(omega * t_phase + offset))
            joints[f"{side}_hip_pitch_joint"] = STAND_POSE[f"{side}_hip_pitch_joint"] - env * MARCH_HIP_AMP * lift
            joints[f"{side}_knee_joint"] = STAND_POSE[f"{side}_knee_joint"] + env * MARCH_KNEE_AMP * lift
            joints[f"{side}_ankle_pitch_joint"] = (
                STAND_POSE[f"{side}_ankle_pitch_joint"] - env * 0.5 * MARCH_KNEE_AMP * lift
            )
    elif mode == "side":
        omega = 2.0 * math.pi * SIDE_FREQ_HZ
        for side, offset, roll_sign in (("left", 0.0, 1.0), ("right", math.pi, -1.0)):
            lift = pos_part(math.sin(omega * t_phase + offset))
            joints[f"{side}_knee_joint"] = STAND_POSE[f"{side}_knee_joint"] + env * SIDE_KNEE_AMP * lift
            joints[f"{side}_ankle_pitch_joint"] = (
                STAND_POSE[f"{side}_ankle_pitch_joint"] - env * 0.5 * SIDE_KNEE_AMP * lift
            )
            joints[f"{side}_hip_roll_joint"] = roll_sign * env * SIDE_HIP_ROLL_AMP * lift
    return joints


def bob_offset(mode: str, t_phase: float, duration: float) -> float:
    if mode not in {"walk", "march"}:
        return 0.0
    env_in = smoothstep(t_phase / GAIT_RAMP_S)
    env_out = smoothstep((duration - t_phase) / GAIT_RAMP_S) if math.isfinite(duration) else 1.0
    env = min(env_in, env_out)
    omega = 2.0 * math.pi * WALK_FREQ_HZ
    return -BOB_AMP * env * 0.5 * (1.0 - math.cos(2.0 * omega * t_phase))


# ---------------------------------------------------------------------------
# 单机器人状态求值
# ---------------------------------------------------------------------------
@dataclass
class RobotState:
    pos: tuple[float, float, float]
    yaw_rad: float
    body_q: np.ndarray


def eval_robot(robot_id: int, timeline: list[Phase], t: float) -> tuple[RobotState, str]:
    spawn = ROBOT_SPAWN[robot_id]
    yaw0 = math.radians(SPAWN_YAW_DEG)
    yaw_face = math.radians(FACE_YAW_DEG[robot_id])

    x, z = spawn[0], spawn[2]
    y = spawn[1]
    yaw = yaw0
    t_rem = t
    active = timeline[-1]
    t_phase = 0.0
    for phase in timeline:
        if t_rem <= phase.duration or not math.isfinite(phase.duration):
            active = phase
            t_phase = t_rem
            u = smoothstep(t_rem / phase.duration) if math.isfinite(phase.duration) and phase.duration > 0 else 1.0
            y += phase.move_y * u
            if phase.turn_to_face:
                yaw = yaw0 + (yaw_face - yaw0) * u
            break
        t_rem -= phase.duration
        y += phase.move_y
        if phase.turn_to_face:
            yaw = yaw_face

    # 手臂关键帧插值 + 步态叠加
    u_arm = smoothstep(t_phase / active.duration) if math.isfinite(active.duration) and active.duration > 0 else 1.0
    arm = ArmPose(
        shoulder_pitch=active.arm_from.shoulder_pitch
        + (active.arm_to.shoulder_pitch - active.arm_from.shoulder_pitch) * u_arm,
        shoulder_roll_left=active.arm_from.shoulder_roll_left
        + (active.arm_to.shoulder_roll_left - active.arm_from.shoulder_roll_left) * u_arm,
        elbow=active.arm_from.elbow + (active.arm_to.elbow - active.arm_from.elbow) * u_arm,
        shoulder_yaw=active.arm_from.shoulder_yaw
        + (active.arm_to.shoulder_yaw - active.arm_from.shoulder_yaw) * u_arm,
        wrist=active.arm_from.wrist + (active.arm_to.wrist - active.arm_from.wrist) * u_arm,
    )
    joints = gait_overlay(active.gait, t_phase, active.duration, arm)
    z += bob_offset(active.gait, t_phase, active.duration)

    body_q = np.zeros(len(ISAACLAB_29DOF_JOINT_NAMES), dtype=np.float32)
    for name, val in joints.items():
        body_q[JOINT_INDEX[name]] = val
    return RobotState(pos=(x, y, z), yaw_rad=yaw, body_q=body_q), active.name


# ---------------------------------------------------------------------------
# UDP 发布
# ---------------------------------------------------------------------------
ROBOT_PORTS = {
    1: {"body": 5557, "root": 5558, "body_topic": "g1_1_debug", "root_topic": "g1_1_root"},
    2: {"body": 5567, "root": 5568, "body_topic": "g1_2_debug", "root_topic": "g1_2_root"},
}


class UdpSender:
    def __init__(self, hosts: list[str]):
        self.hosts = hosts
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, topic: str, port: int, payload: dict) -> None:
        packet = topic.encode("utf-8") + msgpack.packb(payload, use_bin_type=True)
        for host in self.hosts:
            try:
                self.sock.sendto(packet, (host, port))
            except OSError:
                pass  # 目标端未启动时 Windows 会报 ICMP 拒绝，忽略继续发


def publish_robot(
    sender: UdpSender,
    robot_id: int,
    state: RobotState,
    prev: RobotState | None,
    dt: float,
    now: float,
) -> None:
    ports = ROBOT_PORTS[robot_id]
    body_q = state.body_q.astype(float).tolist()
    if prev is not None:
        body_dq = ((state.body_q - prev.body_q) / dt).astype(float).tolist()
        lin_vel = [(state.pos[i] - prev.pos[i]) / dt for i in range(3)]
        dyaw = (state.yaw_rad - prev.yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
        ang_vel = [0.0, 0.0, dyaw / dt]
    else:
        body_dq = [0.0] * len(body_q)
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

    sender.send(
        ports["body_topic"],
        ports["body"],
        {
            "schema": "g1_scripted_carry.v1",
            "time": now,
            "joint_order": "isaaclab",
            "body_q": body_q,
            "body_dq": body_dq,
        },
    )
    sender.send(
        ports["root_topic"],
        ports["root"],
        {
            "schema": "g1_scripted_carry_root.v1",
            "time": now,
            "root_pos_w": list(state.pos),
            "root_quat_w": yaw_quat_wxyz(state.yaw_rad),
            "root_lin_vel_w": lin_vel,
            "root_ang_vel_w": ang_vel,
        },
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双 G1 机器人程序化搬运编排（UDP 镜像发布端）")
    parser.add_argument("--hosts", default="127.0.0.1", help="逗号分隔的 Isaac Lab 主机列表")
    parser.add_argument("--rate", type=float, default=100.0, help="发布频率 Hz")
    parser.add_argument("--settle-s", type=float, default=2.0, help="起始站定时长")
    parser.add_argument("--walk-speed", type=float, default=0.35, help="前进速度 m/s")
    parser.add_argument("--turn-s", type=float, default=2.5, help="相向转身时长")
    parser.add_argument("--reach-s", type=float, default=2.0, help="伸臂时长")
    parser.add_argument("--squeeze-s", type=float, default=1.5, help="夹紧时长")
    parser.add_argument("--lift-s", type=float, default=2.0, help="抬升时长")
    parser.add_argument("--carry-distance", type=float, default=1.5, help="搬运行走距离 m（世界 +Y）")
    parser.add_argument("--carry-speed", type=float, default=0.2, help="搬运行走速度 m/s")
    parser.add_argument("--dry-run", action="store_true", help="只打印时间线与采样数据，不发网络包")
    return parser.parse_args()


def print_timeline(timeline: list[Phase]) -> None:
    print("=" * 68)
    print(f"{'phase':10s} {'start':>8s} {'end':>8s} {'gait':8s} {'dy':>6s}")
    print("-" * 68)
    t0 = 0.0
    for phase in timeline:
        end = t0 + phase.duration
        end_str = f"{end:8.2f}" if math.isfinite(end) else "     inf"
        print(f"{phase.name:10s} {t0:8.2f} {end_str} {phase.gait:8s} {phase.move_y:6.2f}")
        t0 = end
    print("=" * 68)


def dry_run(timeline: list[Phase]) -> None:
    print_timeline(timeline)
    total = sum(p.duration for p in timeline if math.isfinite(p.duration))
    for t in np.arange(0.0, total + 2.0, 1.0):
        for robot_id in (1, 2):
            state, phase_name = eval_robot(robot_id, timeline, float(t))
            q = state.body_q
            print(
                f"t={t:6.2f} R{robot_id} [{phase_name:8s}] "
                f"pos=({state.pos[0]:+.2f},{state.pos[1]:+.2f},{state.pos[2]:.3f}) "
                f"yaw={math.degrees(state.yaw_rad):6.1f}° "
                f"L(sp,sr,el)=({q[JOINT_INDEX['left_shoulder_pitch_joint']]:+.2f},"
                f"{q[JOINT_INDEX['left_shoulder_roll_joint']]:+.2f},"
                f"{q[JOINT_INDEX['left_elbow_joint']]:+.2f}) "
                f"knee=({q[JOINT_INDEX['left_knee_joint']]:+.2f},{q[JOINT_INDEX['right_knee_joint']]:+.2f})"
            )
    # 数据包格式自检
    payload = {"body_q": state.body_q.astype(float).tolist(), "joint_order": "isaaclab"}
    packet = "g1_1_debug".encode() + msgpack.packb(payload, use_bin_type=True)
    decoded = msgpack.unpackb(packet[len(b"g1_1_debug"):], raw=False)
    assert len(decoded["body_q"]) == 29, "body_q 必须为 29 维"
    print(f"\n[OK] 数据包编解码自检通过（{len(packet)} 字节/包）")


def main() -> None:
    args = parse_args()
    timeline = build_timeline(args)
    if args.dry_run:
        dry_run(timeline)
        return

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    sender = UdpSender(hosts)
    print_timeline(timeline)
    print(f"[INFO] 发布到 {hosts}，rate={args.rate:.0f} Hz。Ctrl+C 停止。")

    dt = 1.0 / args.rate
    prev: dict[int, RobotState | None] = {1: None, 2: None}
    last_phase = {1: "", 2: ""}
    t_start = time.perf_counter()
    next_tick = t_start
    try:
        while True:
            now = time.perf_counter()
            t = now - t_start
            for robot_id in (1, 2):
                state, phase_name = eval_robot(robot_id, timeline, t)
                publish_robot(sender, robot_id, state, prev[robot_id], dt, time.time())
                prev[robot_id] = state
                if phase_name != last_phase[robot_id]:
                    last_phase[robot_id] = phase_name
                    if robot_id == 1:
                        print(f"[t={t:6.2f}s] 进入相位: {phase_name}")
            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[INFO] 已停止发布。仿真端机器人将保持最后镜像姿态。")


if __name__ == "__main__":
    main()
