"""Bridge GR00T/SONIC Unitree DDS LowCmd to IsaacLab ZMQ targets.

Run this on the GR00T/Linux machine, not on native Windows. It publishes a
minimal simulated G1 LowState/secondary_imu so GR00T deploy can leave INIT, then
subscribes rt/lowcmd and forwards motor q targets to IsaacLab over ZMQ.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Sequence

import msgpack
import numpy as np
import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUStateDefault,
    unitree_hg_msg_dds__LowState_ as LowStateDefault,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_


G1_NUM_MOTOR = 29

# GR00T/Unitree hardware/MuJoCo order.
G1_DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)


class SonicUnitreeDdsProxy:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.latest_lowcmd_q = G1_DEFAULT_ANGLES_MUJOCO.copy()
        self.latest_lowcmd_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.latest_lowcmd_kp = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.latest_lowcmd_kd = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.have_lowcmd = False
        self.lowcmd_count = 0
        self.lowstate_count = 0
        self.zmq_count = 0
        self.sim_q = G1_DEFAULT_ANGLES_MUJOCO.copy()
        self.sim_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.last_sim_update = time.monotonic()

        if args.interface:
            ChannelFactoryInitialize(args.domain_id, args.interface)
        else:
            ChannelFactoryInitialize(args.domain_id)

        self.lowstate = LowStateDefault()
        self.imu = IMUStateDefault()
        self.lowstate_pub = ChannelPublisher(args.lowstate_topic, LowState_)
        self.lowstate_pub.Init()
        self.imu_pub = ChannelPublisher(args.secondary_imu_topic, IMUState_)
        self.imu_pub.Init()
        self.lowcmd_sub = ChannelSubscriber(args.lowcmd_topic, LowCmd_)
        self.lowcmd_sub.Init(self._lowcmd_handler, 1)

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.setsockopt(zmq.SNDHWM, 1)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.bind(args.zmq_bind)

    def _lowcmd_handler(self, msg) -> None:
        q = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        dq = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        kp = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        kd = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        for i in range(G1_NUM_MOTOR):
            motor_cmd = msg.motor_cmd[i]
            q[i] = float(motor_cmd.q)
            dq[i] = float(motor_cmd.dq)
            kp[i] = float(motor_cmd.kp)
            kd[i] = float(motor_cmd.kd)
        with self.lock:
            self.latest_lowcmd_q = q
            self.latest_lowcmd_dq = dq
            self.latest_lowcmd_kp = kp
            self.latest_lowcmd_kd = kd
            self.have_lowcmd = True
            self.lowcmd_count += 1

    @staticmethod
    def _set_sequence(dst, values: Sequence[float]) -> None:
        for idx, value in enumerate(values):
            dst[idx] = float(value)

    def _step_fake_robot(self, target_q: np.ndarray, now: float) -> None:
        dt = max(now - self.last_sim_update, 1.0 / self.args.lowstate_hz)
        self.last_sim_update = now
        alpha = float(np.clip(self.args.state_follow_alpha, 0.0, 1.0))
        prev_q = self.sim_q.copy()
        self.sim_q = (1.0 - alpha) * self.sim_q + alpha * target_q
        self.sim_dq = (self.sim_q - prev_q) / dt

    def _publish_lowstate(self, now: float) -> None:
        with self.lock:
            target_q = self.latest_lowcmd_q.copy()
            lowcmd_count = self.lowcmd_count
        self._step_fake_robot(target_q, now)

        for i in range(G1_NUM_MOTOR):
            motor_state = self.lowstate.motor_state[i]
            motor_state.q = float(self.sim_q[i])
            motor_state.dq = float(self.sim_dq[i])
            motor_state.ddq = 0.0
            motor_state.tau_est = 0.0

        self.lowstate.mode_machine = int(self.args.mode_machine)
        self.lowstate.tick = int(now * 1000.0) & 0xFFFFFFFF
        self._set_sequence(self.lowstate.imu_state.quaternion, (1.0, 0.0, 0.0, 0.0))
        self._set_sequence(self.lowstate.imu_state.gyroscope, (0.0, 0.0, 0.0))
        self._set_sequence(self.lowstate.imu_state.accelerometer, (0.0, 0.0, 9.81))
        self.lowstate_pub.Write(self.lowstate)

        self._set_sequence(self.imu.quaternion, (1.0, 0.0, 0.0, 0.0))
        self._set_sequence(self.imu.gyroscope, (0.0, 0.0, 0.0))
        self._set_sequence(self.imu.accelerometer, (0.0, 0.0, 9.81))
        self.imu_pub.Write(self.imu)
        self.lowstate_count += 1

        if self.lowstate_count % max(int(self.args.lowstate_hz), 1) == 0:
            print(
                "[DDSProxy] "
                f"lowstate={self.lowstate_count} lowcmd={lowcmd_count} zmq={self.zmq_count} "
                f"q_absmax={float(np.max(np.abs(self.sim_q))):.3f}"
            )

    def _publish_zmq_target(self) -> None:
        with self.lock:
            if not self.have_lowcmd:
                return
            q = self.latest_lowcmd_q.astype(float).tolist()
            dq = self.latest_lowcmd_dq.astype(float).tolist()
            kp = self.latest_lowcmd_kp.astype(float).tolist()
            kd = self.latest_lowcmd_kd.astype(float).tolist()
            lowcmd_count = self.lowcmd_count

        payload = {
            "source": "unitree_dds_lowcmd_proxy",
            "lowcmd_count": lowcmd_count,
            "body_q_target": q,
            "body_dq_target": dq,
            "body_kp": kp,
            "body_kd": kd,
            "target_order": "mujoco",
            "timestamp": time.time(),
        }
        raw = self.args.zmq_topic.encode("utf-8") + msgpack.packb(payload, use_bin_type=True)
        self.zmq_socket.send(raw, flags=zmq.NOBLOCK)
        self.zmq_count += 1

    def run(self) -> None:
        print(
            "[DDSProxy] started "
            f"domain={self.args.domain_id} interface={self.args.interface or '<auto>'} "
            f"lowcmd={self.args.lowcmd_topic} lowstate={self.args.lowstate_topic} "
            f"imu={self.args.secondary_imu_topic} zmq={self.args.zmq_bind}/{self.args.zmq_topic}"
        )
        lowstate_period = 1.0 / self.args.lowstate_hz
        zmq_period = 1.0 / self.args.zmq_hz
        next_lowstate = time.monotonic()
        next_zmq = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= next_lowstate:
                self._publish_lowstate(now)
                next_lowstate += lowstate_period
            if now >= next_zmq:
                self._publish_zmq_target()
                next_zmq += zmq_period
            time.sleep(0.0005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", default="", help="DDS network interface name. Empty means auto.")
    parser.add_argument("--lowcmd-topic", default="rt/lowcmd")
    parser.add_argument("--lowstate-topic", default="rt/lowstate")
    parser.add_argument("--secondary-imu-topic", default="rt/secondary_imu")
    parser.add_argument("--mode-machine", type=int, default=5)
    parser.add_argument("--lowstate-hz", type=float, default=500.0)
    parser.add_argument("--zmq-hz", type=float, default=50.0)
    parser.add_argument("--zmq-bind", default="tcp://*:5557")
    parser.add_argument("--zmq-topic", default="g1_debug")
    parser.add_argument("--state-follow-alpha", type=float, default=0.35)
    return parser.parse_args()


if __name__ == "__main__":
    SonicUnitreeDdsProxy(parse_args()).run()
