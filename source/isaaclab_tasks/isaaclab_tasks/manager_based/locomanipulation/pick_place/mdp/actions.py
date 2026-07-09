# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import socket
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .configs.action_cfg import AgileBasedLowerBodyActionCfg


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

ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

MUJOCO_29DOF_JOINT_NAMES = [ISAACLAB_29DOF_JOINT_NAMES[i] for i in ISAACLAB_TO_MUJOCO_DOF]

LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]


@dataclass
class _MirrorSample:
    joint_pos_mujoco: np.ndarray | None = None
    joint_vel_mujoco: np.ndarray | None = None
    left_hand_pos: np.ndarray | None = None
    right_hand_pos: np.ndarray | None = None
    left_hand_vel: np.ndarray | None = None
    right_hand_vel: np.ndarray | None = None
    root_pos_w: np.ndarray | None = None
    root_quat_w: np.ndarray | None = None
    root_lin_vel_w: np.ndarray | None = None
    root_ang_vel_w: np.ndarray | None = None
    body_source: str = "none"
    root_source: str = "none"
    root_fresh: bool = False
    fresh: bool = False


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-6 or not math.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _body_q_to_mujoco_order(values: np.ndarray, joint_order: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < len(MUJOCO_29DOF_JOINT_NAMES):
        raise ValueError(f"Joint vector has {values.size} values, expected at least 29")
    q29 = values[: len(MUJOCO_29DOF_JOINT_NAMES)]
    if joint_order == "isaaclab":
        return q29[ISAACLAB_TO_MUJOCO_DOF].copy()
    if joint_order == "mujoco":
        return q29.copy()
    raise ValueError(f"Unsupported joint order: {joint_order}")


class _ZmqLatestSubscriber:
    def __init__(self, host: str, port: int, topic: str, timeout: float):
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.zmq = zmq
        self.topic = topic.encode("utf-8")
        self.timeout = timeout
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.endpoint = f"tcp://{host}:{port}"
        self.description = f"{self.endpoint}/{topic}"
        self.socket.connect(self.endpoint)
        self.last_msg: dict[str, Any] | None = None
        self.last_rx_time = 0.0
        print(f"[INFO] MuJoCo G1 mirror ZMQ connected: {self.description}")

    def close(self) -> None:
        self.socket.close(0)
        self.ctx.term()

    @property
    def fresh(self) -> bool:
        return self.last_msg is not None and (time.monotonic() - self.last_rx_time) <= self.timeout

    def _decode(self, parts: list[bytes]) -> dict[str, Any] | None:
        if not parts:
            return None
        if len(parts) >= 2 and parts[0] == self.topic:
            payload = parts[-1]
        else:
            raw = parts[0]
            payload = raw[len(self.topic) :] if raw.startswith(self.topic) else raw
        return self.msgpack.unpackb(payload, raw=False)

    def poll_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                parts = self.socket.recv_multipart(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            latest = self._decode(parts)
        if latest is not None:
            self.last_msg = latest
            self.last_rx_time = time.monotonic()
        return latest if latest is not None else self.last_msg


class _UdpLatestSubscriber:
    def __init__(self, bind_host: str, port: int, topic: str, timeout: float, rcvbuf: int):
        import msgpack

        self.msgpack = msgpack
        self.topic = topic.encode("utf-8")
        self.timeout = timeout
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        self.socket.bind((bind_host, int(port)))
        self.socket.setblocking(False)
        actual_rcvbuf = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.endpoint = f"udp://{bind_host}:{int(port)}"
        self.description = f"{self.endpoint}/{topic}"
        self.last_msg: dict[str, Any] | None = None
        self.last_rx_time = 0.0
        print(f"[INFO] MuJoCo G1 mirror UDP listening: {self.description} SO_RCVBUF={actual_rcvbuf}")

    def close(self) -> None:
        self.socket.close()

    @property
    def fresh(self) -> bool:
        return self.last_msg is not None and (time.monotonic() - self.last_rx_time) <= self.timeout

    def _decode(self, packet: bytes) -> dict[str, Any] | None:
        if not packet.startswith(self.topic):
            return None
        payload = packet[len(self.topic) :]
        if not payload:
            return None
        return self.msgpack.unpackb(payload, raw=False)

    def poll_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            decoded = self._decode(packet)
            if decoded is not None:
                latest = decoded
        if latest is not None:
            self.last_msg = latest
            self.last_rx_time = time.monotonic()
        return latest if latest is not None else self.last_msg


class _ZmqLatestPublisher:
    def __init__(self, port: int, topic: str):
        import zmq

        self.zmq = zmq
        self.topic = topic.encode("utf-8")
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 5)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.endpoint = f"tcp://*:{int(port)}"
        self.description = f"{self.endpoint}/{topic}"
        self.socket.bind(self.endpoint)
        print(f"[INFO] G1 gripper sync ZMQ publishing: {self.description}")

    def close(self) -> None:
        self.socket.close(0)
        self.ctx.term()

    def publish(self, payload: dict[str, Any]) -> None:
        import msgpack

        packed = msgpack.packb(payload, use_bin_type=True)
        try:
            self.socket.send_multipart([self.topic, packed], flags=self.zmq.NOBLOCK)
        except self.zmq.Again:
            pass


class MuJoCoG1MirrorAction(ActionTerm):
    """Mirror MuJoCo/SONIC G1 root and joint state into the Isaac Lab robot.

    The term listens to the same debug streams used by ``isaaclab_g1_sim2sim_viewer.py``
    and writes state directly only after data is received.
    Without a live publisher it stays idle, allowing the normal motion-controller locomotion
    action to continue working.
    """

    cfg: "MuJoCoG1MirrorActionCfg"

    def __init__(self, cfg: "MuJoCoG1MirrorActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False
        self._enabled = cfg.enabled and self.num_envs == 1

        self._body_mujoco_ids, self._body_isaac_ids = self._build_body_joint_ids(cfg.mirror_joint_names)
        self._pd_body_isaac_ids, self._kinematic_body_isaac_ids = self._split_pd_drive_joint_ids(
            cfg.pd_drive_joint_names
        )
        self._left_hand_ids = self._build_joint_ids(LEFT_HAND_JOINT_NAMES)
        self._right_hand_ids = self._build_joint_ids(RIGHT_HAND_JOINT_NAMES)
        self._all_hand_ids = self._left_hand_ids + self._right_hand_ids
        self._foot_body_ids = self._build_body_ids(cfg.foot_body_names)

        self._transport = str(cfg.transport).lower()
        self._body_topic = cfg.udp_topic if self._transport == "udp" else cfg.zmq_topic
        self._root_topic = cfg.root_udp_topic if self._transport == "udp" else cfg.root_zmq_topic
        self._subscriber: _ZmqLatestSubscriber | _UdpLatestSubscriber | None = None
        self._root_subscriber: _ZmqLatestSubscriber | _UdpLatestSubscriber | None = None
        self._last_sample: _MirrorSample | None = None
        self._root_pose = self._asset.data.default_root_state[:, :7].clone()
        self._root_velocity = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._source_root_pos0: torch.Tensor | None = None
        self._target_root_pos0 = self._root_pose[:, :3].clone()
        self._foot_min_z: float | None = None
        self._source_origin_xy: torch.Tensor | None = None
        self._source_root_is_moving = False
        self._stance_slot: int | None = None
        self._anchor_xy: torch.Tensor | None = None
        self._warned_disabled = False
        self._warned_stale = False
        self._warned_root_missing = False
        self._warned_root_position_mode = False
        self._warned_gripper_unavailable = False
        self._printed_first_sample = False
        self._last_root_debug_time = 0.0
        self._last_gripper_debug_time = 0.0
        self._last_mirror_hands_from_mujoco = False

        if self._enabled:
            try:
                if self._transport == "udp":
                    self._subscriber = _UdpLatestSubscriber(
                        cfg.udp_bind_host,
                        cfg.udp_port,
                        cfg.udp_topic,
                        cfg.zmq_timeout,
                        cfg.udp_rcvbuf,
                    )
                    if cfg.root_udp:
                        self._root_subscriber = _UdpLatestSubscriber(
                            cfg.root_udp_bind_host,
                            cfg.root_udp_port,
                            cfg.root_udp_topic,
                            cfg.zmq_timeout,
                            cfg.root_udp_rcvbuf,
                        )
                elif self._transport == "zmq":
                    self._subscriber = _ZmqLatestSubscriber(cfg.zmq_host, cfg.zmq_port, cfg.zmq_topic, cfg.zmq_timeout)
                    if cfg.root_zmq:
                        self._root_subscriber = _ZmqLatestSubscriber(
                            cfg.root_zmq_host,
                            cfg.root_zmq_port,
                            cfg.root_zmq_topic,
                            cfg.zmq_timeout,
                        )
                else:
                    raise ValueError(f"Unsupported MuJoCo G1 mirror transport: {cfg.transport!r}")
            except Exception as exc:
                self._enabled = False
                print(f"[WARN] MuJoCo G1 mirror disabled; failed to create {self._transport.upper()} subscriber: {exc}")
        elif cfg.enabled and self.num_envs != 1:
            print("[WARN] MuJoCo G1 mirror is disabled because it only supports num_envs=1 for XR first-person use.")

    def __del__(self):
        for subscriber in (getattr(self, "_subscriber", None), getattr(self, "_root_subscriber", None)):
            if subscriber is not None:
                try:
                    subscriber.close()
                except Exception:
                    pass
        try:
            super().__del__()
        except Exception:
            pass

    @property
    def action_dim(self) -> int:
        return 4 if self.cfg.controller_gripper_enabled else 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        if self.action_dim == 0:
            self._processed_actions = actions
            return
        target_actions = torch.clamp(actions, 0.0, 1.0)
        alpha = min(max(float(self.cfg.controller_gripper_action_alpha), 0.0), 1.0)
        self._processed_actions = self._processed_actions + alpha * (target_actions - self._processed_actions)

    def apply_actions(self):
        if not self._enabled or self._subscriber is None:
            self._hold_default_body_pose()
            self._apply_controller_gripper_targets()
            return

        sample = self._sample()
        if sample is None:
            self._hold_default_body_pose()
            self._apply_controller_gripper_targets()
            return
        self._last_sample = sample
        if not self._printed_first_sample:
            root_state = "yes" if sample.root_pos_w is not None or sample.root_quat_w is not None else "no"
            print(
                "[INFO] MuJoCo G1 mirror received first packet: "
                f"mirrored_body_joints={len(self._body_isaac_ids)} "
                f"(pd_drive={len(self._pd_body_isaac_ids)}, kinematic={len(self._kinematic_body_isaac_ids)}), "
                f"mirror_hands={self.cfg.mirror_hands}, root={root_state}, "
                f"body_source={sample.body_source}, root_source={sample.root_source}"
            )
            self._printed_first_sample = True

        source_root_applied = self._apply_source_root_state(sample)

        joint_pos = self._asset.data.joint_pos.clone()
        joint_vel = self._asset.data.joint_vel.clone()

        if sample.joint_pos_mujoco is not None:
            q = torch.tensor(sample.joint_pos_mujoco[self._body_mujoco_ids], dtype=torch.float32, device=self.device)
            joint_pos[:, self._body_isaac_ids] = q.unsqueeze(0)
        if sample.joint_vel_mujoco is not None:
            dq = torch.tensor(sample.joint_vel_mujoco[self._body_mujoco_ids], dtype=torch.float32, device=self.device)
            joint_vel[:, self._body_isaac_ids] = dq.unsqueeze(0)

        mirror_hands_from_mujoco = self.cfg.mirror_hands and not self.cfg.controller_gripper_enabled
        self._last_mirror_hands_from_mujoco = mirror_hands_from_mujoco
        if mirror_hands_from_mujoco:
            if sample.left_hand_pos is not None and len(self._left_hand_ids) == len(LEFT_HAND_JOINT_NAMES):
                joint_pos[:, self._left_hand_ids] = torch.tensor(
                    sample.left_hand_pos[:7], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            if sample.right_hand_pos is not None and len(self._right_hand_ids) == len(RIGHT_HAND_JOINT_NAMES):
                joint_pos[:, self._right_hand_ids] = torch.tensor(
                    sample.right_hand_pos[:7], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            if sample.left_hand_vel is not None and len(self._left_hand_ids) == len(LEFT_HAND_JOINT_NAMES):
                joint_vel[:, self._left_hand_ids] = torch.tensor(
                    sample.left_hand_vel[:7], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            if sample.right_hand_vel is not None and len(self._right_hand_ids) == len(RIGHT_HAND_JOINT_NAMES):
                joint_vel[:, self._right_hand_ids] = torch.tensor(
                    sample.right_hand_vel[:7], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

        # PD-drive joints (arms by default) only receive position/velocity targets below,
        # so contact forces from held objects are resolved by the actuator instead of
        # being overwritten by a kinematic state reset every step.
        if self._kinematic_body_isaac_ids:
            self._asset.write_joint_state_to_sim(
                joint_pos[:, self._kinematic_body_isaac_ids],
                joint_vel[:, self._kinematic_body_isaac_ids],
                joint_ids=self._kinematic_body_isaac_ids,
            )
        self._asset.set_joint_position_target(joint_pos[:, self._body_isaac_ids], joint_ids=self._body_isaac_ids)
        self._asset.set_joint_velocity_target(joint_vel[:, self._body_isaac_ids], joint_ids=self._body_isaac_ids)
        if mirror_hands_from_mujoco and self._all_hand_ids:
            self._asset.write_joint_state_to_sim(
                joint_pos[:, self._all_hand_ids],
                joint_vel[:, self._all_hand_ids],
                joint_ids=self._all_hand_ids,
            )
            self._asset.set_joint_position_target(joint_pos[:, self._all_hand_ids], joint_ids=self._all_hand_ids)
            self._asset.set_joint_velocity_target(joint_vel[:, self._all_hand_ids], joint_ids=self._all_hand_ids)
        self._apply_controller_gripper_targets()

        if self.cfg.root_motion_mode in {"stance", "auto"}:
            self._apply_stance_root_if_needed(source_has_root=sample.root_pos_w is not None)
        if self.cfg.ground_lock and not source_root_applied:
            self._apply_ground_lock()
        self._print_root_debug(sample)

    def _apply_source_root_state(self, sample: _MirrorSample) -> bool:
        if sample.root_pos_w is None and sample.root_quat_w is None:
            self._warn_missing_root_once()
            self._root_pose = self._asset.data.root_link_state_w[:, :7].clone()
            return False

        if sample.root_pos_w is not None:
            source_root_pos = torch.tensor(sample.root_pos_w, dtype=torch.float32, device=self.device).view(1, 3)
            self._root_pose[:, :3] = self._map_source_root_position(source_root_pos)
        if sample.root_quat_w is not None:
            self._root_pose[:, 3:7] = torch.tensor(sample.root_quat_w, dtype=torch.float32, device=self.device)
        if sample.root_lin_vel_w is not None:
            self._root_velocity[:, :3] = torch.tensor(
                sample.root_lin_vel_w, dtype=torch.float32, device=self.device
            ).view(1, 3)
        if sample.root_ang_vel_w is not None:
            self._root_velocity[:, 3:6] = torch.tensor(
                sample.root_ang_vel_w, dtype=torch.float32, device=self.device
            ).view(1, 3)

        self._source_root_is_moving |= self._detect_source_root_motion(self._root_pose)
        self._asset.write_root_link_pose_to_sim(self._root_pose)
        self._asset.write_root_link_velocity_to_sim(self._root_velocity)
        return True

    def _map_source_root_position(self, source_root_pos: torch.Tensor) -> torch.Tensor:
        mode = str(self.cfg.root_position_mode).lower()
        if mode in {"relative", "delta"}:
            if self._source_root_pos0 is None:
                self._source_root_pos0 = source_root_pos.clone()
                self._target_root_pos0 = self._root_pose[:, :3].clone()
            return self._target_root_pos0 + (source_root_pos - self._source_root_pos0)
        if mode in {"absolute", "source"}:
            return source_root_pos
        if not self._warned_root_position_mode:
            print(
                f"[WARN] MuJoCo G1 mirror unknown root_position_mode={self.cfg.root_position_mode!r}; "
                "falling back to relative root displacement."
            )
            self._warned_root_position_mode = True
        if self._source_root_pos0 is None:
            self._source_root_pos0 = source_root_pos.clone()
            self._target_root_pos0 = self._root_pose[:, :3].clone()
        return self._target_root_pos0 + (source_root_pos - self._source_root_pos0)

    def _warn_missing_root_once(self) -> None:
        if self._warned_root_missing:
            return
        if self._root_subscriber is not None and self.cfg.root_zmq_required:
            root_stream = getattr(self._root_subscriber, "description", "dedicated root-state stream")
            print(
                "[WARN] MuJoCo G1 mirror has body joint packets but no dedicated root packets yet. "
                f"Expected {root_stream}; "
                "the robot will walk in place until root_pos_w/root_quat_w arrive."
            )
        self._warned_root_missing = True

    def _print_root_debug(self, sample: _MirrorSample) -> None:
        interval = float(self.cfg.root_debug_interval_s)
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_root_debug_time < interval:
            return
        self._last_root_debug_time = now

        applied_pos = self._root_pose[0, :3].detach().cpu().numpy()
        if sample.root_pos_w is None:
            print(
                "[INFO] MuJoCo G1 root mirror: source=none, "
                f"applied_xyz=[{applied_pos[0]:.3f}, {applied_pos[1]:.3f}, {applied_pos[2]:.3f}], "
                "waiting for g1_root."
            )
            return

        src_pos = sample.root_pos_w
        src_delta_xy = np.zeros(2, dtype=np.float32)
        if self._source_root_pos0 is not None:
            src0 = self._source_root_pos0[0, :2].detach().cpu().numpy()
            src_delta_xy = src_pos[:2] - src0
        print(
            "[INFO] MuJoCo G1 root mirror: "
            f"body_source={sample.body_source}, root_source={sample.root_source}, "
            f"fresh={sample.root_fresh}, mode={self.cfg.root_position_mode}, "
            f"src_xyz=[{src_pos[0]:.3f}, {src_pos[1]:.3f}, {src_pos[2]:.3f}], "
            f"src_delta_xy=[{src_delta_xy[0]:.3f}, {src_delta_xy[1]:.3f}], "
            f"applied_xyz=[{applied_pos[0]:.3f}, {applied_pos[1]:.3f}, {applied_pos[2]:.3f}]"
        )

    def _hold_default_body_pose(self) -> None:
        """Hold the robot's default body pose when no mirror data is available.

        Without a live publisher, the DCMotor legs (0 stiffness / 0 damping)
        generate zero torque and collapse under gravity. This method writes the
        initial default joint state and root pose each physics step, keeping the
        robot standing until the first mirror packet arrives.
        """
        # Root pose and velocity → back to default (don't let gravity pull the robot down)
        self._root_pose = self._asset.data.default_root_state[:, :7].clone()
        self._root_velocity = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._target_root_pos0 = self._root_pose[:, :3].clone()
        self._asset.write_root_link_pose_to_sim(self._root_pose)
        self._asset.write_root_link_velocity_to_sim(self._root_velocity)

        # Body joints (legs, arms, waist): revert to the init_state joint positions
        joint_pos = self._asset.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._asset.write_joint_state_to_sim(
            joint_pos[:, self._body_isaac_ids],
            joint_vel[:, self._body_isaac_ids],
            joint_ids=self._body_isaac_ids,
        )
        self._asset.set_joint_position_target(joint_pos[:, self._body_isaac_ids], joint_ids=self._body_isaac_ids)
        self._asset.set_joint_velocity_target(joint_vel[:, self._body_isaac_ids], joint_ids=self._body_isaac_ids)

        # Reset mirror-internal state so a later first-packet triggers a clean take-over
        self._last_sample = None
        self._source_root_pos0 = None
        self._printed_first_sample = False
        self._source_root_is_moving = False
        self._stance_slot = None
        self._anchor_xy = None

    def _apply_controller_gripper_targets(self) -> None:
        if not self.cfg.controller_gripper_enabled or self.action_dim == 0:
            return
        if (
            len(self._left_hand_ids) != len(LEFT_HAND_JOINT_NAMES)
            or len(self._right_hand_ids) != len(RIGHT_HAND_JOINT_NAMES)
        ):
            if not self._warned_gripper_unavailable:
                print(
                    "[WARN] MuJoCo G1 mirror controller gripper disabled; "
                    f"left_hand_joints={len(self._left_hand_ids)}, right_hand_joints={len(self._right_hand_ids)}"
                )
                self._warned_gripper_unavailable = True
            return

        left_target = self._compose_hand_target(
            index_close=self._processed_actions[:, 0],
            middle_close=self._processed_actions[:, 1],
            is_left=True,
        )
        right_target = self._compose_hand_target(
            index_close=self._processed_actions[:, 2],
            middle_close=self._processed_actions[:, 3],
            is_left=False,
        )
        target = torch.cat((left_target, right_target), dim=-1)
        if self.cfg.controller_gripper_use_soft_limits:
            limits = self._asset.data.soft_joint_pos_limits[:, self._all_hand_ids, :]
        else:
            limits = self._asset.data.joint_pos_limits[:, self._all_hand_ids, :]
        unclamped_target = target.clone()
        target = torch.max(torch.min(target, limits[..., 1]), limits[..., 0])
        if self.cfg.controller_gripper_write_joint_state:
            joint_vel = torch.zeros_like(target)
            self._asset.write_joint_state_to_sim(target, joint_vel, joint_ids=self._all_hand_ids)
        self._asset.set_joint_position_target(target, joint_ids=self._all_hand_ids)
        self._print_gripper_debug(unclamped_target, target, limits)

    def _compose_hand_target(self, index_close: torch.Tensor, middle_close: torch.Tensor, is_left: bool) -> torch.Tensor:
        target = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        thumb_close = torch.minimum(index_close, middle_close)

        thumb_yaw = self.cfg.controller_gripper_thumb_yaw_angle * (middle_close - index_close) * thumb_close
        thumb_1 = self.cfg.controller_gripper_thumb_1_angle * thumb_close
        thumb_2 = self.cfg.controller_gripper_thumb_2_angle * thumb_close
        index = self.cfg.controller_gripper_finger_close_angle * index_close
        middle = self.cfg.controller_gripper_finger_close_angle * middle_close

        target[:, 0] = thumb_yaw
        if is_left:
            target[:, 1] = thumb_1
            target[:, 2] = thumb_2
            target[:, 3] = -index
            target[:, 4] = -index
            target[:, 5] = -middle
            target[:, 6] = -middle
        else:
            target[:, 1] = -thumb_1
            target[:, 2] = -thumb_2
            target[:, 3] = index
            target[:, 4] = index
            target[:, 5] = middle
            target[:, 6] = middle
        return target

    def _print_gripper_debug(
        self,
        unclamped_target: torch.Tensor,
        clamped_target: torch.Tensor,
        limits: torch.Tensor,
    ) -> None:
        interval = float(self.cfg.controller_gripper_debug_interval_s)
        if interval <= 0.0:
            return

        now = time.monotonic()
        if now - self._last_gripper_debug_time < interval:
            return
        self._last_gripper_debug_time = now

        close = self._processed_actions[0].detach().cpu().numpy()
        current = self._asset.data.joint_pos[:, self._all_hand_ids][0].detach().cpu().numpy()
        target = clamped_target[0].detach().cpu().numpy()
        raw_target = unclamped_target[0].detach().cpu().numpy()
        limit_np = limits[0].detach().cpu().numpy()
        limit_kind = "soft" if self.cfg.controller_gripper_use_soft_limits else "hard"
        print(
            "[INFO] Controller gripper debug: "
            f"close=[L_idx={close[0]:.3f}, L_mid={close[1]:.3f}, "
            f"R_idx={close[2]:.3f}, R_mid={close[3]:.3f}], "
            f"limit={limit_kind}, mujoco_hand_mirror={self._last_mirror_hands_from_mujoco}, "
            f"write_joint_state={self.cfg.controller_gripper_write_joint_state}, "
            f"raw_target={np.round(raw_target, 3).tolist()}, "
            f"target={np.round(target, 3).tolist()}, "
            f"current={np.round(current, 3).tolist()}, "
            f"limits={np.round(limit_np, 3).tolist()}"
        )

    def _sample(self) -> _MirrorSample | None:
        debug_msg = self._subscriber.poll_latest() if self._subscriber is not None else None
        root_msg = self._root_subscriber.poll_latest() if self._root_subscriber is not None else None

        body_msg = root_msg if self._has_body_state(root_msg) else debug_msg
        if body_msg is None:
            return None

        using_root_full_state = body_msg is root_msg
        fresh = (
            self._root_subscriber.fresh
            if using_root_full_state and self._root_subscriber is not None
            else self._subscriber.fresh
            if self._subscriber is not None
            else False
        )
        if not fresh and not self._warned_stale:
            print(f"[WARN] MuJoCo G1 mirror {self._transport.upper()} stream is stale; holding last mirrored pose.")
            self._warned_stale = True

        q = self._select_body_q(body_msg)
        dq = self._select_body_dq(body_msg)
        msg_order = str(body_msg.get("target_order", body_msg.get("joint_order", self.cfg.zmq_joint_order))).lower()
        if msg_order not in {"mujoco", "isaaclab"}:
            msg_order = self.cfg.zmq_joint_order

        sample = _MirrorSample(
            fresh=fresh,
            body_source=self._root_topic if using_root_full_state else self._body_topic,
        )
        if q is not None:
            sample.joint_pos_mujoco = _body_q_to_mujoco_order(q, msg_order)
        elif self._last_sample is not None:
            sample.joint_pos_mujoco = self._last_sample.joint_pos_mujoco
        if dq is not None and dq.size >= 29:
            sample.joint_vel_mujoco = _body_q_to_mujoco_order(dq, msg_order)
        elif self._last_sample is not None:
            sample.joint_vel_mujoco = self._last_sample.joint_vel_mujoco

        sample.left_hand_pos = self._select_hand_q(body_msg, "left")
        sample.right_hand_pos = self._select_hand_q(body_msg, "right")
        sample.left_hand_vel = self._select_hand_dq(body_msg, "left")
        sample.right_hand_vel = self._select_hand_dq(body_msg, "right")
        if debug_msg is not None and body_msg is not debug_msg:
            if sample.left_hand_pos is None:
                sample.left_hand_pos = self._select_hand_q(debug_msg, "left")
            if sample.right_hand_pos is None:
                sample.right_hand_pos = self._select_hand_q(debug_msg, "right")
            if sample.left_hand_vel is None:
                sample.left_hand_vel = self._select_hand_dq(debug_msg, "left")
            if sample.right_hand_vel is None:
                sample.right_hand_vel = self._select_hand_dq(debug_msg, "right")

        root_source_name = self._body_topic
        root_fresh = fresh
        if root_msg is not None:
            root_source_name = self._root_topic
            root_fresh = self._root_subscriber.fresh if self._root_subscriber is not None else fresh
        elif self.cfg.root_zmq_required:
            root_source_name = "none"
        if self._root_subscriber is not None and self.cfg.root_zmq_required:
            root_source = root_msg
        else:
            root_source = root_msg if root_msg is not None else debug_msg
        root_pos = self._select_root_pos(root_source)
        root_quat = self._select_root_quat(root_source)
        root_lin_vel = self._select_root_lin_vel(root_source)
        root_ang_vel = self._select_root_ang_vel(root_source)
        if root_pos is not None and root_pos.size >= 3:
            sample.root_pos_w = root_pos[:3].copy()
            sample.root_pos_w[2] += self.cfg.root_z_offset
        if root_quat is not None and root_quat.size >= 4:
            sample.root_quat_w = _normalize_quat_wxyz(root_quat[:4])
        if root_lin_vel is not None and root_lin_vel.size >= 3:
            sample.root_lin_vel_w = root_lin_vel[:3].copy()
        if root_ang_vel is not None and root_ang_vel.size >= 3:
            sample.root_ang_vel_w = root_ang_vel[:3].copy()
        if sample.root_pos_w is not None or sample.root_quat_w is not None:
            sample.root_source = root_source_name
            sample.root_fresh = root_fresh

        if sample.joint_pos_mujoco is None:
            return None
        return sample

    @staticmethod
    def _has_body_state(msg: dict[str, Any] | None) -> bool:
        return msg is not None and any(
            key in msg for key in ("body_q", "body_q_measured", "body_q_target", "joint_pos", "q", "dof_pos")
        )

    @staticmethod
    def _first_array(msg: dict[str, Any] | None, keys: tuple[str, ...]) -> np.ndarray | None:
        if msg is None:
            return None
        for key in keys:
            if key in msg:
                arr = np.asarray(msg[key], dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return arr
        return None

    def _select_body_q(self, msg: dict[str, Any]) -> np.ndarray | None:
        if self.cfg.zmq_pose_source == "target":
            return self._first_array(msg, ("body_q_target", "joint_pos", "q", "dof_pos"))
        if self.cfg.zmq_pose_source == "measured":
            return self._first_array(msg, ("body_q_measured", "body_q", "joint_pos", "q", "dof_pos"))
        target = self._first_array(msg, ("body_q_target",))
        if target is not None and float(np.max(np.abs(target[: min(target.size, 29)]))) > 1.0e-4:
            return target
        return self._first_array(msg, ("body_q_measured", "body_q", "joint_pos", "q", "dof_pos"))

    def _select_body_dq(self, msg: dict[str, Any]) -> np.ndarray | None:
        if self.cfg.zmq_pose_source == "target":
            return self._first_array(msg, ("body_dq_target", "joint_vel", "dq", "dof_vel"))
        return self._first_array(msg, ("body_dq_measured", "body_dq", "joint_vel", "dq", "dof_vel"))

    def _select_hand_q(self, msg: dict[str, Any], side: str) -> np.ndarray | None:
        measured_keys = (f"{side}_hand_q", f"{side}_hand_q_measured")
        target_keys = (f"{side}_hand_q_target", f"last_{side}_hand_action")
        if self.cfg.zmq_pose_source == "target":
            return self._first_array(msg, target_keys + measured_keys)
        if self.cfg.zmq_pose_source == "measured":
            return self._first_array(msg, measured_keys + target_keys)
        target = self._first_array(msg, target_keys)
        if target is not None and target.size >= 7 and float(np.max(np.abs(target[:7]))) > 1.0e-4:
            return target[:7].copy()
        measured = self._first_array(msg, measured_keys)
        return measured[:7].copy() if measured is not None and measured.size >= 7 else None

    def _select_hand_dq(self, msg: dict[str, Any], side: str) -> np.ndarray | None:
        keys = (f"{side}_hand_dq", f"{side}_hand_dq_measured", f"{side}_hand_dq_target")
        arr = self._first_array(msg, keys)
        return arr[:7].copy() if arr is not None and arr.size >= 7 else None

    def _select_root_pos(self, msg: dict[str, Any] | None) -> np.ndarray | None:
        if self.cfg.zmq_pose_source == "target":
            return self._first_array(msg, ("root_pos_w", "base_trans_target", "base_pos", "root_pos"))
        if self.cfg.zmq_pose_source == "measured":
            return self._first_array(msg, ("root_pos_w", "base_trans_measured", "base_pos", "root_pos"))
        target = self._first_array(msg, ("root_pos_w", "base_trans_target", "base_pos", "root_pos"))
        if target is not None and float(np.linalg.norm(target[: min(target.size, 3)])) > 1.0e-4:
            return target
        return self._first_array(msg, ("base_trans_measured",))

    def _select_root_quat(self, msg: dict[str, Any] | None) -> np.ndarray | None:
        if self.cfg.zmq_pose_source == "target":
            return self._first_array(msg, ("root_quat_w", "base_quat_target", "base_quat", "root_quat"))
        if self.cfg.zmq_pose_source == "measured":
            return self._first_array(msg, ("root_quat_w", "base_quat_measured", "base_quat", "root_quat"))
        return self._first_array(
            msg,
            ("root_quat_w", "base_quat_measured", "base_quat_target", "base_quat", "root_quat"),
        )

    def _select_root_lin_vel(self, msg: dict[str, Any] | None) -> np.ndarray | None:
        return self._first_array(msg, ("root_lin_vel_w", "base_lin_vel", "root_lin_vel"))

    def _select_root_ang_vel(self, msg: dict[str, Any] | None) -> np.ndarray | None:
        return self._first_array(msg, ("root_ang_vel_w", "base_ang_vel", "root_ang_vel"))

    def _build_body_joint_ids(self, mirror_patterns: list[str]) -> tuple[list[int], list[int]]:
        compiled = [re.compile(pattern) for pattern in mirror_patterns]
        isaac_name_to_id = {name: idx for idx, name in enumerate(self._asset.joint_names)}
        mujoco_ids: list[int] = []
        isaac_ids: list[int] = []
        for mujoco_id, name in enumerate(MUJOCO_29DOF_JOINT_NAMES):
            if not any(pattern.fullmatch(name) for pattern in compiled):
                continue
            isaac_id = isaac_name_to_id.get(name)
            if isaac_id is not None:
                mujoco_ids.append(mujoco_id)
                isaac_ids.append(isaac_id)
        if not isaac_ids:
            raise RuntimeError("MuJoCo G1 mirror did not match any Isaac Lab joints.")
        return mujoco_ids, isaac_ids

    def _build_joint_ids(self, joint_names: list[str]) -> list[int]:
        isaac_name_to_id = {name: idx for idx, name in enumerate(self._asset.joint_names)}
        return [isaac_name_to_id[name] for name in joint_names if name in isaac_name_to_id]

    def _split_pd_drive_joint_ids(self, pd_patterns: list[str]) -> tuple[list[int], list[int]]:
        """Split mirrored body joints into PD-target-only and kinematic hard-write groups."""
        compiled = [re.compile(pattern) for pattern in pd_patterns]
        joint_names = self._asset.joint_names
        pd_ids = [
            isaac_id
            for isaac_id in self._body_isaac_ids
            if any(pattern.fullmatch(joint_names[isaac_id]) for pattern in compiled)
        ]
        pd_id_set = set(pd_ids)
        kinematic_ids = [isaac_id for isaac_id in self._body_isaac_ids if isaac_id not in pd_id_set]
        if pd_ids:
            print(
                f"[INFO] MuJoCo G1 mirror PD-drive joints ({len(pd_ids)}): "
                f"{[joint_names[i] for i in pd_ids]}"
            )
        return pd_ids, kinematic_ids

    def _build_body_ids(self, body_names: list[str]) -> list[int]:
        body_name_to_id = {name: idx for idx, name in enumerate(self._asset.body_names)}
        return [body_name_to_id[name] for name in body_names if name in body_name_to_id]

    def _detect_source_root_motion(self, root_pose: torch.Tensor) -> bool:
        source_xy = root_pose[0, :2].detach().clone()
        if self._source_origin_xy is None:
            self._source_origin_xy = source_xy
            return False
        return torch.linalg.norm(source_xy - self._source_origin_xy).item() > self.cfg.source_root_motion_eps

    def _apply_stance_root_if_needed(self, source_has_root: bool) -> None:
        if not self._foot_body_ids:
            return
        if self.cfg.root_motion_mode == "auto" and source_has_root and self._source_root_is_moving:
            self._stance_slot = None
            self._anchor_xy = None
            return

        self._env.sim.forward()
        self._asset.update(0.0)
        foot_pos = self._asset.data.body_pos_w[0, self._foot_body_ids, :3].detach()
        foot_min_z = self._resolve_foot_min_z()
        foot_height = foot_pos[:, 2] - (self.cfg.ground_height + foot_min_z)
        candidate_slot = int(torch.argmin(foot_height).item())

        if self._stance_slot is None or self._anchor_xy is None:
            self._stance_slot = candidate_slot
            self._anchor_xy = foot_pos[candidate_slot, :2].clone()
            return

        current_height = float(foot_height[self._stance_slot].item())
        candidate_height = float(foot_height[candidate_slot].item())
        current_contact = current_height <= self.cfg.stance_foot_height_tolerance
        candidate_contact = candidate_height <= self.cfg.stance_foot_height_tolerance
        should_switch = (
            candidate_slot != self._stance_slot
            and candidate_contact
            and ((not current_contact) or candidate_height < current_height - self.cfg.stance_foot_switch_margin)
        )
        if should_switch:
            self._stance_slot = candidate_slot
            self._anchor_xy = foot_pos[candidate_slot, :2].clone()
            return

        delta_xy = self._anchor_xy - foot_pos[self._stance_slot, :2]
        delta_norm = float(torch.linalg.norm(delta_xy).item())
        if self.cfg.stance_root_max_step > 0.0 and delta_norm > self.cfg.stance_root_max_step:
            delta_xy = delta_xy * (self.cfg.stance_root_max_step / max(delta_norm, 1.0e-9))
            self._anchor_xy = foot_pos[self._stance_slot, :2] + delta_xy

        if float(torch.linalg.norm(delta_xy).item()) > 1.0e-7:
            self._root_pose[:, :2] += delta_xy.unsqueeze(0)
            self._asset.write_root_link_pose_to_sim(self._root_pose)
            self._asset.write_root_link_velocity_to_sim(self._root_velocity)
            self._env.sim.forward()
            self._asset.update(0.0)

    def _apply_ground_lock(self) -> None:
        if not self._foot_body_ids:
            return
        self._env.sim.forward()
        self._asset.update(0.0)
        target_min_z = self.cfg.ground_height + self._resolve_foot_min_z()
        current_min_z = float(torch.min(self._asset.data.body_pos_w[:, self._foot_body_ids, 2]).item())
        z_correction = max(target_min_z - current_min_z, 0.0)
        if z_correction > 1.0e-5:
            self._root_pose[:, 2] += z_correction
            self._asset.write_root_link_pose_to_sim(self._root_pose)
            self._asset.write_root_link_velocity_to_sim(self._root_velocity)
            self._env.sim.forward()
            self._asset.update(0.0)

    def _resolve_foot_min_z(self) -> float:
        if self._foot_min_z is None:
            if self.cfg.ground_lock_clearance >= 0.0:
                self._foot_min_z = self.cfg.ground_lock_clearance
            elif self._foot_body_ids:
                foot_z = self._asset.data.body_pos_w[:, self._foot_body_ids, 2]
                self._foot_min_z = max(float(torch.min(foot_z).item()) - self.cfg.ground_height, 0.0)
            else:
                self._foot_min_z = 0.0
        return self._foot_min_z


class G1GripperSyncAction(ActionTerm):
    """Apply local OpenXR gripper commands or mirror a peer gripper stream."""

    cfg: "G1GripperSyncActionCfg"

    def __init__(self, cfg: "G1GripperSyncActionCfg", env: "ManagerBasedEnv"):
        self._mode = str(cfg.mode).lower()
        self._transport = str(cfg.transport).lower()
        self._enabled = bool(cfg.enabled)
        self._action_dim = 4 if self._mode == "local_publish" and self._enabled else 0
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False

        self._left_hand_ids = self._build_joint_ids(LEFT_HAND_JOINT_NAMES)
        self._right_hand_ids = self._build_joint_ids(RIGHT_HAND_JOINT_NAMES)
        self._all_hand_ids = self._left_hand_ids + self._right_hand_ids
        self._publisher: _ZmqLatestPublisher | None = None
        self._subscriber: _ZmqLatestSubscriber | None = None
        self._sequence = 0
        self._last_publish_time = 0.0
        self._last_debug_time = 0.0
        self._warned_unavailable = False
        self._warned_stale = False
        self._warned_bad_payload = False

        if self.num_envs != 1:
            self._enabled = False
            print("[WARN] G1 gripper sync disabled because it only supports num_envs=1.")
            return
        if len(self._left_hand_ids) != len(LEFT_HAND_JOINT_NAMES) or len(self._right_hand_ids) != len(
            RIGHT_HAND_JOINT_NAMES
        ):
            self._enabled = False
            print(
                "[WARN] G1 gripper sync disabled; "
                f"left_hand_joints={len(self._left_hand_ids)}, right_hand_joints={len(self._right_hand_ids)}"
            )
            return
        if self._transport != "zmq":
            self._enabled = False
            print(f"[WARN] G1 gripper sync disabled; unsupported transport={cfg.transport!r}.")
            return

        try:
            if self._mode == "local_publish":
                self._publisher = _ZmqLatestPublisher(cfg.zmq_port, cfg.zmq_topic)
            elif self._mode == "remote_subscribe":
                self._subscriber = _ZmqLatestSubscriber(cfg.zmq_host, cfg.zmq_port, cfg.zmq_topic, cfg.timeout)
            else:
                self._enabled = False
                print(f"[WARN] G1 gripper sync disabled; unsupported mode={cfg.mode!r}.")
        except Exception as exc:
            self._enabled = False
            print(f"[WARN] G1 gripper sync disabled; failed to create ZMQ endpoint: {exc}")

    def __del__(self):
        for endpoint in (getattr(self, "_publisher", None), getattr(self, "_subscriber", None)):
            if endpoint is not None:
                try:
                    endpoint.close()
                except Exception:
                    pass
        try:
            super().__del__()
        except Exception:
            pass

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        if self.action_dim == 0:
            self._raw_actions = actions
            self._processed_actions = actions
            return
        self._raw_actions = actions
        target_actions = torch.clamp(actions, 0.0, 1.0)
        alpha = min(max(float(self.cfg.controller_gripper_action_alpha), 0.0), 1.0)
        self._processed_actions = self._processed_actions + alpha * (target_actions - self._processed_actions)

    def apply_actions(self):
        if not self._enabled:
            return
        if self._mode == "local_publish":
            self._apply_local_actions()
        elif self._mode == "remote_subscribe":
            self._apply_remote_state()

    def _apply_local_actions(self) -> None:
        target = self._compose_target_from_actions(self._processed_actions)
        target = self._clamp_target(target)
        self._write_hand_target(target)
        self._publish_target(target)
        self._print_debug("local", target)

    def _apply_remote_state(self) -> None:
        if self._subscriber is None:
            return
        msg = self._subscriber.poll_latest()
        if msg is None:
            return
        if not self._subscriber.fresh and not self._warned_stale:
            print("[WARN] G1 gripper sync stream is stale; holding last gripper pose.")
            self._warned_stale = True
        target = self._target_from_payload(msg)
        if target is None:
            return
        target = self._clamp_target(target)
        self._write_hand_target(target)
        self._print_debug("remote", target)

    def _compose_target_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        left_target = self._compose_hand_target(
            index_close=actions[:, 0],
            middle_close=actions[:, 1],
            is_left=True,
        )
        right_target = self._compose_hand_target(
            index_close=actions[:, 2],
            middle_close=actions[:, 3],
            is_left=False,
        )
        return torch.cat((left_target, right_target), dim=-1)

    def _compose_hand_target(self, index_close: torch.Tensor, middle_close: torch.Tensor, is_left: bool) -> torch.Tensor:
        target = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        thumb_close = torch.minimum(index_close, middle_close)

        thumb_yaw = self.cfg.controller_gripper_thumb_yaw_angle * (middle_close - index_close) * thumb_close
        thumb_1 = self.cfg.controller_gripper_thumb_1_angle * thumb_close
        thumb_2 = self.cfg.controller_gripper_thumb_2_angle * thumb_close
        index = self.cfg.controller_gripper_finger_close_angle * index_close
        middle = self.cfg.controller_gripper_finger_close_angle * middle_close

        target[:, 0] = thumb_yaw
        if is_left:
            target[:, 1] = thumb_1
            target[:, 2] = thumb_2
            target[:, 3] = -index
            target[:, 4] = -index
            target[:, 5] = -middle
            target[:, 6] = -middle
        else:
            target[:, 1] = -thumb_1
            target[:, 2] = -thumb_2
            target[:, 3] = index
            target[:, 4] = index
            target[:, 5] = middle
            target[:, 6] = middle
        return target

    def _clamp_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.controller_gripper_use_soft_limits:
            limits = self._asset.data.soft_joint_pos_limits[:, self._all_hand_ids, :]
        else:
            limits = self._asset.data.joint_pos_limits[:, self._all_hand_ids, :]
        return torch.max(torch.min(target, limits[..., 1]), limits[..., 0])

    def _write_hand_target(self, target: torch.Tensor) -> None:
        joint_vel = torch.zeros_like(target)
        if self.cfg.write_joint_state:
            self._asset.write_joint_state_to_sim(target, joint_vel, joint_ids=self._all_hand_ids)
        self._asset.set_joint_position_target(target, joint_ids=self._all_hand_ids)
        self._asset.set_joint_velocity_target(joint_vel, joint_ids=self._all_hand_ids)

    def _publish_target(self, target: torch.Tensor) -> None:
        if self._publisher is None:
            return
        now = time.monotonic()
        interval = float(self.cfg.publish_interval_s)
        if interval > 0.0 and now - self._last_publish_time < interval:
            return
        self._last_publish_time = now
        target_np = target[0].detach().cpu().numpy()
        actions_np = (
            self._processed_actions[0].detach().cpu().numpy()
            if self._processed_actions.numel() == 4
            else np.zeros(4, dtype=np.float32)
        )
        self._sequence += 1
        self._publisher.publish(
            {
                "schema": "g1_gripper_state.v1",
                "robot_id": int(self.cfg.robot_id),
                "time": time.time(),
                "sequence": int(self._sequence),
                "joint_order": "g1_trihand_7dof_per_hand",
                "left_hand_q": target_np[:7].astype(float).tolist(),
                "right_hand_q": target_np[7:14].astype(float).tolist(),
                "raw_openxr_action": actions_np.astype(float).tolist(),
                "source": "isaaclab_openxr",
            }
        )

    def _target_from_payload(self, msg: dict[str, Any]) -> torch.Tensor | None:
        try:
            left = np.asarray(msg["left_hand_q"], dtype=np.float32).reshape(-1)
            right = np.asarray(msg["right_hand_q"], dtype=np.float32).reshape(-1)
            if left.size < 7 or right.size < 7:
                raise ValueError("left_hand_q/right_hand_q must each contain at least 7 values")
            values = np.concatenate((left[:7], right[:7]), axis=0)
        except Exception as exc:
            if not self._warned_bad_payload:
                print(f"[WARN] G1 gripper sync ignored malformed payload: {exc}")
                self._warned_bad_payload = True
            return None
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(1, 14)

    def _print_debug(self, source: str, target: torch.Tensor) -> None:
        interval = float(self.cfg.debug_interval_s)
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_debug_time < interval:
            return
        self._last_debug_time = now
        target_np = target[0].detach().cpu().numpy()
        print(
            "[INFO] G1 gripper sync: "
            f"robot_id={self.cfg.robot_id}, mode={self._mode}, source={source}, "
            f"target={np.round(target_np, 3).tolist()}"
        )

    def _build_joint_ids(self, joint_names: list[str]) -> list[int]:
        name_to_id = {name: idx for idx, name in enumerate(self._asset.joint_names)}
        return [name_to_id[name] for name in joint_names if name in name_to_id]


class AgileBasedLowerBodyAction(ActionTerm):
    """Action term that is based on Agile lower body RL policy."""

    cfg: AgileBasedLowerBodyActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: AgileBasedLowerBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Save the observation config from cfg
        self._observation_cfg = env.cfg.observations
        self._obs_group_name = cfg.obs_group_name

        # Load policy here if needed
        _temp_policy_path = retrieve_file_path(cfg.policy_path)
        self._policy = load_torchscript_model(_temp_policy_path, device=env.device)
        self._env = env

        # Find joint ids for the lower body joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        # Get the scale and offset from the configuration
        self._policy_output_scale = torch.tensor(cfg.policy_output_scale, device=env.device)
        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # Create tensors to store raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, hip_height]"""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _compose_policy_input(self, base_command: torch.Tensor, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Compose the policy input by concatenating repeated commands with observations.

        Args:
            base_command: The base command tensor [vx, vy, wz, hip_height].
            obs_tensor: The observation tensor from the environment.

        Returns:
            The composed policy input tensor with repeated commands concatenated to observations.
        """
        # Get history length from observation configuration
        history_length = getattr(self._observation_cfg, self._obs_group_name).history_length
        # Default to 1 if history_length is None (no history, just current observation)
        if history_length is None:
            history_length = 1

        # Repeat commands based on history length and concatenate with observations
        repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(base_command.shape[0], -1)
        policy_input = torch.cat([repeated_commands, obs_tensor], dim=-1)

        return policy_input

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions using the locomotion policy.

        Args:
            actions: The lower body commands.
        """

        # Extract base command from the action tensor
        # Assuming the base command [vx, vy, wz, hip_height]
        base_command = actions

        obs_tensor = self._env.obs_buf["lower_body_policy"]

        # Compose policy input using helper function
        policy_input = self._compose_policy_input(base_command, obs_tensor)

        joint_actions = self._policy.forward(policy_input)

        self._raw_actions[:] = joint_actions

        # Apply scaling and offset to the raw actions from the policy
        self._processed_actions = joint_actions * self._policy_output_scale + self._policy_output_offset

        # Clip actions if configured
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        """Apply the actions to the environment."""
        # Store the raw actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
