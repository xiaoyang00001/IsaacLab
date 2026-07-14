# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import math
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..configs.action_cfg import (
        AgileBasedLowerBodyActionCfg,
        AutoWalkActionCfg,
        SonicDeployTargetActionCfg,
        SONICWholeBodyActionCfg,
        SonicRobotStatePublisherActionCfg,
        UnitreeDdsLowCmdActionCfg,
        UnitreeLowStatePublisherActionCfg,
    )

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
                f"mirrored_body_joints={len(self._body_isaac_ids)}, "
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

        self._asset.write_joint_state_to_sim(
            joint_pos[:, self._body_isaac_ids],
            joint_vel[:, self._body_isaac_ids],
            joint_ids=self._body_isaac_ids,
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

# G1 29-DoF MuJoCo/URDF order. Deploy uses this order only at the motor boundary.
SONIC_G1_29DOF_MUJOCO_JOINT_ORDER: tuple[str, ...] = (
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

# Mapping name follows deploy comments: for each MuJoCo index, gives the IsaacLab index.
SONIC_G1_MUJOCO_TO_ISAACLAB_DOF: tuple[int, ...] = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)
SONIC_G1_ISAACLAB_TO_MUJOCO_DOF: tuple[int, ...] = tuple(
    mujoco_idx
    for _, mujoco_idx in sorted(
        zip(SONIC_G1_MUJOCO_TO_ISAACLAB_DOF, range(len(SONIC_G1_MUJOCO_TO_ISAACLAB_DOF)))
    )
)

# SONIC policy/decoder 29D order is IsaacLab order. The deploy C++ converts raw
# action to MuJoCo order only when writing motor targets.
SONIC_G1_29DOF_JOINT_ORDER: tuple[str, ...] = tuple(
    joint_name
    for _, joint_name in sorted(
        zip(SONIC_G1_MUJOCO_TO_ISAACLAB_DOF, SONIC_G1_29DOF_MUJOCO_JOINT_ORDER)
    )
)

_G1_NATURAL_FREQ = 10.0 * 2.0 * math.pi
_G1_STIFFNESS_5020 = 0.003609725 * _G1_NATURAL_FREQ**2
_G1_STIFFNESS_7520_14 = 0.010177520 * _G1_NATURAL_FREQ**2
_G1_STIFFNESS_7520_22 = 0.025101925 * _G1_NATURAL_FREQ**2
_G1_STIFFNESS_4010 = 0.00425 * _G1_NATURAL_FREQ**2


def _sonic_g1_action_scale_for_joint(joint_name: str) -> float:
    """Match gear_sonic G1_MODEL_12_ACTION_SCALE: 0.25 * effort_limit / stiffness."""
    if "_hip_yaw_joint" in joint_name:
        return 0.25 * 88.0 / _G1_STIFFNESS_7520_14
    if any(s in joint_name for s in ("_hip_roll_joint", "_hip_pitch_joint", "_knee_joint")):
        return 0.25 * 139.0 / _G1_STIFFNESS_7520_22
    if any(s in joint_name for s in ("_ankle_pitch_joint", "_ankle_roll_joint")):
        return 0.25 * 50.0 / (2.0 * _G1_STIFFNESS_5020)
    if joint_name in ("waist_roll_joint", "waist_pitch_joint"):
        return 0.25 * 50.0 / (2.0 * _G1_STIFFNESS_5020)
    if joint_name == "waist_yaw_joint":
        return 0.25 * 88.0 / _G1_STIFFNESS_7520_14
    if any(s in joint_name for s in ("_shoulder_pitch_joint", "_shoulder_roll_joint", "_shoulder_yaw_joint", "_elbow_joint", "_wrist_roll_joint")):
        return 0.25 * 25.0 / _G1_STIFFNESS_5020
    if any(s in joint_name for s in ("_wrist_pitch_joint", "_wrist_yaw_joint")):
        return 0.25 * 5.0 / _G1_STIFFNESS_4010
    raise KeyError(f"No SONIC action scale for joint {joint_name!r}")


SONIC_G1_29DOF_ACTION_SCALE: tuple[float, ...] = tuple(
    _sonic_g1_action_scale_for_joint(joint_name) for joint_name in SONIC_G1_29DOF_JOINT_ORDER
)

# Deploy C++ `policy_parameters.hpp::default_angles` in MuJoCo order. SONIC uses
# these offsets for both history joint_pos_rel and final joint target decoding.
SONIC_G1_29DOF_MUJOCO_DEFAULT_ANGLES: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)
SONIC_G1_29DOF_DEFAULT_ANGLES: tuple[float, ...] = tuple(
    default_angle
    for _, default_angle in sorted(
        zip(SONIC_G1_MUJOCO_TO_ISAACLAB_DOF, SONIC_G1_29DOF_MUJOCO_DEFAULT_ANGLES)
    )
)

# SONIC encoder 用的 14 个 body link（来自 sonic_release/config.yaml body_names）。
# command_multi_future_nonflat 返回这 14 个 body 在 pelvis 坐标系下的位置（10 帧）。
SONIC_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)

# SMPL 24-joint → SONIC 14-body 近似映射（用 mocap['smpl_joints'] 替代 forward kinematics）。
# SMPL 标准关节顺序：0=pelvis, 1=l_hip, 2=r_hip, 3=spine1, 4=l_knee, 5=r_knee, 6=spine2,
#                    7=l_ankle, 8=r_ankle, 9=spine3, 10=l_foot, 11=r_foot, 12=neck, 13=l_collar,
#                    14=r_collar, 15=head, 16=l_shoulder, 17=r_shoulder, 18=l_elbow, 19=r_elbow,
#                    20=l_wrist, 21=r_wrist, 22=l_hand, 23=r_hand
SMPL_TO_SONIC_BODY_IDX: tuple[int, ...] = (
    0,   # pelvis
    1,   # left_hip_roll_link  ≈ l_hip
    4,   # left_knee_link      ≈ l_knee
    7,   # left_ankle_roll_link ≈ l_ankle
    2,   # right_hip_roll_link
    5,   # right_knee_link
    8,   # right_ankle_roll_link
    6,   # torso_link          ≈ spine2
    16,  # left_shoulder_roll_link
    18,  # left_elbow_link
    20,  # left_wrist_yaw_link
    17,  # right_shoulder_roll_link
    19,  # right_elbow_link
    21,  # right_wrist_yaw_link
)


class SonicDeployTargetAction(ActionTerm):
    """Drive the SONIC robot from GR00T deploy ZMQ joint targets.

    GR00T deploy publishes a single-part ZMQ message:
        [topic_prefix][msgpack payload]

    The minimal IsaacLab bridge consumes deploy motor targets and writes them
    directly as joint position targets for `sonic_robot`.
    """

    cfg: SonicDeployTargetActionCfg
    _asset: Articulation

    def __init__(self, cfg: SonicDeployTargetActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_joint_pos.clone()
        # 关节限位夹紧（永远生效）：摔倒后 policy 可能饱和输出 7~8 rad 的目标
        # （G1 关节上限 2.88），不夹紧会让 PD 以巨大偏置把关节往限位上撞（剧烈抽搐）
        _limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        self._target_pos_lower = _limits[..., 0].clone()
        self._target_pos_upper = _limits[..., 1].clone()
        self._post_unlock_steps = 0
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        reference_joint_ids = [
            idx
            for idx, joint_name in enumerate(self._joint_names)
            if any(token in joint_name for token in ("_hip_", "_knee_", "_ankle_", "waist_"))
        ]
        self._reference_joint_indices = torch.tensor(reference_joint_ids, device=self.device, dtype=torch.long)

        self._topic = cfg.topic.encode("utf-8")
        self._context = None
        self._socket = None
        self._zmq = None
        self._msgpack = None
        self._receiver_ready = False
        self._last_packet_time = 0.0
        self._packet_count = 0
        self._debug_counter = 0
        self._last_debug_wall_time = 0.0
        self._first_packet_logged = False
        self._first_target_logged = False
        self._last_target_step_delta_absmax = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._last_root_xy_step_norm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._settle_step_counter = 0
        self._unlock_blend_counter = 0
        self._unlock_blend_total = 0
        self._root_pose_anchor: torch.Tensor | None = None
        self._root_velocity_zero = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._root_anchor_logged = False
        self._root_pose_unlocked = False
        self._auto_unlock_after_settle = False
        self._base_trans_target: torch.Tensor | None = None
        self._initial_base_target_pos: torch.Tensor | None = None
        self._previous_base_trans_target: torch.Tensor | None = None
        self._base_translation_goal_xy: torch.Tensor | None = None
        self._last_root_xy_target: torch.Tensor | None = None
        self._previous_lower_body_target: torch.Tensor | None = None
        self._last_reference_target: torch.Tensor | None = None
        self._last_root_motion_source = "none"
        self._base_quat_target: torch.Tensor | None = None
        self._initial_base_target_yaw: torch.Tensor | None = None
        self._root_anchor_yaw: torch.Tensor | None = None
        self._last_root_yaw_target: torch.Tensor | None = None
        self._last_root_z_target: torch.Tensor | None = None
        self._anchor_knee: torch.Tensor | None = None
        knee_ids = [idx for idx, name in enumerate(self._joint_names) if "knee" in name]
        self._knee_joint_indices = torch.tensor(knee_ids, device=self.device, dtype=torch.long)
        if bool(cfg.keep_feet_on_ground):
            if self._knee_joint_indices.numel() == 0:
                self._log_warning("keep_feet_on_ground enabled but no knee joints found; disabling squat")
            else:
                self._log_info(
                    f"keep_feet_on_ground squat via knee joints "
                    f"{[self._joint_names[i] for i in knee_ids]} scale={cfg.foot_ground_scale} max_drop={cfg.max_squat_drop_m}"
                )
        self._last_target_field = "<none>"
        self._last_reference_field = "<none>"
        self._invalid_payload_fields_logged: set[str] = set()

        self._target_order = str(cfg.target_order).lower()
        if self._target_order not in ("mujoco", "isaaclab"):
            raise ValueError(f"target_order must be 'mujoco' or 'isaaclab', got {cfg.target_order!r}")

        self._isaac_to_mujoco_index = torch.tensor(
            SONIC_G1_ISAACLAB_TO_MUJOCO_DOF, device=self.device, dtype=torch.long
        )
        self._connect_receiver()
        self._log_info(
            f"asset={cfg.asset_name} endpoint={cfg.endpoint} topic={cfg.topic!r} "
            f"field={cfg.target_field!r} target_order={self._target_order} "
            f"resolved={len(self._joint_ids)} joints receiver_ready={self._receiver_ready} "
            f"reference_lower_body={bool(cfg.blend_reference_lower_body)} "
            f"reference_joints={len(reference_joint_ids)} "
            f"hold_reference={bool(cfg.hold_last_reference_target)} "
            f"follow_base_translation={bool(cfg.follow_base_translation_target)} "
            f"synthetic_base_motion={bool(cfg.synthetic_base_motion_from_lower_body)} "
            f"follow_base_yaw={bool(cfg.follow_base_yaw_target)} "
            f"env_endpoint={os.environ.get('SONIC_DEPLOY_ENDPOINT', '<unset>')!r} "
            f"env_topic={os.environ.get('SONIC_DEPLOY_TOPIC', '<unset>')!r}"
        )

    def __del__(self):
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
        try:
            super().__del__()
        except AttributeError:
            pass

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @staticmethod
    def _format_log_message(message: str) -> str:
        return f"[SonicDeployTarget] {message}"

    def _log_info(self, message: str) -> None:
        formatted = self._format_log_message(message)
        logger.info(formatted)
        print(f"[IsaacLab] {formatted}")

    def _log_warning(self, message: str) -> None:
        formatted = self._format_log_message(f"WARNING {message}")
        logger.warning(formatted)
        print(f"[IsaacLab] {formatted}")

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            joint_ids, matched_names = self._asset.find_joints([f"^{joint_name}$"])
            if len(joint_ids) != 1:
                raise ValueError(
                    f"Expected exactly one joint match for '{joint_name}' on asset '{self.cfg.asset_name}', "
                    f"but got {len(joint_ids)} matches: {matched_names}"
                )
            resolved_ids.append(int(joint_ids[0]))
            resolved_names.append(matched_names[0])
        return resolved_ids, resolved_names

    def _connect_receiver(self) -> None:
        try:
            import msgpack
            import zmq
        except ModuleNotFoundError as exc:
            self._log_warning(f"{exc.name} is not installed; holding sonic_robot default pose.")
            return

        self._log_info(f"connecting endpoint={self.cfg.endpoint} topic={self.cfg.topic!r}")
        self._zmq = zmq
        self._msgpack = msgpack
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        try:
            self._socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.ZMQError:
            pass
        self._socket.connect(self.cfg.endpoint)
        self._receiver_ready = True
        self._log_info(f"subscriber connected endpoint={self.cfg.endpoint} topic={self.cfg.topic!r}")

    def _drain_latest_packet(self) -> dict | None:
        if not self._receiver_ready or self._socket is None or self._zmq is None or self._msgpack is None:
            return None

        latest_payload = None
        while True:
            try:
                raw = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break

            if not raw.startswith(self._topic):
                continue
            latest_payload = raw[len(self._topic):]

        if latest_payload is None:
            return None

        try:
            payload = self._msgpack.unpackb(latest_payload, raw=False, strict_map_key=False)
        except Exception as exc:
            self._log_warning(f"failed to unpack msgpack payload: {exc}")
            return None

        if not isinstance(payload, dict):
            self._log_warning(f"expected msgpack map, got {type(payload).__name__}")
            return None
        self._last_packet_time = time.monotonic()
        self._packet_count += 1
        if not self._first_packet_logged:
            self._first_packet_logged = True
            self._log_info(
                f"first packet received keys={list(payload.keys())} "
                f"payload_fields={len(payload)}"
            )
        return payload

    @staticmethod
    def _append_field_name(field_names: list[str], field_name: str) -> None:
        field_name = field_name.strip()
        if field_name and field_name not in field_names:
            field_names.append(field_name)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _log_invalid_payload_field_once(self, field_name: str, reason: str) -> None:
        key = f"{field_name}:{reason}"
        if key in self._invalid_payload_fields_logged:
            return
        self._invalid_payload_fields_logged.add(key)
        self._log_warning(f"ignoring deploy field {field_name!r}: {reason}")

    @staticmethod
    def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
        half_yaw = 0.5 * yaw
        quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
        quat[:, 0] = torch.cos(half_yaw)
        quat[:, 3] = torch.sin(half_yaw)
        return quat

    def _primary_target_field_names(self) -> list[str]:
        field_names: list[str] = []
        # Allow comma-separated overrides such as
        # SONIC_DEPLOY_TARGET_FIELD=last_action,body_q_target during debugging.
        for field_name in str(self.cfg.target_field).split(","):
            self._append_field_name(field_names, field_name)
        if self.cfg.fallback_to_last_action:
            self._append_field_name(field_names, "last_action")
        if getattr(self.cfg, "fallback_to_body_q_target", True):
            self._append_field_name(field_names, "body_q_target")
        if self.cfg.fallback_to_measured:
            self._append_field_name(field_names, "body_q_measured")
            self._append_field_name(field_names, "body_q")
        return field_names

    def _extract_joint_target_from_fields(
        self, payload: dict, field_names: list[str], *, log_missing: bool
    ) -> tuple[torch.Tensor | None, str | None]:
        found_field = False
        for field_name in field_names:
            if field_name not in payload:
                continue
            found_field = True
            target_values = payload[field_name]
            target = torch.tensor(target_values, device=self.device, dtype=torch.float32).flatten()
            if target.numel() != len(SONIC_G1_29DOF_JOINT_ORDER):
                self._log_invalid_payload_field_once(
                    field_name, f"has {target.numel()} values, expected 29"
                )
                continue
            finite_mask = torch.isfinite(target)
            if not bool(finite_mask.all()):
                bad_count = int((~finite_mask).sum().item())
                self._log_invalid_payload_field_once(
                    field_name, f"contains {bad_count} non-finite value(s)"
                )
                continue
            packet_target_order = str(payload.get("target_order", self._target_order)).lower()
            if packet_target_order not in ("mujoco", "isaaclab"):
                self._log_warning(
                    f"payload target_order={packet_target_order!r} is invalid; using cfg order {self._target_order!r}"
                )
                packet_target_order = self._target_order
            if packet_target_order == "mujoco":
                target = target[self._isaac_to_mujoco_index]
            return target.unsqueeze(0).repeat(self.num_envs, 1), str(field_name)
        if not found_field:
            if log_missing and self._packet_count <= 3:
                self._log_warning(
                    f"none of target fields {field_names} found; payload keys={list(payload.keys())}"
                )
        return None, None

    def _extract_base_quat_target(self, payload: dict) -> torch.Tensor | None:
        field_name = str(self.cfg.base_quat_target_field).strip()
        if not field_name or field_name not in payload:
            return None
        quat = torch.tensor(payload[field_name], device=self.device, dtype=torch.float32).flatten()
        if quat.numel() != 4:
            if self._packet_count <= 3:
                self._log_warning(f"field {field_name!r} has {quat.numel()} values, expected quaternion size 4")
            return None
        if not bool(torch.isfinite(quat).all()):
            self._log_invalid_payload_field_once(field_name, "contains non-finite quaternion value(s)")
            return None
        quat = quat / torch.linalg.norm(quat).clamp_min(1.0e-6)
        if not bool(torch.isfinite(quat).all()):
            self._log_invalid_payload_field_once(field_name, "normalizes to non-finite quaternion value(s)")
            return None
        return quat.unsqueeze(0).repeat(self.num_envs, 1)

    def _extract_base_trans_target(self, payload: dict) -> torch.Tensor | None:
        field_name = str(self.cfg.base_trans_target_field).strip()
        if not field_name or field_name not in payload:
            return None
        pos = torch.tensor(payload[field_name], device=self.device, dtype=torch.float32).flatten()
        if pos.numel() != 3:
            if self._packet_count <= 3:
                self._log_warning(f"field {field_name!r} has {pos.numel()} values, expected translation size 3")
            return None
        if not bool(torch.isfinite(pos).all()):
            self._log_invalid_payload_field_once(field_name, "contains non-finite translation value(s)")
            return None
        if torch.max(torch.abs(pos)).item() <= 1.0e-4:
            return None
        return pos.unsqueeze(0).repeat(self.num_envs, 1)

    def _extract_target(self, payload: dict) -> torch.Tensor | None:
        target, target_field = self._extract_joint_target_from_fields(
            payload, self._primary_target_field_names(), log_missing=True
        )
        if target is None:
            return None

        reference_field_names: list[str] = []
        for field_name in str(self.cfg.reference_target_field).split(","):
            self._append_field_name(reference_field_names, field_name)
        reference, reference_field = self._extract_joint_target_from_fields(
            payload, reference_field_names, log_missing=False
        )
        reference_used = False
        reference_to_apply = None
        reference_label = "<none>"
        if (
            reference is not None
            and bool(self.cfg.blend_reference_lower_body)
            and self._reference_joint_indices.numel() > 0
        ):
            reference_lower = reference[:, self._reference_joint_indices]
            if torch.max(torch.abs(reference_lower)).item() > 1.0e-4:
                self._last_reference_target = reference.clone()
                reference_to_apply = reference
                reference_label = str(reference_field)
        if (
            reference_to_apply is None
            and bool(self.cfg.hold_last_reference_target)
            and self._last_reference_target is not None
        ):
            reference_to_apply = self._last_reference_target
            reference_label = f"{self.cfg.reference_target_field}:hold"
        if reference_to_apply is not None and self._reference_joint_indices.numel() > 0:
            target[:, self._reference_joint_indices] = reference_to_apply[:, self._reference_joint_indices]
            reference_used = True

        self._last_target_field = str(target_field)
        self._last_reference_field = reference_label if reference_used else "<none>"
        base_trans_target = self._extract_base_trans_target(payload)
        if base_trans_target is not None:
            self._base_trans_target = base_trans_target
        base_quat_target = self._extract_base_quat_target(payload)
        if base_quat_target is not None:
            self._base_quat_target = base_quat_target

        if not self._first_target_logged:
            self._first_target_logged = True
            target_cpu = target.detach().cpu()
            ref_text = (
                f" ref_field={self._last_reference_field!r} ref_joints={int(self._reference_joint_indices.numel())}"
                if reference_used
                else ""
            )
            self._log_info(
                f"first target parsed field={self._last_target_field!r}{ref_text} "
                f"mean={target_cpu.mean():+.4f} absmax={target_cpu.abs().max():.4f}"
            )
        return target

    def _apply_target_rate_limit(self, target: torch.Tensor) -> torch.Tensor:
        # Always clamp targets to joint limits first: a saturated/fallen policy can
        # emit 7+ rad targets and the PD would slam joints into their stops.
        target = torch.clamp(target, min=self._target_pos_lower, max=self._target_pos_upper)
        max_delta = float(self.cfg.target_rate_limit_rad_per_step)
        if bool(self.cfg.rate_limit_only_while_root_locked) and (
            self._root_pose_unlocked or self._unlock_blend_total > 0
        ):
            # Closed loop handover: the policy's raw action is the contract (MuJoCo
            # deploy has no slew limiter), but an instant bypass discharges the
            # backlog accumulated under the locked-phase limit in one step (observed
            # 1.8 rad snap that kicked the robot over at realtime). Release the limit
            # exponentially instead: double the allowed step every 5 env steps,
            # starting at unlock (blend included), fully open after ~1 s.
            self._post_unlock_steps += 1
            # Cap the exponent: the counter keeps running for the rest of the episode
            # and float pow overflows at 2**1024 (OverflowError ~102 s after unlock).
            exponent = min(float(self._post_unlock_steps) / 5.0, 64.0)
            max_delta = max_delta * (2.0 ** exponent)
            if max_delta >= 50.0:
                max_delta = 0.0
        if max_delta <= 0.0:
            self._last_target_step_delta_absmax = torch.max(
                torch.abs(target - self._processed_actions), dim=-1
            ).values
            return target

        delta = torch.clamp(target - self._processed_actions, min=-max_delta, max=max_delta)
        self._last_target_step_delta_absmax = torch.max(torch.abs(delta), dim=-1).values
        return self._processed_actions + delta

    def _stabilize_root_pose(self) -> None:
        if not self.cfg.stabilize_root_pose or self._root_pose_unlocked:
            return
        # Unlock blending: gradually release root velocity damping over N steps
        # while letting PhysX control the root pose. Unlike position lerping
        # (which is a no-op because anchor == physX_current when locked), this
        # progressively scales down velocity damping so physics takes over smoothly.
        if self._unlock_blend_total > 0 and self._root_pose_anchor is not None:
            self._unlock_blend_counter += 1
            alpha = min(float(self._unlock_blend_counter) / float(self._unlock_blend_total), 1.0)
            # Get current PhysX velocity and damp it. Damping decreases from full→none.
            # - XY linear velocity: alpha³ damping (aggressive, prevents horizontal drift at start)
            # - Z linear velocity: alpha¹ damping (light, allows quick ground settling)
            # - Angular velocity: alpha² damping (moderate)
            vel = self._asset.data.root_vel_w.clone()
            vel[:, 0] *= alpha * alpha * alpha  # X
            vel[:, 1] *= alpha * alpha * alpha  # Y
            vel[:, 2] *= alpha                   # Z (light damping)
            vel[:, 3:] *= alpha * alpha          # angular
            self._asset.write_root_velocity_to_sim(vel)
            # Do NOT write root pose — let PhysX move the root freely.
            # The velocity damping above prevents explosive motion at blend start.
            if self._unlock_blend_counter >= self._unlock_blend_total:
                total = self._unlock_blend_total
                self._root_pose_unlocked = True
                self._unlock_blend_total = 0
                self._unlock_blend_counter = 0
                # No settle restart and no final velocity write here: once the root is
                # free, every step without deploy targets is a step of passive PD
                # standing, which is unconditionally unstable (ankle stiffness ≪ mgh).
                # The policy must stay in control through the handover.
                self._log_info(f"root pose unlock blend complete ({total} steps); root is now free")
            return
        if self._root_pose_anchor is None:
            self._root_pose_anchor = torch.cat(
                [self._asset.data.root_pos_w, self._asset.data.root_quat_w], dim=-1
            ).clone()
            self._root_anchor_yaw = self._yaw_from_quat(self._root_pose_anchor[:, 3:7]).clone()
            self._last_root_xy_target = self._root_pose_anchor[:, :2].clone()
            self._last_root_yaw_target = self._root_anchor_yaw.clone()
            if self._knee_joint_indices.numel() > 0:
                self._anchor_knee = self._processed_actions[:, self._knee_joint_indices].mean(dim=-1).clone()
            if not self._root_anchor_logged:
                anchor = self._root_pose_anchor[0].detach().cpu().tolist()
                self._log_info(
                    "root pose stabilized at "
                    f"pos=({anchor[0]:+.3f}, {anchor[1]:+.3f}, {anchor[2]:+.3f})"
                )
                self._root_anchor_logged = True
        root_pose_target = self._root_pose_anchor.clone()
        desired_xy = (
            self._last_root_xy_target.clone()
            if self._last_root_xy_target is not None
            else self._root_pose_anchor[:, :2].clone()
        )
        self._last_root_xy_step_norm.zero_()
        self._last_root_motion_source = "none"
        base_desired_xy = None

        if bool(self.cfg.follow_base_translation_target) and self._base_trans_target is not None:
            if self._initial_base_target_pos is None:
                self._initial_base_target_pos = self._base_trans_target.clone()
                self._previous_base_trans_target = self._base_trans_target.clone()
            if self._previous_base_trans_target is None:
                self._previous_base_trans_target = self._base_trans_target.clone()

            base_target_step = torch.linalg.norm(
                self._base_trans_target[:, :2] - self._previous_base_trans_target[:, :2], dim=-1
            )
            if torch.max(base_target_step).item() > 1.0e-6:
                # deploy 端 base_trans_target 位于 deploy odom 系（机器人初始朝向
                # 为其 +X）。锚定 yaw 非零时（如 banyun 工位面向 +Y 出生）需把
                # 位移旋进世界系，否则机器人会相对自身朝向"横着走"。yaw 跟随
                # （下方）本就是增量式，天然与坐标系无关，无需同款处理。
                delta_xy = (
                    self._base_trans_target[:, :2] - self._initial_base_target_pos[:, :2]
                ) * float(self.cfg.base_translation_scale)
                if self._initial_base_target_yaw is None and self._base_quat_target is not None:
                    self._initial_base_target_yaw = self._yaw_from_quat(self._base_quat_target).clone()
                yaw_offset = (
                    self._root_anchor_yaw.clone()
                    if self._root_anchor_yaw is not None
                    else torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                )
                if self._initial_base_target_yaw is not None:
                    yaw_offset = yaw_offset - self._initial_base_target_yaw
                cos_o = torch.cos(yaw_offset)
                sin_o = torch.sin(yaw_offset)
                self._base_translation_goal_xy = self._root_pose_anchor[:, :2] + torch.stack(
                    [
                        cos_o * delta_xy[:, 0] - sin_o * delta_xy[:, 1],
                        sin_o * delta_xy[:, 0] + cos_o * delta_xy[:, 1],
                    ],
                    dim=-1,
                )
            self._previous_base_trans_target = self._base_trans_target.clone()

            base_desired_xy = self._base_translation_goal_xy
        if base_desired_xy is not None:
            base_goal_xy = base_desired_xy
            max_delta = float(self.cfg.base_translation_rate_limit_m_per_step)
            if max_delta > 0.0 and self._last_root_xy_target is not None:
                xy_delta = base_desired_xy - self._last_root_xy_target
                delta_norm = torch.linalg.norm(xy_delta, dim=-1, keepdim=True).clamp_min(1.0e-6)
                scale = torch.clamp(max_delta / delta_norm, max=1.0)
                base_desired_xy = self._last_root_xy_target + xy_delta * scale
            if (
                self._last_root_xy_target is None
                or torch.max(torch.linalg.norm(base_desired_xy - self._last_root_xy_target, dim=-1)).item() > 1.0e-6
            ):
                desired_xy = base_desired_xy
                self._last_root_motion_source = "base_trans"
            if torch.max(torch.linalg.norm(base_goal_xy - desired_xy, dim=-1)).item() <= 1.0e-5:
                self._base_translation_goal_xy = None

        if (
            self._last_root_motion_source == "none"
            and bool(self.cfg.synthetic_base_motion_from_lower_body)
            and self._reference_joint_indices.numel() > 0
        ):
            lower_target = self._processed_actions[:, self._reference_joint_indices]
            if self._previous_lower_body_target is None:
                self._previous_lower_body_target = lower_target.clone()
            lower_activity = torch.mean(torch.abs(lower_target - self._previous_lower_body_target), dim=-1)
            self._previous_lower_body_target = lower_target.clone()
            synthetic_step = torch.clamp(
                (lower_activity - float(self.cfg.synthetic_base_motion_deadzone))
                * float(self.cfg.synthetic_base_motion_gain),
                min=0.0,
                max=float(self.cfg.synthetic_base_motion_max_step_m),
            )
            if torch.max(synthetic_step).item() > 1.0e-6:
                motion_yaw = (
                    self._last_root_yaw_target
                    if self._last_root_yaw_target is not None
                    else self._root_anchor_yaw
                )
                if motion_yaw is None:
                    motion_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                forward = torch.stack([torch.cos(motion_yaw), torch.sin(motion_yaw)], dim=-1)
                desired_xy = desired_xy + forward * synthetic_step.unsqueeze(-1)
                self._last_root_motion_source = "synthetic"

        if self._last_root_xy_target is not None:
            self._last_root_xy_step_norm = torch.linalg.norm(desired_xy - self._last_root_xy_target, dim=-1)
        self._last_root_xy_target = desired_xy.clone()
        root_pose_target[:, :2] = desired_xy

        if bool(self.cfg.follow_base_height_target) and self._base_trans_target is not None:
            if self._initial_base_target_pos is None:
                self._initial_base_target_pos = self._base_trans_target.clone()
            desired_z = self._root_pose_anchor[:, 2] + (
                self._base_trans_target[:, 2] - self._initial_base_target_pos[:, 2]
            ) * float(self.cfg.base_height_scale)
            max_delta = float(self.cfg.base_height_rate_limit_m_per_step)
            if max_delta > 0.0 and self._last_root_z_target is not None:
                z_delta = torch.clamp(
                    desired_z - self._last_root_z_target, min=-max_delta, max=max_delta
                )
                desired_z = self._last_root_z_target + z_delta
            self._last_root_z_target = desired_z.clone()
            root_pose_target[:, 2] = desired_z

        if bool(self.cfg.keep_feet_on_ground) and self._knee_joint_indices.numel() > 0:
            knee = self._processed_actions[:, self._knee_joint_indices].mean(dim=-1)
            if self._anchor_knee is None:
                self._anchor_knee = knee.clone()
            squat_depth = torch.clamp(
                (knee - self._anchor_knee) * float(self.cfg.foot_ground_scale),
                min=0.0,
                max=float(self.cfg.max_squat_drop_m),
            )
            desired_z = self._root_pose_anchor[:, 2] - squat_depth
            max_delta = float(self.cfg.base_height_rate_limit_m_per_step)
            if max_delta > 0.0 and self._last_root_z_target is not None:
                z_delta = torch.clamp(
                    desired_z - self._last_root_z_target, min=-max_delta, max=max_delta
                )
                desired_z = self._last_root_z_target + z_delta
            self._last_root_z_target = desired_z.clone()
            root_pose_target[:, 2] = desired_z

        if bool(self.cfg.follow_base_yaw_target) and self._base_quat_target is not None:
            base_target_yaw = self._yaw_from_quat(self._base_quat_target)
            if self._initial_base_target_yaw is None:
                self._initial_base_target_yaw = base_target_yaw.clone()
            desired_yaw = self._root_anchor_yaw + self._wrap_to_pi(base_target_yaw - self._initial_base_target_yaw)
            max_delta = float(self.cfg.base_yaw_rate_limit_rad_per_step)
            if max_delta > 0.0 and self._last_root_yaw_target is not None:
                yaw_delta = torch.clamp(
                    self._wrap_to_pi(desired_yaw - self._last_root_yaw_target),
                    min=-max_delta,
                    max=max_delta,
                )
                desired_yaw = self._last_root_yaw_target + yaw_delta
            self._last_root_yaw_target = desired_yaw.clone()
            root_pose_target[:, 3:7] = self._quat_from_yaw(desired_yaw)
        # Physics mode (lock_root_z=False): preserve PhysX Z so the robot can settle to the ground.
        # XY and yaw are still anchored to prevent horizontal drift; Z is left to physics.
        if not self.cfg.lock_root_z:
            root_pose_target[:, 2] = self._asset.data.root_pos_w[:, 2]
        self._asset.write_root_pose_to_sim(root_pose_target)
        if self.cfg.lock_root_z:
            self._asset.write_root_velocity_to_sim(self._root_velocity_zero)
        else:
            # Physics mode: only damp XY + angular velocity; leave Z velocity to PhysX
            # so the robot can fall and settle to the ground.
            vel = self._asset.data.root_vel_w.clone()
            vel[:, :2] = 0.0
            vel[:, 3:] = 0.0
            self._asset.write_root_velocity_to_sim(vel)

    def unlock_root_pose(self) -> None:
        if not self.cfg.stabilize_root_pose:
            self._log_info("root pose stabilization is disabled; unlock request ignored")
            return
        if self._root_pose_unlocked:
            return
        if self._unlock_blend_total > 0:
            return  # already blending
        blend = int(self.cfg.unlock_blend_steps)
        if blend > 0 and self._root_pose_anchor is not None:
            # Start blend: gradually release root velocity damping. Deploy targets
            # keep flowing throughout — re-running the settle phase here would lock
            # the policy out for the exact window where the root goes free, and
            # passive PD standing cannot survive that window.
            self._unlock_blend_total = blend
            self._unlock_blend_counter = 0
            self._post_unlock_steps = 0
            self._log_info(f"root pose unlock blending started ({blend} steps); deploy targets stay live")
        else:
            # Instant unlock (no blending configured).
            self._root_pose_unlocked = True
            self._post_unlock_steps = 0
            self._log_info("root pose unlocked by operator (instant)")

    def recover_standing(self) -> None:
        """摔倒后原地恢复站立：机器人本体回直立默认姿态，重走锁根启动序列。

        SonicSolo/SonicFullscene 场景没有恢复机器人本体的 reset 事件，
        env.reset()（R 键）若不带 reset_scene_to_default 只会复位本 term 的
        状态机——摔倒的机器人会在原地被重新锁根，站不起来。本方法直接写机器人
        状态：保留当前 XY 与 yaw（XR 第一视角视点连续、不丢行走进度），root 回
        默认高度、roll/pitch 归零，关节回默认站姿、速度清零；随后 reset() 重走
        启动状态机（重新锁根 → settle）。deploy 侧无需重启：状态发布持续，
        settle 期间 deploy 目标被 drain，物理模式解锁前 policy 已重新看到约 1s
        的站立状态流（与冷启动序列一致）。

        解锁交接（对齐 MuJoCo 参考 mj_resetData 后 policy 直接接管的语义）：
        若摔倒前 root 已解锁（或在解锁 blend 中）且 ``auto_unlock_after_recover``
        为 True，settle 完成后自动重新解锁；否则保持锁根等 U/START。冷启动
        （从未解锁过）永远走手动门控。
        """
        was_unlocked = self._root_pose_unlocked or self._unlock_blend_total > 0
        default_root_state = self._asset.data.default_root_state.clone()
        root_pos = default_root_state[:, 0:3] + self._env.scene.env_origins
        root_pos[:, 0:2] = self._asset.data.root_pos_w[:, 0:2]
        root_quat = self._quat_from_yaw(self._yaw_from_quat(self._asset.data.root_quat_w))
        self._asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
        self._asset.write_root_velocity_to_sim(self._root_velocity_zero)
        self._asset.write_joint_state_to_sim(
            self._asset.data.default_joint_pos.clone(),
            torch.zeros_like(self._asset.data.default_joint_vel),
        )
        self.reset()
        pending_auto_unlock = bool(self.cfg.auto_unlock_after_recover) and was_unlocked
        if pending_auto_unlock and int(self.cfg.startup_settle_steps) <= 0:
            # 没有 settle 阶段（fixed-root 配置）：立即交还控制权。
            self.unlock_root_pose()
            pending_auto_unlock = False
        self._auto_unlock_after_settle = pending_auto_unlock
        unlock_text = (
            "will auto re-unlock after settle"
            if pending_auto_unlock
            else "press U/START to unlock"
        )
        self._log_info(
            "recover standing: root uprighted in place (kept XY+yaw), joints reset to default; "
            f"root re-locked and settle restarted — {unlock_text}"
        )

    def _maybe_auto_recover_fall(self) -> None:
        """摔倒自动恢复：对齐 MuJoCo 参考环境 base_sim.check_fall()。

        MuJoCo 侧每个 sim step 检测 ``qpos[2] < 0.2``，命中即 mj_resetData 自动
        回初始状态、policy 继续跑，无需人工干预。这里在每个 env step（50Hz）做
        同阈值检测，命中调 recover_standing()（原地扶正而非回出生点）。
        """
        if not bool(self.cfg.auto_recover_on_fall):
            return
        threshold = float(self.cfg.fall_root_height_m)
        if threshold <= 0.0:
            return
        root_height = self._asset.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        min_height = torch.min(root_height).item()
        if min_height >= threshold:
            return
        self._log_warning(
            f"robot has fallen (root height {min_height:.3f} m < {threshold:.2f} m); auto recovering standing"
        )
        self.recover_standing()

    def process_actions(self, actions: torch.Tensor):
        # 摔倒自动恢复检测（MuJoCo check_fall 等价物）。命中后本步直接进入下面的
        # settle 分支驱动默认站姿。
        self._maybe_auto_recover_fall()
        settle_steps = int(self.cfg.startup_settle_steps)
        if settle_steps > 0 and self._settle_step_counter < settle_steps:
            # Startup/post-unlock settle: gradually transition joints to default
            # standing pose and drain stale deploy packets. Each step moves one
            # rate-limited increment towards default so the transition is smooth.
            self._drain_latest_packet()
            default_target = self._default_joint_pos.clone()
            self._processed_actions = self._apply_target_rate_limit(default_target)
            self._settle_step_counter += 1
            if self._settle_step_counter == settle_steps:
                self._log_info(f"startup settle complete ({settle_steps} steps); begin consuming deploy targets")
                if self._auto_unlock_after_settle:
                    # 摔倒恢复的自动交接：settle 完成即重新解锁（带 blend），
                    # 对齐 MuJoCo reset 后 policy 直接接管的行为。
                    self._auto_unlock_after_settle = False
                    self._log_info("auto re-unlocking root after recovery settle")
                    self.unlock_root_pose()
            return
        # hold_after_unlock: after root is freed, ignore deploy targets and drive joints
        # toward the default standing pose (rate-limited). This isolates physics-only
        # standing — with soft SONIC gains it is expected to fall (ankle stiffness ≪ mgh);
        # use only as a diagnostic. Drain packets to avoid queue buildup.
        if self._root_pose_unlocked and self.cfg.hold_after_unlock:
            self._drain_latest_packet()
            self._processed_actions = self._apply_target_rate_limit(self._default_joint_pos.clone())
            return
        payload = self._drain_latest_packet()
        if payload is not None:
            target = self._extract_target(payload)
            if target is not None:
                self._processed_actions = self._apply_target_rate_limit(target)
                if self.cfg.clip is not None:
                    self._processed_actions = torch.clamp(
                        self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                    )

        stale_timeout_s = float(self.cfg.stale_timeout_s)
        if (
            stale_timeout_s > 0.0
            and self._last_packet_time > 0.0
            and time.monotonic() - self._last_packet_time > stale_timeout_s
        ):
            self._last_packet_time = 0.0
            self._log_warning(f"no deploy target received for {stale_timeout_s:.2f}s; holding last target.")

        self._debug_counter += 1
        if self.cfg.debug_log_interval > 0 and self._debug_counter % self.cfg.debug_log_interval == 0:
            proc = self._processed_actions[0].detach().cpu()
            bt = self._base_trans_target
            bt_text = (
                f"({bt[0, 0].item():+.3f},{bt[0, 1].item():+.3f},{bt[0, 2].item():+.3f})"
                if bt is not None
                else "<none>"
            )
            rz_text = (
                f"{self._last_root_z_target[0].item():+.3f}"
                if self._last_root_z_target is not None
                else "<none>"
            )
            # Wall-clock env rate: deploy paces its gait phase at wall 50 Hz, so the
            # closed loop is only meaningful when this reads close to 50.
            now = time.monotonic()
            if self._last_debug_wall_time > 0.0:
                env_hz = float(self.cfg.debug_log_interval) / max(now - self._last_debug_wall_time, 1.0e-6)
                hz_text = f"{env_hz:.1f}"
            else:
                hz_text = "n/a"
            self._last_debug_wall_time = now
            # Measured robot state, not the deploy target: base_trans above is the
            # commanded base pose and never falls — a real fall is only visible here.
            rp = self._asset.data.root_pos_w[0]
            gb = self._asset.data.projected_gravity_b[0]
            tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -gb[2].item()))))
            real_text = f"({rp[0].item():+.3f},{rp[1].item():+.3f},{rp[2].item():+.3f}) tilt={tilt_deg:.1f}deg"
            self._log_info(
                f"step={self._debug_counter} packets={self._packet_count} "
                f"field={self._last_target_field} ref={self._last_reference_field} "
                f"target_mean={proc.mean():+.4f} target_absmax={proc.abs().max():.4f} "
                f"step_delta_absmax={self._last_target_step_delta_absmax[0].item():.4f} "
                f"root_xy_step={self._last_root_xy_step_norm[0].item():.4f} "
                f"root_src={self._last_root_motion_source} "
                f"base_trans={bt_text} root_z={rz_text} real_root={real_text} env_hz={hz_text}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._stabilize_root_pose()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._processed_actions.copy_(self._default_joint_pos)
            self._last_target_step_delta_absmax.zero_()
            self._last_root_xy_step_norm.zero_()
            self._root_pose_anchor = None
            self._root_pose_unlocked = False
            self._auto_unlock_after_settle = False
            self._unlock_blend_counter = 0
            self._unlock_blend_total = 0
            self._settle_step_counter = 0
            self._post_unlock_steps = 0
            self._base_trans_target = None
            self._initial_base_target_pos = None
            self._previous_base_trans_target = None
            self._base_translation_goal_xy = None
            self._last_root_xy_target = None
            self._previous_lower_body_target = None
            self._last_reference_target = None
            self._last_root_motion_source = "none"
            self._base_quat_target = None
            self._initial_base_target_yaw = None
            self._root_anchor_yaw = None
            self._last_root_yaw_target = None
            self._last_root_z_target = None
            self._anchor_knee = None
            return
        self._processed_actions[env_ids] = self._default_joint_pos[env_ids]
        self._last_target_step_delta_absmax[env_ids] = 0.0
        self._last_root_xy_step_norm[env_ids] = 0.0
        self._root_pose_anchor = None
        self._root_pose_unlocked = False
        self._auto_unlock_after_settle = False
        self._unlock_blend_counter = 0
        self._unlock_blend_total = 0
        self._settle_step_counter = 0
        self._post_unlock_steps = 0
        self._base_trans_target = None
        self._initial_base_target_pos = None
        self._previous_base_trans_target = None
        self._base_translation_goal_xy = None
        self._last_root_xy_target = None
        self._previous_lower_body_target = None
        self._last_reference_target = None
        self._last_root_motion_source = "none"
        self._base_quat_target = None
        self._initial_base_target_yaw = None
        self._root_anchor_yaw = None
        self._last_root_yaw_target = None
        self._last_root_z_target = None
        self._anchor_knee = None


class SonicRobotStatePublisherAction(ActionTerm):
    """Publish ``sonic_robot`` physical state over ZMQ/msgpack for a C++ Unitree LowState bridge."""

    cfg: SonicRobotStatePublisherActionCfg
    _asset: Articulation

    def __init__(self, cfg: SonicRobotStatePublisherActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = self._raw_actions
        self._mujoco_to_isaac_index = torch.tensor(
            SONIC_G1_MUJOCO_TO_ISAACLAB_DOF, device=self.device, dtype=torch.long
        )
        self._target_order = str(cfg.target_order).lower()
        if self._target_order not in ("mujoco", "isaaclab"):
            raise ValueError(f"target_order must be 'mujoco' or 'isaaclab', got {cfg.target_order!r}")

        self._zmq = None
        self._msgpack = None
        self._context = None
        self._socket = None
        self._publisher_ready = False
        self._packet_count = 0
        self._debug_counter = 0
        self._connect_publisher()
        self._log_info(
            f"asset={cfg.asset_name} bind={cfg.bind_endpoint} topic={cfg.topic!r} "
            f"target_order={self._target_order} resolved={len(self._joint_ids)} joints "
            f"publisher_ready={self._publisher_ready}"
        )

    def __del__(self):
        if self._socket is not None:
            try:
                self._socket.close(0)
            except Exception:
                pass
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
        try:
            super().__del__()
        except AttributeError:
            pass

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @staticmethod
    def _format_log_message(message: str) -> str:
        return f"[SonicRobotStatePublisher] {message}"

    def _log_info(self, message: str) -> None:
        formatted = self._format_log_message(message)
        logger.info(formatted)
        print(f"[IsaacLab] {formatted}")

    def _log_warning(self, message: str) -> None:
        formatted = self._format_log_message(f"WARNING {message}")
        logger.warning(formatted)
        print(f"[IsaacLab] {formatted}")

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            joint_ids, matched_names = self._asset.find_joints([f"^{joint_name}$"])
            if len(joint_ids) != 1:
                raise ValueError(
                    f"Expected exactly one joint match for '{joint_name}' on asset '{self.cfg.asset_name}', "
                    f"but got {len(joint_ids)} matches: {matched_names}"
                )
            resolved_ids.append(int(joint_ids[0]))
            resolved_names.append(matched_names[0])
        return resolved_ids, resolved_names

    def _connect_publisher(self) -> None:
        try:
            import msgpack
            import zmq
        except ModuleNotFoundError as exc:
            self._log_warning(f"{exc.name} is not installed; sonic_robot state ZMQ publisher disabled.")
            return

        self._zmq = zmq
        self._msgpack = msgpack
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        try:
            self._socket.setsockopt(zmq.CONFLATE, 1)
        except zmq.ZMQError:
            pass
        try:
            self._socket.bind(self.cfg.bind_endpoint)
        except Exception as exc:
            self._log_warning(f"failed to bind {self.cfg.bind_endpoint}: {exc}")
            return
        self._publisher_ready = True

    def _publish_state(self) -> None:
        if not self._publisher_ready or self._socket is None or self._msgpack is None:
            return

        env_idx = 0
        joint_pos = self._asset.data.joint_pos[env_idx, self._joint_ids].detach().cpu()
        joint_vel = self._asset.data.joint_vel[env_idx, self._joint_ids].detach().cpu()
        joint_acc = self._asset.data.joint_acc[env_idx, self._joint_ids].detach().cpu()
        torque_src = getattr(self._asset.data, "applied_torque", None)
        if torque_src is None:
            joint_tau = torch.zeros_like(joint_pos)
        else:
            joint_tau = torque_src[env_idx, self._joint_ids].detach().cpu()
        root_quat = self._asset.data.root_quat_w[env_idx].detach().cpu()
        root_ang_vel = self._asset.data.root_ang_vel_b[env_idx].detach().cpu()

        # IMU accelerometer: specific force = (a_kinematic_w - g_w) in body frame.
        # g_w = [0,0,-9.81], so: a_imu_w = a_kinematic_w + [0,0,9.81].
        # Requires retain_accelerations=True on the articulation (set in physics mode).
        try:
            a_kin_w = self._asset.data.body_com_lin_acc_w[env_idx, 0, :3].detach().cpu()
            a_imu_w = a_kin_w + torch.tensor([0.0, 0.0, 9.81])
            root_accel_b = quat_apply_inverse(root_quat.unsqueeze(0), a_imu_w.unsqueeze(0)).squeeze(0).tolist()
        except Exception:
            root_accel_b = [0.0, 0.0, 9.81]

        if self._target_order == "mujoco":
            remap = self._mujoco_to_isaac_index.cpu()
            joint_pos = joint_pos[remap]
            joint_vel = joint_vel[remap]
            joint_acc = joint_acc[remap]
            joint_tau = joint_tau[remap]

        payload = {
            "source": "isaaclab_sonic_robot_state",
            "sequence": self._packet_count,
            "timestamp": time.time(),
            "mode_machine": int(self.cfg.mode_machine),
            "target_order": self._target_order,
            "joint_pos": joint_pos.tolist(),
            "joint_vel": joint_vel.tolist(),
            "joint_acc": joint_acc.tolist(),
            "joint_tau": joint_tau.tolist(),
            "root_quat_w": root_quat.tolist(),
            "root_ang_vel_b": root_ang_vel.tolist(),
            "root_accel_b": root_accel_b,
        }
        raw = self.cfg.topic.encode("utf-8") + self._msgpack.packb(payload, use_bin_type=True)
        try:
            self._socket.send(raw, flags=self._zmq.NOBLOCK)
        except Exception as exc:
            self._log_warning(f"failed to publish state packet: {exc}")
            return
        self._packet_count += 1

    def process_actions(self, actions: torch.Tensor):
        del actions

    def apply_actions(self):
        self._publish_state()
        self._debug_counter += 1
        if self.cfg.debug_log_interval > 0 and self._debug_counter % self.cfg.debug_log_interval == 0:
            self._log_info(f"step={self._debug_counter} packets={self._packet_count}")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids


class UnitreeDdsLowCmdAction(ActionTerm):
    """Drive sonic_robot from Unitree DDS LowCmd and publish simulated LowState.

    This makes IsaacLab behave like a virtual G1 for GR00T/SONIC deploy:
    deploy publishes `rt/lowcmd`, while IsaacLab publishes `rt/lowstate`.
    """

    cfg: UnitreeDdsLowCmdActionCfg
    _asset: Articulation

    def __init__(self, cfg: UnitreeDdsLowCmdActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_joint_pos.clone()
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._isaac_to_mujoco_index = torch.tensor(
            SONIC_G1_ISAACLAB_TO_MUJOCO_DOF, device=self.device, dtype=torch.long
        )
        self._mujoco_to_isaac_index = torch.tensor(
            SONIC_G1_MUJOCO_TO_ISAACLAB_DOF, device=self.device, dtype=torch.long
        )
        self._target_order = str(cfg.target_order).lower()
        if self._target_order not in ("mujoco", "isaaclab"):
            raise ValueError(f"target_order must be 'mujoco' or 'isaaclab', got {cfg.target_order!r}")

        self._low_cmd = None
        self._low_cmd_lock = threading.Lock()
        self._new_low_cmd = False
        self._lowcmd_count = 0
        self._lowstate_count = 0
        self._last_lowcmd_time = 0.0
        self._last_target_step_delta_absmax = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._debug_counter = 0
        self._first_lowcmd_logged = False
        self._root_pose_anchor: torch.Tensor | None = None
        self._root_velocity_zero = torch.zeros(self.num_envs, 6, device=self.device, dtype=torch.float32)
        self._root_anchor_logged = False
        self._root_pose_unlocked = False
        self._dds_ready = False
        self._lowstate_msg = None
        self._secondary_imu_msg = None
        self._lowstate_publisher = None
        self._secondary_imu_publisher = None
        self._lowcmd_subscriber = None
        self._connect_dds()
        self._log_info(
            f"asset={cfg.asset_name} domain_id={cfg.domain_id} "
            f"interface={cfg.network_interface or '<auto>'} lowcmd={cfg.lowcmd_topic!r} "
            f"lowstate={cfg.lowstate_topic!r} target_order={self._target_order} "
            f"resolved={len(self._joint_ids)} joints dds_ready={self._dds_ready}"
        )

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @staticmethod
    def _format_log_message(message: str) -> str:
        return f"[UnitreeDdsLowCmd] {message}"

    def _log_info(self, message: str) -> None:
        formatted = self._format_log_message(message)
        logger.info(formatted)
        print(f"[IsaacLab] {formatted}")

    def _log_warning(self, message: str) -> None:
        formatted = self._format_log_message(f"WARNING {message}")
        logger.warning(formatted)
        print(f"[IsaacLab] {formatted}")

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            joint_ids, matched_names = self._asset.find_joints([f"^{joint_name}$"])
            if len(joint_ids) != 1:
                raise ValueError(
                    f"Expected exactly one joint match for '{joint_name}' on asset '{self.cfg.asset_name}', "
                    f"but got {len(joint_ids)} matches: {matched_names}"
                )
            resolved_ids.append(int(joint_ids[0]))
            resolved_names.append(matched_names[0])
        return resolved_ids, resolved_names

    def _connect_dds(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUStateDefault,
                unitree_hg_msg_dds__LowState_ as LowStateDefault,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_
        except ModuleNotFoundError as exc:
            self._log_warning(f"{exc.name} is not installed; holding sonic_robot default pose.")
            return

        try:
            if self.cfg.network_interface:
                ChannelFactoryInitialize(int(self.cfg.domain_id), self.cfg.network_interface)
            else:
                ChannelFactoryInitialize(int(self.cfg.domain_id))
        except Exception as exc:
            self._log_warning(f"ChannelFactoryInitialize failed: {exc}")
            return

        self._lowstate_msg = LowStateDefault()
        self._secondary_imu_msg = IMUStateDefault()
        self._lowstate_publisher = ChannelPublisher(self.cfg.lowstate_topic, LowState_)
        self._lowstate_publisher.Init()
        self._secondary_imu_publisher = ChannelPublisher(self.cfg.secondary_imu_topic, IMUState_)
        self._secondary_imu_publisher.Init()
        self._lowcmd_subscriber = ChannelSubscriber(self.cfg.lowcmd_topic, LowCmd_)
        self._lowcmd_subscriber.Init(self._lowcmd_handler, 1)
        self._dds_ready = True

    def _lowcmd_handler(self, msg) -> None:
        with self._low_cmd_lock:
            self._low_cmd = msg
            self._new_low_cmd = True
            self._lowcmd_count += 1
            self._last_lowcmd_time = time.monotonic()

    def _take_latest_lowcmd(self):
        with self._low_cmd_lock:
            msg = self._low_cmd
            is_new = self._new_low_cmd
            self._new_low_cmd = False
        return msg, is_new

    def _extract_lowcmd_target(self, low_cmd) -> torch.Tensor | None:
        try:
            target_values = [float(low_cmd.motor_cmd[i].q) for i in range(len(SONIC_G1_29DOF_JOINT_ORDER))]
        except Exception as exc:
            self._log_warning(f"failed to read LowCmd motor_cmd q fields: {exc}")
            return None

        target = torch.tensor(target_values, device=self.device, dtype=torch.float32)
        if self._target_order == "mujoco":
            target = target[self._isaac_to_mujoco_index]
        if not self._first_lowcmd_logged:
            self._first_lowcmd_logged = True
            target_cpu = target.detach().cpu()
            self._log_info(
                f"first LowCmd parsed order={self._target_order} "
                f"mean={target_cpu.mean():+.4f} absmax={target_cpu.abs().max():.4f}"
            )
        return target.unsqueeze(0).repeat(self.num_envs, 1)

    def _apply_target_rate_limit(self, target: torch.Tensor) -> torch.Tensor:
        max_delta = float(self.cfg.target_rate_limit_rad_per_step)
        if max_delta <= 0.0:
            self._last_target_step_delta_absmax = torch.max(
                torch.abs(target - self._processed_actions), dim=-1
            ).values
            return target
        delta = torch.clamp(target - self._processed_actions, min=-max_delta, max=max_delta)
        self._last_target_step_delta_absmax = torch.max(torch.abs(delta), dim=-1).values
        return self._processed_actions + delta

    def _stabilize_root_pose(self) -> None:
        if not self.cfg.stabilize_root_pose or self._root_pose_unlocked:
            return
        if self._root_pose_anchor is None:
            self._root_pose_anchor = torch.cat(
                [self._asset.data.root_pos_w, self._asset.data.root_quat_w], dim=-1
            ).clone()
            if not self._root_anchor_logged:
                anchor = self._root_pose_anchor[0].detach().cpu().tolist()
                self._log_info(
                    "root pose stabilized at "
                    f"pos=({anchor[0]:+.3f}, {anchor[1]:+.3f}, {anchor[2]:+.3f})"
                )
                self._root_anchor_logged = True
        self._asset.write_root_pose_to_sim(self._root_pose_anchor)
        self._asset.write_root_velocity_to_sim(self._root_velocity_zero)

    def unlock_root_pose(self) -> None:
        if not self.cfg.stabilize_root_pose:
            self._log_info("root pose stabilization is disabled; unlock request ignored")
            return
        if self._root_pose_unlocked:
            return
        self._root_pose_unlocked = True
        self._asset.write_root_velocity_to_sim(self._root_velocity_zero)
        self._log_info("root pose unlocked by operator")

    @staticmethod
    def _set_sequence(dst, values) -> None:
        for idx, value in enumerate(values):
            dst[idx] = float(value)

    def _publish_lowstate(self) -> None:
        if not self._dds_ready or self._lowstate_publisher is None or self._lowstate_msg is None:
            return

        env_idx = 0
        joint_pos = self._asset.data.joint_pos[env_idx, self._joint_ids].detach().cpu()
        joint_vel = self._asset.data.joint_vel[env_idx, self._joint_ids].detach().cpu()
        joint_acc = self._asset.data.joint_acc[env_idx, self._joint_ids].detach().cpu()
        torque_src = getattr(self._asset.data, "applied_torque", None)
        if torque_src is None:
            joint_tau = torch.zeros_like(joint_pos)
        else:
            joint_tau = torque_src[env_idx, self._joint_ids].detach().cpu()
        root_quat = self._asset.data.root_quat_w[env_idx].detach().cpu()
        root_ang_vel = self._asset.data.root_ang_vel_b[env_idx].detach().cpu()

        if self._target_order == "mujoco":
            joint_pos = joint_pos[self._mujoco_to_isaac_index.cpu()]
            joint_vel = joint_vel[self._mujoco_to_isaac_index.cpu()]
            joint_acc = joint_acc[self._mujoco_to_isaac_index.cpu()]
            joint_tau = joint_tau[self._mujoco_to_isaac_index.cpu()]

        for i in range(len(SONIC_G1_29DOF_JOINT_ORDER)):
            motor_state = self._lowstate_msg.motor_state[i]
            motor_state.q = float(joint_pos[i])
            motor_state.dq = float(joint_vel[i])
            motor_state.ddq = float(joint_acc[i])
            motor_state.tau_est = float(joint_tau[i])
        if hasattr(self._lowstate_msg, "mode_machine"):
            self._lowstate_msg.mode_machine = int(self.cfg.mode_machine)
        if hasattr(self._lowstate_msg, "tick"):
            self._lowstate_msg.tick = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
        self._set_sequence(self._lowstate_msg.imu_state.quaternion, root_quat.tolist())
        self._set_sequence(self._lowstate_msg.imu_state.gyroscope, root_ang_vel.tolist())
        self._set_sequence(self._lowstate_msg.imu_state.accelerometer, (0.0, 0.0, 0.0))
        self._lowstate_publisher.Write(self._lowstate_msg)

        if self._secondary_imu_publisher is not None and self._secondary_imu_msg is not None:
            self._set_sequence(self._secondary_imu_msg.quaternion, root_quat.tolist())
            self._set_sequence(self._secondary_imu_msg.gyroscope, root_ang_vel.tolist())
            self._secondary_imu_publisher.Write(self._secondary_imu_msg)
        self._lowstate_count += 1

    def process_actions(self, actions: torch.Tensor):
        low_cmd, is_new = self._take_latest_lowcmd()
        if low_cmd is not None and is_new:
            target = self._extract_lowcmd_target(low_cmd)
            if target is not None:
                self._processed_actions = self._apply_target_rate_limit(target)
                if self.cfg.clip is not None:
                    self._processed_actions = torch.clamp(
                        self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                    )

        stale_timeout_s = float(self.cfg.stale_timeout_s)
        if (
            stale_timeout_s > 0.0
            and self._last_lowcmd_time > 0.0
            and time.monotonic() - self._last_lowcmd_time > stale_timeout_s
        ):
            self._last_lowcmd_time = 0.0
            self._log_warning(f"no LowCmd received for {stale_timeout_s:.2f}s; holding last target.")

        self._debug_counter += 1
        if self.cfg.debug_log_interval > 0 and self._debug_counter % self.cfg.debug_log_interval == 0:
            proc = self._processed_actions[0].detach().cpu()
            self._log_info(
                f"step={self._debug_counter} lowcmd={self._lowcmd_count} lowstate={self._lowstate_count} "
                f"target_mean={proc.mean():+.4f} target_absmax={proc.abs().max():.4f} "
                f"step_delta_absmax={self._last_target_step_delta_absmax[0].item():.4f}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._stabilize_root_pose()
        if self.cfg.publish_lowstate_every_apply:
            self._publish_lowstate()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._processed_actions.copy_(self._default_joint_pos)
            self._last_target_step_delta_absmax.zero_()
            self._root_pose_anchor = None
            self._root_pose_unlocked = False
            return
        self._processed_actions[env_ids] = self._default_joint_pos[env_ids]
        self._last_target_step_delta_absmax[env_ids] = 0.0
        self._root_pose_anchor = None
        self._root_pose_unlocked = False


class UnitreeLowStatePublisherAction(ActionTerm):
    """Publish sonic_robot state as Unitree DDS ``rt/lowstate`` without consuming commands.

    Unlike :class:`UnitreeDdsLowCmdAction`, this term does not subscribe to ``rt/lowcmd``
    and does not drive any joint. It only mirrors the simulated ``sonic_robot`` state
    (joint q/dq/ddq/tau_est + base IMU) onto ``rt/lowstate`` so GR00T/SONIC deploy can use
    IsaacLab as the state source while joint targets are still driven over ZMQ by
    :class:`SonicDeployTargetAction`.
    """

    cfg: UnitreeLowStatePublisherActionCfg
    _asset: Articulation

    def __init__(self, cfg: UnitreeLowStatePublisherActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = self._raw_actions
        self._mujoco_to_isaac_index = torch.tensor(
            SONIC_G1_MUJOCO_TO_ISAACLAB_DOF, device=self.device, dtype=torch.long
        )
        self._target_order = str(cfg.target_order).lower()
        if self._target_order not in ("mujoco", "isaaclab"):
            raise ValueError(f"target_order must be 'mujoco' or 'isaaclab', got {cfg.target_order!r}")

        self._lowstate_count = 0
        self._debug_counter = 0
        self._dds_ready = False
        self._lowstate_msg = None
        self._secondary_imu_msg = None
        self._lowstate_publisher = None
        self._secondary_imu_publisher = None
        self._connect_dds()
        secondary = self.cfg.secondary_imu_topic if self.cfg.publish_secondary_imu else "<off>"
        self._log_info(
            f"asset={cfg.asset_name} domain_id={cfg.domain_id} "
            f"interface={cfg.network_interface or '<auto>'} lowstate={cfg.lowstate_topic!r} "
            f"secondary_imu={secondary!r} target_order={self._target_order} "
            f"resolved={len(self._joint_ids)} joints dds_ready={self._dds_ready}"
        )

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @staticmethod
    def _format_log_message(message: str) -> str:
        return f"[UnitreeLowStatePublisher] {message}"

    def _log_info(self, message: str) -> None:
        formatted = self._format_log_message(message)
        logger.info(formatted)
        print(f"[IsaacLab] {formatted}")

    def _log_warning(self, message: str) -> None:
        formatted = self._format_log_message(f"WARNING {message}")
        logger.warning(formatted)
        print(f"[IsaacLab] {formatted}")

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            joint_ids, matched_names = self._asset.find_joints([f"^{joint_name}$"])
            if len(joint_ids) != 1:
                raise ValueError(
                    f"Expected exactly one joint match for '{joint_name}' on asset '{self.cfg.asset_name}', "
                    f"but got {len(joint_ids)} matches: {matched_names}"
                )
            resolved_ids.append(int(joint_ids[0]))
            resolved_names.append(matched_names[0])
        return resolved_ids, resolved_names

    def _connect_dds(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__IMUState_ as IMUStateDefault,
                unitree_hg_msg_dds__LowState_ as LowStateDefault,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_
        except ModuleNotFoundError as exc:
            self._log_warning(f"{exc.name} is not installed; rt/lowstate will not be published.")
            return

        try:
            if self.cfg.network_interface:
                ChannelFactoryInitialize(int(self.cfg.domain_id), self.cfg.network_interface)
            else:
                ChannelFactoryInitialize(int(self.cfg.domain_id))
        except Exception as exc:
            self._log_warning(f"ChannelFactoryInitialize failed: {exc}")
            return

        self._lowstate_msg = LowStateDefault()
        self._lowstate_publisher = ChannelPublisher(self.cfg.lowstate_topic, LowState_)
        self._lowstate_publisher.Init()
        if self.cfg.publish_secondary_imu:
            self._secondary_imu_msg = IMUStateDefault()
            self._secondary_imu_publisher = ChannelPublisher(self.cfg.secondary_imu_topic, IMUState_)
            self._secondary_imu_publisher.Init()
        self._dds_ready = True

    @staticmethod
    def _set_sequence(dst, values) -> None:
        for idx, value in enumerate(values):
            dst[idx] = float(value)

    def _publish_lowstate(self) -> None:
        if not self._dds_ready or self._lowstate_publisher is None or self._lowstate_msg is None:
            return

        env_idx = 0
        joint_pos = self._asset.data.joint_pos[env_idx, self._joint_ids].detach().cpu()
        joint_vel = self._asset.data.joint_vel[env_idx, self._joint_ids].detach().cpu()
        joint_acc = self._asset.data.joint_acc[env_idx, self._joint_ids].detach().cpu()
        torque_src = getattr(self._asset.data, "applied_torque", None)
        if torque_src is None:
            joint_tau = torch.zeros_like(joint_pos)
        else:
            joint_tau = torque_src[env_idx, self._joint_ids].detach().cpu()
        root_quat = self._asset.data.root_quat_w[env_idx].detach().cpu()
        root_ang_vel = self._asset.data.root_ang_vel_b[env_idx].detach().cpu()

        if self._target_order == "mujoco":
            joint_pos = joint_pos[self._mujoco_to_isaac_index.cpu()]
            joint_vel = joint_vel[self._mujoco_to_isaac_index.cpu()]
            joint_acc = joint_acc[self._mujoco_to_isaac_index.cpu()]
            joint_tau = joint_tau[self._mujoco_to_isaac_index.cpu()]

        for i in range(len(SONIC_G1_29DOF_JOINT_ORDER)):
            motor_state = self._lowstate_msg.motor_state[i]
            motor_state.q = float(joint_pos[i])
            motor_state.dq = float(joint_vel[i])
            motor_state.ddq = float(joint_acc[i])
            motor_state.tau_est = float(joint_tau[i])
        if hasattr(self._lowstate_msg, "mode_machine"):
            self._lowstate_msg.mode_machine = int(self.cfg.mode_machine)
        if hasattr(self._lowstate_msg, "tick"):
            self._lowstate_msg.tick = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
        self._set_sequence(self._lowstate_msg.imu_state.quaternion, root_quat.tolist())
        self._set_sequence(self._lowstate_msg.imu_state.gyroscope, root_ang_vel.tolist())
        self._set_sequence(self._lowstate_msg.imu_state.accelerometer, (0.0, 0.0, 0.0))
        self._lowstate_publisher.Write(self._lowstate_msg)

        if self._secondary_imu_publisher is not None and self._secondary_imu_msg is not None:
            self._set_sequence(self._secondary_imu_msg.quaternion, root_quat.tolist())
            self._set_sequence(self._secondary_imu_msg.gyroscope, root_ang_vel.tolist())
            self._secondary_imu_publisher.Write(self._secondary_imu_msg)
        self._lowstate_count += 1

    def process_actions(self, actions: torch.Tensor):
        del actions

    def apply_actions(self):
        self._publish_lowstate()
        self._debug_counter += 1
        if self.cfg.debug_log_interval > 0 and self._debug_counter % self.cfg.debug_log_interval == 0:
            self._log_info(f"step={self._debug_counter} lowstate={self._lowstate_count}")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids


class AgileBasedLowerBodyAction(ActionTerm):
    """Action term that drives robot A lower-body walking from a locomotion policy."""

    cfg: AgileBasedLowerBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: AgileBasedLowerBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._observation_cfg = env.cfg.observations
        self._obs_group_name = cfg.obs_group_name
        self._env = env
        self._joint_ids, self._joint_names = self._resolve_joint_order(self.cfg.joint_names)
        self._policy_output_scale = torch.tensor(cfg.policy_output_scale, device=env.device, dtype=torch.float32)
        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._action_smoothing = float(cfg.action_smoothing)
        self._command_scale = float(cfg.command_scale)
        self._stand_command_deadzone = float(cfg.stand_command_deadzone)
        self._enable_policy_when_moving = bool(cfg.enable_policy_when_moving)
        self._root_motion_scale = float(cfg.root_motion_scale)
        self._root_motion_smoothing = float(cfg.root_motion_smoothing)
        self._stabilize_root_pose = bool(cfg.stabilize_root_pose)
        self._default_hip_height = torch.tensor([cfg.hip_height], device=env.device, dtype=torch.float32)
        self._policy_path = retrieve_file_path(cfg.policy_path)
        self._policy_kind = Path(self._policy_path).suffix.lower()
        self._policy = None
        self._onnx_input_name: str | None = None
        self._onnx_output_name: str | None = None
        self._expected_input_dim: int | None = None
        self._shape_warning_printed = False
        self._debug_counter = 0
        self._runtime_state_logged = False
        self._load_policy()
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = self._policy_output_offset.clone()
        self._stable_root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._stable_root_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._last_root_target_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._last_root_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._root_reference_sync_threshold = 0.35
        self._turn_in_place_xy_deadzone = 0.05
        self._turn_in_place_yaw_deadzone = 0.10
        print(
            "[IsaacLab] [LowerBodyONNX] "
            f"asset={cfg.asset_name} joints={list(self._joint_names)} "
            f"scale={float(cfg.policy_output_scale):.3f} smoothing={self._action_smoothing:.2f} "
            f"cmd_scale={self._command_scale:.2f}"
        )

    def _resolve_joint_order(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            joint_ids, matched_names = self._asset.find_joints([f"^{joint_name}$"])
            if len(joint_ids) != 1:
                raise ValueError(
                    f"Expected exactly one joint match for '{joint_name}' on asset '{self.cfg.asset_name}', "
                    f"but got {len(joint_ids)} matches: {matched_names}"
                )
            resolved_ids.append(int(joint_ids[0]))
            resolved_names.append(matched_names[0])
        return resolved_ids, resolved_names

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, hip_height]."""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _compose_policy_input(self, base_command: torch.Tensor, obs_tensor: torch.Tensor) -> torch.Tensor:
        history_length = getattr(self._observation_cfg, self._obs_group_name).history_length
        if history_length is None:
            history_length = 1
        repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(base_command.shape[0], -1)
        return torch.cat([repeated_commands, obs_tensor], dim=-1)

    @staticmethod
    def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        qw = quat_wxyz[:, 0]
        qx = quat_wxyz[:, 1]
        qy = quat_wxyz[:, 2]
        qz = quat_wxyz[:, 3]
        return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    @staticmethod
    def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
        half_yaw = 0.5 * yaw
        quat[:, 0] = torch.cos(half_yaw)
        quat[:, 3] = torch.sin(half_yaw)
        return quat

    def _apply_absolute_root_pose(self, base_command: torch.Tensor) -> None:
        if not self._stabilize_root_pose:
            return

        current_root_xy = self._asset.data.root_pos_w[:, :2]
        current_root_yaw = self._yaw_from_quat(self._asset.data.root_quat_w)
        command_xy_norm = torch.linalg.norm(base_command[:, :2], dim=-1)
        command_yaw_abs = torch.abs(base_command[:, 2])

        # When the headset retargeter recenters, it emits a near-zero command at
        # the robot's current pose. Sync the stable root reference in that case
        # so subsequent absolute targets use the new pose as their baseline.
        root_reference_error = torch.linalg.norm(current_root_xy - self._stable_root_pos[:, :2], dim=-1)
        should_sync_reference = (
            (command_xy_norm <= self._turn_in_place_xy_deadzone)
            & (command_yaw_abs <= self._turn_in_place_yaw_deadzone)
            & (root_reference_error >= self._root_reference_sync_threshold)
        )
        if torch.any(should_sync_reference):
            self._stable_root_pos[should_sync_reference] = self._asset.data.root_pos_w[should_sync_reference]
            self._stable_root_yaw[should_sync_reference] = current_root_yaw[should_sync_reference]
            self._last_root_target_xy[should_sync_reference] = current_root_xy[should_sync_reference]
            self._last_root_target_yaw[should_sync_reference] = current_root_yaw[should_sync_reference]

        target_xy = self._stable_root_pos[:, :2] + base_command[:, :2] * self._root_motion_scale
        target_yaw = self._stable_root_yaw + base_command[:, 2]

        # Turn-in-place commands should keep the current/root-target position
        # instead of snapping back to the stable-root origin when xy is near zero.
        is_turn_in_place = (
            (command_xy_norm <= self._turn_in_place_xy_deadzone)
            & (command_yaw_abs > self._turn_in_place_yaw_deadzone)
        )
        if torch.any(is_turn_in_place):
            target_xy[is_turn_in_place] = current_root_xy[is_turn_in_place]
            target_yaw[is_turn_in_place] = current_root_yaw[is_turn_in_place] + base_command[is_turn_in_place, 2]

        self._last_root_target_xy.copy_(target_xy)
        self._last_root_target_yaw.copy_(target_yaw)

        root_pose = torch.cat([self._asset.data.root_pos_w, self._asset.data.root_quat_w], dim=-1).clone()
        alpha = float(min(max(self._root_motion_smoothing, 0.0), 1.0))
        root_pose[:, :2] = torch.lerp(root_pose[:, :2], target_xy, alpha)
        root_pose[:, 2] = self._stable_root_pos[:, 2]
        target_quat = self._quat_from_yaw(target_yaw)
        if alpha >= 0.999:
            root_pose[:, 3:7] = target_quat
        else:
            root_pose[:, 3:7] = torch.lerp(root_pose[:, 3:7], target_quat, alpha)
            quat_norm = torch.linalg.norm(root_pose[:, 3:7], dim=-1, keepdim=True).clamp_min(1e-6)
            root_pose[:, 3:7] = root_pose[:, 3:7] / quat_norm
        root_velocity = torch.zeros(root_pose.shape[0], 6, device=root_pose.device, dtype=root_pose.dtype)
        self._asset.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1))

    def _load_policy(self):
        if self._policy_kind == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "ONNX walking policy requested, but `onnxruntime` is not installed in the IsaacLab environment."
                ) from exc

            self._policy = ort.InferenceSession(self._policy_path, providers=["CPUExecutionProvider"])
            input_meta = self._policy.get_inputs()[0]
            self._onnx_input_name = input_meta.name
            if isinstance(input_meta.shape[-1], int):
                self._expected_input_dim = input_meta.shape[-1]
            self._onnx_output_name = self._policy.get_outputs()[0].name
            return

        self._policy = load_torchscript_model(self._policy_path, device=self.device)

    def _run_policy(self, policy_input: torch.Tensor) -> torch.Tensor:
        if self._policy_kind == ".onnx":
            assert self._onnx_input_name is not None
            assert self._onnx_output_name is not None
            output = self._policy.run(
                [self._onnx_output_name],
                {self._onnx_input_name: policy_input.detach().cpu().numpy().astype(np.float32)},
            )[0]
            return torch.from_numpy(output).to(device=self.device, dtype=torch.float32)

        return self._policy.forward(policy_input)

    def process_actions(self, actions: torch.Tensor):
        if not self._runtime_state_logged:
            robot_pos = self._asset.data.root_pos_w[0].detach().cpu().tolist()
            robot_prim = getattr(self._asset.cfg, "prim_path", "<unknown>")
            print(
                "[IsaacLab] [RuntimeAssetState] "
                f"controlled_asset={self.cfg.asset_name} prim={robot_prim} "
                f"root_pos={tuple(round(float(v), 4) for v in robot_pos)}"
            )
            self._runtime_state_logged = True

        if actions.shape[-1] >= 4:
            base_command = actions[:, :4].clone()
        else:
            base_command = torch.cat(
                [actions[:, :3], self._default_hip_height.repeat(actions.shape[0], 1)],
                dim=-1,
            )

        self._apply_absolute_root_pose(base_command)

        # Fallback teleop mode: keep the legs in a stable standing pose and
        # drive only the articulated root from the headset-derived command.
        self._raw_actions.zero_()
        self._processed_actions = torch.lerp(
            self._processed_actions,
            self._policy_output_offset,
            self._action_smoothing,
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            cmd = base_command[0].detach().cpu().numpy()
            proc = self._processed_actions[0].detach().cpu()
            root_xy = self._asset.data.root_pos_w[0, :2].detach().cpu().numpy()
            target_xy = self._last_root_target_xy[0].detach().cpu().numpy()
            print(
                "[IsaacLab] [LowerBodyONNX] "
                f"stand_root cmd=[{cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f}, {cmd[3]:+.3f}] "
                f"root_xy=[{root_xy[0]:+.3f}, {root_xy[1]:+.3f}] "
                f"target_xy=[{target_xy[0]:+.3f}, {target_xy[1]:+.3f}] "
                f"proc_mean={proc.mean():+.4f} proc_absmax={proc.abs().max():+.4f}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._processed_actions.copy_(self._policy_output_offset)
            self._raw_actions.zero_()
            self._stable_root_pos.copy_(self._asset.data.root_pos_w)
            self._stable_root_yaw.copy_(self._yaw_from_quat(self._asset.data.root_quat_w))
            self._last_root_target_xy.copy_(self._stable_root_pos[:, :2])
            self._last_root_target_yaw.copy_(self._stable_root_yaw)
            return

        self._processed_actions[env_ids] = self._policy_output_offset[env_ids]
        self._raw_actions[env_ids].zero_()
        self._stable_root_pos[env_ids] = self._asset.data.root_pos_w[env_ids]
        self._stable_root_yaw[env_ids] = self._yaw_from_quat(self._asset.data.root_quat_w[env_ids])
        self._last_root_target_xy[env_ids] = self._stable_root_pos[env_ids, :2]
        self._last_root_target_yaw[env_ids] = self._stable_root_yaw[env_ids]


class AutoWalkAction(ActionTerm):
    """全身骨骼捕捉数据驱动的物理行走（腿+腰+手臂+手），含自然摆臂。

    数据流（概念上）::

        time → SkeletonPoseSimulator.sample(phase) → 各关节目标角度 → robot

    内部不接收外部输入，由 `_sample_skeleton_pose` 产生与 walking 阶段同步的
    全身关节角度。这模拟了一个本地 mocap 流：法线交互/重定向部分内嵌实现。

    机器人通过物理引擎自然行走，脚与地面产生真实接触力。
    """

    cfg: AutoWalkActionCfg
    _asset: Articulation

    def __init__(self, cfg: AutoWalkActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._env = env

        # 关节解析（缺失关节直接跳过，不抛错）
        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        self._idx = {n: i for i, n in enumerate(self._joint_names)}
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_joint_pos.clone()

        self._phase = torch.zeros(self.num_envs, device=self.device)

        # 把上下身关节按区分组缓存，避免每步反复查字典
        self._leg_groups = self._collect_side_indices(
            patterns=("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")
        )
        self._arm_groups = self._collect_side_indices(
            patterns=("shoulder_pitch_joint", "elbow_joint")
        )
        self._waist_yaw_idx = self._idx.get("waist_yaw_joint")
        self._waist_roll_idx = self._idx.get("waist_roll_joint")
        self._waist_pitch_idx = self._idx.get("waist_pitch_joint")
        # 髋 yaw（用于骨盆旋转）
        self._left_hip_yaw_idx = self._idx.get("left_hip_yaw_joint")
        self._right_hip_yaw_idx = self._idx.get("right_hip_yaw_joint")
        # 收集手部关节索引
        self._hand_indices = [i for n, i in self._idx.items() if "_hand_" in n]

        print(
            f"[IsaacLab] [AutoWalkAction] asset={cfg.asset_name} "
            f"freq={cfg.walk_frequency:.2f}Hz "
            f"resolved_joints={len(self._joint_ids)}/{len(cfg.joint_names)} "
            f"(legs={sum(len(v) for v in self._leg_groups.values())} "
            f"arms={sum(len(v) for v in self._arm_groups.values())} "
            f"hands={len(self._hand_indices)})"
        )

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        """逐个解析关节。缺失关节会被跳过并打印警告（保持代码对 G1 变体兼容）。"""
        ids, names = [], []
        for name in joint_names:
            jids, jnames = self._asset.find_joints([f"^{name}$"])
            if len(jids) == 1:
                ids.append(int(jids[0]))
                names.append(jnames[0])
            else:
                print(f"[IsaacLab] [AutoWalkAction] skip joint '{name}' (matches={len(jids)})")
        return ids, names

    def _collect_side_indices(self, patterns: tuple[str, ...]) -> dict[str, dict[str, int]]:
        """返回形如 {'left': {'hip_pitch_joint': idx, ...}, 'right': {...}} 的索引表。"""
        groups: dict[str, dict[str, int]] = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for p in patterns:
                key = f"{side}_{p}"
                if key in self._idx:
                    groups[side][p] = self._idx[key]
        return groups

    @property
    def action_dim(self) -> int:
        return 1  # 占位；外部不发送命令

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _sample_skeleton_pose(self, phase: torch.Tensor) -> torch.Tensor:
        """模拟骨骼捕捉数据流，输出全身关节目标角度。

        这一函数等价于 ``retarget(mocap_data_at_time(t), robot_skeleton)``，
        但 mocap_data 用解析公式合成而非从外部读取。
        """
        targets = self._default_joint_pos.clone()

        phase_l = phase           # 左腿相位
        phase_r = phase + math.pi  # 右腿相位（180° 偏移）

        # ── LEGS：行走步态 ────────────────────────────────────
        A_hip = self.cfg.hip_pitch_amplitude
        A_knee = self.cfg.knee_amplitude
        A_ankle = self.cfg.ankle_pitch_amplitude

        for side, ph in (("left", phase_l), ("right", phase_r)):
            leg = self._leg_groups[side]
            if "hip_pitch_joint" in leg:
                i = leg["hip_pitch_joint"]
                targets[:, i] = self._default_joint_pos[:, i] + A_hip * torch.sin(ph)
            if "knee_joint" in leg:
                i = leg["knee_joint"]
                # 膝关节在腿前摆中段（mid-swing）弯曲最大
                targets[:, i] = self._default_joint_pos[:, i] + A_knee * torch.clamp(torch.sin(ph), min=0.0)
            if "ankle_pitch_joint" in leg:
                i = leg["ankle_pitch_joint"]
                targets[:, i] = self._default_joint_pos[:, i] - A_ankle * torch.sin(ph)

        # ── ARMS：反向摆动（与同侧腿 180° 相位） ─────────────
        A_arm = self.cfg.arm_swing_amplitude
        A_elbow = self.cfg.elbow_bend_amplitude

        for side, ph_arm in (("left", phase_l), ("right", phase_r)):
            arm = self._arm_groups[side]
            if "shoulder_pitch_joint" in arm:
                i = arm["shoulder_pitch_joint"]
                # 手臂与同侧腿"前后位置"反相：腿后摆 → 同侧臂前摆
                targets[:, i] = self._default_joint_pos[:, i] + A_arm * torch.sin(ph_arm)
            if "elbow_joint" in arm:
                i = arm["elbow_joint"]
                # 前摆时肘部轻微弯曲
                targets[:, i] = self._default_joint_pos[:, i] + A_elbow * torch.clamp(torch.sin(ph_arm + 0.5), min=0.0)

        # ── WAIST：小幅反向扭转，增加自然感 ────────────────
        A_waist_yaw = self.cfg.waist_yaw_amplitude
        if self._waist_yaw_idx is not None:
            # 与腿运动反相（腿前摆，腰反扭）
            targets[:, self._waist_yaw_idx] = (
                self._default_joint_pos[:, self._waist_yaw_idx] - A_waist_yaw * torch.sin(phase_l)
            )

        # ── WAIST ROLL：行走时的身体侧倾（重心转移） ──────────
        A_waist_roll = self.cfg.waist_roll_amplitude
        if self._waist_roll_idx is not None:
            # 左腿支撑时身体向左倾，右腿支撑时向右倾
            targets[:, self._waist_roll_idx] = (
                self._default_joint_pos[:, self._waist_roll_idx] + A_waist_roll * torch.sin(phase_l)
            )

        # ── HIP YAW：骨盆旋转（与腰部 yaw 协同） ───────────────
        A_hip_yaw = self.cfg.hip_yaw_amplitude
        if self._left_hip_yaw_idx is not None:
            # 左髋与腰部同向旋转
            targets[:, self._left_hip_yaw_idx] = (
                self._default_joint_pos[:, self._left_hip_yaw_idx] - A_hip_yaw * torch.sin(phase_l)
            )
        if self._right_hip_yaw_idx is not None:
            # 右髋与腰部同向旋转
            targets[:, self._right_hip_yaw_idx] = (
                self._default_joint_pos[:, self._right_hip_yaw_idx] - A_hip_yaw * torch.sin(phase_l)
            )

        # ── HANDS：保持微弱放松卷曲（恒定，不随相位变化） ──
        if self._hand_indices and self.cfg.hand_curl_amount != 0.0:
            curl = self.cfg.hand_curl_amount
            for hi in self._hand_indices:
                targets[:, hi] = self._default_joint_pos[:, hi] + curl

        return targets

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt

        # ── 1. 更新相位 ──────────────────────────────────────
        self._phase += 2.0 * math.pi * self.cfg.walk_frequency * dt

        # ── 2. 从"骨骼数据"生成全身关节目标 ───────────────────
        self._processed_actions = self._sample_skeleton_pose(self._phase)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._phase.zero_()
            self._processed_actions.copy_(self._default_joint_pos)
        else:
            self._phase[env_ids] = 0.0
            self._processed_actions[env_ids] = self._default_joint_pos[env_ids]


class SONICWholeBodyAction(ActionTerm):
    """GEAR-SONIC encoder-decoder 全身追踪 Action Term。

    - encoder 端 1762D 输入按 deploy observation_config.yaml 构造 g1 mocap reference。
    - decoder 端 994D 输入按 deploy observation_config.yaml 偏移精确构造：
      token_state(64) + his_base_ang_vel(30) + his_joint_pos(290) + his_joint_vel(290)
      + his_last_actions(290) + his_gravity_dir(30) = 994
    """

    HISTORY_LEN = 10  # decoder 端 _10frame_step1 历史长度

    cfg: SONICWholeBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: SONICWholeBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._env = env

        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        if len(self._joint_ids) != cfg.sonic_action_dim:
            print(
                f"[IsaacLab] [SONIC] WARNING resolved {len(self._joint_ids)}/{cfg.sonic_action_dim} joints; "
                "SONIC was trained on 29 DoF — outputs for missing joints will be discarded."
            )

        self._asset_default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._default_joint_pos = torch.tensor(
            SONIC_G1_29DOF_DEFAULT_ANGLES, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).clone()
        self._processed_actions = self._default_joint_pos.clone()
        self._last_action = torch.zeros(self.num_envs, cfg.sonic_action_dim, device=self.device)
        self._sonic_action_scale = torch.tensor(
            SONIC_G1_29DOF_ACTION_SCALE, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self._target_rate_limit = self._build_target_rate_limit()
        self._mocap_target_blend = self._build_mocap_target_blend()
        self._episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._last_target_step_delta_absmax = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._reset_root_pos_w = self._asset.data.root_pos_w.clone()
        self._reset_mocap_root_trans = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        self._init_history()
        self._init_sonic_body_indices()
        self._load_policies()
        self._load_mocap()
        self._load_action_noise_std()
        self._debug_counter = 0

        if self.num_envs > 1:
            print(
                f"[IsaacLab] [SONIC] WARNING num_envs={self.num_envs}; ONNX runs in a per-env loop "
                "(no batch dim in encoder/decoder); expect ~6ms × num_envs per step."
            )
        print(
            f"[IsaacLab] [SONIC] asset={cfg.asset_name} resolved={len(self._joint_ids)}/{cfg.sonic_action_dim} joints "
            f"action_scale={cfg.action_scale:.2f} per_joint_scale=[{self._sonic_action_scale.min().item():.4f}, "
            f"{self._sonic_action_scale.max().item():.4f}] enc_in={self._encoder_input_dim}D "
            f"dec_in={self._decoder_input_dim}D history_len={self.HISTORY_LEN} "
            f"sonic_default_absmax={self._default_joint_pos.abs().max().item():.3f} "
            f"asset_default_absmax={self._asset_default_joint_pos.abs().max().item():.3f} "
            f"startup_blend_steps={cfg.startup_blend_steps} "
            f"upper_rate_limit={cfg.upper_body_target_rate_limit_rad_per_step:.3f} "
            f"wrist_rate_limit={cfg.wrist_target_rate_limit_rad_per_step:.3f} "
            f"follow_root_xy={cfg.follow_mocap_root_xy} "
            f"follow_root_z={cfg.follow_mocap_root_z} "
            f"follow_root_rot={cfg.follow_mocap_root_rot} "
            f"upper_mocap_blend={cfg.upper_body_mocap_target_blend:.2f} "
            f"wrist_mocap_blend={cfg.wrist_mocap_target_blend:.2f}"
        )

    def _init_history(self):
        N, J = self.num_envs, self.cfg.sonic_action_dim
        H = self.HISTORY_LEN
        dev = self.device
        # 注意：SONIC 训练用 joint_pos_rel = joint_pos - default → 静止时 = 0
        # 之前用绝对 default joint pos 初始化是 bug，与训练对齐应当全 zero
        self._hist_base_ang_vel = torch.zeros(N, H, 3, device=dev)
        self._hist_joint_pos = torch.zeros(N, H, J, device=dev)
        self._hist_joint_vel = torch.zeros(N, H, J, device=dev)
        self._hist_last_actions = torch.zeros(N, H, J, device=dev)
        self._hist_gravity_dir = torch.zeros(N, H, 3, device=dev)
        self._hist_gravity_dir[:, :, 2] = -1.0

    def _load_mocap(self):
        """加载 walking mocap 序列，提供时变 motion anchor orientation 给 encoder。

        E3 第一版：仅用 mocap 的 root_rot 给 anchor_ori，body_pos 仍 self-ref。
        后续可加 forward kinematics 让 body_pos 也跟 mocap pose。
        """
        path = self.cfg.mocap_path
        if not path or path == "":
            self._mocap_root_rot_wxyz = None
            self._mocap_dof = None
            self._mocap_root_trans = None
            self._mocap_num_frames = 0
            print("[IsaacLab] [SONIC] no mocap_path; encoder anchor_ori 用 identity（self-ref）")
            return
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("SONIC mocap 加载需要 joblib，请 pip install joblib") from exc
        raw = joblib.load(path)
        # 顶层是 {motion_name: motion_dict}
        motion_name = next(iter(raw.keys()))
        motion = raw[motion_name]
        src_fps = float(motion.get("fps", 30))
        target_fps = 50.0  # 与 SONIC 训练 motion_lib.target_fps=50 对齐

        # 50fps 重采样（与训练 motion_lib 行为一致）
        # - root_rot 用 SLERP
        # - dof / root_trans 用线性插值
        # 重采样后 _mocap_num_frames = round(T_src × 50/30) ≈ 2003，与 F4 .npy 帧数对齐
        from scipy.spatial.transform import Rotation as _SR
        from scipy.spatial.transform import Slerp as _Slerp
        T_src = motion["root_rot"].shape[0]
        # 与 Humanoid_Batch.interploate_pose 完全一致：duration = (T_src-1)/src_fps，
        # 时间戳 arange(0, duration, 1/target_fps)（不含 duration 端点）
        duration = (T_src - 1) / src_fps  # 秒
        t_src = np.arange(T_src) / src_fps  # (T_src,) 源时间戳
        t_out = np.arange(0.0, duration, 1.0 / target_fps)  # (n_out,) 50fps 时间戳
        n_out = t_out.shape[0]

        # root_rot SLERP（mocap PKL 是 xyzw 顺序，scipy 也是 xyzw）
        root_rot_xyzw_src = motion["root_rot"]  # (T_src, 4)
        slerp = _Slerp(t_src, _SR.from_quat(root_rot_xyzw_src))
        root_rot_xyzw_out = slerp(t_out).as_quat()  # (n_out, 4) xyzw
        # 转 IsaacLab wxyz
        root_rot_wxyz_out = root_rot_xyzw_out[:, [3, 0, 1, 2]]
        self._mocap_root_rot_wxyz = (
            torch.from_numpy(root_rot_wxyz_out).to(self.device).float()
        )
        self._mocap_num_frames = self._mocap_root_rot_wxyz.shape[0]
        self._mocap_fps = target_fps

        # dof + root_trans 线性插值（关节角是连续函数，root_trans 是位置）
        def _interp(arr_src: np.ndarray) -> torch.Tensor:
            # arr_src: (T_src, D) → (n_out, D)
            out = np.empty((n_out, arr_src.shape[1]), dtype=np.float32)
            for d in range(arr_src.shape[1]):
                out[:, d] = np.interp(t_out, t_src, arr_src[:, d])
            return torch.from_numpy(out).to(self.device).float()

        mocap_dof_mujoco = _interp(motion["dof"])  # (n_out, 29), raw PKL uses MuJoCo/URDF order
        self._mocap_dof = mocap_dof_mujoco[:, SONIC_G1_ISAACLAB_TO_MUJOCO_DOF]
        self._mocap_root_trans = _interp(motion["root_trans_offset"])  # (n_out, 3)

        # 消掉 mocap 第 0 帧的全局姿态，让 robot 从 identity 朝向开始（reset 时 robot
        # 也设 identity，避免 root_pos 不动 + 大 yaw 导致初始侧站不稳）。root translation
        # 必须应用同一个对齐，否则 walking delta 仍在原始 mocap world frame 中，root_lag
        # 和 diagnostic root replay 会沿错误方向解释参考轨迹。
        # 公式：
        #   q_aligned[t] = q_inv(q[0]) * q[t]
        #   p_aligned[t] = p[0] + rotate(q_inv(q[0]), p[t] - p[0])
        q0 = self._mocap_root_rot_wxyz[0]
        root_trans_origin = self._mocap_root_trans[0].unsqueeze(0)
        root_trans_rel = self._mocap_root_trans - root_trans_origin
        self._mocap_root_trans = root_trans_origin + quat_apply_inverse(
            q0.unsqueeze(0).expand(root_trans_rel.shape[0], -1),
            root_trans_rel,
        )
        q0_inv = torch.tensor(
            [q0[0], -q0[1], -q0[2], -q0[3]], device=self.device  # conjugate for unit quat
        )
        # quat_mul (wxyz): (w1w2-x1x2-y1y2-z1z2, w1x2+x1w2+y1z2-z1y2, w1y2-x1z2+y1w2+z1x2, w1z2+x1y2-y1x2+z1w2)
        q = self._mocap_root_rot_wxyz  # (T, 4)
        w1, x1, y1, z1 = q0_inv[0], q0_inv[1], q0_inv[2], q0_inv[3]
        w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        self._mocap_root_rot_wxyz = torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )
        # 训练 (sonic_release/config.yaml commands.motion):
        #   target_fps=50, dt_future_ref_frames=0.1s, num_future_frames=10
        # → 取未来 10 帧、相邻帧间隔 0.1s（在 50Hz 训练时 = 5 sim steps）
        # 重采样到 50fps 后，0.1s 间隔 = 5 mocap-50fps 帧，advance = 1 frame/sim step
        self._mocap_step = max(1, round(0.1 * self._mocap_fps))  # 5
        self._sim_fps = 50.0
        self._mocap_advance_per_step = self._mocap_fps / self._sim_fps  # 1.0
        self._mocap_frame = 0
        self._mocap_frame_f = 0.0

        # 14 body in pelvis frame：优先用 F4 预算的 .npy（gear_sonic Humanoid_Batch FK 输出），
        # 否则 fallback SMPL 24-joint 近似，最后 fallback self-ref。
        # F4 预算脚本：scripts/tools/precompute_mocap_body_pos.py
        self._mocap_body_pos_b: torch.Tensor | None = None
        body_pos_npy = Path(path).with_name(Path(path).stem + "__body_pos14_pelvis.npy")
        smpl = motion.get("smpl_joints")
        if body_pos_npy.exists():
            arr = np.load(body_pos_npy)
            if arr.shape[0] == self._mocap_num_frames and arr.shape[1:] == (14, 3):
                self._mocap_body_pos_b = torch.from_numpy(arr).to(self.device).float()
                body_src = f"F4 FK npy (absmax={np.abs(arr).max():.3f}) @ {body_pos_npy.name}"
            else:
                body_src = (
                    f"F4 npy shape mismatch {arr.shape} vs ({self._mocap_num_frames}, 14, 3); "
                    "fall through"
                )
        if self._mocap_body_pos_b is None and smpl is not None and np.abs(smpl).max() > 1e-6:
            pelvis = smpl[:, 0:1, :]
            rel = smpl[:, list(SMPL_TO_SONIC_BODY_IDX), :] - pelvis
            self._mocap_body_pos_b = torch.from_numpy(rel).to(self.device).float()
            body_src = f"SMPL approx (absmax={np.abs(rel).max():.3f})"
        elif self._mocap_body_pos_b is None:
            body_src = "self-ref fallback (no F4 npy, smpl_joints zero/missing)"
        print(
            f"[IsaacLab] [SONIC] loaded mocap from {path}: motion={motion_name!r} "
            f"frames={self._mocap_num_frames} fps={self._mocap_fps:.1f} body_pos={body_src}"
        )

    def _load_action_noise_std(self):
        """加载 per-joint action noise std (29,)，覆盖 scalar fallback。"""
        path = self.cfg.action_noise_std_path
        if not path:
            self._action_noise_std_np = None
            return
        if not os.path.exists(path):
            print(f"[IsaacLab] [SONIC] WARN action_noise_std_path 不存在: {path}，fallback scalar")
            self._action_noise_std_np = None
            return
        arr = np.load(path).astype(np.float32)
        if arr.shape != (self.cfg.sonic_action_dim,):
            print(
                f"[IsaacLab] [SONIC] WARN std shape {arr.shape} != ({self.cfg.sonic_action_dim},)，"
                "fallback scalar"
            )
            self._action_noise_std_np = None
            return
        self._action_noise_std_np = arr
        print(
            f"[IsaacLab] [SONIC] loaded per-joint action noise std (29,) from {path}: "
            f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}"
        )

    def _init_sonic_body_indices(self):
        """找 SONIC 训练用 14 个 body link 在 USD articulation 中的索引。"""
        all_body_names = list(self._asset.data.body_names)
        print(f"[IsaacLab] [SONIC INIT] USD has {len(all_body_names)} bodies: {all_body_names}")

        self._sonic_body_ids: list[int] = []
        missing = []
        for name in SONIC_BODY_NAMES:
            ids, _ = self._asset.find_bodies([f"^{name}$"])
            if len(ids) == 1:
                self._sonic_body_ids.append(int(ids[0]))
            else:
                missing.append(name)
                self._sonic_body_ids.append(0)  # fallback to root link

        resolved = 14 - len(missing)
        print(f"[IsaacLab] [SONIC INIT] body indices resolved: {resolved}/14, ids={self._sonic_body_ids}")
        if missing:
            print(f"[IsaacLab] [SONIC INIT] MISSING SONIC bodies (fall back to root): {missing}")

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        ids, names = [], []
        for name in joint_names:
            jids, jnames = self._asset.find_joints([f"^{name}$"])
            if len(jids) == 1:
                ids.append(int(jids[0]))
                names.append(jnames[0])
            else:
                print(f"[IsaacLab] [SONIC] skip joint '{name}' (matches={len(jids)})")
        return ids, names

    def _load_policies(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "SONIC requires `onnxruntime` in the IsaacLab env. "
                "Install via `pip install onnxruntime-gpu`."
            ) from exc

        enc_path = retrieve_file_path(self.cfg.encoder_path)
        dec_path = retrieve_file_path(self.cfg.decoder_path)
        self._encoder = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
        self._decoder = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

        enc_in = self._encoder.get_inputs()[0]
        dec_in = self._decoder.get_inputs()[0]
        self._enc_input_name = enc_in.name
        self._dec_input_name = dec_in.name
        self._enc_output_name = self._encoder.get_outputs()[0].name
        self._dec_output_name = self._decoder.get_outputs()[0].name
        self._encoder_input_dim = int(enc_in.shape[-1])
        self._decoder_input_dim = int(dec_in.shape[-1])
        self._token_dim = int(self._encoder.get_outputs()[0].shape[-1])

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _build_target_rate_limit(self) -> torch.Tensor:
        """Build per-joint target rate limits in SONIC/IsaacLab joint order."""
        limit = torch.full((self.cfg.sonic_action_dim,), float("inf"), device=self.device, dtype=torch.float32)
        if self.cfg.target_rate_limit_rad_per_step > 0.0:
            limit.fill_(float(self.cfg.target_rate_limit_rad_per_step))

        upper_limit = float(self.cfg.upper_body_target_rate_limit_rad_per_step)
        wrist_limit = float(self.cfg.wrist_target_rate_limit_rad_per_step)
        for i, name in enumerate(self.cfg.joint_names[: self.cfg.sonic_action_dim]):
            is_upper = any(s in name for s in ("_shoulder_", "_elbow_", "_wrist_"))
            if is_upper and upper_limit > 0.0:
                limit[i] = min(float(limit[i].item()), upper_limit)
            if "_wrist_" in name and wrist_limit > 0.0:
                limit[i] = min(float(limit[i].item()), wrist_limit)
        return limit.unsqueeze(0)

    def _build_mocap_target_blend(self) -> torch.Tensor:
        """Build per-joint mocap target blend factors in SONIC/IsaacLab joint order."""
        blend = torch.zeros((self.cfg.sonic_action_dim,), device=self.device, dtype=torch.float32)
        upper_blend = float(self.cfg.upper_body_mocap_target_blend)
        wrist_blend = float(self.cfg.wrist_mocap_target_blend)
        for i, name in enumerate(self.cfg.joint_names[: self.cfg.sonic_action_dim]):
            if any(s in name for s in ("_shoulder_", "_elbow_", "_wrist_")):
                blend[i] = max(float(blend[i].item()), upper_blend)
            if "_wrist_" in name:
                blend[i] = max(float(blend[i].item()), wrist_blend)
        return torch.clamp(blend, 0.0, 1.0).unsqueeze(0)

    def _apply_mocap_target_blend(self, sonic_target: torch.Tensor) -> torch.Tensor:
        """Blend upper-body target toward mocap DoF without changing raw action history."""
        if self._mocap_dof is None or self._mocap_num_frames <= 0:
            return sonic_target
        if self._mocap_target_blend.max().item() <= 0.0:
            return sonic_target

        frame = self._resolve_mocap_frame(self._mocap_frame)
        mocap_target = self._mocap_dof[frame, : sonic_target.shape[1]].unsqueeze(0).expand_as(sonic_target)
        blend = self._mocap_target_blend[:, : sonic_target.shape[1]]
        return torch.lerp(sonic_target, mocap_target, blend)

    def _stabilize_target(self, sonic_target: torch.Tensor) -> torch.Tensor:
        """Apply reset startup blend and per-step target rate limits."""
        previous_target = self._processed_actions
        next_steps = self._episode_steps + 1
        target = sonic_target

        blend_steps = max(0, int(self.cfg.startup_blend_steps))
        if blend_steps > 0:
            alpha = torch.clamp(next_steps.to(dtype=torch.float32) / float(blend_steps), max=1.0).unsqueeze(-1)
            target = torch.lerp(previous_target, target, alpha)

        rate_limit = self._target_rate_limit[:, : target.shape[1]]
        target_delta = torch.minimum(torch.maximum(target - previous_target, -rate_limit), rate_limit)
        self._last_target_step_delta_absmax = target_delta.abs().max(dim=1).values
        self._episode_steps = next_steps
        return previous_target + target_delta

    def _push_history(self):
        """FIFO 推入当前观测，最新帧在 [-1] 位置。所有按 SONIC 关节顺序取。"""
        ang_vel = self._asset.data.root_ang_vel_b  # (N, 3) body frame IMU
        # SONIC 训练用 joint_pos_rel = 当前 - default（与 sonic_release/config.yaml 一致）
        jp_abs = self._asset.data.joint_pos[:, self._joint_ids]  # (N, 29) absolute
        jp = jp_abs - self._default_joint_pos  # (N, 29) relative
        jv = self._asset.data.joint_vel[:, self._joint_ids]  # (N, 29)
        gravity = self._asset.data.projected_gravity_b  # (N, 3) 重力投影到 body frame

        self._hist_base_ang_vel = torch.roll(self._hist_base_ang_vel, shifts=-1, dims=1)
        self._hist_joint_pos = torch.roll(self._hist_joint_pos, shifts=-1, dims=1)
        self._hist_joint_vel = torch.roll(self._hist_joint_vel, shifts=-1, dims=1)
        self._hist_last_actions = torch.roll(self._hist_last_actions, shifts=-1, dims=1)
        self._hist_gravity_dir = torch.roll(self._hist_gravity_dir, shifts=-1, dims=1)

        # B1：可选 obs noise 注入，匹配训练 AdditiveUniformNoise 分布
        # 训练时 obs term 加 noise，SONIC 学到"对 noise robust 的特征"，推理无 noise
        # 反 OOD —— 是 decoder history 反馈循环的候选根因之一
        if self.cfg.obs_noise_enabled:
            n_jp = self.cfg.obs_noise_joint_pos
            n_jv = self.cfg.obs_noise_joint_vel
            n_av = self.cfg.obs_noise_base_ang_vel
            n_g = self.cfg.obs_noise_gravity_dir
            ang_vel = ang_vel + (torch.rand_like(ang_vel) * 2.0 - 1.0) * n_av
            jp = jp + (torch.rand_like(jp) * 2.0 - 1.0) * n_jp
            jv = jv + (torch.rand_like(jv) * 2.0 - 1.0) * n_jv
            gravity = gravity + (torch.rand_like(gravity) * 2.0 - 1.0) * n_g

        self._hist_base_ang_vel[:, -1, :] = ang_vel
        self._hist_joint_pos[:, -1, :] = jp
        self._hist_joint_vel[:, -1, :] = jv
        self._hist_last_actions[:, -1, :] = self._last_action
        self._hist_gravity_dir[:, -1, :] = gravity

    def _build_decoder_input(self, tokens: np.ndarray, env_idx: int) -> np.ndarray:
        """按 deploy obs_config.yaml + C++ ObservationRegistry 真实顺序拼 994D decoder。

        Deploy yaml `observations:` 段字段顺序（gravity_dir **在最后**）：
            [0:64]    token_state (64D，encoder 当帧输出)
            [64:94]   his_base_angular_velocity_10frame_step1 (3*10 = 30D)
            [94:384]  his_body_joint_positions_10frame_step1  (29*10 = 290D)
            [384:674] his_body_joint_velocities_10frame_step1 (29*10 = 290D)
            [674:964] his_last_actions_10frame_step1          (29*10 = 290D)
            [964:994] his_gravity_dir_10frame_step1           (3*10 = 30D)

        flatten 约定: **row-major time-major** — 匹配 deploy C++ GatherHisXxx 函数
        `frame_offset = offset + f * joints` 的循环 (line 1472-1494)，layout
        = [t0_f0..fN, t1_f0..fN, ..., t9_f0..fN]。

        B6 修正：B5 把字段顺序按 sonic_release/config.yaml PolicyCfg dict 顺序
        排列（gravity_dir 在最前）是错的 —— deploy ONNX 是独立 export，
        以 deploy yaml 为准。保留 B5 flatten 约定修复（已确认正确）。
        """
        dec = np.zeros((1, self._decoder_input_dim), dtype=np.float32)
        dec[:, :64] = tokens
        if self.cfg.force_zero_decoder_history:
            return dec
        # row-major flatten (time-major) 按 deploy yaml 字段顺序
        dec[0, 64:94] = self._hist_base_ang_vel[env_idx].flatten().cpu().numpy()
        dec[0, 94:384] = self._hist_joint_pos[env_idx].flatten().cpu().numpy()
        dec[0, 384:674] = self._hist_joint_vel[env_idx].flatten().cpu().numpy()
        if not self.cfg.force_zero_last_action_history:
            dec[0, 674:964] = self._hist_last_actions[env_idx].flatten().cpu().numpy()
        dec[0, 964:994] = self._hist_gravity_dir[env_idx].flatten().cpu().numpy()
        return dec

    def _compute_self_ref_body_pos_b(self) -> torch.Tensor:
        """计算 14 个 SONIC body 在 pelvis (root) 坐标系下的位置。

        self-reference 时 reference == 当前 robot → 这些就是当前姿态下 body 相对 pelvis 的位置。
        Returns: (N, 14, 3)
        """
        # Match gear_sonic's robot_body_pos_b: use body COM positions in the pelvis body frame.
        body_pos_w = self._asset.data.body_pos_w[:, self._sonic_body_ids, :]  # (N, 14, 3)
        root_pos_w = self._asset.data.body_pos_w[:, self._sonic_body_ids[0], :]  # (N, 3)
        root_quat_w = self._asset.data.body_quat_w[:, self._sonic_body_ids[0], :]  # (N, 4)
        rel_w = body_pos_w - root_pos_w.unsqueeze(1)  # (N, 14, 3)
        quat_expanded = root_quat_w.unsqueeze(1).expand(-1, 14, -1)  # (N, 14, 4)
        return quat_apply_inverse(quat_expanded, rel_w)  # (N, 14, 3)

    def _build_encoder_input(self, env_idx: int) -> np.ndarray:
        """Encoder 1762D 输入（B6 完整重写：从 deploy obs_config.yaml + C++ Gather 函数推得真实 layout）。

        Deploy obs_config.yaml encoder_observations 字段顺序 + C++ ObservationRegistry
        维度交叉验证后的 g1 mode 完整 layout：

            [0:4]      encoder_mode_4 ([0]=mode_id 0/1/2, [1:4]=zero)
            [4:294]    motion_joint_positions_10frame_step5  (10 frames × 29 joints = 290D)
            [294:584]  motion_joint_velocities_10frame_step5 (10 × 29 = 290D)
            [584:594]  motion_root_z_position_10frame_step5  (10D) — g1 不用, zero
            [594:595]  motion_root_z_position                (1D)  — g1 不用, zero
            [595:601]  motion_anchor_orientation             (6D)  — g1 不用, zero
            [601:661]  motion_anchor_orientation_10frame_step5 (10 × 6 = 60D)
            [661:1762] 其他 (lowerbody/vr/smpl/wrists)              — g1 不用, zero

        flatten 约定: row-major (time-major) `[t0_f0..fN, t1_f0..fN, ...]` 匹配 ONNX
        wrapper reshape 行为 + IsaacLab CircularBuffer 默认 flatten。

        B6 关键纠正（vs B5 之前 / F4 阶段）：
          - encoder_mode_4 占 4D 不是 1D，后续 obs 从 offset 4 起
          - g1 encoder 实际吃 mocap.dof 的 joint_pos+joint_vel（290+290=580D），
            不是 F4 阶段算的 mocap body_pos 在 pelvis frame 下的 14×3=420D
          - joint_vel 由 mocap.dof finite difference 算（mocap.dof 自带，无需 FK）
        """
        enc = np.zeros((1, self._encoder_input_dim), dtype=np.float32)

        # [0:4] encoder_mode_4: [0] = mode_id (0=g1), [1:4] zero-fill
        enc[0, 0] = float(self.cfg.probe_encoder_mode)

        if self._mocap_dof is not None and self._mocap_num_frames > 0:
            n = self._mocap_num_frames
            indices = self._mocap_future_indices(10)

            # [4:294] motion_joint_positions_10frame_step5 = mocap.dof[future 10 frames]
            mocap_jp = self._mocap_dof[indices]  # (10, 29)
            enc[0, 4:294] = mocap_jp.flatten().cpu().numpy()

            # [294:584] motion_joint_velocities_10frame_step5 = finite diff on mocap.dof
            # dt = 1 mocap frame interval = 1 / mocap_fps (e.g., 1/50 = 0.02s)
            idx_next = (indices + 1) % n if self.cfg.loop_mocap else torch.clamp(indices + 1, max=n - 1)
            dt = 1.0 / self._mocap_fps
            mocap_jv = (self._mocap_dof[idx_next] - self._mocap_dof[indices]) / dt  # (10, 29)
            enc[0, 294:584] = mocap_jv.flatten().cpu().numpy()

            # [601:661] motion_anchor_orientation_10frame_step5 (10 × 6D rotation diff)
            if self._mocap_root_rot_wxyz is not None:
                ref_quat = self._mocap_root_rot_wxyz[indices]  # (10, 4) wxyz
                # Deploy GatherMotionAnchorOrientationMutiFrame(mode=0):
                # base_to_ref = quat_inv(current_robot_base) * future_ref_root.
                # This is equal to absolute ref_quat only at frame-0 identity reset; arbitrary
                # reset frames need the live robot root orientation to stay in-distribution.
                base_quat = self._asset.data.root_quat_w[env_idx].unsqueeze(0).expand_as(ref_quat)
                base_to_ref_quat = quat_mul(quat_conjugate(base_quat), ref_quat)
                mat = matrix_from_quat(base_to_ref_quat)  # (10, 3, 3)
                ori_6d = mat[..., :2].reshape(10, 6)  # (10, 6) row-major [m00,m01,m10,m11,m20,m21]
                enc[0, 601:661] = ori_6d.flatten().cpu().numpy()
            else:
                # fallback identity 10 frames tile (row-major)
                identity_6d = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
                enc[0, 601:661] = np.tile(identity_6d, 10)

        return enc

    def _resolve_mocap_frame(self, frame: int | float) -> int:
        if self._mocap_num_frames <= 0:
            return 0
        frame_i = int(frame)
        if self.cfg.loop_mocap:
            return frame_i % self._mocap_num_frames
        return max(0, min(frame_i, self._mocap_num_frames - 1))

    def _mocap_future_indices(self, count: int) -> torch.Tensor:
        n = self._mocap_num_frames
        indices = self._mocap_frame + torch.arange(count, device=self.device, dtype=torch.long) * self._mocap_step
        if self.cfg.loop_mocap:
            return indices % n
        return torch.clamp(indices, max=n - 1)

    def _advance_mocap(self):
        """每个 sim step 推进 mocap_fps/sim_fps 帧，使 mocap 真实速度与 sim 时钟同步。

        例：mocap 30fps + sim 50Hz → 每 step 推 0.6 帧 → 50 个 sim step 推 30 mocap 帧 = 0.6s
        mocap = 0.6s 真实时间，正确匹配。错用 1 frame/step 会让 mocap 1.67× 慢播。
        """
        if self._mocap_root_rot_wxyz is None:
            return
        self._mocap_frame_f += self._mocap_advance_per_step
        if self.cfg.loop_mocap:
            self._mocap_frame = int(self._mocap_frame_f) % self._mocap_num_frames
        else:
            self._mocap_frame = max(0, min(int(self._mocap_frame_f), self._mocap_num_frames - 1))
            self._mocap_frame_f = float(self._mocap_frame)

    def _apply_mocap_root_follow(self):
        """Diagnostic root follower for visualizing mocap trajectory tracking."""
        if not (self.cfg.follow_mocap_root_xy or self.cfg.follow_mocap_root_z or self.cfg.follow_mocap_root_rot):
            return
        if self._mocap_num_frames <= 0:
            return

        frame = self._resolve_mocap_frame(self._mocap_frame)
        current_pos = self._asset.data.root_pos_w.clone()
        current_quat = self._asset.data.root_quat_w.clone()
        next_pos = current_pos.clone()
        next_quat = current_quat

        if self.cfg.follow_mocap_root_xy:
            if getattr(self, "_mocap_root_trans", None) is None:
                return
            target_xy = self._reset_root_pos_w[:, :2] + (
                self._mocap_root_trans[frame, :2].unsqueeze(0) - self._reset_mocap_root_trans[:, :2]
            )
            delta_xy = target_xy - current_pos[:, :2]

            max_speed = float(self.cfg.follow_mocap_root_xy_rate_limit_mps)
            if max_speed > 0.0:
                max_step = max_speed * float(getattr(self._env, "step_dt", 0.02))
                delta_norm = torch.linalg.norm(delta_xy, dim=-1, keepdim=True).clamp(min=1e-6)
                delta_xy = delta_xy * torch.clamp(max_step / delta_norm, max=1.0)
            next_pos[:, :2] = current_pos[:, :2] + delta_xy

        if self.cfg.follow_mocap_root_z:
            if getattr(self, "_mocap_root_trans", None) is None:
                return
            target_z = self._reset_root_pos_w[:, 2] + (
                self._mocap_root_trans[frame, 2].unsqueeze(0) - self._reset_mocap_root_trans[:, 2]
            )
            next_pos[:, 2] = target_z

        if self.cfg.follow_mocap_root_rot:
            if getattr(self, "_mocap_root_rot_wxyz", None) is None:
                return
            next_quat = self._mocap_root_rot_wxyz[frame].unsqueeze(0).expand(self.num_envs, -1)

        self._asset.write_root_pose_to_sim(torch.cat([next_pos, next_quat], dim=-1))

        root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self._asset.write_root_velocity_to_sim(root_vel)

    def _run_sonic(self) -> torch.Tensor:
        """Run g1 mocap-reference encoder + decoder with real 10-frame proprioception history."""
        n_act = self.cfg.sonic_action_dim
        # 缓存当前帧 body_pos_b（per-env loop 内复用）
        self._self_ref_body_pos_b = self._compute_self_ref_body_pos_b()  # (N, 14, 3)
        out = np.zeros((self.num_envs, n_act), dtype=np.float32)
        for i in range(self.num_envs):
            enc_in = self._build_encoder_input(env_idx=i)
            tokens = self._encoder.run([self._enc_output_name], {self._enc_input_name: enc_in})[0]
            dec_in = self._build_decoder_input(tokens, env_idx=i)
            action = self._decoder.run([self._dec_output_name], {self._dec_input_name: dec_in})[0][0]
            out[i] = action

        if (
            self.cfg.probe_encoder_mode != 0
            or self.cfg.force_zero_body_pos
            or self.cfg.force_zero_last_action_history
            or self.cfg.force_zero_decoder_history
        ):
            absmax = float(np.abs(out).max())
            print(
                f"[SONIC PROBE] mode={self.cfg.probe_encoder_mode} "
                f"zero_body={self.cfg.force_zero_body_pos} "
                f"zero_la_hist={self.cfg.force_zero_last_action_history} "
                f"zero_dec_hist={self.cfg.force_zero_decoder_history} "
                f"action_absmax={absmax:.4f} mean={out.mean():.4f} std={out.std():.4f}"
            )

        # 探针开关：部署时默认关闭。打开后会把随机关节偏移直接写进 PD target。
        if self.cfg.action_noise_enabled:
            if self._action_noise_std_np is not None:
                noise = np.random.normal(0.0, 1.0, out.shape).astype(np.float32) * self._action_noise_std_np
            else:
                noise = np.random.normal(0.0, self.cfg.action_noise_std, out.shape).astype(np.float32)
            out = out + noise

        return torch.from_numpy(out).to(device=self.device, dtype=torch.float32)

    def process_actions(self, actions: torch.Tensor):
        if self.cfg.follow_mocap_root_xy or self.cfg.follow_mocap_root_z or self.cfg.follow_mocap_root_rot:
            self._apply_mocap_root_follow()
        self._push_history()
        self._advance_mocap()
        self._apply_mocap_root_follow()
        action_rel = self._run_sonic()
        n_resolved = len(self._joint_ids)
        # SONIC 训练用 per-joint JointPositionActionCfg scale；raw action 仍进入 last_action history。
        action_delta = action_rel[:, :n_resolved] * self._sonic_action_scale[:, :n_resolved] * self.cfg.action_scale
        sonic_target = self._default_joint_pos + action_delta
        sonic_target = self._apply_mocap_target_blend(sonic_target)
        self._processed_actions = self._stabilize_target(sonic_target)
        self._last_action = action_rel

        self._debug_counter += 1
        # B path: dump step 1 obs to CSV for PyTorch vs ONNX comparison
        if self._debug_counter == 1:
            for i in range(self.num_envs):
                enc_in = self._build_encoder_input(env_idx=i)
                dec_in = self._build_decoder_input(
                    self._encoder.run([self._enc_output_name], {self._enc_input_name: enc_in})[0],
                    env_idx=i,
                )
                np.savetxt("enc_obs_step1.csv", enc_in.reshape(-1), delimiter=",", fmt="%.8f")
                np.savetxt("dec_obs_step1.csv", dec_in.reshape(-1), delimiter=",", fmt="%.8f")
                print(f"[SONIC DUMP] step=1 obs saved: enc={enc_in.shape} dec={dec_in.shape}")
        if self._debug_counter % 50 == 0:
            a = action_rel[0].detach().cpu()
            jp = self._hist_joint_pos[0, -1].detach().cpu()
            bp = self._self_ref_body_pos_b[0].detach().cpu()  # (14, 3) self-ref motion
            print(
                f"[IsaacLab] [SONIC] step={self._debug_counter} "
                f"action mean={a.mean():+.4f} absmax={a.abs().max():.4f} std={a.std():.4f} "
                f"| joint_pos absmax={jp.abs().max():.4f} "
                f"| self_ref_body_pos absmax={bp.abs().max():.4f} mean={bp.mean():+.4f} "
                f"| target_step_delta={self._last_target_step_delta_absmax[0].item():.4f}"
            )
            def _group_absmax(prefix: str | None = None, contains: tuple[str, ...] = ()) -> float:
                indices = [
                    i for i, name in enumerate(self.cfg.joint_names[: a.numel()])
                    if (prefix is None or name.startswith(prefix)) and any(s in name for s in contains)
                ]
                if not indices:
                    return 0.0
                return a[indices].abs().max().item()

            legs = _group_absmax(contains=("_hip_", "_knee_", "_ankle_"))
            waist = _group_absmax(prefix="waist", contains=("_joint",))
            l_arm = _group_absmax(prefix="left", contains=("_shoulder_", "_elbow_", "_wrist_"))
            r_arm = _group_absmax(prefix="right", contains=("_shoulder_", "_elbow_", "_wrist_"))
            print(
                f"[IsaacLab] [SONIC] step={self._debug_counter} action by joint group: "
                f"legs={legs:.3f} waist={waist:.3f} l_arm={l_arm:.3f} r_arm={r_arm:.3f}"
            )
            # 同步打印 articulation joint_names 顺序 (一次性) 以便核对 SONIC index → USD joint name 映射
            if self._debug_counter == 50:
                print(
                    f"[IsaacLab] [SONIC] articulation joint_names "
                    f"(SONIC index → USD joint name via self._joint_ids):"
                )
                art_names = self._asset.data.joint_names
                for i_sonic, jid in enumerate(self._joint_ids):
                    print(f"  SONIC[{i_sonic:2d}] → USD[{jid:2d}] {art_names[jid]}")

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # B3: 随机选 mocap 帧（所有 env 共享；mocap 不可用时 fallback 0）
        if (
            self.cfg.reset_to_random_mocap_frame
            and getattr(self, "_mocap_num_frames", 0) > 0
        ):
            max_start = self._mocap_num_frames - 1
            if not self.cfg.loop_mocap:
                max_start = max(0, max_start - 9 * int(getattr(self, "_mocap_step", 1)))
            frame_idx = int(np.random.randint(0, max_start + 1))
        else:
            frame_idx = max(0, int(self.cfg.reset_mocap_frame))
            if getattr(self, "_mocap_num_frames", 0) > 0:
                frame_idx = self._resolve_mocap_frame(frame_idx)

        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if env_ids is None or isinstance(env_ids, slice):
            reset_env_ids = all_env_ids
            is_full_reset = True
        elif isinstance(env_ids, torch.Tensor):
            reset_env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
            is_full_reset = (
                reset_env_ids.numel() == self.num_envs
                and torch.equal(torch.sort(reset_env_ids).values, all_env_ids)
            )
        else:
            reset_env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long).flatten()
            is_full_reset = (
                reset_env_ids.numel() == self.num_envs
                and torch.equal(torch.sort(reset_env_ids).values, all_env_ids)
            )

        self._processed_actions[reset_env_ids] = self._default_joint_pos[reset_env_ids]
        self._last_action[reset_env_ids] = 0.0
        self._episode_steps[reset_env_ids] = 0
        self._last_target_step_delta_absmax[reset_env_ids] = 0.0
        self._hist_base_ang_vel[reset_env_ids] = 0.0
        self._hist_joint_pos[reset_env_ids] = 0.0
        self._hist_joint_vel[reset_env_ids] = 0.0
        self._hist_last_actions[reset_env_ids] = 0.0
        self._hist_gravity_dir[reset_env_ids] = 0.0
        self._hist_gravity_dir[reset_env_ids, :, 2] = -1.0

        if is_full_reset and getattr(self, "_mocap_root_rot_wxyz", None) is not None:
            # ActionManager passes all env ids as a tensor during env.reset(). Treat that
            # as a full reset so the global mocap playback pointer stays aligned with
            # the robot pose synced below.
            self._mocap_frame = frame_idx
            self._mocap_frame_f = float(frame_idx)

        # 下游修复 2：把 sonic_robot 同步到 mocap[frame_idx]（joint_pos + root_rot）。
        # 部分 reset 时 mocap pointer 不动（其他 env 仍在推进），仅同步 reset envs 的 robot。
        self._sync_robot_to_mocap_frame(frame_idx, reset_env_ids)

        if self.cfg.reset_to_random_mocap_frame and getattr(self, "_mocap_num_frames", 0) > 0:
            print(f"[IsaacLab] [SONIC RESET] mocap frame_idx={frame_idx}/{self._mocap_num_frames}")

    def _sync_robot_to_mocap_frame(
        self, frame_idx: int = 0, env_ids: torch.Tensor | None = None
    ) -> None:
        """把 sonic_robot 的 29 个 SONIC 关节设为 mocap.dof[frame_idx]，root_rot 同步。

        root_pos 保持 sonic_robot spawn 位置（不用 mocap.root_trans[frame_idx]，那是 mocap
        坐标系起点的累积位移）。velocity 全清零，避免初速度带来 obs 偏差。

        训练 (sonic_release/config.yaml commands.motion + motion_lib) 在 episode 开始时把
        robot 设为 mocap *随机帧* 姿态。这里 frame_idx=0 是 reset 基线；
        B3 (reset_to_random_mocap_frame=True) 时调用方传入 randint(0, num_frames-1)。
        """
        if getattr(self, "_mocap_dof", None) is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        n_e = len(env_ids)
        if n_e == 0:
            return

        # 防御：frame_idx 越界 clamp 到 [0, num_frames-1]
        f = max(0, min(int(frame_idx), int(self._mocap_dof.shape[0]) - 1))

        joint_ids_t = torch.tensor(self._joint_ids, device=self.device, dtype=torch.long)
        dof_t = self._mocap_dof[f].unsqueeze(0).expand(n_e, -1)  # (n_e, 29)
        zero_jvel = torch.zeros(n_e, 29, device=self.device)
        self._asset.write_joint_state_to_sim(
            dof_t, zero_jvel, joint_ids=joint_ids_t, env_ids=env_ids
        )

        # root pose: XY 保持当前 root_pos；Z 可跟随 mocap 相对第 0 帧的高度差，避免脚底高度突变。
        current_root_pos = self._asset.data.root_pos_w[env_ids].clone()  # (n_e, 3)
        if self.cfg.align_root_height_to_mocap and getattr(self, "_mocap_root_trans", None) is not None:
            root_z_delta = self._mocap_root_trans[f, 2] - self._mocap_root_trans[0, 2]
            current_root_pos[:, 2] = current_root_pos[:, 2] + root_z_delta
        mocap_rot_f = self._mocap_root_rot_wxyz[f].unsqueeze(0).expand(n_e, -1)  # (n_e, 4)
        root_pose = torch.cat([current_root_pos, mocap_rot_f], dim=-1)
        self._asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        zero_root_vel = torch.zeros(n_e, 6, device=self.device)
        self._asset.write_root_velocity_to_sim(zero_root_vel, env_ids=env_ids)
        self._reset_root_pos_w[env_ids] = current_root_pos
        if getattr(self, "_mocap_root_trans", None) is not None:
            self._reset_mocap_root_trans[env_ids] = self._mocap_root_trans[f].unsqueeze(0)

        # _processed_actions 同步到 mocap[frame_idx]，apply_actions 第一帧不会拉回 default
        self._processed_actions[env_ids] = dof_t
        if self.cfg.seed_history_from_reset_pose:
            self._seed_history_from_pose(env_ids, dof_t, zero_jvel, mocap_rot_f)

        # debug: 一次性打印同步是否生效
        if not getattr(self, "_sync_debug_printed", False):
            actual = self._asset.data.joint_pos[env_ids[0], joint_ids_t]
            mocap_f_cpu = self._mocap_dof[f].cpu()
            default_cpu = self._default_joint_pos[0].cpu()
            print(
                f"[IsaacLab] [SONIC RESET] sync to mocap.dof[{f}]: "
                f"target absmax={mocap_f_cpu.abs().max():.3f} mean={mocap_f_cpu.mean():+.3f} "
                f"| default absmax={default_cpu.abs().max():.3f} "
                f"| post-write actual absmax={actual.abs().max().item():.3f} "
                f"| dof[{f}]-default absmax={(mocap_f_cpu - default_cpu).abs().max():.3f}"
            )
            print(
                f"[IsaacLab] [SONIC RESET] mocap root_rot[{f}] wxyz={self._mocap_root_rot_wxyz[f].cpu().tolist()}"
            )
            self._sync_debug_printed = True

    def _seed_history_from_pose(
        self,
        env_ids: torch.Tensor,
        joint_pos_abs: torch.Tensor,
        joint_vel: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> None:
        """Initialize decoder history from the reset pose instead of starting with stale zeros."""
        joint_pos_rel = joint_pos_abs - self._default_joint_pos[env_ids]
        gravity_w = torch.zeros((len(env_ids), 3), device=self.device)
        gravity_w[:, 2] = -1.0
        gravity_b = quat_apply_inverse(root_quat_w, gravity_w)

        self._hist_base_ang_vel[env_ids] = 0.0
        self._hist_joint_pos[env_ids] = joint_pos_rel.unsqueeze(1)
        self._hist_joint_vel[env_ids] = joint_vel.unsqueeze(1)
        self._hist_last_actions[env_ids] = 0.0
        self._hist_gravity_dir[env_ids] = gravity_b.unsqueeze(1)
