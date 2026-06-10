# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""ZeroMQ MGXR device for Isaac Lab teleoperation.

This module is a Python rewrite of the C++ ``ZeroMqGameSub`` subscriber.  It
subscribes to the MGXR ``state`` topic, parses the packed C++ payloads from
``PubDataType.h``, caches the latest tracking packet per remote player, and
exposes Isaac Lab style raw device data through ``DeviceBase._get_raw_data``.

Raw data shape intentionally follows ``openxr_device.py``:

* ``DeviceBase.TrackingTarget.HEAD`` -> numpy array ``[x, y, z, qw, qx, qy, qz]``
* ``DeviceBase.TrackingTarget.CONTROLLER_LEFT`` -> ``2 x 7`` numpy array
* ``DeviceBase.TrackingTarget.CONTROLLER_RIGHT`` -> ``2 x 7`` numpy array
* ``DeviceBase.TrackingTarget.HAND_LEFT`` -> dict of 26 joint poses
* ``DeviceBase.TrackingTarget.HAND_RIGHT`` -> dict of 26 joint poses

Additional optional string keys are also emitted for non-standard Isaac Lab data:
``remote_player_id``, ``hand_left_pinch``, ``hand_right_pinch``,
``hand_left_bend``, ``hand_right_bend``, and ``whole_body``.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from pxr import Gf

try:
    import zmq
except ModuleNotFoundError:  # Isaac Lab install may not include pyzmq by default.
    zmq = None

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase

logger = logging.getLogger(__name__)


MGXR_MAGIC = 0x4D475852  # 'MGXR'
MGXR_VERSION = 1
XR_HAND_JOINT_COUNT = 26
XR_HAND_JOINT_PINCH_NUM = 4
XR_HAND_JOINT_BEND_NUM = 5
XR_WHOLE_BODY_JOINT_COUNT = 22


class MgxrMsgType(IntEnum):
    PLAYER_ONLINE = 0
    PLAYER_OFFLINE = 1
    MOTION_CONTROLLER_TRACKING_INFO = 2
    HEAD_TRACKING_INFO = 3
    HAND_TRACKING_INFO = 4
    WHOLE_BODY_TRACKING_INFO = 5


class WholeBodyJointType(IntEnum):
    BODY_Hips = 0
    BODY_Spine = 1
    BODY_Chest = 2
    BODY_UpperChest = 3
    Head_Neck = 4
    Head_Head = 5
    LEFT_ARM_Shoulder = 6
    LEFT_ARM_UpperArm = 7
    LEFT_ARM_LowerArm = 8
    LEFT_ARM_Hand = 9
    RIGHT_ARM_Shoulder = 10
    RIGHT_ARM_UpperArm = 11
    RIGHT_ARM_LowerArm = 12
    RIGHT_ARM_Hand = 13
    LEFT_LEG_UpperLeg = 14
    LEFT_LEG_LowerLeg = 15
    LEFT_LEG_Foot = 16
    LEFT_LEG_ToeBase = 17
    RIGHT_LEG_UpperLeg = 18
    RIGHT_LEG_LowerLeg = 19
    RIGHT_LEG_Foot = 20
    RIGHT_LEG_ToeBase = 21


WHOLE_BODY_JOINT_NAMES = [joint.name for joint in WholeBodyJointType]


# C++ structs are declared under ``#pragma pack(push, 1)``.
_HEADER_STRUCT = struct.Struct("<IIIII")
_POSE_STRUCT = struct.Struct("<fffffff")  # position xyz, quaternion xyzw
_CONTROLLER_STATES_STRUCT = struct.Struct("<IIfff")  # buttons, touches, thumbstick xy, trigger
_HAND_JOINT_STRUCT = struct.Struct("<ffffffff")  # pose xyz xyzw, radius

_HEAD_PAYLOAD_SIZE = 4 + _POSE_STRUCT.size
_CONTROLLER_PAYLOAD_SIZE = 4 + _POSE_STRUCT.size + _CONTROLLER_STATES_STRUCT.size + _POSE_STRUCT.size + _CONTROLLER_STATES_STRUCT.size
_HAND_PAYLOAD_SIZE = (
    4
    + 2 * XR_HAND_JOINT_COUNT * _HAND_JOINT_STRUCT.size
    + 2 * XR_HAND_JOINT_PINCH_NUM * 4
    + 2 * XR_HAND_JOINT_BEND_NUM * 4
)
_WHOLE_BODY_PAYLOAD_SIZE = 4 + XR_WHOLE_BODY_JOINT_COUNT * _POSE_STRUCT.size


def _pose_xyzw_to_wxyz(values: tuple[float, ...]) -> np.ndarray:
    """Convert C++ XrPosef layout xyz + quaternion xyzw to Isaac Lab xyz + qwxyz.

    Applies coordinate system conversion:
    MGXR/OpenXR convention: +X = right, +Y = up, -Z = forward
    Isaac Lab convention:   +X = right, +Y = forward, +Z = up
    """
    px, py, pz, qx, qy, qz, qw = values

    oxr_quat = Gf.Quatd(qw, qx, qy, qz)
    oxr_pos = Gf.Vec3d(px, py, pz)

    oxr_matrix = Gf.Matrix4d()
    oxr_matrix.SetTransform(Gf.Rotation(oxr_quat), oxr_pos)

    # +X -> +X, +Y -> +Z, -Z -> +Y: +90 degrees around X.
    transform_matrix = Gf.Matrix4d()
    transform_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), 90.0))

    # In pxr, row-vectors are used, so M_world = M_local * M_transform.
    pose = oxr_matrix * transform_matrix

    position = pose.ExtractTranslation()
    quat = pose.ExtractRotationQuat()

    return np.array([
        position[0],
        position[1],
        position[2],
        quat.GetReal(),
        quat.GetImaginary()[0],
        quat.GetImaginary()[1],
        quat.GetImaginary()[2],
    ], dtype=np.float32)


def _zero_pose() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class ZeroMqGameSubDevice(DeviceBase):
    """Isaac Lab hardware device that consumes MGXR tracking packets over ZeroMQ."""

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"

    def __init__(self, cfg: ZeroMqGameSubDeviceCfg, retargeters: list[RetargeterBase] | None = None):
        super().__init__(retargeters)
        if zmq is None:
            raise ModuleNotFoundError("pyzmq is required. Install it with `pip install pyzmq` in the Isaac Lab environment.")

        self._cfg = cfg
        self._additional_callbacks: dict[str, Callable] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: Any | None = None
        self._socket: Any | None = None
        self._latest_remote_player_id: int | None = None

        self._head_tracking_infos: dict[int, np.ndarray] = {}
        self._motion_controller_tracking_infos: dict[int, dict[str, np.ndarray]] = {}
        self._hand_tracking_infos: dict[int, dict[str, Any]] = {}
        self._whole_body_tracking_infos: dict[int, dict[str, np.ndarray]] = {}

        self._previous_headpose = _zero_pose()
        self._previous_left_controller = np.zeros((2, 7), dtype=np.float32)
        self._previous_right_controller = np.zeros((2, 7), dtype=np.float32)
        self._previous_left_controller[0] = _zero_pose()
        self._previous_right_controller[0] = _zero_pose()
        self._previous_joint_poses_left = {name: _zero_pose() for name in HAND_JOINT_NAMES}
        self._previous_joint_poses_right = {name: _zero_pose() for name in HAND_JOINT_NAMES}
        self._previous_whole_body = {name: _zero_pose() for name in WHOLE_BODY_JOINT_NAMES}

        if cfg.auto_start:
            self.start()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    def __str__(self) -> str:
        msg = f"ZeroMQ MGXR Device: {self.__class__.__name__}\n"
        msg += f"\tEndpoint: {self._cfg.endpoint}\n"
        msg += f"\tTopic: {self._cfg.topic!r}\n"
        msg += f"\tLocal Player ID: {self._cfg.local_player_id}\n"
        msg += f"\tTarget Remote Player ID: {self._cfg.target_remote_player_id}\n"
        msg += f"\tRetargeters: {', '.join(r.__class__.__name__ for r in self._retargeters) if self._retargeters else 'None'}\n"
        return msg

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._context = zmq.Context()
        self._thread = threading.Thread(target=self._thread_sub_fun, name="ZeroMqGameSubDevice", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        context = self._context

        if thread is not None and thread.is_alive():
            join_timeout = max(1.0, self._cfg.receive_timeout_ms / 1000.0 + 0.5)
            thread.join(timeout=join_timeout)

        if thread is not None and thread.is_alive():
            logger.warning("ZeroMQ subscriber thread did not stop within timeout; terminating context.")
            context = self._context
            try:
                if context is not None:
                    context.term()
                    self._context = None
            except Exception:
                pass
            thread.join(timeout=1.0)
            context = None

        if self._socket is not None and (thread is None or not thread.is_alive()):
            try:
                self._socket.close(0)
            except Exception:
                pass
            self._socket = None

        if context is not None and (thread is None or not thread.is_alive()):
            try:
                context.term()
            except Exception:
                pass
            self._context = None

        if thread is None or not thread.is_alive():
            self._thread = None

    def reset(self) -> None:
        with self._lock:
            self._head_tracking_infos.clear()
            self._motion_controller_tracking_infos.clear()
            self._hand_tracking_infos.clear()
            self._whole_body_tracking_infos.clear()
            self._latest_remote_player_id = None
            self._previous_headpose = _zero_pose()
            self._previous_left_controller = np.zeros((2, 7), dtype=np.float32)
            self._previous_right_controller = np.zeros((2, 7), dtype=np.float32)
            self._previous_left_controller[0] = _zero_pose()
            self._previous_right_controller[0] = _zero_pose()
            self._previous_joint_poses_left = {name: _zero_pose() for name in HAND_JOINT_NAMES}
            self._previous_joint_poses_right = {name: _zero_pose() for name in HAND_JOINT_NAMES}
            self._previous_whole_body = {name: _zero_pose() for name in WHOLE_BODY_JOINT_NAMES}

    def add_callback(self, key: str, func: Callable):
        """Keep the same callback API as OpenXRDevice for START/STOP/RESET hooks."""
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor | None:
        """Process current device state and return control commands.

        Returns:
            None if no remote player is connected or no tracking packets have been received.
            Otherwise, a torch.Tensor containing the retargeted outputs.
        """
        with self._lock:
            if self._select_remote_player_id_locked() is None:
                return None
        return super().advance()

    def _get_raw_data(self) -> Any:
        """Return latest remote tracking data in Isaac Lab device format."""
        with self._lock:
            remote_player_id = self._select_remote_player_id_locked()
            data: dict[Any, Any] = {}
            if remote_player_id is None:
                return data

            data["remote_player_id"] = remote_player_id

            if RetargeterBase.Requirement.HEAD_TRACKING in self._required_features:
                self._previous_headpose = self._head_tracking_infos.get(remote_player_id, self._previous_headpose).copy()
                data[DeviceBase.TrackingTarget.HEAD] = self._previous_headpose.copy()

            if RetargeterBase.Requirement.MOTION_CONTROLLER in self._required_features:
                controller = self._motion_controller_tracking_infos.get(remote_player_id)
                if controller is not None:
                    self._previous_left_controller = controller["left"].copy()
                    self._previous_right_controller = controller["right"].copy()
                data[DeviceBase.TrackingTarget.CONTROLLER_LEFT] = self._previous_left_controller.copy()
                data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] = self._previous_right_controller.copy()

            if RetargeterBase.Requirement.HAND_TRACKING in self._required_features:
                hand = self._hand_tracking_infos.get(remote_player_id)
                if hand is not None:
                    self._previous_joint_poses_left = {k: v.copy() for k, v in hand["left_joints"].items()}
                    self._previous_joint_poses_right = {k: v.copy() for k, v in hand["right_joints"].items()}
                    data["hand_left_pinch"] = hand["left_pinch"].copy()
                    data["hand_right_pinch"] = hand["right_pinch"].copy()
                    data["hand_left_bend"] = hand["left_bend"].copy()
                    data["hand_right_bend"] = hand["right_bend"].copy()
                data[DeviceBase.TrackingTarget.HAND_LEFT] = {k: v.copy() for k, v in self._previous_joint_poses_left.items()}
                data[DeviceBase.TrackingTarget.HAND_RIGHT] = {k: v.copy() for k, v in self._previous_joint_poses_right.items()}

            whole_body = self._whole_body_tracking_infos.get(remote_player_id)
            if whole_body is not None:
                self._previous_whole_body = {k: v.copy() for k, v in whole_body.items()}
            data["whole_body"] = {k: v.copy() for k, v in self._previous_whole_body.items()}

            return data

    # ---------------------------------------------------------------------
    # ZeroMQ receive and packet parsing
    # ---------------------------------------------------------------------

    def _thread_sub_fun(self) -> None:
        socket = None
        try:
            context = self._context
            if context is None:
                return

            socket = context.socket(zmq.SUB)
            self._socket = socket
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVHWM, self._cfg.receive_high_water_mark)
            socket.setsockopt_string(zmq.SUBSCRIBE, self._cfg.topic)
            socket.setsockopt(zmq.RCVTIMEO, self._cfg.receive_timeout_ms)

            try:
                socket.connect(self._cfg.endpoint)
            except Exception as exc:
                logger.error("ZeroMQ subscriber connect failed: %s", exc)
                return

            while not self._stop_event.is_set():
                try:
                    frames = socket.recv_multipart()
                except zmq.Again:
                    continue
                except zmq.ZMQError:
                    if not self._stop_event.is_set():
                        logger.exception("ZeroMQ subscriber receive failed")
                    break

                if not frames:
                    continue
                payload = frames[-1]
                try:
                    self._handle_remote_packet(payload)
                except Exception:
                    logger.exception("Failed to handle MGXR packet")
        finally:
            if socket is not None:
                try:
                    socket.close(0)
                except Exception:
                    pass
                if self._socket is socket:
                    self._socket = None

    def _handle_remote_packet(self, data: bytes) -> None:
        header = self._parse_header(data)
        if header is None:
            return

        player_id, msg_type, payload = header
        if player_id == self._cfg.local_player_id:
            return
        # print(f"[IsaacLab] [ZeroMQ] Received message from player {player_id} of type {msg_type}")
        if msg_type == MgxrMsgType.HEAD_TRACKING_INFO:
            info = self._parse_head_tracking_info(payload)
            self._on_remote_head_tracking(player_id, info)
        elif msg_type == MgxrMsgType.MOTION_CONTROLLER_TRACKING_INFO:
            info = self._parse_motion_controller_tracking_info(payload)
            self._on_remote_motion_controller_tracking(player_id, info)
        elif msg_type == MgxrMsgType.HAND_TRACKING_INFO:
            info = self._parse_hand_tracking_info(payload)
            self._on_remote_hand_tracking(player_id, info)
        elif msg_type == MgxrMsgType.WHOLE_BODY_TRACKING_INFO:
            info = self._parse_whole_body_tracking_info(payload)
            self._on_remote_whole_body_tracking(player_id, info)

    def _parse_header(self, data: bytes) -> tuple[int, MgxrMsgType, bytes] | None:
        if len(data) < _HEADER_STRUCT.size:
            return None
        magic, _version, player_id, msg_type_raw, payload_size = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != MGXR_MAGIC:
            return None
        if _HEADER_STRUCT.size + payload_size != len(data):
            return None
        try:
            msg_type = MgxrMsgType(msg_type_raw)
        except ValueError:
            return None
        return player_id, msg_type, data[_HEADER_STRUCT.size:]

    def _parse_head_tracking_info(self, payload: bytes) -> np.ndarray:
        if len(payload) != _HEAD_PAYLOAD_SIZE:
            raise ValueError(f"invalid HEAD_TRACKING_INFO size: {len(payload)}")
        offset = 4  # uint32_t type
        return _pose_xyzw_to_wxyz(_POSE_STRUCT.unpack_from(payload, offset))

    def _parse_motion_controller_tracking_info(self, payload: bytes) -> dict[str, np.ndarray]:
        if len(payload) != _CONTROLLER_PAYLOAD_SIZE:
            raise ValueError(f"invalid MOTION_CONTROLLER_TRACKING_INFO size: {len(payload)}")
        offset = 4  # uint32_t type
        left_pose = _pose_xyzw_to_wxyz(_POSE_STRUCT.unpack_from(payload, offset))
        offset += _POSE_STRUCT.size
        left_inputs = self._parse_controller_states(payload, offset, is_left=True)
        offset += _CONTROLLER_STATES_STRUCT.size
        right_pose = _pose_xyzw_to_wxyz(_POSE_STRUCT.unpack_from(payload, offset))
        offset += _POSE_STRUCT.size
        right_inputs = self._parse_controller_states(payload, offset, is_left=False)

        left = np.stack([left_pose, left_inputs]).astype(np.float32)
        right = np.stack([right_pose, right_inputs]).astype(np.float32)
        return {"left": left, "right": right}

    def _parse_controller_states(self, payload: bytes, offset: int, is_left: bool) -> np.ndarray:
        buttons, _touches, thumbstick_x, thumbstick_y, trigger = _CONTROLLER_STATES_STRUCT.unpack_from(payload, offset)
        trigger_button_mask = getattr(self._cfg, "trigger_button_mask", 0)
        trigger_pressed = 1.0 if trigger_button_mask and (buttons & trigger_button_mask) else 0.0
        trigger = max(float(trigger), trigger_pressed)
        squeeze = 1.0 if (buttons & self._cfg.squeeze_button_mask) else 0.0
        button_0 = 1.0 if (buttons & self._cfg.button_0_mask) else 0.0
        button_1 = 1.0 if (buttons & self._cfg.button_1_mask) else 0.0
        thumbstick_click = 1.0 if (buttons & self._cfg.thumbstick_button_mask) else 0.0

        if is_left:
            if button_0 > 0.5:
                if "START" in self._additional_callbacks:
                    self._additional_callbacks["START"]()
                print("[IsaacLab] [ZeroMQ] Button 0 pressed")
            if button_1 > 0.5:
                if "STOP" in self._additional_callbacks:
                    self._additional_callbacks["STOP"]()
                print("[IsaacLab] [ZeroMQ] Button 1 pressed")
        else:
            if button_0 > 0.5:
                if "RESET" in self._additional_callbacks:
                    self._additional_callbacks["RESET"]()
                print("[IsaacLab] [ZeroMQ] Button 0 pressed")

        return np.array(
            [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, thumbstick_click], dtype=np.float32
        )

    def _parse_hand_tracking_info(self, payload: bytes) -> dict[str, Any]:
        if len(payload) != _HAND_PAYLOAD_SIZE:
            raise ValueError(f"invalid HAND_TRACKING_INFO size: {len(payload)}")
        offset = 4  # uint32_t type
        left_joints: dict[str, np.ndarray] = {}
        right_joints: dict[str, np.ndarray] = {}

        for i in range(XR_HAND_JOINT_COUNT):
            joint_values = _HAND_JOINT_STRUCT.unpack_from(payload, offset)
            left_joints[HAND_JOINT_NAMES[i]] = _pose_xyzw_to_wxyz(joint_values[:7])
            offset += _HAND_JOINT_STRUCT.size

        for i in range(XR_HAND_JOINT_COUNT):
            joint_values = _HAND_JOINT_STRUCT.unpack_from(payload, offset)
            right_joints[HAND_JOINT_NAMES[i]] = _pose_xyzw_to_wxyz(joint_values[:7])
            offset += _HAND_JOINT_STRUCT.size

        left_pinch = np.frombuffer(payload, dtype="<f4", count=XR_HAND_JOINT_PINCH_NUM, offset=offset).copy()
        offset += XR_HAND_JOINT_PINCH_NUM * 4
        right_pinch = np.frombuffer(payload, dtype="<f4", count=XR_HAND_JOINT_PINCH_NUM, offset=offset).copy()
        offset += XR_HAND_JOINT_PINCH_NUM * 4
        left_bend = np.frombuffer(payload, dtype="<f4", count=XR_HAND_JOINT_BEND_NUM, offset=offset).copy()
        offset += XR_HAND_JOINT_BEND_NUM * 4
        right_bend = np.frombuffer(payload, dtype="<f4", count=XR_HAND_JOINT_BEND_NUM, offset=offset).copy()

        return {
            "left_joints": left_joints,
            "right_joints": right_joints,
            "left_pinch": left_pinch.astype(np.float32),
            "right_pinch": right_pinch.astype(np.float32),
            "left_bend": left_bend.astype(np.float32),
            "right_bend": right_bend.astype(np.float32),
        }

    def _parse_whole_body_tracking_info(self, payload: bytes) -> dict[str, np.ndarray]:
        if len(payload) != _WHOLE_BODY_PAYLOAD_SIZE:
            raise ValueError(f"invalid WHOLE_BODY_TRACKING_INFO size: {len(payload)}")
        offset = 4  # uint32_t type
        whole_body: dict[str, np.ndarray] = {}
        for joint_name in WHOLE_BODY_JOINT_NAMES:
            whole_body[joint_name] = _pose_xyzw_to_wxyz(_POSE_STRUCT.unpack_from(payload, offset))
            offset += _POSE_STRUCT.size
        return whole_body

    # ---------------------------------------------------------------------
    # Cache callbacks.  These mirror the original C++ OnRemote* methods.
    # ---------------------------------------------------------------------

    def _on_remote_head_tracking(self, remote_player_id: int, info: np.ndarray) -> None:
        with self._lock:
            self._latest_remote_player_id = remote_player_id
            self._head_tracking_infos[remote_player_id] = info

    def _on_remote_motion_controller_tracking(self, remote_player_id: int, info: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._latest_remote_player_id = remote_player_id
            self._motion_controller_tracking_infos[remote_player_id] = info

    def _on_remote_hand_tracking(self, remote_player_id: int, info: dict[str, Any]) -> None:
        with self._lock:
            self._latest_remote_player_id = remote_player_id
            self._hand_tracking_infos[remote_player_id] = info

    def _on_remote_whole_body_tracking(self, remote_player_id: int, info: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._latest_remote_player_id = remote_player_id
            self._whole_body_tracking_infos[remote_player_id] = info

    def _select_remote_player_id_locked(self) -> int | None:
        if self._cfg.target_remote_player_id is not None:
            return self._cfg.target_remote_player_id
        return self._latest_remote_player_id

    # Optional helper useful for scripts that want to poll until the first packet arrives.
    def wait_for_first_packet(self, timeout_s: float = 2.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            with self._lock:
                if self._latest_remote_player_id is not None:
                    return True
            time.sleep(0.01)
        return False


@dataclass
class ZeroMqGameSubDeviceCfg(DeviceCfg):
    """Configuration for the ZeroMQ MGXR Isaac Lab device."""

    endpoint: str = "tcp://127.0.0.1:5555"
    topic: str = "state"
    local_player_id: int = 0
    target_remote_player_id: int | None = None
    auto_start: bool = True
    receive_timeout_ms: int = 20
    receive_high_water_mark: int = 10

    # ControllerStates only contains buttons/touches/thumbstick/trigger_value.
    # These bit masks map packed ``buttons`` to the 7-value Isaac Lab controller input row:
    # [thumbstick_x, thumbstick_y, trigger, squeeze, button_0, button_1, padding].
    squeeze_button_mask: int = 1 << 3
    trigger_button_mask: int = 1 << 4
    button_0_mask: int = 1 << 0
    button_1_mask: int = 1 << 1
    thumbstick_button_mask: int = 1 << 2

    class_type: type[DeviceBase] = ZeroMqGameSubDevice
