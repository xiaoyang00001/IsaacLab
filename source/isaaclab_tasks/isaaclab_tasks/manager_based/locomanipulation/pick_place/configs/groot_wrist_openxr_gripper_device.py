# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device that combines GR00T wrist poses with OpenXR Trigger/Grip inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np
import torch
import zmq

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.sim import SimulationContext

class G1GrootWristOpenXRGripperDevice(OpenXRDevice):
    """Output GR00T wrists plus OpenXR Trigger/Grip hand-joint targets.

    The OpenXR controller pose row is intentionally never read. OpenXR is used
    only for the analog Trigger and Squeeze/Grip inputs and for its normal XR
    callbacks/anchor integration inherited from :class:`OpenXRDevice`.
    """

    def __init__(
        self,
        cfg: G1GrootWristOpenXRGripperDeviceCfg,
        retargeters: list[RetargeterBase] | None = None,
    ):
        # No retargeter participates in this device's output path.
        super().__init__(cfg, retargeters=[])
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._required_features = {RetargeterBase.Requirement.MOTION_CONTROLLER}
        self._last_valid_wrist_pose: np.ndarray | None = None
        self._hip_height = cfg.hip_height
        self._last_wrist_packet_time = 0.0
        self._last_debug_time = 0.0
        self._zmq_context = zmq.Context.instance()
        self._wrist_socket = self._zmq_context.socket(zmq.SUB)
        self._wrist_socket.setsockopt(zmq.SUBSCRIBE, cfg.wrist_zmq_topic.encode("utf-8"))
        self._wrist_socket.setsockopt(zmq.CONFLATE, 1)
        self._wrist_socket.setsockopt(zmq.RCVHWM, 1)
        self._wrist_socket.setsockopt(zmq.LINGER, 0)
        self._wrist_socket.connect(f"tcp://{cfg.wrist_zmq_host}:{cfg.wrist_zmq_port}")
        print(
            "[INFO] GR00T wrist device connected: "
            f"tcp://{cfg.wrist_zmq_host}:{cfg.wrist_zmq_port}, topic={cfg.wrist_zmq_topic}"
        )

    def advance(self) -> torch.Tensor:
        raw_data = self._get_raw_data()
        wrist_pose, wrist_source = self._read_groot_wrist_pose()
        if wrist_pose is not None:
            self._last_valid_wrist_pose = wrist_pose
        elif self._last_valid_wrist_pose is not None:
            wrist_pose = self._last_valid_wrist_pose
            wrist_source = f"held:{wrist_source}"
        else:
            # The mirror action fills the cache on the first environment step.
            # A neutral pose is used only for this startup boundary; controller
            # absolute poses are never used as a fallback.
            wrist_pose = np.array(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2,
                dtype=np.float32,
            )
            wrist_source = "waiting-for-groot"

        left_controller_data = raw_data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_controller_data = raw_data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))
        left_hand_raw, left_trigger, left_grip = self._map_trigger_grip_hand_joints(left_controller_data)
        right_hand_joints, right_trigger, right_grip = self._map_trigger_grip_hand_joints(right_controller_data)
        left_hand_joints = -left_hand_raw

        hand_joints = np.array(
            [
                left_hand_joints[3],
                left_hand_joints[5],
                left_hand_joints[0],
                right_hand_joints[3],
                right_hand_joints[5],
                right_hand_joints[0],
                left_hand_joints[4],
                left_hand_joints[6],
                left_hand_joints[1],
                right_hand_joints[4],
                right_hand_joints[6],
                right_hand_joints[1],
                left_hand_joints[2],
                right_hand_joints[2],
            ],
            dtype=np.float32,
        )
        lower_body_command = self._map_agile_locomotion(
            left_controller_data, right_controller_data
        )

        now = time.monotonic()
        if self.cfg.debug_interval_s > 0.0 and now - self._last_debug_time >= self.cfg.debug_interval_s:
            self._last_debug_time = now
            print(
                "[INFO] GR00T-wrist/OpenXR-gripper device: "
                f"wrist_source={wrist_source}, cache_key={self.cfg.cache_key}, "
                f"trigger=[{left_trigger:.3f}, {right_trigger:.3f}], "
                f"grip=[{left_grip:.3f}, {right_grip:.3f}], "
                f"hand_max={float(np.max(np.abs(hand_joints))):.3f}, "
                f"agile_cmd={np.array2string(lower_body_command, precision=3)}"
            )

        return torch.tensor(
            np.concatenate((wrist_pose, hand_joints, lower_body_command)),
            dtype=torch.float32,
            device=self._sim_device,
        )

    def _read_groot_wrist_pose(self) -> tuple[np.ndarray | None, str]:
        """Read calibrated, root-local GROOT wrist targets from the pose/planner stream."""
        latest_message: bytes | None = None
        while True:
            try:
                latest_message = self._wrist_socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_message is not None:
            try:
                fields = self._unpack_message(latest_message)
                positions = np.asarray(fields["vr_position"], dtype=np.float32).reshape(-1)
                orientations = np.asarray(fields["vr_orientation"], dtype=np.float32).reshape(-1)
                if positions.size != 9 or orientations.size != 12:
                    raise ValueError(
                        f"expected vr_position[9]/vr_orientation[12], got {positions.size}/{orientations.size}"
                    )
                # GROOT ordering is left wrist, right wrist, head. Quaternions are wxyz.
                wrist_pose = np.concatenate(
                    (
                        positions[0:3],
                        orientations[0:4],
                        positions[3:6],
                        orientations[4:8],
                    )
                ).astype(np.float32, copy=False)
                self._last_wrist_packet_time = time.monotonic()
                return wrist_pose, f"zmq:{self.cfg.wrist_zmq_topic}:base-relative"
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return None, f"invalid-groot-packet:{exc}"

        if self._last_wrist_packet_time <= 0.0:
            return None, "no-groot-packet"
        age = time.monotonic() - self._last_wrist_packet_time
        if age > self.cfg.wrist_timeout_s:
            return None, f"stale-groot-packet:{age:.3f}s"
        return None, "between-groot-packets"

    def _unpack_message(self, message: bytes) -> dict[str, np.ndarray]:
        topic = self.cfg.wrist_zmq_topic.encode("utf-8")
        if not message.startswith(topic):
            raise ValueError("topic prefix mismatch")
        offset = len(topic)
        header_size = self.cfg.wrist_header_size
        if len(message) < offset + header_size:
            raise ValueError("packet shorter than topic plus header")
        header = json.loads(message[offset : offset + header_size].rstrip(b"\x00").decode("utf-8"))
        payload_offset = offset + header_size
        dtype_map = {
            "f32": np.dtype("<f4"),
            "f64": np.dtype("<f8"),
            "i32": np.dtype("<i4"),
            "i64": np.dtype("<i8"),
            "u8": np.dtype("u1"),
            "bool": np.dtype("?"),
        }
        output: dict[str, np.ndarray] = {}
        for field in header.get("fields", []):
            dtype_name = field["dtype"]
            if dtype_name not in dtype_map:
                raise ValueError(f"unsupported dtype {dtype_name}")
            dtype = dtype_map[dtype_name]
            shape = tuple(int(value) for value in field.get("shape", []))
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            byte_count = count * dtype.itemsize
            field_end = payload_offset + byte_count
            if field_end > len(message):
                raise ValueError(f"truncated field {field['name']}")
            output[field["name"]] = np.frombuffer(
                message[payload_offset:field_end], dtype=dtype, count=count
            ).reshape(shape or (1,))
            payload_offset = field_end
        return output

    def __del__(self):
        if hasattr(self, "_wrist_socket"):
            self._wrist_socket.close(linger=0)
        super().__del__()

    def _map_trigger_grip_hand_joints(self, controller_data: np.ndarray) -> tuple[np.ndarray, float, float]:
        hand_joints = np.zeros(7, dtype=np.float32)
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return hand_joints, 0.0, 0.0
        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) <= DeviceBase.MotionControllerInputIndex.SQUEEZE.value:
            return hand_joints, 0.0, 0.0

        # Only the input row is accessed. The controller absolute pose row is ignored.
        trigger = self._normalize_analog_close(
            float(inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value])
        )
        grip = self._normalize_analog_close(
            float(inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value])
        )
        close = max(trigger, grip)

        hand_joints[1] = -0.4 * close
        hand_joints[2] = -0.7 * close
        hand_joints[3] = close
        hand_joints[4] = close
        hand_joints[5] = close
        hand_joints[6] = close
        return hand_joints, trigger, grip

    def _map_agile_locomotion(
        self, left_controller_data: np.ndarray, right_controller_data: np.ndarray
    ) -> np.ndarray:
        """Match the official G1 motion-controller mapping for Agile [vx, vy, wz, height]."""

        left_x = left_y = right_x = right_y = 0.0
        input_row = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
        if len(left_controller_data) > input_row:
            inputs = left_controller_data[input_row]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value:
                left_x = float(inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])
                left_y = float(inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value])
        if len(right_controller_data) > input_row:
            inputs = right_controller_data[input_row]
            if len(inputs) > DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value:
                right_x = float(inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])
                right_y = float(inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value])

        dt = SimulationContext.instance().get_rendering_dt()
        self._hip_height -= right_y * dt * self.cfg.rotation_scale
        self._hip_height = min(max(self._hip_height, self.cfg.min_hip_height), self.cfg.max_hip_height)
        return np.array(
            [
                -left_y * self.cfg.movement_scale,
                -left_x * self.cfg.movement_scale,
                -right_x * self.cfg.rotation_scale,
                self._hip_height,
            ],
            dtype=np.float32,
        )

    def _normalize_analog_close(self, value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        if value < self.cfg.input_deadzone:
            return 0.0
        full_press_threshold = min(
            max(self.cfg.full_press_threshold, self.cfg.input_deadzone + 1.0e-6),
            1.0,
        )
        normalized = (value - self.cfg.input_deadzone) / (full_press_threshold - self.cfg.input_deadzone)
        return min(max(normalized, 0.0), 1.0)


@dataclass
class G1GrootWristOpenXRGripperDeviceCfg(OpenXRDeviceCfg):
    """Configuration for the GR00T-wrist/OpenXR-gripper device."""

    cache_key: str = "robot"
    wrist_zmq_host: str = "127.0.0.1"
    wrist_zmq_port: int = 5556
    wrist_zmq_topic: str = "pose"
    wrist_header_size: int = 1280
    wrist_timeout_s: float = 0.5
    debug_interval_s: float = 1.0
    input_deadzone: float = 0.04
    full_press_threshold: float = 0.85
    hip_height: float = 0.72
    min_hip_height: float = 0.4
    max_hip_height: float = 1.0
    movement_scale: float = 0.5
    rotation_scale: float = 0.35
    class_type: type[DeviceBase] = G1GrootWristOpenXRGripperDevice
