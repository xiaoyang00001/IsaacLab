# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""RobotYao wheeled-base retargeter for Unity XR controller input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8 or not np.isfinite(norm):
        return None
    return (quat / norm).astype(np.float32)


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _axis_angle_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat)
    if quat is None:
        return np.zeros(3, dtype=np.float32)
    if quat[0] < 0.0:
        quat = -quat

    vector = quat[1:4]
    sin_half_angle = float(np.linalg.norm(vector))
    if sin_half_angle < 1.0e-8:
        return (2.0 * vector).astype(np.float32)

    angle = 2.0 * np.arctan2(sin_half_angle, float(quat[0]))
    return (vector / sin_half_angle * angle).astype(np.float32)


def _relative_rotvec_wxyz(current_quat: np.ndarray, previous_quat: np.ndarray) -> np.ndarray:
    delta_quat = _quat_mul_wxyz(current_quat, _quat_conjugate_wxyz(previous_quat))
    return _axis_angle_from_quat_wxyz(delta_quat)


_MOCOPI_ARM_JOINT_NAMES = {
    "left": (
        "LEFT_ARM_UpperArm",
        "LEFT_ARM_LowerArm",
        "LEFT_ARM_Hand",
    ),
    "right": (
        "RIGHT_ARM_UpperArm",
        "RIGHT_ARM_LowerArm",
        "RIGHT_ARM_Hand",
    ),
}


def _parse_joint_signs(signs: str, *, label: str) -> np.ndarray:
    values = [value.strip() for value in signs.split(",") if value.strip()]
    if len(values) != 7:
        raise ValueError(f"{label} must contain exactly 7 comma-separated signs.")
    parsed = np.asarray([float(value) for value in values], dtype=np.float32)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{label} contains NaN/Inf.")
    return parsed


class RobotYaoWheeledXrRetargeter(RetargeterBase):
    """Map Unity XR controller data to a compact wheeled-robot command.

    Output tensor layout:

    ``[
        base_forward_mps,
        base_lateral_mps,
        base_yaw_radps,
        arm_follow_active,
        left_arm_delta_x,
        left_arm_delta_y,
        left_arm_delta_z,
        right_arm_delta_x,
        right_arm_delta_y,
        right_arm_delta_z,
        left_grip,
        right_grip,
        left_primary,
        left_secondary,
        left_thumbstick_click,
        right_primary,
        right_secondary,
        right_thumbstick_click,
        left_trigger,
        right_trigger,
        unscaled_left_delta_x,
        unscaled_left_delta_y,
        unscaled_left_delta_z,
        unscaled_right_delta_x,
        unscaled_right_delta_y,
        unscaled_right_delta_z,
        left_arm_rot_delta_x,
        left_arm_rot_delta_y,
        left_arm_rot_delta_z,
        right_arm_rot_delta_x,
        right_arm_rot_delta_y,
        right_arm_rot_delta_z,
        left_arm_follow_active,
        right_arm_follow_active,
        base_height_vel,
        left_arm_joint_delta_0..6,
        right_arm_joint_delta_0..6,
    ]``

    Right-hand B/A starts/stops right-arm follow; left-hand Y/X starts/stops
    left-arm follow. Controller poses are expected to already be converted to Isaac Lab coordinates by
    ``ZeroMqGameSubDevice``; this retargeter differences consecutive controller
    poses and applies position/rotation delta scales. Holding the left grip enters
    body-lift mode, routes the left stick Y axis to the lift command, and freezes
    arm deltas so the arms remain fixed relative to the moving body.
    """

    OUTPUT_SIZE = 49
    LEGACY_OUTPUT_SIZE = 35
    BASE_FORWARD = 0
    BASE_LATERAL = 1
    BASE_YAW = 2
    ARM_FOLLOW_ACTIVE = 3
    LEFT_ARM_DELTA_START = 4
    RIGHT_ARM_DELTA_START = 7
    LEFT_GRIP = 10
    RIGHT_GRIP = 11
    LEFT_PRIMARY = 12
    LEFT_SECONDARY = 13
    LEFT_THUMBSTICK_CLICK = 14
    RIGHT_PRIMARY = 15
    RIGHT_SECONDARY = 16
    RIGHT_THUMBSTICK_CLICK = 17
    LEFT_TRIGGER = 18
    RIGHT_TRIGGER = 19
    RAW_LEFT_DELTA_START = 20
    RAW_RIGHT_DELTA_START = 23
    LEFT_ARM_ROT_DELTA_START = 26
    RIGHT_ARM_ROT_DELTA_START = 29
    LEFT_ARM_FOLLOW_ACTIVE = 32
    RIGHT_ARM_FOLLOW_ACTIVE = 33
    BASE_HEIGHT_VEL = 34
    LEFT_ARM_JOINT_DELTA_START = 35
    RIGHT_ARM_JOINT_DELTA_START = 42
    ARM_JOINT_DELTA_SIZE = 7

    def __init__(self, cfg: RobotYaoWheeledXrRetargeterCfg):
        super().__init__(cfg)
        if cfg.follow_button_mode not in ("toggle", "hold"):
            raise ValueError("follow_button_mode must be either 'toggle' or 'hold'.")

        self._dead_zone = float(cfg.dead_zone)
        self._max_forward_speed = float(cfg.max_forward_speed)
        self._max_lateral_speed = float(cfg.max_lateral_speed)
        self._max_yaw_rate = float(cfg.max_yaw_rate)
        self._arm_delta_scale = float(cfg.arm_delta_scale)
        self._arm_rotation_delta_scale = float(cfg.arm_rotation_delta_scale)
        self._arm_position_delta_dead_zone = max(0.0, float(cfg.arm_position_delta_dead_zone))
        self._arm_rotation_delta_dead_zone = max(0.0, float(cfg.arm_rotation_delta_dead_zone))
        self._follow_button_mode = cfg.follow_button_mode
        self._debug_deltas = bool(cfg.debug_deltas)
        self._mocopi_arm_joint_control = bool(cfg.mocopi_arm_joint_control)
        self._mocopi_arm_joint_delta_scale = float(cfg.mocopi_arm_joint_delta_scale)
        self._mocopi_arm_joint_dead_zone = max(0.0, float(cfg.mocopi_arm_joint_dead_zone))
        self._mocopi_arm_joint_max_step = max(0.0, float(cfg.mocopi_arm_joint_max_step))
        self._mocopi_left_joint_signs = _parse_joint_signs(
            cfg.mocopi_left_joint_signs, label="mocopi_left_joint_signs"
        )
        self._mocopi_right_joint_signs = _parse_joint_signs(
            cfg.mocopi_right_joint_signs, label="mocopi_right_joint_signs"
        )

        self._left_arm_follow_active = False
        self._right_arm_follow_active = False
        self._previous_left_follow_start_button = False
        self._previous_left_follow_stop_button = False
        self._previous_right_follow_start_button = False
        self._previous_right_follow_stop_button = False
        self._previous_left_controller_position: np.ndarray | None = None
        self._previous_right_controller_position: np.ndarray | None = None
        self._previous_left_controller_quat: np.ndarray | None = None
        self._previous_right_controller_quat: np.ndarray | None = None
        self._previous_mocopi_arm_quats: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
        self._previous_left_grip_active = False

    def retarget(self, data: dict) -> torch.Tensor:
        left_controller = np.asarray(data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([])), dtype=np.float32)
        right_controller = np.asarray(
            data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([])), dtype=np.float32
        )

        left_inputs = self._extract_inputs(left_controller)
        right_inputs = self._extract_inputs(right_controller)

        left_grip_val = left_inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]
        left_grip_active = left_grip_val > 0.5
        left_grip_released = self._previous_left_grip_active and not left_grip_active
        thumbstick_y = self._apply_dead_zone(left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value])
        thumbstick_x = self._apply_dead_zone(left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])
        if left_grip_active:
            forward = 0.0
            lateral = 0.0
            height_vel = thumbstick_y
            arm_delta_enabled = False
        else:
            forward = thumbstick_y
            lateral = thumbstick_x
            height_vel = 0.0
            arm_delta_enabled = True

        yaw = self._apply_dead_zone(right_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])

        left_follow_start_button = left_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value] > 0.5
        left_follow_stop_button = left_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value] > 0.5
        right_follow_start_button = right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value] > 0.5
        right_follow_stop_button = right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value] > 0.5
        self._update_follow_state("left", left_follow_start_button, left_follow_stop_button)
        self._update_follow_state("right", right_follow_start_button, right_follow_stop_button)
        if left_grip_released:
            self._stop_all_arm_follow()
        arm_follow_active = self._left_arm_follow_active or self._right_arm_follow_active

        left_raw_delta = np.zeros(3, dtype=np.float32)
        right_raw_delta = np.zeros(3, dtype=np.float32)
        left_delta = np.zeros(3, dtype=np.float32)
        right_delta = np.zeros(3, dtype=np.float32)
        left_rot_delta = np.zeros(3, dtype=np.float32)
        right_rot_delta = np.zeros(3, dtype=np.float32)
        left_joint_delta = np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)
        right_joint_delta = np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)
        left_position = self._extract_position(left_controller)
        right_position = self._extract_position(right_controller)
        left_quat = self._extract_quaternion(left_controller)
        right_quat = self._extract_quaternion(right_controller)

        if left_position is not None:
            if self._previous_left_controller_position is not None:
                left_raw_delta = left_position - self._previous_left_controller_position
                left_delta = left_raw_delta.copy()
            self._previous_left_controller_position = left_position.copy()
        else:
            self._previous_left_controller_position = None

        if right_position is not None:
            if self._previous_right_controller_position is not None:
                right_raw_delta = right_position - self._previous_right_controller_position
                right_delta = right_raw_delta.copy()
            self._previous_right_controller_position = right_position.copy()
        else:
            self._previous_right_controller_position = None

        if left_quat is not None:
            if self._previous_left_controller_quat is not None:
                left_rot_delta = _relative_rotvec_wxyz(left_quat, self._previous_left_controller_quat)
            self._previous_left_controller_quat = left_quat.copy()
        else:
            self._previous_left_controller_quat = None

        if right_quat is not None:
            if self._previous_right_controller_quat is not None:
                right_rot_delta = _relative_rotvec_wxyz(right_quat, self._previous_right_controller_quat)
            self._previous_right_controller_quat = right_quat.copy()
        else:
            self._previous_right_controller_quat = None

        left_delta = self._apply_vector_dead_zone(left_delta, self._arm_position_delta_dead_zone)
        right_delta = self._apply_vector_dead_zone(right_delta, self._arm_position_delta_dead_zone)
        left_rot_delta = self._apply_vector_dead_zone(left_rot_delta, self._arm_rotation_delta_dead_zone)
        right_rot_delta = self._apply_vector_dead_zone(right_rot_delta, self._arm_rotation_delta_dead_zone)
        if self._mocopi_arm_joint_control:
            whole_body = data.get("whole_body", {})
            if self._left_arm_follow_active and arm_delta_enabled:
                left_joint_delta = self._compute_mocopi_arm_joint_delta("left", whole_body)
            else:
                self._reset_mocopi_arm_history("left")
            if self._right_arm_follow_active and arm_delta_enabled:
                right_joint_delta = self._compute_mocopi_arm_joint_delta("right", whole_body)
            else:
                self._reset_mocopi_arm_history("right")

        if self._debug_deltas and (np.any(left_raw_delta != 0.0) or np.any(left_rot_delta != 0.0)):
            print(
                f"[DEBUG Retargeter] Left Hand Delta - "
                f"Controller (Isaac xyz): [{left_raw_delta[0]:.6f}, {left_raw_delta[1]:.6f}, {left_raw_delta[2]:.6f}], "
                f"Arm delta before scale: [{left_delta[0]:.6f}, {left_delta[1]:.6f}, {left_delta[2]:.6f}], "
                f"Rot delta axis-angle: [{left_rot_delta[0]:.6f}, {left_rot_delta[1]:.6f}, {left_rot_delta[2]:.6f}], "
                f"FollowActive: {self._left_arm_follow_active}",
                flush=True
            )
        if self._debug_deltas and (np.any(right_raw_delta != 0.0) or np.any(right_rot_delta != 0.0)):
            print(
                f"[DEBUG Retargeter] Right Hand Delta - "
                f"Controller (Isaac xyz): [{right_raw_delta[0]:.6f}, {right_raw_delta[1]:.6f}, {right_raw_delta[2]:.6f}], "
                f"Arm delta before scale: [{right_delta[0]:.6f}, {right_delta[1]:.6f}, {right_delta[2]:.6f}], "
                f"Rot delta axis-angle: [{right_rot_delta[0]:.6f}, {right_rot_delta[1]:.6f}, {right_rot_delta[2]:.6f}], "
                f"FollowActive: {self._right_arm_follow_active}",
                flush=True
            )

        output = np.zeros(self.OUTPUT_SIZE, dtype=np.float32)
        output[self.BASE_FORWARD] = forward * self._max_forward_speed
        output[self.BASE_LATERAL] = lateral * self._max_lateral_speed
        output[self.BASE_YAW] = yaw * self._max_yaw_rate
        output[self.ARM_FOLLOW_ACTIVE] = 1.0 if arm_follow_active else 0.0
        output[self.LEFT_ARM_DELTA_START : self.LEFT_ARM_DELTA_START + 3] = (
            left_delta * self._arm_delta_scale if self._left_arm_follow_active and arm_delta_enabled else 0.0
        )
        output[self.RIGHT_ARM_DELTA_START : self.RIGHT_ARM_DELTA_START + 3] = (
            right_delta * self._arm_delta_scale if self._right_arm_follow_active and arm_delta_enabled else 0.0
        )
        output[self.LEFT_GRIP] = left_inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]
        output[self.RIGHT_GRIP] = right_inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]
        output[self.LEFT_PRIMARY] = left_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value]
        output[self.LEFT_SECONDARY] = left_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value]
        output[self.LEFT_THUMBSTICK_CLICK] = left_inputs[DeviceBase.MotionControllerInputIndex.PADDING.value]
        output[self.RIGHT_PRIMARY] = right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value]
        output[self.RIGHT_SECONDARY] = right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value]
        output[self.RIGHT_THUMBSTICK_CLICK] = right_inputs[DeviceBase.MotionControllerInputIndex.PADDING.value]
        output[self.LEFT_TRIGGER] = left_inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
        output[self.RIGHT_TRIGGER] = right_inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]
        output[self.RAW_LEFT_DELTA_START : self.RAW_LEFT_DELTA_START + 3] = left_raw_delta
        output[self.RAW_RIGHT_DELTA_START : self.RAW_RIGHT_DELTA_START + 3] = right_raw_delta
        output[self.LEFT_ARM_ROT_DELTA_START : self.LEFT_ARM_ROT_DELTA_START + 3] = (
            left_rot_delta * self._arm_rotation_delta_scale if self._left_arm_follow_active and arm_delta_enabled else 0.0
        )
        output[self.RIGHT_ARM_ROT_DELTA_START : self.RIGHT_ARM_ROT_DELTA_START + 3] = (
            right_rot_delta * self._arm_rotation_delta_scale if self._right_arm_follow_active and arm_delta_enabled else 0.0
        )
        output[self.LEFT_ARM_FOLLOW_ACTIVE] = 1.0 if self._left_arm_follow_active else 0.0
        output[self.RIGHT_ARM_FOLLOW_ACTIVE] = 1.0 if self._right_arm_follow_active else 0.0
        output[self.BASE_HEIGHT_VEL] = height_vel
        output[
            self.LEFT_ARM_JOINT_DELTA_START : self.LEFT_ARM_JOINT_DELTA_START + self.ARM_JOINT_DELTA_SIZE
        ] = left_joint_delta
        output[
            self.RIGHT_ARM_JOINT_DELTA_START : self.RIGHT_ARM_JOINT_DELTA_START + self.ARM_JOINT_DELTA_SIZE
        ] = right_joint_delta
        self._previous_left_grip_active = left_grip_active
        return torch.tensor(output, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _update_follow_state(self, side: str, follow_start_button: bool, follow_stop_button: bool) -> None:
        if side == "left":
            was_active = self._left_arm_follow_active
            previous_start_button = self._previous_left_follow_start_button
            previous_stop_button = self._previous_left_follow_stop_button
        elif side == "right":
            was_active = self._right_arm_follow_active
            previous_start_button = self._previous_right_follow_start_button
            previous_stop_button = self._previous_right_follow_stop_button
        else:
            raise ValueError(f"Unsupported follow side: {side}")

        is_active = was_active
        if self._follow_button_mode == "hold":
            if follow_start_button and not was_active:
                self._reset_controller_pose_history(side)
            if follow_stop_button:
                is_active = False
                self._reset_controller_pose_history(side)
            else:
                is_active = follow_start_button
        elif follow_stop_button and not previous_stop_button:
            is_active = False
            self._reset_controller_pose_history(side)
        elif follow_start_button and not previous_start_button:
            is_active = True
            self._reset_controller_pose_history(side)

        if side == "left":
            self._left_arm_follow_active = is_active
            self._previous_left_follow_start_button = follow_start_button
            self._previous_left_follow_stop_button = follow_stop_button
        else:
            self._right_arm_follow_active = is_active
            self._previous_right_follow_start_button = follow_start_button
            self._previous_right_follow_stop_button = follow_stop_button

    def _reset_controller_pose_history(self, side: str | None = None) -> None:
        if side is None or side == "left":
            self._previous_left_controller_position = None
            self._previous_left_controller_quat = None
            self._reset_mocopi_arm_history("left")
        if side is None or side == "right":
            self._previous_right_controller_position = None
            self._previous_right_controller_quat = None
            self._reset_mocopi_arm_history("right")

    def _stop_all_arm_follow(self) -> None:
        self._left_arm_follow_active = False
        self._right_arm_follow_active = False
        self._reset_controller_pose_history()

    def _extract_inputs(self, controller_data: np.ndarray) -> np.ndarray:
        inputs = np.zeros(len(DeviceBase.MotionControllerInputIndex), dtype=np.float32)
        row = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
        if controller_data.ndim == 2 and controller_data.shape[0] > row:
            count = min(inputs.size, controller_data.shape[1])
            inputs[:count] = controller_data[row, :count]
        return inputs

    def _extract_position(self, controller_data: np.ndarray) -> np.ndarray | None:
        row = DeviceBase.MotionControllerDataRowIndex.POSE.value
        if controller_data.ndim == 2 and controller_data.shape[0] > row and controller_data.shape[1] >= 3:
            return controller_data[row, :3].copy()
        return None

    def _extract_quaternion(self, controller_data: np.ndarray) -> np.ndarray | None:
        row = DeviceBase.MotionControllerDataRowIndex.POSE.value
        if controller_data.ndim == 2 and controller_data.shape[0] > row and controller_data.shape[1] >= 7:
            return _normalize_quat_wxyz(controller_data[row, 3:7].copy())
        return None

    def _reset_mocopi_arm_history(self, side: str | None = None) -> None:
        if side is None:
            self._previous_mocopi_arm_quats = {"left": {}, "right": {}}
        elif side in self._previous_mocopi_arm_quats:
            self._previous_mocopi_arm_quats[side].clear()

    def _compute_mocopi_arm_joint_delta(self, side: str, whole_body: dict) -> np.ndarray:
        if side not in _MOCOPI_ARM_JOINT_NAMES:
            raise ValueError(f"Unsupported Mocopi arm side: {side}")
        if not isinstance(whole_body, dict):
            self._reset_mocopi_arm_history(side)
            return np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)

        current_quats: dict[str, np.ndarray] = {}
        for joint_name in _MOCOPI_ARM_JOINT_NAMES[side]:
            pose = whole_body.get(joint_name)
            if pose is None:
                self._reset_mocopi_arm_history(side)
                return np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)
            pose_array = np.asarray(pose, dtype=np.float32)
            if pose_array.shape[0] < 7:
                self._reset_mocopi_arm_history(side)
                return np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)
            quat = _normalize_quat_wxyz(pose_array[3:7].copy())
            if quat is None:
                self._reset_mocopi_arm_history(side)
                return np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)
            current_quats[joint_name] = quat

        previous_quats = self._previous_mocopi_arm_quats[side]
        if len(previous_quats) != len(current_quats):
            self._previous_mocopi_arm_quats[side] = {name: quat.copy() for name, quat in current_quats.items()}
            return np.zeros(self.ARM_JOINT_DELTA_SIZE, dtype=np.float32)

        upper_name, lower_name, hand_name = _MOCOPI_ARM_JOINT_NAMES[side]
        upper = _relative_rotvec_wxyz(current_quats[upper_name], previous_quats[upper_name])
        lower = _relative_rotvec_wxyz(current_quats[lower_name], previous_quats[lower_name])
        hand = _relative_rotvec_wxyz(current_quats[hand_name], previous_quats[hand_name])
        self._previous_mocopi_arm_quats[side] = {name: quat.copy() for name, quat in current_quats.items()}

        # First-pass human-arm to 7-DoF robot-arm delta map.  The signs are CLI-tunable
        # because Mocopi/avatar rig axes and robot joint axes are not guaranteed to match.
        joint_delta = np.asarray(
            [
                upper[1],
                -upper[0],
                upper[2],
                lower[1],
                lower[0],
                hand[1],
                hand[2],
            ],
            dtype=np.float32,
        )
        signs = self._mocopi_left_joint_signs if side == "left" else self._mocopi_right_joint_signs
        joint_delta = joint_delta * signs * self._mocopi_arm_joint_delta_scale
        joint_delta = self._apply_vector_dead_zone(joint_delta, self._mocopi_arm_joint_dead_zone)
        if self._mocopi_arm_joint_max_step > 0.0:
            joint_delta = np.clip(joint_delta, -self._mocopi_arm_joint_max_step, self._mocopi_arm_joint_max_step)
        return joint_delta.astype(np.float32)

    def _apply_dead_zone(self, value: float) -> float:
        value = float(value)
        if abs(value) < self._dead_zone:
            return 0.0
        return float(np.clip(value, -1.0, 1.0))

    @staticmethod
    def _apply_vector_dead_zone(vector: np.ndarray, dead_zone: float) -> np.ndarray:
        if dead_zone <= 0.0:
            return vector
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= dead_zone:
            return np.zeros_like(vector)
        return vector


@dataclass
class RobotYaoWheeledXrRetargeterCfg(RetargeterCfg):
    """Configuration for Unity XR control of the RobotYao wheeled scene."""

    dead_zone: float = 0.12
    max_forward_speed: float = 1.0
    max_lateral_speed: float = 0.6
    max_yaw_rate: float = 1.2
    arm_delta_scale: float = 1.0
    arm_rotation_delta_scale: float = 1.0
    arm_position_delta_dead_zone: float = 0.0015
    arm_rotation_delta_dead_zone: float = 0.006
    follow_button_mode: str = "toggle"
    debug_deltas: bool = False
    mocopi_arm_joint_control: bool = False
    mocopi_arm_joint_delta_scale: float = 1.0
    mocopi_arm_joint_dead_zone: float = 0.002
    mocopi_arm_joint_max_step: float = 0.08
    mocopi_left_joint_signs: str = "1,1,1,1,1,1,1"
    mocopi_right_joint_signs: str = "1,1,1,1,1,1,1"
    retargeter_type: type[RetargeterBase] = RobotYaoWheeledXrRetargeter
