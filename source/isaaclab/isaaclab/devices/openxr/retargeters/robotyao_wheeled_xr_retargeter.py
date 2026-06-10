# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""RobotYao wheeled-base retargeter for Unity XR controller input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


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
    ]``

    Right-hand B starts arm-follow, right-hand A stops arm-follow. Controller
    poses are expected to already be converted to Isaac Lab coordinates by
    ``ZeroMqGameSubDevice``; this retargeter only differences consecutive
    controller positions and applies ``arm_delta_scale``.
    """

    OUTPUT_SIZE = 26
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

    def __init__(self, cfg: RobotYaoWheeledXrRetargeterCfg):
        super().__init__(cfg)
        if cfg.follow_button_mode not in ("toggle", "hold"):
            raise ValueError("follow_button_mode must be either 'toggle' or 'hold'.")

        self._dead_zone = float(cfg.dead_zone)
        self._max_forward_speed = float(cfg.max_forward_speed)
        self._max_lateral_speed = float(cfg.max_lateral_speed)
        self._max_yaw_rate = float(cfg.max_yaw_rate)
        self._arm_delta_scale = float(cfg.arm_delta_scale)
        self._follow_button_mode = cfg.follow_button_mode

        self._arm_follow_active = False
        self._previous_follow_start_button = False
        self._previous_follow_stop_button = False
        self._previous_left_controller_position: np.ndarray | None = None
        self._previous_right_controller_position: np.ndarray | None = None

    def retarget(self, data: dict) -> torch.Tensor:
        left_controller = np.asarray(data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([])), dtype=np.float32)
        right_controller = np.asarray(
            data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([])), dtype=np.float32
        )

        left_inputs = self._extract_inputs(left_controller)
        right_inputs = self._extract_inputs(right_controller)

        forward = self._apply_dead_zone(left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value])
        lateral = self._apply_dead_zone(left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])
        yaw = self._apply_dead_zone(right_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])

        follow_start_button = right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value] > 0.5
        follow_stop_button = right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value] > 0.5
        self._update_follow_state(follow_start_button, follow_stop_button)

        left_raw_delta = np.zeros(3, dtype=np.float32)
        right_raw_delta = np.zeros(3, dtype=np.float32)
        left_delta = np.zeros(3, dtype=np.float32)
        right_delta = np.zeros(3, dtype=np.float32)
        left_position = self._extract_position(left_controller)
        right_position = self._extract_position(right_controller)

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

        if np.any(right_raw_delta != 0.0):
            print(
                f"[DEBUG Retargeter] Right Hand Delta - "
                f"Controller (Isaac xyz): [{right_raw_delta[0]:.6f}, {right_raw_delta[1]:.6f}, {right_raw_delta[2]:.6f}], "
                f"Arm delta before scale: [{right_delta[0]:.6f}, {right_delta[1]:.6f}, {right_delta[2]:.6f}], "
                f"FollowActive: {self._arm_follow_active}",
                flush=True
            )

        output = np.zeros(self.OUTPUT_SIZE, dtype=np.float32)
        output[self.BASE_FORWARD] = forward * self._max_forward_speed
        output[self.BASE_LATERAL] = lateral * self._max_lateral_speed
        output[self.BASE_YAW] = yaw * self._max_yaw_rate
        output[self.ARM_FOLLOW_ACTIVE] = 1.0 if self._arm_follow_active else 0.0
        output[self.LEFT_ARM_DELTA_START : self.LEFT_ARM_DELTA_START + 3] = (
            left_delta * self._arm_delta_scale if self._arm_follow_active else 0.0
        )
        output[self.RIGHT_ARM_DELTA_START : self.RIGHT_ARM_DELTA_START + 3] = (
            right_delta * self._arm_delta_scale if self._arm_follow_active else 0.0
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
        return torch.tensor(output, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _update_follow_state(self, follow_start_button: bool, follow_stop_button: bool) -> None:
        if self._follow_button_mode == "hold":
            if follow_start_button and not self._arm_follow_active:
                self._previous_left_controller_position = None
                self._previous_right_controller_position = None
            if follow_stop_button:
                self._arm_follow_active = False
                self._previous_left_controller_position = None
                self._previous_right_controller_position = None
            else:
                self._arm_follow_active = follow_start_button
        elif follow_stop_button and not self._previous_follow_stop_button:
            self._arm_follow_active = False
            self._previous_left_controller_position = None
            self._previous_right_controller_position = None
        elif follow_start_button and not self._previous_follow_start_button:
            self._arm_follow_active = True
            self._previous_left_controller_position = None
            self._previous_right_controller_position = None

        self._previous_follow_start_button = follow_start_button
        self._previous_follow_stop_button = follow_stop_button

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

    def _apply_dead_zone(self, value: float) -> float:
        value = float(value)
        if abs(value) < self._dead_zone:
            return 0.0
        return float(np.clip(value, -1.0, 1.0))


@dataclass
class RobotYaoWheeledXrRetargeterCfg(RetargeterCfg):
    """Configuration for Unity XR control of the RobotYao wheeled scene."""

    dead_zone: float = 0.12
    max_forward_speed: float = 1.0
    max_lateral_speed: float = 0.6
    max_yaw_rate: float = 1.2
    arm_delta_scale: float = 1.0
    follow_button_mode: str = "toggle"
    retargeter_type: type[RetargeterBase] = RobotYaoWheeledXrRetargeter
