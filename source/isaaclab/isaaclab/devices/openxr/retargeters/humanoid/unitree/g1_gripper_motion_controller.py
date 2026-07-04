# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class G1GripperMotionControllerRetargeter(RetargeterBase):
    """Retarget OpenXR motion-controller inputs to G1 hand close ratios.

    Output layout:
    ``[left_index, left_middle, right_index, right_middle]``.

    Each value is in ``[0, 1]``. Right A closes the right index finger, right B closes the
    right middle finger, and trigger or grip/squeeze on either controller closes both
    fingers on that side. The right A/B bindings can be disabled from the config when those
    buttons are reserved for task-level controls. The action term closes the thumb when both
    finger commands are active. Left X is intentionally not consumed here because it is bound
    to environment reset.
    """

    def __init__(self, cfg: G1GripperMotionControllerRetargeterCfg):
        super().__init__(cfg)
        self.cfg = cfg

    def retarget(self, data: dict) -> torch.Tensor:
        left_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))

        left_index, left_middle = self._extract_inputs(left_controller_data, use_ab_buttons=False)
        right_index, right_middle = self._extract_inputs(
            right_controller_data,
            use_ab_buttons=True,
            use_button_0=self.cfg.use_right_a_button,
            use_button_1=self.cfg.use_right_b_button,
        )

        return torch.tensor(
            [left_index, left_middle, right_index, right_middle],
            dtype=torch.float32,
            device=self._sim_device,
        )

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def _extract_inputs(
        self,
        controller_data: np.ndarray,
        use_ab_buttons: bool,
        use_button_0: bool = True,
        use_button_1: bool = True,
    ) -> tuple[float, float]:
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return 0.0, 0.0

        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) <= DeviceBase.MotionControllerInputIndex.BUTTON_1.value:
            return 0.0, 0.0

        trigger = self._normalize_analog_close(float(inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value]))
        grip = self._normalize_analog_close(float(inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value]))
        button_0 = float(inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value])
        button_1 = float(inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value])

        full_grip = max(trigger, grip)
        index_close = max(full_grip, button_0 if use_ab_buttons and use_button_0 else 0.0)
        middle_close = max(full_grip, button_1 if use_ab_buttons and use_button_1 else 0.0)

        index_close = 0.0 if index_close < self.cfg.deadzone else index_close
        middle_close = 0.0 if middle_close < self.cfg.deadzone else middle_close

        return min(max(index_close, 0.0), 1.0), min(max(middle_close, 0.0), 1.0)

    def _normalize_analog_close(self, value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        if value < self.cfg.deadzone:
            return 0.0

        full_press_threshold = min(
            max(self.cfg.full_press_threshold, self.cfg.deadzone + 1.0e-6),
            1.0,
        )
        normalized = (value - self.cfg.deadzone) / (full_press_threshold - self.cfg.deadzone)
        return min(normalized, 1.0)


@dataclass
class G1GripperMotionControllerRetargeterCfg(RetargeterCfg):
    """Configuration for G1 motion-controller gripper retargeting."""

    deadzone: float = 0.04
    """Ignore small analog trigger/grip noise below this threshold."""

    full_press_threshold: float = 0.85
    """Analog trigger/grip value treated as a full close command after deadzone normalization."""

    use_right_a_button: bool = True
    """Whether right controller A contributes to the right index finger close command."""

    use_right_b_button: bool = True
    """Whether right controller B contributes to the right middle finger close command."""

    retargeter_type: type[RetargeterBase] = G1GripperMotionControllerRetargeter
