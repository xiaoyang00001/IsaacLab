from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class G1OpenXRGripperRetargeter(RetargeterBase):
    """Generate Pink gripper channels from native OpenXR controller inputs.

    The output order is ``[left_index, left_middle, right_index, right_middle]``.
    It follows the native G1 controller mapping: trigger controls the index
    finger and squeeze controls the middle finger. Digital face buttons are
    intentionally ignored.
    """

    def __init__(self, cfg: G1OpenXRGripperRetargeterCfg):
        super().__init__(cfg)
        self._sim_device = cfg.sim_device

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        left = self._extract_channels(data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([])))
        right = self._extract_channels(data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([])))
        return torch.tensor([*left, *right], dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _extract_channels(controller_data: np.ndarray) -> tuple[float, float]:
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return 0.0, 0.0
        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        required_index = max(
            DeviceBase.MotionControllerInputIndex.TRIGGER.value,
            DeviceBase.MotionControllerInputIndex.SQUEEZE.value,
        )
        if len(inputs) <= required_index:
            return 0.0, 0.0
        trigger = float(inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value])
        squeeze = float(inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value])
        index_close = min(max(trigger, 0.0), 1.0)
        middle_close = min(max(squeeze, 0.0), 1.0)
        return index_close, middle_close


@dataclass
class G1OpenXRGripperRetargeterCfg(RetargeterCfg):
    """Configuration for the project-local OpenXR gripper retargeter."""

    retargeter_type: type[RetargeterBase] = G1OpenXRGripperRetargeter
