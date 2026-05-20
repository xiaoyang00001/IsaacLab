from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.utils.math import wrap_to_pi


class G1LowerBodyStandingHeadPoseRetargeter(RetargeterBase):
    """Map headset planar motion to lower-body locomotion commands for G1."""

    def __init__(self, cfg: G1LowerBodyStandingHeadPoseRetargeterCfg):
        super().__init__(cfg)
        self.cfg = cfg
        self._reference_xy: np.ndarray | None = None
        self._reference_yaw: float | None = None
        self._previous_head_yaw: float | None = None
        self._continuous_head_yaw: float | None = None
        self._reference_continuous_yaw: float | None = None
        self._filtered_command = torch.tensor(
            [0.0, 0.0, 0.0, cfg.hip_height], device=cfg.sim_device, dtype=torch.float32
        )
        self._debug_counter = 0

    def retarget(self, data: dict) -> torch.Tensor:
        head_pose = data.get(DeviceBase.TrackingTarget.HEAD)
        if head_pose is None or len(head_pose) < 7:
            return self._filtered_command.clone()

        head_pose = np.asarray(head_pose, dtype=np.float32)
        head_xy = head_pose[list(self.cfg.planar_axis_indices)]
        head_yaw = self._quat_wxyz_to_yaw(head_pose[3:7])
        if np.any(np.abs(head_xy) > self.cfg.invalid_pose_abs_limit):
            return self._filtered_command.clone()

        continuous_head_yaw = self._update_continuous_yaw(head_yaw)

        if self._reference_xy is None or self._reference_yaw is None or self._reference_continuous_yaw is None:
            self._reference_xy = head_xy.copy()
            self._reference_yaw = head_yaw
            self._reference_continuous_yaw = continuous_head_yaw
            print(
                "[IsaacLab] [HeadPoseLocomotion] "
                f"reference set xy=({head_xy[0]:+.3f}, {head_xy[1]:+.3f}) yaw={head_yaw:+.3f}"
            )
            return self._filtered_command.clone()

        delta_xy_world = head_xy - self._reference_xy
        delta_xy_local = self._rotate_into_reference_frame(delta_xy_world, self._reference_yaw)

        vx = self._apply_deadzone(delta_xy_local[0], self.cfg.position_deadzone) * self.cfg.forward_position_scale
        vy = self._apply_deadzone(delta_xy_local[1], self.cfg.position_deadzone) * self.cfg.lateral_position_scale

        yaw_delta = float(continuous_head_yaw - self._reference_continuous_yaw)
        wz = self._apply_deadzone(yaw_delta, self.cfg.yaw_deadzone) * self.cfg.yaw_scale

        vx = float(np.clip(vx, -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity))
        vy = float(np.clip(vy, -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity))
        wz = float(np.clip(wz, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate))

        target_command = torch.tensor([vx, vy, wz, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32)
        self._filtered_command = (
            (1.0 - self.cfg.smoothing_factor) * self._filtered_command
            + self.cfg.smoothing_factor * target_command
        )
        self._filtered_command[3] = self.cfg.hip_height

        self._debug_counter += 1
        if self._debug_counter % 60 == 0:
            debug_cmd = self._filtered_command.detach().cpu().numpy()
            print(
                "[IsaacLab] [HeadPoseLocomotion] "
                f"head_xy=[{head_xy[0]:+.3f}, {head_xy[1]:+.3f}] "
                f"delta=[{delta_xy_local[0]:+.3f}, {delta_xy_local[1]:+.3f}] "
                f"cmd=[{debug_cmd[0]:+.3f}, {debug_cmd[1]:+.3f}, {debug_cmd[2]:+.3f}, {debug_cmd[3]:+.3f}]"
            )
        return self._filtered_command.clone()

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HEAD_TRACKING]

    def reset(self) -> None:
        self._reference_xy = None
        self._reference_yaw = None
        self._previous_head_yaw = None
        self._continuous_head_yaw = None
        self._reference_continuous_yaw = None
        self._filtered_command = torch.tensor(
            [0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32
        )
        print("[IsaacLab] [HeadPoseLocomotion] reference cleared")

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
        qw, qx, qy, qz = quat_wxyz
        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(sin_yaw, cos_yaw))

    def _update_continuous_yaw(self, head_yaw: float) -> float:
        if self._previous_head_yaw is None or self._continuous_head_yaw is None:
            self._previous_head_yaw = head_yaw
            self._continuous_head_yaw = head_yaw
            return head_yaw

        delta_yaw = wrap_to_pi(torch.tensor([head_yaw - self._previous_head_yaw], dtype=torch.float32)).item()
        self._continuous_head_yaw += float(delta_yaw)
        self._previous_head_yaw = head_yaw
        return self._continuous_head_yaw

    @staticmethod
    def _rotate_into_reference_frame(delta_xy_world: np.ndarray, reference_yaw: float) -> np.ndarray:
        cos_yaw = np.cos(reference_yaw)
        sin_yaw = np.sin(reference_yaw)
        return np.array(
            [
                cos_yaw * delta_xy_world[0] + sin_yaw * delta_xy_world[1],
                -sin_yaw * delta_xy_world[0] + cos_yaw * delta_xy_world[1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float) -> float:
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        return float(np.sign(value) * (magnitude - deadzone))


@dataclass
class G1LowerBodyStandingHeadPoseRetargeterCfg(RetargeterCfg):
    """Configuration for headset-driven lower-body locomotion."""

    hip_height: float = 0.72
    position_scale: float = 2.0
    forward_position_scale: float = 2.0
    lateral_position_scale: float = 2.0
    yaw_scale: float = 0.75
    position_deadzone: float = 0.04
    yaw_deadzone: float = 0.08
    invalid_pose_abs_limit: float = 100.0
    planar_axis_indices: tuple[int, int] = (0, 1)
    max_linear_velocity: float = 0.5
    max_yaw_rate: float = 1.5
    smoothing_factor: float = 0.2
    retargeter_type: type[RetargeterBase] = G1LowerBodyStandingHeadPoseRetargeter
