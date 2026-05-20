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
        self._previous_head_xy: np.ndarray | None = None
        self._previous_head_yaw: float | None = None
        self._continuous_head_yaw: float | None = None
        self._reference_continuous_yaw: float | None = None
        self._filtered_command = torch.tensor(
            [0.0, 0.0, 0.0, cfg.hip_height], device=cfg.sim_device, dtype=torch.float32
        )
        self._debug_counter = 0
        self._has_tracking_reference = False

    def retarget(self, data: dict) -> torch.Tensor:
        head_pose = data.get(DeviceBase.TrackingTarget.HEAD)
        if head_pose is None or len(head_pose) < 7:
            return self._filtered_command.clone()

        head_pose = np.asarray(head_pose, dtype=np.float32)
        # Different OpenXR runtimes expose planar headset translation on different axes.
        # We keep the axis pair configurable so the locomotion mapping can match the
        # actual runtime instead of assuming x/z.
        head_xy = head_pose[list(self.cfg.planar_axis_indices)]
        head_yaw = self._quat_wxyz_to_yaw(head_pose[3:7])
        continuous_head_yaw = self._update_continuous_yaw(head_yaw)
        if np.any(np.abs(head_xy) > self.cfg.invalid_pose_abs_limit):
            return self._filtered_command.clone()

        if not self._has_tracking_reference:
            if np.linalg.norm(head_xy) < self.cfg.tracking_origin_epsilon and abs(head_yaw) < self.cfg.yaw_origin_epsilon:
                return self._filtered_command.clone()
            self._set_reference(head_xy, head_yaw)
            return self._filtered_command.clone()

        if self._previous_head_xy is None:
            self._previous_head_xy = head_xy.copy()
            return self._filtered_command.clone()

        step_xy_world = head_xy - self._previous_head_xy
        self._previous_head_xy = head_xy.copy()
        if np.linalg.norm(step_xy_world) > self.cfg.recenter_distance:
            self._set_reference(head_xy, head_yaw)
            self._filtered_command = torch.tensor(
                [0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32
            )
            return self._filtered_command.clone()

        # Convert per-step headset translation into a velocity command in the
        # initial heading frame. This matches the "move inside a hole and the
        # robot follows with the same directional motion" interaction model.
        delta_xy_local = self._rotate_into_reference_frame(step_xy_world, self._reference_yaw)
        command_xy_local = np.array(
            [
                delta_xy_local[1] / self.cfg.translation_time_step,
                -delta_xy_local[0] / self.cfg.translation_time_step,
            ],
            dtype=np.float32,
        )

        vx = self._apply_deadzone(command_xy_local[0], self.cfg.position_deadzone) * self.cfg.forward_position_scale
        vy = self._apply_deadzone(command_xy_local[1], self.cfg.position_deadzone) * self.cfg.lateral_position_scale

        yaw_delta = float(continuous_head_yaw - self._reference_continuous_yaw)
        if (
            np.linalg.norm(step_xy_world) <= self.cfg.yaw_glitch_linear_threshold
            and abs(yaw_delta) >= self.cfg.yaw_glitch_recenter_threshold
        ):
            self._reference_yaw = head_yaw
            self._reference_continuous_yaw = continuous_head_yaw
            yaw_delta = 0.0
        wz = self._apply_deadzone(yaw_delta, self.cfg.yaw_deadzone) * self.cfg.yaw_scale

        vx = float(np.clip(vx, -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity))
        vy = float(np.clip(vy, -self.cfg.max_linear_velocity, self.cfg.max_linear_velocity))
        wz = float(np.clip(wz, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate))

        # Head rotation often introduces a small planar arc at the headset. Treat
        # strong heading changes as in-place turning so the robot does not "chase"
        # the headset position and block the user's view.
        if abs(yaw_delta) >= self.cfg.in_place_yaw_threshold:
            vx *= self.cfg.turn_translation_scale
            vy *= self.cfg.turn_translation_scale
            self._reference_xy = head_xy.copy()

        if not self.cfg.enable_vx:
            vx = 0.0
        if not self.cfg.enable_vy:
            vy = 0.0
        linear_speed = float(np.hypot(vx, vy))
        if linear_speed > self.cfg.yaw_suppression_linear_speed:
            wz = 0.0

        target_command = torch.tensor(
            [vx, vy, wz, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32
        )
        filtered_command = self._filtered_command.clone()
        filtered_command[:2] = (
            (1.0 - self.cfg.smoothing_factor) * filtered_command[:2]
            + self.cfg.smoothing_factor * target_command[:2]
        )
        filtered_command[2] = (
            (1.0 - self.cfg.yaw_smoothing_factor) * filtered_command[2]
            + self.cfg.yaw_smoothing_factor * target_command[2]
        )
        filtered_command[3] = self.cfg.hip_height
        self._filtered_command = filtered_command
        self._debug_counter += 1
        if self._debug_counter % 60 == 0:
            debug_cmd = self._filtered_command.detach().cpu().numpy()
            print(
                "[IsaacLab] [HeadPoseLocomotion] "
                f"head_xy=[{head_xy[0]:+.3f}, {head_xy[1]:+.3f}] "
                f"delta=[{delta_xy_local[0]:+.3f}, {delta_xy_local[1]:+.3f}] "
                f"cmd_xy=[{command_xy_local[0]:+.3f}, {command_xy_local[1]:+.3f}] "
                f"yaw_delta={yaw_delta:+.3f} "
                f"cmd=[{debug_cmd[0]:+.3f}, {debug_cmd[1]:+.3f}, {debug_cmd[2]:+.3f}, {debug_cmd[3]:+.3f}]"
            )
        return self._filtered_command.clone()

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HEAD_TRACKING]

    def reset(self) -> None:
        self._reference_xy = None
        self._reference_yaw = None
        self._previous_head_xy = None
        self._previous_head_yaw = None
        self._continuous_head_yaw = None
        self._reference_continuous_yaw = None
        self._has_tracking_reference = False
        self._filtered_command = torch.tensor(
            [0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32
        )
        print("[IsaacLab] [HeadPoseLocomotion] reference cleared")

    def _set_reference(self, head_xy: np.ndarray, head_yaw: float) -> None:
        self._reference_xy = head_xy.copy()
        self._reference_yaw = head_yaw
        self._previous_head_xy = head_xy.copy()
        if self._continuous_head_yaw is None:
            self._continuous_head_yaw = head_yaw
        self._reference_continuous_yaw = self._continuous_head_yaw
        self._has_tracking_reference = True
        print(
            "[IsaacLab] [HeadPoseLocomotion] "
            f"reference set xy=({head_xy[0]:+.3f}, {head_xy[1]:+.3f}) yaw={head_yaw:+.3f}"
        )

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
    tracking_origin_epsilon: float = 0.01
    yaw_origin_epsilon: float = 0.01
    invalid_pose_abs_limit: float = 3.0
    planar_axis_indices: tuple[int, int] = (0, 1)
    translation_time_step: float = 0.02
    max_position_offset: float = 0.18
    comfort_position_radius: float = 0.04
    reference_follow_rate: float = 0.05
    recenter_distance: float = 0.75
    max_linear_velocity: float = 0.5
    max_yaw_rate: float = 0.5
    yaw_suppression_linear_speed: float = 0.03
    yaw_smoothing_factor: float = 0.35
    yaw_glitch_linear_threshold: float = 0.12
    yaw_glitch_recenter_threshold: float = 2.4
    in_place_yaw_threshold: float = 0.55
    turn_translation_scale: float = 0.0
    enable_vx: bool = True
    enable_vy: bool = True
    smoothing_factor: float = 0.2
    retargeter_type: type[RetargeterBase] = G1LowerBodyStandingHeadPoseRetargeter
