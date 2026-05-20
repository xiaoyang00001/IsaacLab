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
        self._reference_body_xy: np.ndarray | None = None
        self._reference_yaw: float | None = None
        self._filtered_command = torch.tensor(
            [0.0, 0.0, 0.0, cfg.hip_height], device=cfg.sim_device, dtype=torch.float32
        )
        self._debug_counter = 0
        self._waiting_for_valid_pose_logged = False
        self._turn_in_place_active = False
        self._last_head_yaw: float | None = None

    def retarget(self, data: dict) -> torch.Tensor:
        head_pose = data.get(DeviceBase.TrackingTarget.HEAD)
        if head_pose is None or len(head_pose) < 7:
            return self._filtered_command.clone()

        head_pose = np.asarray(head_pose, dtype=np.float32)
        head_xy = head_pose[list(self.cfg.planar_axis_indices)]
        head_yaw = self._quat_wxyz_to_yaw(head_pose[3:7])
        if np.any(np.abs(head_xy) > self.cfg.invalid_pose_abs_limit):
            return self._filtered_command.clone()
        if not self._is_valid_tracking_pose(head_xy, head_yaw):
            if not self._waiting_for_valid_pose_logged:
                print("[IsaacLab] [HeadPoseLocomotion] waiting for valid head pose")
                self._waiting_for_valid_pose_logged = True
            return self._filtered_command.clone()

        self._waiting_for_valid_pose_logged = False

        if self._reference_xy is None or self._reference_yaw is None:
            self._reference_xy = head_xy.copy()
            self._reference_body_xy = self._estimate_body_xy(head_xy, head_yaw)
            self._reference_yaw = head_yaw
            self._last_head_yaw = head_yaw
            print(
                "[IsaacLab] [HeadPoseLocomotion] "
                f"reference set xy=({head_xy[0]:+.3f}, {head_xy[1]:+.3f}) yaw={head_yaw:+.3f}"
            )
            return self._filtered_command.clone()

        body_xy = self._estimate_body_xy(head_xy, head_yaw)
        delta_xy_world = body_xy - self._reference_body_xy
        yaw_delta = wrap_to_pi(torch.tensor([head_yaw - self._reference_yaw], dtype=torch.float32)).item()
        delta_xy_norm = float(np.linalg.norm(delta_xy_world))
        instantaneous_yaw_delta = 0.0
        if self._last_head_yaw is not None:
            instantaneous_yaw_delta = wrap_to_pi(
                torch.tensor([head_yaw - self._last_head_yaw], dtype=torch.float32)
            ).item()
        self._last_head_yaw = head_yaw

        # Stable-operation profile: disable headset-driven turn-in-place and
        # continuously absorb yaw drift into the reference so only planar motion
        # contributes to locomotion.
        self._reference_yaw = head_yaw
        yaw_delta = 0.0
        self._turn_in_place_active = False

        if delta_xy_norm > self.cfg.reference_recenter_threshold:
            self._reference_xy = head_xy.copy()
            self._reference_body_xy = body_xy.copy()
            self._reference_yaw = head_yaw
            self._turn_in_place_active = False
            self._filtered_command = torch.tensor(
                [0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32
            )
            print(
                "[IsaacLab] [HeadPoseLocomotion] "
                f"reference recentered xy=({head_xy[0]:+.3f}, {head_xy[1]:+.3f}) yaw={head_yaw:+.3f}"
            )
            return self._filtered_command.clone()

        # For the current stable-operation branch, locomotion translation is
        # fully disabled. We still keep the reference updated for diagnostics
        # and future re-enable work, but the lower body remains rooted.
        turning_in_place = abs(instantaneous_yaw_delta) >= self.cfg.turn_motion_lock_yaw_delta
        if turning_in_place:
            self._reference_xy = head_xy.copy()
            self._reference_body_xy = body_xy.copy()
        delta_xy_world = np.zeros_like(delta_xy_world)
        vx = 0.0
        vy = 0.0

        # The root follower interprets yaw as a target heading offset from the
        # reset pose, so use the headset yaw delta directly instead of a
        # rate-like scaled signal.
        wz = 0.0

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
                f"delta=[{delta_xy_world[0]:+.3f}, {delta_xy_world[1]:+.3f}] "
                f"yaw=[head={head_yaw:+.3f}, ref={self._reference_yaw:+.3f}, delta={yaw_delta:+.3f}, inst={instantaneous_yaw_delta:+.3f}, active={turning_in_place}] "
                f"cmd_xy=[{vx:+.3f}, {vy:+.3f}] "
                f"cmd=[{debug_cmd[0]:+.3f}, {debug_cmd[1]:+.3f}, {debug_cmd[2]:+.3f}, {debug_cmd[3]:+.3f}]"
            )
        return self._filtered_command.clone()

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HEAD_TRACKING]

    def reset(self) -> None:
        self._reference_xy = None
        self._reference_body_xy = None
        self._reference_yaw = None
        self._filtered_command = torch.tensor(
            [0.0, 0.0, 0.0, self.cfg.hip_height], device=self.cfg.sim_device, dtype=torch.float32
        )
        self._turn_in_place_active = False
        self._last_head_yaw = None
        print("[IsaacLab] [HeadPoseLocomotion] reference cleared")

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
        qw, qx, qy, qz = quat_wxyz
        sin_yaw = 2.0 * (qw * qz + qx * qy)
        cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(sin_yaw, cos_yaw))

    def _estimate_body_xy(self, head_xy: np.ndarray, head_yaw: float) -> np.ndarray:
        forward_dir = np.array([np.cos(head_yaw), np.sin(head_yaw)], dtype=np.float32)
        return head_xy - self.cfg.head_to_body_forward_offset * forward_dir

    def _is_valid_tracking_pose(self, head_xy: np.ndarray, head_yaw: float) -> bool:
        if float(np.linalg.norm(head_xy)) < self.cfg.min_valid_head_radius:
            return False
        return True

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
    forward_position_scale: float = 1.2
    lateral_position_scale: float = 1.2
    yaw_scale: float = 1.0
    position_deadzone: float = 0.04
    yaw_deadzone: float = 0.08
    invalid_pose_abs_limit: float = 100.0
    planar_axis_indices: tuple[int, int] = (0, 1)
    max_linear_velocity: float = 1.0
    max_yaw_rate: float = 2.4
    smoothing_factor: float = 0.2
    head_to_body_forward_offset: float = 0.12
    min_valid_head_radius: float = 0.5
    min_valid_yaw_abs: float = 0.0
    reference_recenter_threshold: float = 2.0
    reference_recenter_yaw_threshold: float = 3.2
    turn_in_place_start_yaw: float = 0.25
    turn_in_place_full_suppression_yaw: float = 0.60
    turn_in_place_activate_yaw: float = 0.35
    turn_in_place_deactivate_yaw: float = 0.18
    yaw_reference_stationary_radius: float = 0.20
    turn_motion_lock_yaw_delta: float = 0.10
    retargeter_type: type[RetargeterBase] = G1LowerBodyStandingHeadPoseRetargeter
