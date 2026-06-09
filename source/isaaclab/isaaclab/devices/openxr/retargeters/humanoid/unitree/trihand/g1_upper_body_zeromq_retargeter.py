# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 upper-body retargeter for ``ZeroMqGameSubDevice`` data.

This retargeter follows the structure of Isaac Lab's G1 motion-controller
retargeter, but it is tolerant of the extra ZeroMQ hand data emitted by
``zeromq_game_sub_device.py``.  It uses controller wrist poses for the robot
wrists and maps trigger/squeeze/button or hand-pinch/bend values to the 14 G1
TriHand joints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class G1TriHandUpperBodyZeroMqRetargeter(RetargeterBase):
    """Map ZeroMQ MGXR controller/hand data to G1 upper-body teleop commands.

    Output layout matches ``G1TriHandUpperBodyMotionControllerRetargeter``:

    ``[left_wrist(7), right_wrist(7), hand_joints(14)]``

    Hand joint order:
    ``[left_proximal(3), right_proximal(3), left_distal(2), left_thumb_middle(1),
    right_distal(2), right_thumb_middle(1), left_thumb_tip(1), right_thumb_tip(1)]``.
    """

    def __init__(self, cfg: G1TriHandUpperBodyZeroMqRetargeterCfg):
        super().__init__(cfg)
        self._sim_device = cfg.sim_device
        self._hand_joint_names = cfg.hand_joint_names
        self._use_hand_tracking_if_available = cfg.use_hand_tracking_if_available
        self._enable_wrist_pose_retargeting = cfg.enable_wrist_pose_retargeting
        self._enable_visualization = cfg.enable_visualization
        self._wrist_position_offset = torch.tensor(cfg.wrist_position_offset, dtype=torch.float32)
        if cfg.hand_joint_names is None:
            raise ValueError("hand_joint_names must be provided")
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/g1_controller_markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        left_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_LEFT, np.array([]))
        right_controller_data = data.get(DeviceBase.TrackingTarget.CONTROLLER_RIGHT, np.array([]))
        #print(f"[IsaacLab] [ZeroMQ] Left controller data: {left_controller_data}")
       # print(f"[IsaacLab] [ZeroMQ] Right controller data: {right_controller_data}")
        default_wrist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        left_wrist = self._extract_wrist_pose(left_controller_data, default_wrist)
        right_wrist = self._extract_wrist_pose(right_controller_data, default_wrist)

        if self._use_hand_tracking_if_available and ("hand_left_bend" in data or "hand_left_pinch" in data):
            left_hand_joints = self._map_hand_tracking_to_hand_joints(data, is_left=True)
            right_hand_joints = self._map_hand_tracking_to_hand_joints(data, is_left=False)
        else:
            left_hand_joints = self._map_controller_to_hand_joints(left_controller_data, is_left=True)
            right_hand_joints = self._map_controller_to_hand_joints(right_controller_data, is_left=False)

        left_hand_joints = -left_hand_joints

        all_hand_joints = np.array(
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

        if self._enable_wrist_pose_retargeting:
            left_wrist_cmd = self._retarget_abs(left_wrist, is_left=True)
            right_wrist_cmd = self._retarget_abs(right_wrist, is_left=False)
        else:
            left_wrist_cmd = left_wrist
            right_wrist_cmd = right_wrist

        left_wrist_tensor = torch.tensor(left_wrist_cmd, dtype=torch.float32, device=self._sim_device)
        right_wrist_tensor = torch.tensor(right_wrist_cmd, dtype=torch.float32, device=self._sim_device)
        hand_joints_tensor = torch.tensor(all_hand_joints, dtype=torch.float32, device=self._sim_device)
        return torch.cat([left_wrist_tensor, right_wrist_tensor, hand_joints_tensor])

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        requirements = [RetargeterBase.Requirement.MOTION_CONTROLLER]
        if self._use_hand_tracking_if_available:
            requirements.append(RetargeterBase.Requirement.HAND_TRACKING)
        return requirements

    def _extract_wrist_pose(self, controller_data: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
        if len(controller_data) > DeviceBase.MotionControllerDataRowIndex.POSE.value:
            return controller_data[DeviceBase.MotionControllerDataRowIndex.POSE.value]
        return default_pose

    def _map_controller_to_hand_joints(self, controller_data: np.ndarray, is_left: bool) -> np.ndarray:
        hand_joints = np.zeros(7, dtype=np.float32)
        if len(controller_data) <= DeviceBase.MotionControllerDataRowIndex.INPUTS.value:
            return hand_joints

        inputs = controller_data[DeviceBase.MotionControllerDataRowIndex.INPUTS.value]
        if len(inputs) < len(DeviceBase.MotionControllerInputIndex):
            return hand_joints

        trigger = float(inputs[DeviceBase.MotionControllerInputIndex.TRIGGER.value])
        squeeze = float(inputs[DeviceBase.MotionControllerInputIndex.SQUEEZE.value])
        button_0 = float(inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value])
        button_1 = float(inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value])

        squeeze = max(squeeze, button_1)
        thumb_button = max(trigger, squeeze, button_0)

        thumb_angle = -thumb_button
        thumb_rotation = 0.5 * trigger - 0.5 * squeeze
        if not is_left:
            thumb_rotation = -thumb_rotation

        hand_joints[0] = thumb_rotation
        hand_joints[1] = thumb_angle * 0.4
        hand_joints[2] = thumb_angle * 0.7
        hand_joints[3] = trigger
        hand_joints[4] = trigger
        hand_joints[5] = squeeze
        hand_joints[6] = squeeze
        return hand_joints

    def _map_hand_tracking_to_hand_joints(self, data: dict, is_left: bool) -> np.ndarray:
        hand_joints = np.zeros(7, dtype=np.float32)
        side = "left" if is_left else "right"
        bend = np.asarray(data.get(f"hand_{side}_bend", np.zeros(5, dtype=np.float32)), dtype=np.float32)
        pinch = np.asarray(data.get(f"hand_{side}_pinch", np.zeros(4, dtype=np.float32)), dtype=np.float32)

        if bend.size < 3:
            return hand_joints

        thumb_bend = float(np.clip(bend[0], 0.0, 1.0))
        index_bend = float(np.clip(bend[1], 0.0, 1.0))
        middle_bend = float(np.clip(bend[2], 0.0, 1.0))
        index_pinch = float(np.clip(pinch[0], 0.0, 1.0)) if pinch.size > 0 else 0.0
        middle_pinch = float(np.clip(pinch[1], 0.0, 1.0)) if pinch.size > 1 else 0.0

        thumb_close = max(thumb_bend, index_pinch, middle_pinch)
        thumb_rotation = 0.5 * index_pinch - 0.5 * middle_pinch
        if not is_left:
            thumb_rotation = -thumb_rotation

        hand_joints[0] = thumb_rotation
        hand_joints[1] = -thumb_close * 0.4
        hand_joints[2] = -thumb_close * 0.7
        hand_joints[3] = index_bend
        hand_joints[4] = index_bend
        hand_joints[5] = middle_bend
        hand_joints[6] = middle_bend
        return hand_joints

    def _retarget_abs(self, wrist: np.ndarray, is_left: bool) -> np.ndarray:
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)

        combined_quat = torch.tensor([0.5358, -0.4619, 0.5358, 0.4619], dtype=torch.float32)

        openxr_pose = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))
        transform_pose = PoseUtils.make_pose(torch.zeros(3), PoseUtils.matrix_from_quat(combined_quat))
        result_pose = PoseUtils.pose_in_A_to_pose_in_B(transform_pose, openxr_pose)
        pos, rot_mat = PoseUtils.unmake_pose(result_pose)
        pos = pos + self._wrist_position_offset
        quat = PoseUtils.quat_from_matrix(rot_mat)
        return np.concatenate([pos.numpy(), quat.numpy()]).astype(np.float32)


@dataclass
class G1TriHandUpperBodyZeroMqRetargeterCfg(RetargeterCfg):
    """Configuration for the G1 ZeroMQ upper-body retargeter."""

    enable_visualization: bool = False
    hand_joint_names: list[str] | None = None
    use_hand_tracking_if_available: bool = False
    enable_wrist_pose_retargeting: bool = True
    wrist_position_offset: tuple[float, float, float] = (-0.16, 0.0, 0.0)
    retargeter_type: type[RetargeterBase] = G1TriHandUpperBodyZeroMqRetargeter