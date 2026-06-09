# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from pink.tasks import FrameTask

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.managers.action_manager import ActionTerm

XRCore = None
try:
    from omni.kit.xr.core import XRCore as _XRCore

    XRCore = _XRCore
except Exception:
    XRCore = None

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import pink_actions_cfg


class PinkInverseKinematicsAction(ActionTerm):
    r"""Pink Inverse Kinematics action term.

    This action term processes the action tensor and sets these setpoints in the pink IK framework.
    The action tensor is ordered in the order of the tasks defined in PinkIKControllerCfg.
    """

    cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg
    """Configuration for the Pink Inverse Kinematics action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg, env: ManagerBasedEnv):
        """Initialize the Pink Inverse Kinematics action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._env = env
        self._sim_dt = env.sim.get_physics_dt()

        # Initialize joint information
        self._initialize_joint_info()

        # Initialize IK controllers
        self._initialize_ik_controllers()

        # Initialize action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_hand_joint_positions = torch.zeros(self.num_envs, self.hand_joint_dim, device=self.device)

        # PhysX Articulation Floating joint indices offset from IsaacLab Articulation joint indices
        self._physx_floating_joint_indices_offset = 6

        # Pre-allocate tensors for runtime use
        self._initialize_helper_tensors()
        self._initialize_waist_yaw_assist()

    def _initialize_joint_info(self) -> None:
        """Initialize joint IDs and names based on configuration."""
        # Resolve pink controlled joints
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        # Resolve hand joints
        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(self.cfg.hand_joint_names)

        # Combine all joint information
        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

    def _initialize_ik_controllers(self) -> None:
        """Initialize Pink IK controllers for all environments."""
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."

        self._ik_controllers = []
        for _ in range(self._env.num_envs):
            self._ik_controllers.append(
                PinkIKController(
                    cfg=self.cfg.controller.copy(),
                    robot_cfg=self._env.scene.cfg.robot,
                    device=self.device,
                    controlled_joint_indices=self._isaaclab_controlled_joint_ids,
                )
            )

    def _initialize_helper_tensors(self) -> None:
        """Pre-allocate tensors and cache values for performance optimization."""
        self._controlled_joint_ids_tensor = torch.tensor(self._controlled_joint_ids, device=self.device)

        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        self._base_link_idx = articulation_data.body_names.index(self.cfg.controller.base_link_name)

        num_frame_tasks = sum(
            1 for task in self._ik_controllers[0].cfg.variable_input_tasks if isinstance(task, FrameTask)
        )
        self._num_frame_tasks = num_frame_tasks
        self._controlled_frame_poses = torch.zeros(num_frame_tasks, self.num_envs, 4, 4, device=self.device)

        self._base_link_frame_buffer = torch.zeros(self.num_envs, 4, 4, device=self.device)
        self._frame_positions_in_base = torch.zeros(self._num_frame_tasks, self.num_envs, 3, device=self.device)

    def _initialize_waist_yaw_assist(self) -> None:
        """Initialize state for the separate waist-yaw helper."""
        self._waist_yaw_assist_enabled = bool(self.cfg.enable_waist_yaw_assist)
        self._waist_yaw_joint_id: int | None = None
        self._waist_yaw_default_position = None
        self._waist_yaw_target = None
        self._waist_yaw_filtered_lateral = None
        self._waist_yaw_is_active = None
        self._waist_yaw_active_task_slot = None
        self._waist_yaw_source = "hand"
        self._waist_yaw_head_gain = 1.0
        self._waist_yaw_reference_head_yaw = None
        self._waist_yaw_head_initialized = None
        self._waist_yaw_refresh_posture_target = None

        if not self._waist_yaw_assist_enabled:
            return

        joint_ids, joint_names = self._asset.find_joints([f"^{self.cfg.waist_yaw_joint_name}$"])
        if len(joint_ids) != 1:
            raise ValueError(
                "Expected exactly one waist-yaw joint match for "
                f"'{self.cfg.waist_yaw_joint_name}', got {len(joint_ids)}: {joint_names}"
            )

        self._waist_yaw_joint_id = int(joint_ids[0])
        self._waist_yaw_default_position = self._asset.data.default_joint_pos[:, self._waist_yaw_joint_id].clone()
        self._waist_yaw_target = self._waist_yaw_default_position.clone()
        self._waist_yaw_source = str(self.cfg.waist_yaw_source).lower()
        self._waist_yaw_task_indices = tuple(int(index) for index in self.cfg.waist_yaw_task_indices)
        self._waist_yaw_lateral_axis = int(self.cfg.waist_yaw_lateral_axis)
        self._waist_yaw_direction = float(self.cfg.waist_yaw_direction)
        self._waist_yaw_head_gain = float(self.cfg.waist_yaw_head_gain)
        self._waist_yaw_deadzone = float(self.cfg.waist_yaw_deadzone)
        self._waist_yaw_scale = float(self.cfg.waist_yaw_scale)
        self._waist_yaw_max_angle = float(self.cfg.waist_yaw_max_angle)
        self._waist_yaw_signal_smoothing = float(self.cfg.waist_yaw_signal_smoothing)
        self._waist_yaw_turn_smoothing = float(self.cfg.waist_yaw_turn_smoothing)
        self._waist_yaw_return_smoothing = float(self.cfg.waist_yaw_return_smoothing)
        self._waist_yaw_release_deadzone = float(self.cfg.waist_yaw_release_deadzone)
        self._waist_yaw_max_step = float(self.cfg.waist_yaw_max_step)
        primary_task_index = self.cfg.waist_yaw_primary_task_index
        self._waist_yaw_primary_task_index = None if primary_task_index is None else int(primary_task_index)
        self._waist_yaw_filtered_lateral = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._waist_yaw_is_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._waist_yaw_active_task_slot = torch.full(
            (self.num_envs,),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        self._waist_yaw_reference_head_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._waist_yaw_head_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._waist_yaw_refresh_posture_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    @property
    def hand_joint_dim(self) -> int:
        """Dimension for hand joint positions."""
        return self.cfg.controller.num_hand_joints

    @property
    def position_dim(self) -> int:
        """Dimension for position (x, y, z)."""
        return 3

    @property
    def orientation_dim(self) -> int:
        """Dimension for orientation (w, x, y, z)."""
        return 4

    @property
    def pose_dim(self) -> int:
        """Total pose dimension (position + orientation)."""
        return self.position_dim + self.orientation_dim

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        frame_tasks_count = sum(
            1 for task in self._ik_controllers[0].cfg.variable_input_tasks if isinstance(task, FrameTask)
        )
        return frame_tasks_count * self.pose_dim + self.hand_joint_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term."""
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "PinkInverseKinematicsAction"
        self._IO_descriptor.pink_controller_joint_names = self._isaaclab_controlled_joint_names
        self._IO_descriptor.hand_joint_names = self._hand_joint_names
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the input actions and set targets for each task."""
        self._raw_actions[:] = actions
        self._target_hand_joint_positions = actions[:, -self.hand_joint_dim :]

        self.base_link_frame_in_world_rf = self._get_base_link_frame_transform()

        controlled_frame_poses = self._extract_controlled_frame_poses(actions)
        transformed_poses = self._transform_poses_to_base_link_frame(controlled_frame_poses)

        self._set_task_targets(transformed_poses)
        self._update_waist_yaw_target(*transformed_poses)

    def _get_base_link_frame_transform(self) -> torch.Tensor:
        """Get the base link frame transformation matrix."""
        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        base_link_frame_in_world_origin = articulation_data.body_link_state_w[:, self._base_link_idx, :7]

        torch.sub(
            base_link_frame_in_world_origin[:, :3],
            self._env.scene.env_origins,
            out=self._base_link_frame_buffer[:, :3, 3],
        )

        base_link_frame_quat = base_link_frame_in_world_origin[:, 3:7]
        return math_utils.make_pose(
            self._base_link_frame_buffer[:, :3, 3], math_utils.matrix_from_quat(base_link_frame_quat)
        )

    def _extract_controlled_frame_poses(self, actions: torch.Tensor) -> torch.Tensor:
        """Extract controlled frame poses from action tensor."""
        for task_index in range(self._num_frame_tasks):
            pos_start = task_index * self.pose_dim
            pos_end = pos_start + self.position_dim
            quat_start = pos_end
            quat_end = (task_index + 1) * self.pose_dim

            position = actions[:, pos_start:pos_end]
            quaternion = actions[:, quat_start:quat_end]

            self._controlled_frame_poses[task_index] = math_utils.make_pose(
                position, math_utils.matrix_from_quat(quaternion)
            )

        return self._controlled_frame_poses

    def _transform_poses_to_base_link_frame(self, poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform poses from world frame to base link frame."""
        base_link_inv = math_utils.pose_inv(self.base_link_frame_in_world_rf)
        transformed_poses = math_utils.pose_in_A_to_pose_in_B(poses, base_link_inv)

        positions, rotation_matrices = math_utils.unmake_pose(transformed_poses)
        self._frame_positions_in_base.copy_(positions)

        return positions, rotation_matrices

    def _update_waist_yaw_target(self, positions: torch.Tensor, rotation_matrices: torch.Tensor) -> None:
        """Update the separate waist-yaw helper from transformed hand target poses."""
        if not self._waist_yaw_assist_enabled or self._waist_yaw_target is None or positions.shape[0] == 0:
            return

        if self._waist_yaw_source == "head":
            self._update_waist_yaw_target_from_head()
            return

        task_indices = [index for index in self._waist_yaw_task_indices if 0 <= index < rotation_matrices.shape[0]]
        if not task_indices:
            return

        task_yaws = []
        for index in task_indices:
            rot = rotation_matrices[index]
            yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
            task_yaws.append(yaw)

        task_yaws = torch.stack(task_yaws, dim=0)

        mean_sin = torch.sin(task_yaws).mean(dim=0)
        mean_cos = torch.cos(task_yaws).mean(dim=0)
        yaw_signal = torch.atan2(mean_sin, mean_cos)

        self._waist_yaw_filtered_lateral = torch.lerp(
            self._waist_yaw_filtered_lateral,
            yaw_signal,
            self._waist_yaw_signal_smoothing,
        )

        yaw_signal = self._waist_yaw_filtered_lateral
        yaw_abs = yaw_signal.abs()

        activate_mask = yaw_abs > self._waist_yaw_deadzone
        keep_active_mask = yaw_abs > self._waist_yaw_release_deadzone
        active_mask = torch.where(self._waist_yaw_is_active, keep_active_mask, activate_mask)

        newly_activated_mask = active_mask & ~self._waist_yaw_is_active
        if torch.any(newly_activated_mask):
            self._waist_yaw_refresh_posture_target[newly_activated_mask] = True

        self._waist_yaw_is_active = active_mask

        desired_offset = torch.zeros_like(yaw_signal)
        if torch.any(active_mask):
            effective_yaw = yaw_signal[active_mask] - torch.sign(yaw_signal[active_mask]) * self._waist_yaw_deadzone
            desired_offset[active_mask] = effective_yaw * self._waist_yaw_scale * self._waist_yaw_direction

        desired_offset = torch.clamp(desired_offset, -self._waist_yaw_max_angle, self._waist_yaw_max_angle)
        desired_waist = self._waist_yaw_default_position + desired_offset

        blend = torch.full_like(desired_waist, self._waist_yaw_return_smoothing)
        blend[active_mask] = self._waist_yaw_turn_smoothing
        smoothed_target = torch.lerp(self._waist_yaw_target, desired_waist, blend)
        delta = torch.clamp(
            smoothed_target - self._waist_yaw_target,
            -self._waist_yaw_max_step,
            self._waist_yaw_max_step,
        )
        self._waist_yaw_target = self._waist_yaw_target + delta

    def _update_waist_yaw_target_from_head(self) -> None:
        """Drive the waist helper directly from the current headset yaw."""
        head_yaw = self._get_openxr_head_yaw()
        if head_yaw is None:
            return

        head_yaw_tensor = torch.full((self.num_envs,), head_yaw, device=self.device, dtype=torch.float32)
        uninitialized_mask = ~self._waist_yaw_head_initialized
        if torch.any(uninitialized_mask):
            self._waist_yaw_reference_head_yaw[uninitialized_mask] = head_yaw_tensor[uninitialized_mask]
            self._waist_yaw_head_initialized[uninitialized_mask] = True

        yaw_delta = torch.atan2(
            torch.sin(head_yaw_tensor - self._waist_yaw_reference_head_yaw),
            torch.cos(head_yaw_tensor - self._waist_yaw_reference_head_yaw),
        )
        self._waist_yaw_filtered_lateral = torch.lerp(
            self._waist_yaw_filtered_lateral,
            yaw_delta,
            self._waist_yaw_signal_smoothing,
        )

        yaw_delta = self._waist_yaw_filtered_lateral
        yaw_abs = yaw_delta.abs()
        activate_mask = yaw_abs > self._waist_yaw_deadzone
        keep_active_mask = yaw_abs > self._waist_yaw_release_deadzone
        active_mask = torch.where(self._waist_yaw_is_active, keep_active_mask, activate_mask)
        self._waist_yaw_is_active = active_mask

        desired_offset = torch.zeros_like(yaw_delta)
        if torch.any(active_mask):
            effective_delta = yaw_delta[active_mask] - torch.sign(yaw_delta[active_mask]) * self._waist_yaw_deadzone
            desired_offset[active_mask] = effective_delta * self._waist_yaw_head_gain * self._waist_yaw_direction

        desired_offset = torch.clamp(desired_offset, -self._waist_yaw_max_angle, self._waist_yaw_max_angle)
        desired_waist = self._waist_yaw_default_position + desired_offset
        blend = torch.full_like(desired_waist, self._waist_yaw_return_smoothing)
        blend[active_mask] = self._waist_yaw_turn_smoothing
        smoothed_target = torch.lerp(self._waist_yaw_target, desired_waist, blend)
        delta = torch.clamp(
            smoothed_target - self._waist_yaw_target,
            -self._waist_yaw_max_step,
            self._waist_yaw_max_step,
        )
        self._waist_yaw_target = self._waist_yaw_target + delta

    def _get_openxr_head_yaw(self) -> float | None:
        """Query the current OpenXR headset yaw in world coordinates."""
        if XRCore is None:
            return None

        xr_core = XRCore.get_singleton()
        if xr_core is None:
            return None

        head_device = xr_core.get_input_device("/user/head")
        if head_device is None:
            return None

        try:
            hmd = head_device.get_virtual_world_pose("")
            quat = hmd.ExtractRotationQuat()
            imag = quat.GetImaginary()
            quat_wxyz = torch.tensor(
                [quat.GetReal(), imag[0], imag[1], imag[2]],
                device=self.device,
                dtype=torch.float32,
            )
        except Exception:
            return None

        if not torch.isfinite(quat_wxyz).all():
            return None

        quat_wxyz = quat_wxyz / torch.clamp(torch.linalg.norm(quat_wxyz), min=1e-6)
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(torch.atan2(siny_cosp, cosy_cosp).item())

    def _set_task_targets(self, transformed_poses: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set targets for all tasks across all environments."""
        positions, rotation_matrices = transformed_poses

        for env_index, ik_controller in enumerate(self._ik_controllers):
            for frame_task_index, task in enumerate(ik_controller.cfg.variable_input_tasks):
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                elif isinstance(task, FrameTask):
                    target = task.transform_target_to_world
                else:
                    continue

                target.translation = positions[frame_task_index, env_index, :].cpu().numpy()
                target.rotation = rotation_matrices[frame_task_index, env_index, :].cpu().numpy()
                task.set_target(target)

    def apply_actions(self) -> None:
        """Apply the computed joint positions based on the inverse kinematics solution."""
        ik_joint_positions = self._compute_ik_solutions()

        all_joint_positions = torch.cat((ik_joint_positions, self._target_hand_joint_positions), dim=1)
        self._processed_actions = all_joint_positions

        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()

        self._asset.set_joint_position_target(self._processed_actions, self._controlled_joint_ids)
        if self._waist_yaw_assist_enabled and self._waist_yaw_joint_id is not None and self._waist_yaw_target is not None:
            self._asset.set_joint_position_target(
                self._waist_yaw_target.unsqueeze(-1),
                [self._waist_yaw_joint_id],
            )

    def _apply_gravity_compensation(self) -> None:
        """Apply gravity compensation to arm joints if not disabled in props."""
        if not self._asset.cfg.spawn.rigid_props.disable_gravity:
            if self._asset.is_fixed_base:
                gravity = torch.zeros_like(
                    self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._controlled_joint_ids_tensor]
                )
            else:
                gravity = self._asset.root_physx_view.get_gravity_compensation_forces()[
                    :, self._controlled_joint_ids_tensor + self._physx_floating_joint_indices_offset
                ]

            self._asset.set_joint_effort_target(gravity, self._controlled_joint_ids)

    def _compute_ik_solutions(self) -> torch.Tensor:
        """Compute IK solutions for all environments."""
        ik_solutions = []

        for env_index, ik_controller in enumerate(self._ik_controllers):
            current_joint_pos = self._asset.data.joint_pos.cpu().numpy()[env_index]

            if (
                self._waist_yaw_assist_enabled
                and self._waist_yaw_refresh_posture_target is not None
                and self._waist_yaw_refresh_posture_target[env_index]
            ):
                #ik_controller.update_null_space_joint_targets(current_joint_pos)
                current_controlled_joint_pos = current_joint_pos[self._isaaclab_controlled_joint_ids]
                current_joint_pos_pink = current_controlled_joint_pos[ik_controller.isaac_lab_to_pink_controlled_ordering]
                ik_controller.update_null_space_joint_targets(current_joint_pos_pink)
                self._waist_yaw_refresh_posture_target[env_index] = False

            joint_pos_des = ik_controller.compute(current_joint_pos, self._sim_dt)
            ik_solutions.append(joint_pos_des)

        return torch.stack(ik_solutions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments."""
        if env_ids is None:
            self._raw_actions.zero_()
            self._target_hand_joint_positions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._target_hand_joint_positions[env_ids] = 0.0

        if self._waist_yaw_assist_enabled and self._waist_yaw_target is not None and self._waist_yaw_default_position is not None:
            if env_ids is None:
                self._waist_yaw_target.copy_(self._waist_yaw_default_position)
                self._waist_yaw_filtered_lateral.zero_()
                self._waist_yaw_is_active.zero_()
                self._waist_yaw_active_task_slot.fill_(-1)
                self._waist_yaw_reference_head_yaw.zero_()
                self._waist_yaw_head_initialized.zero_()
                self._waist_yaw_refresh_posture_target.zero_()
            else:
                self._waist_yaw_target[env_ids] = self._waist_yaw_default_position[env_ids]
                self._waist_yaw_filtered_lateral[env_ids] = 0.0
                self._waist_yaw_is_active[env_ids] = False
                self._waist_yaw_active_task_slot[env_ids] = -1
                self._waist_yaw_reference_head_yaw[env_ids] = 0.0
                self._waist_yaw_head_initialized[env_ids] = False
                self._waist_yaw_refresh_posture_target[env_ids] = False