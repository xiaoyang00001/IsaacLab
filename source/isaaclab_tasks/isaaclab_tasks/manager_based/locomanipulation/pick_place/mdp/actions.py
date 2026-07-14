# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
        AgileBasedLowerBodyActionCfg,
        GrootWholeBodyJointTargetActionCfg,
    )


logger = logging.getLogger(__name__)


class AgileBasedLowerBodyAction(ActionTerm):
    """Action term that is based on Agile lower body RL policy."""

    cfg: AgileBasedLowerBodyActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: AgileBasedLowerBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Save the observation config from cfg
        self._observation_cfg = env.cfg.observations
        self._obs_group_name = cfg.obs_group_name

        # Load policy here if needed
        _temp_policy_path = retrieve_file_path(cfg.policy_path)
        self._policy = load_torchscript_model(_temp_policy_path, device=env.device)
        self._env = env

        # Find joint ids for the lower body joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        # Get the scale and offset from the configuration
        self._policy_output_scale = torch.tensor(cfg.policy_output_scale, device=env.device)
        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # Create tensors to store raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Lower Body Action: [vx, vy, wz, hip_height]"""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _compose_policy_input(self, base_command: torch.Tensor, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Compose the policy input by concatenating repeated commands with observations.

        Args:
            base_command: The base command tensor [vx, vy, wz, hip_height].
            obs_tensor: The observation tensor from the environment.

        Returns:
            The composed policy input tensor with repeated commands concatenated to observations.
        """
        # Get history length from observation configuration
        history_length = getattr(self._observation_cfg, self._obs_group_name).history_length
        # Default to 1 if history_length is None (no history, just current observation)
        if history_length is None:
            history_length = 1

        # Repeat commands based on history length and concatenate with observations
        repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(base_command.shape[0], -1)
        policy_input = torch.cat([repeated_commands, obs_tensor], dim=-1)

        return policy_input

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions using the locomotion policy.

        Args:
            actions: The lower body commands.
        """

        # Extract base command from the action tensor
        # Assuming the base command [vx, vy, wz, hip_height]
        base_command = actions

        obs_tensor = self._env.obs_buf["lower_body_policy"]

        # Compose policy input using helper function
        policy_input = self._compose_policy_input(base_command, obs_tensor)

        joint_actions = self._policy.forward(policy_input)

        self._raw_actions[:] = joint_actions

        # Apply scaling and offset to the raw actions from the policy
        self._processed_actions = joint_actions * self._policy_output_scale + self._policy_output_offset

        # Clip actions if configured
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        """Apply the actions to the environment."""
        # Store the raw actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


class GrootWholeBodyJointTargetAction(ActionTerm):
    """Track GROOT 29-DOF whole-body G1 joint targets through IsaacLab position targets."""

    cfg: GrootWholeBodyJointTargetActionCfg
    """The configuration of the action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: GrootWholeBodyJointTargetActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        if len(self._joint_ids) != 29:
            raise ValueError(
                "GrootWholeBodyJointTargetAction expects exactly 29 G1 body joints, "
                f"but resolved {len(self._joint_ids)} joints: {self._joint_names}"
            )

        self._max_joint_delta_per_step = float(self.cfg.max_joint_delta_per_step)
        self._raw_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._raw_actions.clone()

        logger.info(
            "Resolved GROOT whole-body action joints in IsaacLab order: %s [%s]",
            self._joint_names,
            self._joint_ids,
        )

    @property
    def action_dim(self) -> int:
        """GROOT whole-body target action dimension."""
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw absolute joint targets received from the teleoperation device."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint position targets after per-step delta limiting and joint-limit clipping."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Convert absolute GROOT joint targets into bounded simulation joint targets."""
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} GROOT joint targets, received {actions.shape[1]}.")

        self._raw_actions[:] = actions.to(self.device)

        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        target_joint_pos = torch.where(torch.isfinite(self._raw_actions), self._raw_actions, current_joint_pos)
        joint_delta = target_joint_pos - current_joint_pos

        if self._max_joint_delta_per_step > 0.0:
            joint_delta = torch.clamp(
                joint_delta,
                min=-self._max_joint_delta_per_step,
                max=self._max_joint_delta_per_step,
            )

        processed_actions = current_joint_pos + joint_delta

        if self.cfg.clip_to_soft_limits:
            joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            processed_actions = torch.clamp(processed_actions, joint_limits[..., 0], joint_limits[..., 1])

        self._processed_actions[:] = processed_actions

    def apply_actions(self):
        """Apply processed joint position targets to the G1 articulation."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action buffers to the articulation default joint positions."""
        if env_ids is None:
            env_ids = slice(None)
        default_joint_pos = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]
        self._raw_actions[env_ids] = default_joint_pos
        self._processed_actions[env_ids] = default_joint_pos
