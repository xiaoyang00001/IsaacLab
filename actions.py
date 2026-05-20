# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..configs.action_cfg import AgileBasedLowerBodyActionCfg


class AgileBasedLowerBodyAction(ActionTerm):
    """Drive A's legs from the locomotion policy instead of moving the root directly."""

    cfg: AgileBasedLowerBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: AgileBasedLowerBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._observation_cfg = env.cfg.observations
        self._obs_group_name = cfg.obs_group_name
        self._env = env
        self._joint_ids, self._joint_names = self._resolve_joint_order(self.cfg.joint_names)
        self._policy_output_scale = torch.tensor(cfg.policy_output_scale, device=env.device, dtype=torch.float32)
        self._policy_output_offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._action_smoothing = float(cfg.action_smoothing)
        self._command_scale = float(cfg.command_scale)
        self._stand_command_deadzone = float(cfg.stand_command_deadzone)
        self._enable_policy_when_moving = bool(cfg.enable_policy_when_moving)
        self._root_motion_deadzone = float(cfg.root_motion_deadzone)
        self._root_motion_scale = float(cfg.root_motion_scale)
        self._root_motion_smoothing = float(cfg.root_motion_smoothing)
        self._stabilize_root_pose = bool(cfg.stabilize_root_pose)
        self._kinematic_dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))
        self._default_hip_height = torch.tensor([cfg.hip_height], device=env.device, dtype=torch.float32)
        self._policy_path = retrieve_file_path(cfg.policy_path)
        self._policy_kind = Path(self._policy_path).suffix.lower()
        self._policy = None
        self._onnx_input_name: str | None = None
        self._onnx_output_name: str | None = None
        self._expected_input_dim: int | None = None
        self._shape_warning_printed = False
        self._debug_counter = 0
        self._runtime_state_logged = False
        self._load_policy()
        self._raw_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._processed_actions = self._policy_output_offset.clone()
        self._stable_root_pos = torch.tensor(cfg.root_anchor_pos, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._stable_root_quat = torch.tensor(cfg.root_anchor_rot, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._stable_root_yaw = self._yaw_from_quat(self._stable_root_quat)
        self._root_target_xy = self._stable_root_pos[:, :2].clone()
        self._root_target_yaw = self._stable_root_yaw.clone()
        self._last_root_target_xy = self._stable_root_pos[:, :2].clone()
        self._last_world_planar_command = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        print(
            "[IsaacLab] [LowerBodyONNX] "
            f"asset={cfg.asset_name} joints={list(self._joint_names)} "
            f"scale={float(cfg.policy_output_scale):.3f} smoothing={self._action_smoothing:.2f} "
            f"cmd_scale={self._command_scale:.2f}"
        )

    def _resolve_joint_order(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        """Resolve joint ids one-by-one to preserve the configured order exactly."""
        resolved_ids: list[int] = []
        resolved_names: list[str] = []
        for joint_name in joint_names:
            joint_ids, matched_names = self._asset.find_joints([f"^{joint_name}$"])
            if len(joint_ids) != 1:
                raise ValueError(
                    f"Expected exactly one joint match for '{joint_name}' on asset '{self.cfg.asset_name}', "
                    f"but got {len(joint_ids)} matches: {matched_names}"
                )
            resolved_ids.append(int(joint_ids[0]))
            resolved_names.append(matched_names[0])
        return resolved_ids, resolved_names

    @property
    def action_dim(self) -> int:
        # Head retargeter emits [vx, vy, wz, hip_height].
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _compose_policy_input(self, base_command: torch.Tensor, obs_tensor: torch.Tensor) -> torch.Tensor:
        history_length = getattr(self._observation_cfg, self._obs_group_name).history_length
        if history_length is None:
            history_length = 1
        repeated_commands = base_command.unsqueeze(1).repeat(1, history_length, 1).reshape(base_command.shape[0], -1)
        return torch.cat([repeated_commands, obs_tensor], dim=-1)

    def _world_to_body_planar_command(self, base_command: torch.Tensor) -> torch.Tensor:
        """Rotate planar velocity commands from world frame into the robot body frame."""
        command_in_body = base_command.clone()
        root_quat = self._asset.data.root_quat_w
        qw = root_quat[:, 0]
        qx = root_quat[:, 1]
        qy = root_quat[:, 2]
        qz = root_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_world = base_command[:, 0]
        vy_world = base_command[:, 1]
        command_in_body[:, 0] = cos_yaw * vx_world + sin_yaw * vy_world
        command_in_body[:, 1] = -sin_yaw * vx_world + cos_yaw * vy_world
        return command_in_body

    @staticmethod
    def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        qw = quat_wxyz[:, 0]
        qx = quat_wxyz[:, 1]
        qy = quat_wxyz[:, 2]
        qz = quat_wxyz[:, 3]
        return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    @staticmethod
    def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
        half_yaw = 0.5 * yaw
        quat[:, 0] = torch.cos(half_yaw)
        quat[:, 3] = torch.sin(half_yaw)
        return quat

    def _body_to_world_planar_command(self, planar_command: torch.Tensor) -> torch.Tensor:
        """Rotate local planar commands into the scene frame using the current root yaw."""
        command_in_world = planar_command.clone()
        root_quat = self._asset.data.root_quat_w
        qw = root_quat[:, 0]
        qx = root_quat[:, 1]
        qy = root_quat[:, 2]
        qz = root_quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx_body = planar_command[:, 0]
        vy_body = planar_command[:, 1]
        command_in_world[:, 0] = cos_yaw * vx_body - sin_yaw * vy_body
        command_in_world[:, 1] = sin_yaw * vx_body + cos_yaw * vy_body
        return command_in_world

    def _load_policy(self):
        if self._policy_kind == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "ONNX walking policy requested, but `onnxruntime` is not installed in the IsaacLab environment."
                ) from exc

            self._policy = ort.InferenceSession(self._policy_path, providers=["CPUExecutionProvider"])
            input_meta = self._policy.get_inputs()[0]
            self._onnx_input_name = input_meta.name
            if isinstance(input_meta.shape[-1], int):
                self._expected_input_dim = input_meta.shape[-1]
            self._onnx_output_name = self._policy.get_outputs()[0].name
            return

        self._policy = load_torchscript_model(self._policy_path, device=self.device)

    def _run_policy(self, policy_input: torch.Tensor) -> torch.Tensor:
        if self._policy_kind == ".onnx":
            assert self._onnx_input_name is not None
            assert self._onnx_output_name is not None
            output = self._policy.run(
                [self._onnx_output_name],
                {self._onnx_input_name: policy_input.detach().cpu().numpy().astype(np.float32)},
            )[0]
            return torch.from_numpy(output).to(device=self.device, dtype=torch.float32)

        return self._policy.forward(policy_input)

    def _apply_kinematic_root_motion(self, base_command: torch.Tensor) -> torch.Tensor:
        planar_command = base_command[:, :2].clone()
        planar_command = torch.where(
            torch.abs(planar_command) > self._root_motion_deadzone,
            planar_command,
            torch.zeros_like(planar_command),
        )
        moving = torch.any(torch.abs(planar_command) > 0.0, dim=-1)
        yaw_command = base_command[:, 2].clone()
        yaw_command = torch.where(
            torch.abs(yaw_command) > self._root_motion_deadzone,
            yaw_command,
            torch.zeros_like(yaw_command),
        )
        yaw_moving = torch.abs(yaw_command) > 0.0
        if not self._stabilize_root_pose and not torch.any(moving | yaw_moving):
            return moving

        world_planar_command = self._body_to_world_planar_command(planar_command)
        self._last_world_planar_command.copy_(world_planar_command)

        root_pose = torch.cat([self._asset.data.root_pos_w, self._asset.data.root_quat_w], dim=-1).clone()
        # Treat the incoming command as a velocity-like body command and integrate a persistent
        # root target. Using a fixed anchor makes turning and backing up collapse toward the
        # reset pose instead of continuing from the robot's current position.
        self._root_target_xy += world_planar_command * (self._root_motion_scale * self._kinematic_dt)
        target_xy = self._root_target_xy
        self._last_root_target_xy.copy_(target_xy)
        root_pose[:, :2] = torch.lerp(root_pose[:, :2], target_xy, self._root_motion_smoothing)
        self._root_target_yaw += yaw_command * self._kinematic_dt
        target_yaw = self._root_target_yaw
        if self._stabilize_root_pose:
            root_pose[:, 2] = self._stable_root_pos[:, 2]
            root_pose[:, 3:7] = self._quat_from_yaw(target_yaw)
            root_velocity = torch.zeros(root_pose.shape[0], 6, device=root_pose.device, dtype=root_pose.dtype)
            self._asset.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1))
        else:
            self._asset.write_root_pose_to_sim(root_pose)
        return moving | yaw_moving

    def process_actions(self, actions: torch.Tensor):
        if not self._runtime_state_logged:
            remote_asset = self._env.scene["remote_robot"]
            robot_pos = self._asset.data.root_pos_w[0].detach().cpu().tolist()
            remote_pos = remote_asset.data.root_pos_w[0].detach().cpu().tolist()
            robot_prim = getattr(self._asset.cfg, "prim_path", "<unknown>")
            remote_prim = getattr(remote_asset.cfg, "prim_path", "<unknown>")
            print(
                "[IsaacLab] [RuntimeAssetState] "
                f"controlled_asset={self.cfg.asset_name} prim={robot_prim} "
                f"root_pos={tuple(round(float(v), 4) for v in robot_pos)}; "
                f"remote_asset=remote_robot prim={remote_prim} "
                f"root_pos={tuple(round(float(v), 4) for v in remote_pos)}"
            )
            self._runtime_state_logged = True
        if actions.shape[-1] >= 4:
            base_command = actions[:, :4]
        else:
            base_command = torch.cat(
                [actions[:, :3], self._default_hip_height.repeat(actions.shape[0], 1)],
                dim=-1,
            )

        obs_tensor = self._env.obs_buf[self._obs_group_name]
        # The head retargeter already emits commands in its calibrated local frame.
        # Feeding root-yaw-rotated commands back into the policy makes the target drift as soon as the robot leans.
        command_for_policy = base_command.clone()
        command_for_policy[:, :3] *= self._command_scale
        root_moving = self._apply_kinematic_root_motion(base_command)
        active_command = torch.linalg.norm(base_command[:, :3], dim=-1) > self._stand_command_deadzone
        if not self._enable_policy_when_moving or not torch.any(active_command):
            self._raw_actions.zero_()
            self._processed_actions = torch.lerp(
                self._processed_actions,
                self._policy_output_offset,
                self._action_smoothing,
            )
            self._debug_counter += 1
            if self._debug_counter % 20 == 0:
                cmd = base_command[0].detach().cpu().numpy()
                world = self._last_world_planar_command[0].detach().cpu().numpy()
                root_xy = self._asset.data.root_pos_w[0, :2].detach().cpu().numpy()
                target_xy = self._last_root_target_xy[0].detach().cpu().numpy()
                proc = self._processed_actions[0].detach().cpu()
                print(
                    "[IsaacLab] [LowerBodyONNX] "
                    f"stand_hold cmd=[{cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f}, {cmd[3]:+.3f}] "
                    f"world=[{world[0]:+.3f}, {world[1]:+.3f}] "
                    f"root_motion={'on' if bool(root_moving[0].item()) else 'off'} "
                    f"root_xy=[{root_xy[0]:+.3f}, {root_xy[1]:+.3f}] "
                    f"target_xy=[{target_xy[0]:+.3f}, {target_xy[1]:+.3f}] "
                    f"proc_mean={proc.mean():+.4f} proc_absmax={proc.abs().max():+.4f}"
                )
            return

        policy_input = self._compose_policy_input(command_for_policy, obs_tensor)
        if self._expected_input_dim is not None and policy_input.shape[-1] != self._expected_input_dim:
            current_dim = policy_input.shape[-1]
            if current_dim < self._expected_input_dim:
                padding = torch.zeros(
                    policy_input.shape[0],
                    self._expected_input_dim - current_dim,
                    device=policy_input.device,
                    dtype=policy_input.dtype,
                )
                policy_input = torch.cat([policy_input, padding], dim=-1)
            else:
                policy_input = policy_input[:, : self._expected_input_dim]

            if not self._shape_warning_printed:
                print(
                    f"[IsaacLab] [LowerBodyONNX] Adjusted policy input dim from {current_dim} to {self._expected_input_dim}."
                )
                self._shape_warning_printed = True

        joint_actions = self._run_policy(policy_input)
        self._raw_actions[:] = joint_actions
        target_actions = joint_actions * self._policy_output_scale + self._policy_output_offset
        self._processed_actions = torch.lerp(
            self._processed_actions,
            target_actions,
            self._action_smoothing,
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            cmd = base_command[0].detach().cpu().numpy()
            cmd_scaled = command_for_policy[0].detach().cpu().numpy()
            raw = joint_actions[0].detach().cpu()
            proc = self._processed_actions[0].detach().cpu()
            print(
                "[IsaacLab] [LowerBodyONNX] "
                f"cmd=[{cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f}, {cmd[3]:+.3f}] "
                f"cmd_scaled=[{cmd_scaled[0]:+.3f}, {cmd_scaled[1]:+.3f}, {cmd_scaled[2]:+.3f}, {cmd_scaled[3]:+.3f}] "
                f"raw_mean={raw.mean():+.4f} raw_absmax={raw.abs().max():+.4f} "
                f"proc_mean={proc.mean():+.4f} proc_absmax={proc.abs().max():+.4f}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._processed_actions.copy_(self._policy_output_offset)
            self._raw_actions.zero_()
            self._root_target_xy.copy_(self._asset.data.root_pos_w[:, :2])
            self._root_target_yaw.copy_(self._yaw_from_quat(self._asset.data.root_quat_w))
            self._last_root_target_xy.copy_(self._root_target_xy)
            return

        self._processed_actions[env_ids] = self._policy_output_offset[env_ids]
        self._raw_actions[env_ids].zero_()
        self._root_target_xy[env_ids] = self._asset.data.root_pos_w[env_ids, :2]
        self._root_target_yaw[env_ids] = self._yaw_from_quat(self._asset.data.root_quat_w[env_ids])
        self._last_root_target_xy[env_ids] = self._root_target_xy[env_ids]
