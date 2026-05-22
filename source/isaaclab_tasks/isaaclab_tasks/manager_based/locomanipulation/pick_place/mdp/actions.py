# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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

    from ..configs.action_cfg import AgileBasedLowerBodyActionCfg, AutoWalkActionCfg


class AgileBasedLowerBodyAction(ActionTerm):
    """Action term that drives robot A lower-body walking from a locomotion policy."""

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
        self._root_motion_scale = float(cfg.root_motion_scale)
        self._root_motion_smoothing = float(cfg.root_motion_smoothing)
        self._stabilize_root_pose = bool(cfg.stabilize_root_pose)
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
        self._stable_root_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._stable_root_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._last_root_target_xy = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._last_root_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._root_reference_sync_threshold = 0.35
        self._turn_in_place_xy_deadzone = 0.05
        self._turn_in_place_yaw_deadzone = 0.10
        print(
            "[IsaacLab] [LowerBodyONNX] "
            f"asset={cfg.asset_name} joints={list(self._joint_names)} "
            f"scale={float(cfg.policy_output_scale):.3f} smoothing={self._action_smoothing:.2f} "
            f"cmd_scale={self._command_scale:.2f}"
        )

    def _resolve_joint_order(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
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
        """Lower Body Action: [vx, vy, wz, hip_height]."""
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

    def _apply_absolute_root_pose(self, base_command: torch.Tensor) -> None:
        if not self._stabilize_root_pose:
            return

        current_root_xy = self._asset.data.root_pos_w[:, :2]
        current_root_yaw = self._yaw_from_quat(self._asset.data.root_quat_w)
        command_xy_norm = torch.linalg.norm(base_command[:, :2], dim=-1)
        command_yaw_abs = torch.abs(base_command[:, 2])

        # When the headset retargeter recenters, it emits a near-zero command at
        # the robot's current pose. Sync the stable root reference in that case
        # so subsequent absolute targets use the new pose as their baseline.
        root_reference_error = torch.linalg.norm(current_root_xy - self._stable_root_pos[:, :2], dim=-1)
        should_sync_reference = (
            (command_xy_norm <= self._turn_in_place_xy_deadzone)
            & (command_yaw_abs <= self._turn_in_place_yaw_deadzone)
            & (root_reference_error >= self._root_reference_sync_threshold)
        )
        if torch.any(should_sync_reference):
            self._stable_root_pos[should_sync_reference] = self._asset.data.root_pos_w[should_sync_reference]
            self._stable_root_yaw[should_sync_reference] = current_root_yaw[should_sync_reference]
            self._last_root_target_xy[should_sync_reference] = current_root_xy[should_sync_reference]
            self._last_root_target_yaw[should_sync_reference] = current_root_yaw[should_sync_reference]

        target_xy = self._stable_root_pos[:, :2] + base_command[:, :2] * self._root_motion_scale
        target_yaw = self._stable_root_yaw + base_command[:, 2]

        # Turn-in-place commands should keep the current/root-target position
        # instead of snapping back to the stable-root origin when xy is near zero.
        is_turn_in_place = (
            (command_xy_norm <= self._turn_in_place_xy_deadzone)
            & (command_yaw_abs > self._turn_in_place_yaw_deadzone)
        )
        if torch.any(is_turn_in_place):
            target_xy[is_turn_in_place] = current_root_xy[is_turn_in_place]
            target_yaw[is_turn_in_place] = current_root_yaw[is_turn_in_place] + base_command[is_turn_in_place, 2]

        self._last_root_target_xy.copy_(target_xy)
        self._last_root_target_yaw.copy_(target_yaw)

        root_pose = torch.cat([self._asset.data.root_pos_w, self._asset.data.root_quat_w], dim=-1).clone()
        alpha = float(min(max(self._root_motion_smoothing, 0.0), 1.0))
        root_pose[:, :2] = torch.lerp(root_pose[:, :2], target_xy, alpha)
        root_pose[:, 2] = self._stable_root_pos[:, 2]
        target_quat = self._quat_from_yaw(target_yaw)
        if alpha >= 0.999:
            root_pose[:, 3:7] = target_quat
        else:
            root_pose[:, 3:7] = torch.lerp(root_pose[:, 3:7], target_quat, alpha)
            quat_norm = torch.linalg.norm(root_pose[:, 3:7], dim=-1, keepdim=True).clamp_min(1e-6)
            root_pose[:, 3:7] = root_pose[:, 3:7] / quat_norm
        root_velocity = torch.zeros(root_pose.shape[0], 6, device=root_pose.device, dtype=root_pose.dtype)
        self._asset.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1))

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

    def process_actions(self, actions: torch.Tensor):
        if not self._runtime_state_logged:
            robot_pos = self._asset.data.root_pos_w[0].detach().cpu().tolist()
            robot_prim = getattr(self._asset.cfg, "prim_path", "<unknown>")
            print(
                "[IsaacLab] [RuntimeAssetState] "
                f"controlled_asset={self.cfg.asset_name} prim={robot_prim} "
                f"root_pos={tuple(round(float(v), 4) for v in robot_pos)}"
            )
            self._runtime_state_logged = True

        if actions.shape[-1] >= 4:
            base_command = actions[:, :4].clone()
        else:
            base_command = torch.cat(
                [actions[:, :3], self._default_hip_height.repeat(actions.shape[0], 1)],
                dim=-1,
            )

        self._apply_absolute_root_pose(base_command)

        # Fallback teleop mode: keep the legs in a stable standing pose and
        # drive only the articulated root from the headset-derived command.
        self._raw_actions.zero_()
        self._processed_actions = torch.lerp(
            self._processed_actions,
            self._policy_output_offset,
            self._action_smoothing,
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            cmd = base_command[0].detach().cpu().numpy()
            proc = self._processed_actions[0].detach().cpu()
            root_xy = self._asset.data.root_pos_w[0, :2].detach().cpu().numpy()
            target_xy = self._last_root_target_xy[0].detach().cpu().numpy()
            print(
                "[IsaacLab] [LowerBodyONNX] "
                f"stand_root cmd=[{cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f}, {cmd[3]:+.3f}] "
                f"root_xy=[{root_xy[0]:+.3f}, {root_xy[1]:+.3f}] "
                f"target_xy=[{target_xy[0]:+.3f}, {target_xy[1]:+.3f}] "
                f"proc_mean={proc.mean():+.4f} proc_absmax={proc.abs().max():+.4f}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._processed_actions.copy_(self._policy_output_offset)
            self._raw_actions.zero_()
            self._stable_root_pos.copy_(self._asset.data.root_pos_w)
            self._stable_root_yaw.copy_(self._yaw_from_quat(self._asset.data.root_quat_w))
            self._last_root_target_xy.copy_(self._stable_root_pos[:, :2])
            self._last_root_target_yaw.copy_(self._stable_root_yaw)
            return

        self._processed_actions[env_ids] = self._policy_output_offset[env_ids]
        self._raw_actions[env_ids].zero_()
        self._stable_root_pos[env_ids] = self._asset.data.root_pos_w[env_ids]
        self._stable_root_yaw[env_ids] = self._yaw_from_quat(self._asset.data.root_quat_w[env_ids])
        self._last_root_target_xy[env_ids] = self._stable_root_pos[env_ids, :2]
        self._last_root_target_yaw[env_ids] = self._stable_root_yaw[env_ids]


class AutoWalkAction(ActionTerm):
    """模拟骨骼捕捉数据驱动的全身行走（腿+腰+手臂+手），含自然摆臂。

    数据流（概念上）::

        time → SkeletonPoseSimulator.sample(phase) → 各关节目标角度 → robot

    内部不接收外部输入，由 `_sample_skeleton_pose` 产生与 walking 阶段同步的
    全身关节角度。这模拟了一个本地 mocap 流：法线交互/重定向部分内嵌实现。

    根节点通过 ``write_root_state_to_sim`` 沿机器人朝向匀速平移。
    """

    cfg: AutoWalkActionCfg
    _asset: Articulation

    def __init__(self, cfg: AutoWalkActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._env = env

        # 关节解析（缺失关节直接跳过，不抛错）
        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        self._idx = {n: i for i, n in enumerate(self._joint_names)}
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_joint_pos.clone()

        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._walk_xy: torch.Tensor | None = None
        self._init_root_z: torch.Tensor | None = None

        # 把上下身关节按区分组缓存，避免每步反复查字典
        self._leg_groups = self._collect_side_indices(
            patterns=("hip_pitch_joint", "knee_joint", "ankle_pitch_joint")
        )
        self._arm_groups = self._collect_side_indices(
            patterns=("shoulder_pitch_joint", "elbow_joint")
        )
        self._waist_yaw_idx = self._idx.get("waist_yaw_joint")
        self._waist_roll_idx = self._idx.get("waist_roll_joint")
        self._waist_pitch_idx = self._idx.get("waist_pitch_joint")
        # 髋 yaw（用于骨盆旋转）
        self._left_hip_yaw_idx = self._idx.get("left_hip_yaw_joint")
        self._right_hip_yaw_idx = self._idx.get("right_hip_yaw_joint")
        # 收集手部关节索引
        self._hand_indices = [i for n, i in self._idx.items() if "_hand_" in n]

        print(
            f"[IsaacLab] [AutoWalkAction] asset={cfg.asset_name} "
            f"speed={cfg.forward_speed:.2f}m/s freq={cfg.walk_frequency:.2f}Hz "
            f"resolved_joints={len(self._joint_ids)}/{len(cfg.joint_names)} "
            f"(legs={sum(len(v) for v in self._leg_groups.values())} "
            f"arms={sum(len(v) for v in self._arm_groups.values())} "
            f"hands={len(self._hand_indices)})"
        )

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        """逐个解析关节。缺失关节会被跳过并打印警告（保持代码对 G1 变体兼容）。"""
        ids, names = [], []
        for name in joint_names:
            jids, jnames = self._asset.find_joints([f"^{name}$"])
            if len(jids) == 1:
                ids.append(int(jids[0]))
                names.append(jnames[0])
            else:
                print(f"[IsaacLab] [AutoWalkAction] skip joint '{name}' (matches={len(jids)})")
        return ids, names

    def _collect_side_indices(self, patterns: tuple[str, ...]) -> dict[str, dict[str, int]]:
        """返回形如 {'left': {'hip_pitch_joint': idx, ...}, 'right': {...}} 的索引表。"""
        groups: dict[str, dict[str, int]] = {"left": {}, "right": {}}
        for side in ("left", "right"):
            for p in patterns:
                key = f"{side}_{p}"
                if key in self._idx:
                    groups[side][p] = self._idx[key]
        return groups

    @staticmethod
    def _forward_dir_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        """返回 XY 平面上机器人朝向的单位向量（模型正面为 +X 轴）。"""
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        fx = 1.0 - 2.0 * (y * y + z * z)
        fy = 2.0 * (x * y + w * z)
        return torch.stack([fx, fy], dim=-1)

    @property
    def action_dim(self) -> int:
        return 1  # 占位；外部不发送命令

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _sample_skeleton_pose(self, phase: torch.Tensor) -> torch.Tensor:
        """模拟骨骼捕捉数据流，输出全身关节目标角度。

        这一函数等价于 ``retarget(mocap_data_at_time(t), robot_skeleton)``，
        但 mocap_data 用解析公式合成而非从外部读取。
        """
        targets = self._default_joint_pos.clone()

        phase_l = phase           # 左腿相位
        phase_r = phase + math.pi  # 右腿相位（180° 偏移）

        # ── LEGS：行走步态 ────────────────────────────────────
        A_hip = self.cfg.hip_pitch_amplitude
        A_knee = self.cfg.knee_amplitude
        A_ankle = self.cfg.ankle_pitch_amplitude

        for side, ph in (("left", phase_l), ("right", phase_r)):
            leg = self._leg_groups[side]
            if "hip_pitch_joint" in leg:
                i = leg["hip_pitch_joint"]
                targets[:, i] = self._default_joint_pos[:, i] + A_hip * torch.sin(ph)
            if "knee_joint" in leg:
                i = leg["knee_joint"]
                targets[:, i] = self._default_joint_pos[:, i] + A_knee * torch.clamp(torch.sin(ph + 0.5), min=0.0)
            if "ankle_pitch_joint" in leg:
                i = leg["ankle_pitch_joint"]
                targets[:, i] = self._default_joint_pos[:, i] - A_ankle * torch.sin(ph)

        # ── ARMS：反向摆动（与同侧腿 180° 相位） ─────────────
        A_arm = self.cfg.arm_swing_amplitude
        A_elbow = self.cfg.elbow_bend_amplitude

        for side, ph_arm in (("left", phase_l), ("right", phase_r)):
            arm = self._arm_groups[side]
            if "shoulder_pitch_joint" in arm:
                i = arm["shoulder_pitch_joint"]
                # 手臂与同侧腿"前后位置"反相：腿后摆 → 同侧臂前摆
                targets[:, i] = self._default_joint_pos[:, i] + A_arm * torch.sin(ph_arm)
            if "elbow_joint" in arm:
                i = arm["elbow_joint"]
                # 前摆时肘部轻微弯曲
                targets[:, i] = self._default_joint_pos[:, i] + A_elbow * torch.clamp(torch.sin(ph_arm + 0.5), min=0.0)

        # ── WAIST：小幅反向扭转，增加自然感 ────────────────
        A_waist_yaw = self.cfg.waist_yaw_amplitude
        if self._waist_yaw_idx is not None:
            # 与腿运动反相（腿前摆，腰反扭）
            targets[:, self._waist_yaw_idx] = (
                self._default_joint_pos[:, self._waist_yaw_idx] - A_waist_yaw * torch.sin(phase_l)
            )

        # ── WAIST ROLL：行走时的身体侧倾（重心转移） ──────────
        A_waist_roll = self.cfg.waist_roll_amplitude
        if self._waist_roll_idx is not None:
            # 左腿支撑时身体向左倾，右腿支撑时向右倾
            targets[:, self._waist_roll_idx] = (
                self._default_joint_pos[:, self._waist_roll_idx] + A_waist_roll * torch.sin(phase_l)
            )

        # ── HIP YAW：骨盆旋转（与腰部 yaw 协同） ───────────────
        A_hip_yaw = self.cfg.hip_yaw_amplitude
        if self._left_hip_yaw_idx is not None:
            # 左髋与腰部同向旋转
            targets[:, self._left_hip_yaw_idx] = (
                self._default_joint_pos[:, self._left_hip_yaw_idx] - A_hip_yaw * torch.sin(phase_l)
            )
        if self._right_hip_yaw_idx is not None:
            # 右髋与腰部同向旋转
            targets[:, self._right_hip_yaw_idx] = (
                self._default_joint_pos[:, self._right_hip_yaw_idx] - A_hip_yaw * torch.sin(phase_l)
            )

        # ── HANDS：保持微弱放松卷曲（恒定，不随相位变化） ──
        if self._hand_indices and self.cfg.hand_curl_amount != 0.0:
            curl = self.cfg.hand_curl_amount
            for hi in self._hand_indices:
                targets[:, hi] = self._default_joint_pos[:, hi] + curl

        return targets

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt

        # 首次调用记录初始位置
        if self._walk_xy is None:
            self._walk_xy = self._asset.data.root_pos_w[:, :2].clone()
            self._init_root_z = self._asset.data.root_pos_w[:, 2:3].clone()

        # ── 1. 更新相位 ──────────────────────────────────────
        self._phase += 2.0 * math.pi * self.cfg.walk_frequency * dt

        # ── 2. 移动根节点（沿当前朝向） ───────────────────────
        fwd = self._forward_dir_from_quat(self._asset.data.root_quat_w)  # [N, 2]
        self._walk_xy += fwd * (self.cfg.forward_speed * dt)

        target_pos = torch.cat([self._walk_xy, self._init_root_z], dim=-1)
        root_state = torch.cat([
            target_pos,
            self._asset.data.root_quat_w,
            torch.zeros(self.num_envs, 6, device=self.device),
        ], dim=-1)
        self._asset.write_root_state_to_sim(root_state)

        # ── 3. 从"骨骼数据"生成全身关节目标 ───────────────────
        self._processed_actions = self._sample_skeleton_pose(self._phase)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._phase.zero_()
            self._processed_actions.copy_(self._default_joint_pos)
            self._walk_xy = None
            self._init_root_z = None
        else:
            self._phase[env_ids] = 0.0
            self._processed_actions[env_ids] = self._default_joint_pos[env_ids]
            if self._walk_xy is not None:
                self._walk_xy[env_ids] = self._asset.data.root_pos_w[env_ids, :2]
