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
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..configs.action_cfg import AgileBasedLowerBodyActionCfg, AutoWalkActionCfg, SONICWholeBodyActionCfg


# SONIC 训练时使用的 G1 29-DoF 关节顺序（来自 g1_29dof_rev_1_0.xml MJCF 树遍历）。
# decoder 输出的 29D action 与此顺序一一对应；切勿改变。
SONIC_G1_29DOF_JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# SONIC encoder 用的 14 个 body link（来自 sonic_release/config.yaml body_names）。
# command_multi_future_nonflat 返回这 14 个 body 在 pelvis 坐标系下的位置（10 帧）。
SONIC_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


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
    """全身骨骼捕捉数据驱动的物理行走（腿+腰+手臂+手），含自然摆臂。

    数据流（概念上）::

        time → SkeletonPoseSimulator.sample(phase) → 各关节目标角度 → robot

    内部不接收外部输入，由 `_sample_skeleton_pose` 产生与 walking 阶段同步的
    全身关节角度。这模拟了一个本地 mocap 流：法线交互/重定向部分内嵌实现。

    机器人通过物理引擎自然行走，脚与地面产生真实接触力。
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
            f"freq={cfg.walk_frequency:.2f}Hz "
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
                # 膝关节在腿前摆中段（mid-swing）弯曲最大
                targets[:, i] = self._default_joint_pos[:, i] + A_knee * torch.clamp(torch.sin(ph), min=0.0)
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

        # ── 1. 更新相位 ──────────────────────────────────────
        self._phase += 2.0 * math.pi * self.cfg.walk_frequency * dt

        # ── 2. 从"骨骼数据"生成全身关节目标 ───────────────────
        self._processed_actions = self._sample_skeleton_pose(self._phase)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._phase.zero_()
            self._processed_actions.copy_(self._default_joint_pos)
        else:
            self._phase[env_ids] = 0.0
            self._processed_actions[env_ids] = self._default_joint_pos[env_ids]


class SONICWholeBodyAction(ActionTerm):
    """GEAR-SONIC encoder-decoder 全身控制 Action Term（阶段 3.1：真实 decoder 观测 + encoder zero-fill）。

    - decoder 端 994D 输入按 [policy/release/observation_config.yaml] 偏移精确构造：
      token_state(64) + his_base_ang_vel(30) + his_joint_pos(290) + his_joint_vel(290)
      + his_last_actions(290) + his_gravity_dir(30) = 994
    - encoder 端 1762D 仍 zero-fill；motion reference / mode 切换留待阶段 3.2/3.3。
    """

    HISTORY_LEN = 10  # decoder 端 _10frame_step1 历史长度

    cfg: SONICWholeBodyActionCfg
    _asset: Articulation

    def __init__(self, cfg: SONICWholeBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._env = env

        self._joint_ids, self._joint_names = self._resolve_joints(cfg.joint_names)
        if len(self._joint_ids) != cfg.sonic_action_dim:
            print(
                f"[IsaacLab] [SONIC] WARNING resolved {len(self._joint_ids)}/{cfg.sonic_action_dim} joints; "
                "SONIC was trained on 29 DoF — outputs for missing joints will be discarded."
            )

        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_joint_pos.clone()
        self._last_action = torch.zeros(self.num_envs, cfg.sonic_action_dim, device=self.device)

        self._init_history()
        self._init_sonic_body_indices()
        self._load_policies()
        self._debug_counter = 0

        if self.num_envs > 1:
            print(
                f"[IsaacLab] [SONIC] WARNING num_envs={self.num_envs}; ONNX runs in a per-env loop "
                "(no batch dim in encoder/decoder); expect ~6ms × num_envs per step."
            )
        print(
            f"[IsaacLab] [SONIC] asset={cfg.asset_name} resolved={len(self._joint_ids)}/{cfg.sonic_action_dim} joints "
            f"action_scale={cfg.action_scale:.2f} enc_in={self._encoder_input_dim}D dec_in={self._decoder_input_dim}D "
            f"history_len={self.HISTORY_LEN}"
        )

    def _init_history(self):
        N, J = self.num_envs, self.cfg.sonic_action_dim
        H = self.HISTORY_LEN
        dev = self.device
        # joint_pos 用 default 初始化，gravity_dir 用 (0, 0, -1)，其余清零
        self._hist_base_ang_vel = torch.zeros(N, H, 3, device=dev)
        self._hist_joint_pos = self._default_joint_pos.unsqueeze(1).expand(N, H, J).clone()
        self._hist_joint_vel = torch.zeros(N, H, J, device=dev)
        self._hist_last_actions = torch.zeros(N, H, J, device=dev)
        self._hist_gravity_dir = torch.zeros(N, H, 3, device=dev)
        self._hist_gravity_dir[:, :, 2] = -1.0

    def _init_sonic_body_indices(self):
        """找 SONIC 训练用 14 个 body link 在 USD articulation 中的索引。"""
        all_body_names = list(self._asset.data.body_names)
        print(f"[IsaacLab] [SONIC INIT] USD has {len(all_body_names)} bodies: {all_body_names}")

        self._sonic_body_ids: list[int] = []
        missing = []
        for name in SONIC_BODY_NAMES:
            ids, _ = self._asset.find_bodies([f"^{name}$"])
            if len(ids) == 1:
                self._sonic_body_ids.append(int(ids[0]))
            else:
                missing.append(name)
                self._sonic_body_ids.append(0)  # fallback to root link

        resolved = 14 - len(missing)
        print(f"[IsaacLab] [SONIC INIT] body indices resolved: {resolved}/14, ids={self._sonic_body_ids}")
        if missing:
            print(f"[IsaacLab] [SONIC INIT] MISSING SONIC bodies (fall back to root): {missing}")

    def _resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[str]]:
        ids, names = [], []
        for name in joint_names:
            jids, jnames = self._asset.find_joints([f"^{name}$"])
            if len(jids) == 1:
                ids.append(int(jids[0]))
                names.append(jnames[0])
            else:
                print(f"[IsaacLab] [SONIC] skip joint '{name}' (matches={len(jids)})")
        return ids, names

    def _load_policies(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "SONIC requires `onnxruntime` in the IsaacLab env. "
                "Install via `pip install onnxruntime-gpu`."
            ) from exc

        enc_path = retrieve_file_path(self.cfg.encoder_path)
        dec_path = retrieve_file_path(self.cfg.decoder_path)
        self._encoder = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
        self._decoder = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

        enc_in = self._encoder.get_inputs()[0]
        dec_in = self._decoder.get_inputs()[0]
        self._enc_input_name = enc_in.name
        self._dec_input_name = dec_in.name
        self._enc_output_name = self._encoder.get_outputs()[0].name
        self._dec_output_name = self._decoder.get_outputs()[0].name
        self._encoder_input_dim = int(enc_in.shape[-1])
        self._decoder_input_dim = int(dec_in.shape[-1])
        self._token_dim = int(self._encoder.get_outputs()[0].shape[-1])

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _push_history(self):
        """FIFO 推入当前观测，最新帧在 [-1] 位置。所有按 SONIC 关节顺序取。"""
        ang_vel = self._asset.data.root_ang_vel_b  # (N, 3) body frame IMU
        jp = self._asset.data.joint_pos[:, self._joint_ids]  # (N, 29)
        jv = self._asset.data.joint_vel[:, self._joint_ids]  # (N, 29)
        gravity = self._asset.data.projected_gravity_b  # (N, 3) 重力投影到 body frame

        self._hist_base_ang_vel = torch.roll(self._hist_base_ang_vel, shifts=-1, dims=1)
        self._hist_joint_pos = torch.roll(self._hist_joint_pos, shifts=-1, dims=1)
        self._hist_joint_vel = torch.roll(self._hist_joint_vel, shifts=-1, dims=1)
        self._hist_last_actions = torch.roll(self._hist_last_actions, shifts=-1, dims=1)
        self._hist_gravity_dir = torch.roll(self._hist_gravity_dir, shifts=-1, dims=1)

        self._hist_base_ang_vel[:, -1, :] = ang_vel
        self._hist_joint_pos[:, -1, :] = jp
        self._hist_joint_vel[:, -1, :] = jv
        self._hist_last_actions[:, -1, :] = self._last_action
        self._hist_gravity_dir[:, -1, :] = gravity

    def _build_decoder_input(self, tokens: np.ndarray, env_idx: int) -> np.ndarray:
        """按 observation_config.yaml 顺序拼 994D decoder 输入。

        flatten layout: **dim-major** —— 每维的 10 帧时间序列连续存放。
        即 (10, K) 张量先 transpose 到 (K, 10) 再 flatten；frame-major（初版试过）输出 garbage。

        offsets::
          [0:64]    token_state (64D，来自 encoder 当帧输出)
          [64:94]   his_base_angular_velocity_10frame_step1 (3*10, dim-major)
          [94:384]  his_body_joint_positions_10frame_step1 (29*10, dim-major)
          [384:674] his_body_joint_velocities_10frame_step1 (29*10, dim-major)
          [674:964] his_last_actions_10frame_step1 (29*10, dim-major)
          [964:994] his_gravity_dir_10frame_step1 (3*10, dim-major)
        """
        dec = np.zeros((1, self._decoder_input_dim), dtype=np.float32)
        dec[:, :64] = tokens
        # (10, K).T → (K, 10) → flatten → [d0_f0..f9, d1_f0..f9, ...]
        dec[0, 64:94] = self._hist_base_ang_vel[env_idx].t().flatten().cpu().numpy()
        dec[0, 94:384] = self._hist_joint_pos[env_idx].t().flatten().cpu().numpy()
        dec[0, 384:674] = self._hist_joint_vel[env_idx].t().flatten().cpu().numpy()
        dec[0, 674:964] = self._hist_last_actions[env_idx].t().flatten().cpu().numpy()
        dec[0, 964:994] = self._hist_gravity_dir[env_idx].t().flatten().cpu().numpy()
        return dec

    def _compute_self_ref_body_pos_b(self) -> torch.Tensor:
        """计算 14 个 SONIC body 在 pelvis (root) 坐标系下的位置。

        self-reference 时 reference == 当前 robot → 这些就是当前姿态下 body 相对 pelvis 的位置。
        Returns: (N, 14, 3)
        """
        body_pos_w = self._asset.data.body_link_pos_w[:, self._sonic_body_ids, :]  # (N, 14, 3)
        root_pos_w = self._asset.data.root_link_pos_w  # (N, 3)
        root_quat_w = self._asset.data.root_link_quat_w  # (N, 4)
        rel_w = body_pos_w - root_pos_w.unsqueeze(1)  # (N, 14, 3)
        quat_expanded = root_quat_w.unsqueeze(1).expand(-1, 14, -1)  # (N, 14, 4)
        return quat_apply_inverse(quat_expanded, rel_w)  # (N, 14, 3)

    def _build_encoder_input(self, env_idx: int) -> np.ndarray:
        """Encoder 1762D 输入（阶段 3.2 D1：self-reference for g1 mode）。

        layout（按 sonic_release/config.yaml tokenizer 段顺序）::
            [0]         encoder_index = 0 (mode_id=0 = g1)
            [1:421]     command_multi_future_nonflat (420D = 10 × 14 × 3) — body_pos_b × 10 帧
            [421:431]   command_z_multi_future_nonflat (10D) — g1 不用，zero
            [431:491]   motion_anchor_ori_b_mf_nonflat (60D = 10 × 6) — identity 6D × 10 帧
            [491:1762]  其他字段（teleop / smpl）g1 mode 下全 zero
        """
        enc = np.zeros((1, self._encoder_input_dim), dtype=np.float32)

        # offset 0: encoder_index = 0 → g1 mode (cast to long in encoder forward)
        enc[0, 0] = 0.0

        # offset 1:421 = command_multi_future_nonflat (10 frames × 14 bodies × 3)
        # self-reference: motion target = 当前 body pose，10 帧重复
        body_pos_b = self._self_ref_body_pos_b[env_idx]  # (14, 3)
        body_flat = body_pos_b.flatten().cpu().numpy()  # (42,)
        enc[0, 1:421] = np.tile(body_flat, 10)  # frame-major: f0_b0_xyz..f0_b13_xyz, f1_...

        # offset 431:491 = motion_anchor_ori_b_mf_nonflat (10 × 6D rotation diff)
        # self-reference: ori diff = identity → 6D = rotation matrix 前两列 flatten = [1,0,0,0,1,0]
        identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        enc[0, 431:491] = np.tile(identity_6d, 10)

        return enc

    def _run_sonic(self) -> torch.Tensor:
        """Encoder self-reference g1 mode (阶段 3.2 D1), decoder 用真实 10-frame history。"""
        n_act = self.cfg.sonic_action_dim
        # 缓存当前帧 body_pos_b（per-env loop 内复用）
        self._self_ref_body_pos_b = self._compute_self_ref_body_pos_b()  # (N, 14, 3)
        out = np.zeros((self.num_envs, n_act), dtype=np.float32)
        for i in range(self.num_envs):
            enc_in = self._build_encoder_input(env_idx=i)
            tokens = self._encoder.run([self._enc_output_name], {self._enc_input_name: enc_in})[0]
            dec_in = self._build_decoder_input(tokens, env_idx=i)
            action = self._decoder.run([self._dec_output_name], {self._dec_input_name: dec_in})[0][0]
            out[i] = action
        return torch.from_numpy(out).to(device=self.device, dtype=torch.float32)

    def process_actions(self, actions: torch.Tensor):
        self._push_history()
        action_rel = self._run_sonic()
        n_resolved = len(self._joint_ids)
        # SONIC 训练用 JointPositionActionCfg(use_default_offset=true)，输出 = 相对 default 的偏移
        self._processed_actions = self._default_joint_pos + self.cfg.action_scale * action_rel[:, :n_resolved]
        self._last_action = action_rel

        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            a = action_rel[0].detach().cpu()
            jp = self._hist_joint_pos[0, -1].detach().cpu()
            bp = self._self_ref_body_pos_b[0].detach().cpu()  # (14, 3) self-ref motion
            print(
                f"[IsaacLab] [SONIC] step={self._debug_counter} "
                f"action mean={a.mean():+.4f} absmax={a.abs().max():.4f} std={a.std():.4f} "
                f"| joint_pos absmax={jp.abs().max():.4f} "
                f"| self_ref_body_pos absmax={bp.abs().max():.4f} mean={bp.mean():+.4f}"
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._processed_actions.copy_(self._default_joint_pos)
            self._last_action.zero_()
            # history 重置：joint_pos 回到 default，其余清零，gravity 回 -z
            self._hist_base_ang_vel.zero_()
            self._hist_joint_pos[:] = self._default_joint_pos.unsqueeze(1)
            self._hist_joint_vel.zero_()
            self._hist_last_actions.zero_()
            self._hist_gravity_dir.zero_()
            self._hist_gravity_dir[:, :, 2] = -1.0
        else:
            self._processed_actions[env_ids] = self._default_joint_pos[env_ids]
            self._last_action[env_ids] = 0.0
            self._hist_base_ang_vel[env_ids] = 0.0
            self._hist_joint_pos[env_ids] = self._default_joint_pos[env_ids].unsqueeze(1)
            self._hist_joint_vel[env_ids] = 0.0
            self._hist_last_actions[env_ids] = 0.0
            self._hist_gravity_dir[env_ids] = 0.0
            self._hist_gravity_dir[env_ids, :, 2] = -1.0
