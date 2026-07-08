# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from ..configs.action_cfg import MuJoCoG1MirrorActionCfg


class GraspAttachController:
    """把可抓物体锁定到手掌坐标系的抓取附着控制器。

    镜像遥操作路线下机器人根节点与关节状态每步被直写，PhysX 无法通过接触
    力形成稳定夹持（手指会穿透物体、物体被去穿透修正挤出）。此控制器沿用
    与身体镜像相同的"状态直写"思路：夹爪闭合且手腕贴近可抓物体时记录物体
    相对手掌的位姿，此后每步将物体根位姿写为 手掌位姿 × 相对位姿；夹爪松开
    时释放物体并清零速度，恢复自由动力学。仅支持 num_envs=1。
    """

    _HAND_LABELS = ("left", "right")

    def __init__(self, env: ManagerBasedEnv, robot: Articulation, cfg: MuJoCoG1MirrorActionCfg) -> None:
        self._env = env
        self._robot = robot
        self._cfg = cfg
        self._device = robot.device

        palm_names = list(cfg.grasp_attach_palm_body_names)
        if len(palm_names) != 2:
            raise ValueError(f"grasp_attach_palm_body_names must list [left, right] bodies, got {palm_names}")
        self._palm_ids: list[int] = []
        for name in palm_names:
            ids, _ = robot.find_bodies(name)
            if len(ids) != 1:
                raise ValueError(f"Palm body {name!r} matched {len(ids)} bodies; expected exactly 1.")
            self._palm_ids.append(ids[0])

        self._asset_names = [name for name in cfg.grasp_attach_asset_names if self._resolve_asset(name) is not None]
        missing = sorted(set(cfg.grasp_attach_asset_names) - set(self._asset_names))
        if missing:
            print(f"[WARN] Grasp attach: scene has no rigid objects named {missing}; they are ignored.")

        self._attached_asset: list[str | None] = [None, None]
        self._rel_pos: list[torch.Tensor | None] = [None, None]
        self._rel_quat: list[torch.Tensor | None] = [None, None]
        self._zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=self._device)

    def update(self, processed_actions: torch.Tensor) -> None:
        """Attach/follow/release per hand from the smoothed gripper close commands.

        Args:
            processed_actions: shape (1, 4) tensor ``[L_index, L_middle, R_index, R_middle]`` in [0, 1].
        """
        close = torch.clamp(processed_actions[0], 0.0, 1.0)
        hand_close = (
            float(torch.max(close[0], close[1]).item()),
            float(torch.max(close[2], close[3]).item()),
        )
        for hand in range(2):
            if self._attached_asset[hand] is None:
                if hand_close[hand] >= self._cfg.grasp_attach_close_threshold:
                    self._try_attach(hand)
            elif hand_close[hand] <= self._cfg.grasp_attach_release_threshold:
                self._release(hand)
            else:
                self._follow(hand)

    def _resolve_asset(self, name: str) -> RigidObject | None:
        try:
            asset = self._env.scene[name]
        except (KeyError, ValueError):
            return None
        return asset if isinstance(asset, RigidObject) else None

    def _palm_pose(self, hand: int) -> tuple[torch.Tensor, torch.Tensor]:
        body_id = self._palm_ids[hand]
        return self._robot.data.body_pos_w[0, body_id], self._robot.data.body_quat_w[0, body_id]

    def _try_attach(self, hand: int) -> None:
        palm_pos, palm_quat = self._palm_pose(hand)
        best_name: str | None = None
        best_dist = float(self._cfg.grasp_attach_distance)
        for name in self._asset_names:
            if name in self._attached_asset:
                continue  # 已被另一只手附着
            asset = self._resolve_asset(name)
            if asset is None:
                continue
            dist = float(torch.linalg.norm(asset.data.root_pos_w[0] - palm_pos).item())
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_name is None:
            return
        asset = self._resolve_asset(best_name)
        inv_palm_quat = quat_inv(palm_quat.unsqueeze(0))
        self._rel_pos[hand] = quat_apply(inv_palm_quat, (asset.data.root_pos_w[0] - palm_pos).unsqueeze(0))[0]
        self._rel_quat[hand] = quat_mul(inv_palm_quat, asset.data.root_quat_w[0:1])[0]
        self._attached_asset[hand] = best_name
        if self._cfg.grasp_attach_debug:
            print(f"[INFO] Grasp attach: {self._HAND_LABELS[hand]} hand grabbed {best_name!r} at {best_dist:.3f} m.")

    def _follow(self, hand: int) -> None:
        name = self._attached_asset[hand]
        asset = self._resolve_asset(name) if name is not None else None
        if asset is None:
            self._attached_asset[hand] = None
            return
        palm_pos, palm_quat = self._palm_pose(hand)
        target_pos = palm_pos + quat_apply(palm_quat.unsqueeze(0), self._rel_pos[hand].unsqueeze(0))[0]
        target_quat = quat_mul(palm_quat.unsqueeze(0), self._rel_quat[hand].unsqueeze(0))[0]
        asset.write_root_pose_to_sim(torch.cat((target_pos, target_quat)).unsqueeze(0))
        asset.write_root_velocity_to_sim(self._zero_velocity)

    def _release(self, hand: int) -> None:
        name = self._attached_asset[hand]
        self._attached_asset[hand] = None
        self._rel_pos[hand] = None
        self._rel_quat[hand] = None
        if name is None:
            return
        asset = self._resolve_asset(name)
        if asset is not None:
            asset.write_root_velocity_to_sim(self._zero_velocity)
        if self._cfg.grasp_attach_debug:
            print(f"[INFO] Grasp attach: {self._HAND_LABELS[hand]} hand released {name!r}.")
