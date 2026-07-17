# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detect successful packing into ``container_h20`` and reset only the boxes."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

logger = logging.getLogger(__name__)


@configclass
class BoxSuccessResetActionCfg(ActionTermCfg):
    """Configuration for authoritative three-box success detection."""

    class_type: type = None  # set in __post_init__

    enabled: bool = True
    """Only enable this term on the PC1 physics-authority instance."""

    box_names: tuple[str, ...] = ()
    """Scene names of all boxes that must be inside the container."""

    box_sizes: tuple[tuple[float, float, float], ...] = ()
    """Box dimensions used to calculate complete oriented world bounds."""

    container_root_path: str = "/World/envs/env_0/PackingTable"
    """USD subtree that contains the target container."""

    container_prim_name: str = "container_h20"
    """USD prim name of the target container."""

    clearance: float = 0.005
    """Required clearance from the container's world-aligned outer bounds."""

    hold_time_s: float = 0.25
    """Continuous settled time required before the boxes are reset."""

    max_linear_speed: float = 0.15
    """Maximum box linear speed considered settled, in metres per second."""

    max_angular_speed: float = 1.0
    """Maximum box angular speed considered settled, in radians per second."""

    def __post_init__(self):
        self.class_type = BoxSuccessResetAction


class BoxSuccessResetAction(ActionTerm):
    """Reset only the three boxes after they remain inside ``container_h20``."""

    cfg: BoxSuccessResetActionCfg

    def __init__(self, cfg: BoxSuccessResetActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False

        self._enabled = bool(cfg.enabled) and self.num_envs == 1
        self._box_names = tuple(cfg.box_names)
        self._box_sizes = tuple(cfg.box_sizes)
        if len(self._box_names) != len(self._box_sizes):
            raise ValueError("box_names and box_sizes must contain the same number of entries")

        self._container_min: torch.Tensor | None = None
        self._container_max: torch.Tensor | None = None
        self._inside_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        physics_dt = float(getattr(self._env, "physics_dt", self._env.cfg.sim.dt))
        self._required_frames = max(1, int(round(float(cfg.hold_time_s) / physics_dt)))
        self._next_container_lookup_time = 0.0
        self._container_lookup_warning_emitted = False

        if cfg.enabled and self.num_envs != 1:
            logger.warning("[Box Success] Disabled because the current implementation requires num_envs=1")

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        self._processed_actions = actions

    def reset(self, env_ids=None) -> None:
        self._raw_actions.zero_()
        self._processed_actions.zero_()
        if env_ids is None:
            self._inside_frames.zero_()
        else:
            self._inside_frames[env_ids] = 0

    def apply_actions(self):
        if not self._enabled:
            return
        if self._container_min is None or self._container_max is None:
            self._resolve_container_bounds()
            if self._container_min is None or self._container_max is None:
                return

        all_inside = self._all_boxes_inside_and_settled()
        self._inside_frames = torch.where(all_inside, self._inside_frames + 1, torch.zeros_like(self._inside_frames))
        success_env_ids = torch.nonzero(self._inside_frames >= self._required_frames, as_tuple=False).flatten()
        if success_env_ids.numel() == 0:
            return

        self._reset_boxes(success_env_ids)
        self._inside_frames[success_env_ids] = 0
        logger.info(
            "[Box Success] All boxes stayed inside %s for %.3fs; reset boxes only",
            self.cfg.container_prim_name,
            float(self.cfg.hold_time_s),
        )

    def _resolve_container_bounds(self) -> None:
        now = time.monotonic()
        if now < self._next_container_lookup_time:
            return
        self._next_container_lookup_time = now + 1.0

        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            root_prim = stage.GetPrimAtPath(self.cfg.container_root_path)
            if not root_prim.IsValid():
                raise RuntimeError(f"USD root prim not found: {self.cfg.container_root_path}")

            requested_name = str(self.cfg.container_prim_name).casefold()
            container_prim = None
            for prim in Usd.PrimRange(root_prim):
                prim_name = prim.GetName().casefold()
                if prim_name == requested_name:
                    container_prim = prim
                    break
                if container_prim is None and requested_name in prim_name:
                    container_prim = prim
            if container_prim is None:
                raise RuntimeError(
                    f"USD prim {self.cfg.container_prim_name!r} was not found under {self.cfg.container_root_path}"
                )

            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                useExtentsHint=True,
            )
            aligned_range = bbox_cache.ComputeWorldBound(container_prim).ComputeAlignedRange()
            minimum = tuple(float(value) for value in aligned_range.GetMin())
            maximum = tuple(float(value) for value in aligned_range.GetMax())
            if not all(high > low for low, high in zip(minimum, maximum, strict=True)):
                raise RuntimeError(f"invalid container bounds: min={minimum}, max={maximum}")

            self._container_min = torch.tensor(minimum, dtype=torch.float32, device=self.device)
            self._container_max = torch.tensor(maximum, dtype=torch.float32, device=self.device)
            logger.info(
                "[Box Success] Using USD container prim=%s bounds_min=%s bounds_max=%s hold_frames=%d",
                container_prim.GetPath(),
                minimum,
                maximum,
                self._required_frames,
            )
        except Exception as exc:
            if not self._container_lookup_warning_emitted:
                logger.warning("[Box Success] Waiting for container_h20 bounds: %s", exc)
                self._container_lookup_warning_emitted = True

    def _all_boxes_inside_and_settled(self) -> torch.Tensor:
        all_inside = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        clearance = float(self.cfg.clearance)

        for box_name, box_size in zip(self._box_names, self._box_sizes, strict=True):
            box: RigidObject = self._env.scene[box_name]
            rotation = math_utils.matrix_from_quat(box.data.root_quat_w)
            local_half_extents = 0.5 * torch.tensor(
                box_size,
                dtype=box.data.root_pos_w.dtype,
                device=self.device,
            )
            world_half_extents = torch.matmul(torch.abs(rotation), local_half_extents)
            box_min = box.data.root_pos_w - world_half_extents
            box_max = box.data.root_pos_w + world_half_extents

            inside = torch.all(box_min >= self._container_min + clearance, dim=1)
            inside &= torch.all(box_max <= self._container_max - clearance, dim=1)
            inside &= torch.linalg.vector_norm(box.data.root_lin_vel_w, dim=1) < float(
                self.cfg.max_linear_speed
            )
            inside &= torch.linalg.vector_norm(box.data.root_ang_vel_w, dim=1) < float(
                self.cfg.max_angular_speed
            )
            all_inside &= inside

        return all_inside

    def _reset_boxes(self, env_ids: torch.Tensor) -> None:
        for box_name in self._box_names:
            box: RigidObject = self._env.scene[box_name]
            default_root_state = box.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
            box.write_root_state_to_sim(default_root_state, env_ids=env_ids)
