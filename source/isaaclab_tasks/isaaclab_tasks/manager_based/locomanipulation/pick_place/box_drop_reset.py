# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset all task boxes when any box falls below the tabletop threshold."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

logger = logging.getLogger(__name__)


@configclass
class BoxDropResetActionCfg(ActionTermCfg):
    """Configuration for resetting all boxes after any box falls from the table."""

    class_type: type = None  # set in __post_init__

    enabled: bool = True
    """Only enable this term on the PC1 physics-authority instance."""

    box_names: tuple[str, ...] = ()
    """Scene names of the boxes that are monitored and reset together."""

    minimum_height: float = 0.0
    """Minimum box-center height relative to the environment origin."""

    def __post_init__(self):
        self.class_type = BoxDropResetAction


class BoxDropResetAction(ActionTerm):
    """Restore all boxes to their default root states after any box falls."""

    cfg: BoxDropResetActionCfg

    def __init__(self, cfg: BoxDropResetActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False

        self._enabled = bool(cfg.enabled)
        self._box_names = tuple(cfg.box_names)
        if self._enabled and not self._box_names:
            raise ValueError("box_names must contain at least one box when box-drop reset is enabled")

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

    def apply_actions(self):
        if not self._enabled:
            return

        dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for box_name in self._box_names:
            box: RigidObject = self._env.scene[box_name]
            relative_height = box.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
            dropped |= relative_height < float(self.cfg.minimum_height)

        dropped_env_ids = torch.nonzero(dropped, as_tuple=False).flatten()
        if dropped_env_ids.numel() == 0:
            return

        for box_name in self._box_names:
            box: RigidObject = self._env.scene[box_name]
            default_root_state = box.data.default_root_state[dropped_env_ids].clone()
            default_root_state[:, :3] += self._env.scene.env_origins[dropped_env_ids]
            box.write_root_state_to_sim(default_root_state, env_ids=dropped_env_ids)

        logger.info(
            "[Box Drop Reset] A box fell below %.3fm; restored %d boxes to their default root states",
            float(self.cfg.minimum_height),
            len(self._box_names),
        )
