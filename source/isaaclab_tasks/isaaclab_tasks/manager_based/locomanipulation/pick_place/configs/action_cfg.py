# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..mdp.actions import AgileBasedLowerBodyAction, GrootWholeBodyJointTargetAction


@configclass
class AgileBasedLowerBodyActionCfg(ActionTermCfg):
    """Configuration for the lower body action term that is based on Agile lower body RL policy."""

    class_type: type[ActionTerm] = AgileBasedLowerBodyAction
    """The class type for the lower body action term."""

    joint_names: list[str] = MISSING
    """The names of the joints to control."""

    obs_group_name: str = MISSING
    """The name of the observation group to use."""

    policy_path: str = MISSING
    """The path to the policy model."""

    policy_output_offset: float = 0.0
    """Offsets the output of the policy."""

    policy_output_scale: float = 1.0
    """Scales the output of the policy."""


@configclass
class GrootWholeBodyJointTargetActionCfg(ActionTermCfg):
    """Configuration for GROOT whole-body G1 joint target tracking."""

    class_type: type[ActionTerm] = GrootWholeBodyJointTargetAction
    """The class type for the GROOT whole-body action term."""

    joint_names: list[str] = MISSING
    """The 29 G1 body joints to control, in IsaacLab action order."""

    preserve_order: bool = True
    """Whether to preserve :attr:`joint_names` order when resolving joint ids."""

    max_joint_delta_per_step: float = 0.05
    """Maximum commanded joint change per environment step, in radians."""

    clip_to_soft_limits: bool = True
    """Whether to clamp processed joint targets to the articulation soft joint limits."""
