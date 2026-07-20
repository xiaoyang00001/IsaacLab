# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch

from .g1_upper_body_motion_ctrl_retargeter import (
    G1TriHandUpperBodyMotionControllerRetargeter,
    G1TriHandUpperBodyMotionControllerRetargeterCfg,
)


G1_TRIHAND_PINK_JOINT_NAMES = [
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
]
"""G1 TriHand joint order used by :class:`PinkInverseKinematicsAction`."""


class G1TriHandMotionControllerHandRetargeter(G1TriHandUpperBodyMotionControllerRetargeter):
    """Return only the Pink-ordered hand targets from the upper-body controller mapping."""

    def retarget(self, data: dict) -> torch.Tensor:
        upper_body_command = super().retarget(data)
        return upper_body_command[-len(G1_TRIHAND_PINK_JOINT_NAMES) :]


@dataclass
class G1TriHandMotionControllerHandRetargeterCfg(G1TriHandUpperBodyMotionControllerRetargeterCfg):
    """Configuration for Pink-ordered G1 TriHand motion-controller retargeting."""

    hand_joint_names: list[str] | None = None
    retargeter_type: type = G1TriHandMotionControllerHandRetargeter

    def __post_init__(self):
        if self.hand_joint_names is None:
            self.hand_joint_names = list(G1_TRIHAND_PINK_JOINT_NAMES)
        if self.hand_joint_names != G1_TRIHAND_PINK_JOINT_NAMES:
            raise ValueError(
                "G1TriHandMotionControllerHandRetargeterCfg.hand_joint_names must use the Pink 14-joint order."
            )
