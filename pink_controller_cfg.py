# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for pink controller.

This module provides configurations for humanoid robot pink IK controllers,
including both fixed base and mobile configurations for upper body manipulation.
"""

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg

##
# Pink IK Controller Configuration for G1
##

G1_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=14,
    show_ik_warnings=False,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=4.0,
            orientation_cost=0.8,
            lm_damping=40,
            gain=0.15,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=4.0,
            orientation_cost=0.8,
            lm_damping=40,
            gain=0.15,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=4,
            controlled_frames=[
                "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            ],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
            ],
            gain=0.05,
        ),
    ],
    fixed_input_tasks=[],
)


##
# Pink IK Action Configuration for G1
##

G1_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
    ],
    hand_joint_names=[
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
    ],
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    asset_name="robot",
    controller=G1_UPPER_BODY_IK_CONTROLLER_CFG,
)
