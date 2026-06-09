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
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=5.5,
            orientation_cost=6.0,
            lm_damping=16,
            gain=0.35,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=5.5,
            orientation_cost=6.0,
            lm_damping=16,
            gain=0.35,
        ),
        NullSpacePostureTask(
            cost=1.8,
            lm_damping=3,
            controlled_frames=[
                "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            ],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
              #  "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            gain=0.55,
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
       # "waist_.*_joint",
        "waist_pitch_joint",
        "waist_roll_joint",
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
    enable_waist_yaw_assist=True,
    waist_yaw_joint_name="waist_yaw_joint",
    waist_yaw_source="hand",
    waist_yaw_task_indices=(0, 1),
    waist_yaw_primary_task_index=None,
    waist_yaw_lateral_axis=1,
    waist_yaw_direction=1.0,
    waist_yaw_head_gain=1.0,
    waist_yaw_deadzone=0.03,
    waist_yaw_release_deadzone=0.015,
    waist_yaw_scale=1.0,
    waist_yaw_max_angle=1.57,
    waist_yaw_signal_smoothing=0.22,
    waist_yaw_turn_smoothing=0.35,
    waist_yaw_return_smoothing=0.32,
    waist_yaw_max_step=0.045,
)