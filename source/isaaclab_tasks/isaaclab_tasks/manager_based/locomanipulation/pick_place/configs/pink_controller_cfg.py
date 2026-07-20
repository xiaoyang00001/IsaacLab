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
            position_cost=8.0,  # [cost] / [m]
            orientation_cost=2.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.5,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,  # [cost] / [m]
            orientation_cost=2.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
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
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
)
"""Base configuration for the G1 pink IK controller.

This configuration sets up the pink IK controller for the G1 humanoid robot with
left and right wrist control tasks. The controller is designed for upper body
manipulation tasks.
"""


##
# Pink IK Controller Configuration for G1 -- wrist orientation only
##

G1_WRIST_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=0,
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=0.0,  # orientation-only: 3 wrist joints cannot reach an arbitrary position
            orientation_cost=2.0,  # [cost] / [rad]
            lm_damping=10,  # dampening for solver for step jumps
            gain=0.5,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=0.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
    ],
    fixed_input_tasks=[],
)
"""Wrist-only Pink IK controller for G1.

Unlike :data:`G1_UPPER_BODY_IK_CONTROLLER_CFG`, this only optimizes the three wrist joints
(roll/pitch/yaw) per arm. Shoulder/elbow are assumed to be driven by something else -- the MuJoCo
mirror, in the locomanipulation task -- and enter the solve only as the live forward-kinematics base
the wrist chain hangs off of (see ``PinkKinematicsConfiguration``'s controlled/full model split).
Position is left untracked (``position_cost=0``) because a 3-DoF wrist cannot reach an arbitrary
wrist position on its own; only the mirrored shoulder/elbow chain can do that.
"""


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
        "waist_.*_joint",
    ],
    hand_joint_names=[
        "left_hand_index_0_joint",  # Index finger proximal
        "left_hand_middle_0_joint",  # Middle finger proximal
        "left_hand_thumb_0_joint",  # Thumb base (yaw axis)
        "right_hand_index_0_joint",  # Index finger proximal
        "right_hand_middle_0_joint",  # Middle finger proximal
        "right_hand_thumb_0_joint",  # Thumb base (yaw axis)
        "left_hand_index_1_joint",  # Index finger distal
        "left_hand_middle_1_joint",  # Middle finger distal
        "left_hand_thumb_1_joint",  # Thumb middle (pitch axis)
        "right_hand_index_1_joint",  # Index finger distal
        "right_hand_middle_1_joint",  # Middle finger distal
        "right_hand_thumb_1_joint",  # Thumb middle (pitch axis)
        "left_hand_thumb_2_joint",  # Thumb tip
        "right_hand_thumb_2_joint",  # Thumb tip
    ],
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    # the robot in the sim scene we are controlling
    asset_name="robot",
    # Configuration for the IK controller
    # The frames names are the ones present in the URDF file
    # The urdf has to be generated from the USD that is being used in the scene
    controller=G1_UPPER_BODY_IK_CONTROLLER_CFG,
)
"""Base configuration for the G1 pink IK action.

This configuration sets up the pink IK action for the G1 humanoid robot,
defining which joints are controlled by the IK solver and which are fixed.
The configuration includes:
- Upper body joints controlled by IK (shoulders, elbows, wrists)
- Fixed joints (pelvis, legs, hands)
- Hand joint names for additional control
- Reference to the pink IK controller configuration
"""
