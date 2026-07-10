# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

WHOLE_BODY_POLICY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


@configclass
class AgileTeacherPolicyObservationsCfg(ObsGroup):
    """Observation specification aligned with the Unitree whole-body walking policy.

    The ONNX policy expects 14 frames of 65-d observations:
    56 dims from repeated [vx, vy, yaw, height] commands plus 14 frames of 61-d robot state.
    """

    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        scale=1.0,
    )

    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHOLE_BODY_POLICY_JOINT_NAMES,
            ),
        },
    )

    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=WHOLE_BODY_POLICY_JOINT_NAMES,
            ),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 14
