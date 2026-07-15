# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal floating-base G1 environment for closed-loop GR00T WBC simulation."""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from .configs.action_cfg import GrootWholeBodyJointTargetActionCfg
from .configs.groot_joint_cfg import G1_29DOF_JOINT_NAMES_ISAACLAB_ORDER


GROOT_WBC_G1_CFG = G1_29DOF_CFG.copy()
GROOT_WBC_G1_CFG.init_state.pos = (0.0, 0.0, 0.80)
GROOT_WBC_G1_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)
GROOT_WBC_G1_CFG.init_state.joint_pos = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
GROOT_WBC_G1_CFG.init_state.joint_vel = {".*": 0.0}
# Match gear_sonic_deploy/policy_parameters.hpp. These values are part of the
# trained controller's plant model, not optional visual tuning.
GROOT_WBC_G1_CFG.actuators["legs"].stiffness = {
    ".*_hip_pitch_joint": 99.09,
    ".*_hip_roll_joint": 99.09,
    ".*_hip_yaw_joint": 40.18,
    ".*_knee_joint": 99.09,
}
GROOT_WBC_G1_CFG.actuators["legs"].damping = {
    ".*_hip_pitch_joint": 6.31,
    ".*_hip_roll_joint": 6.31,
    ".*_hip_yaw_joint": 2.56,
    ".*_knee_joint": 6.31,
}
GROOT_WBC_G1_CFG.actuators["legs"].effort_limit = {
    ".*_hip_pitch_joint": 139.0,
    ".*_hip_roll_joint": 139.0,
    ".*_hip_yaw_joint": 88.0,
    ".*_knee_joint": 139.0,
}
GROOT_WBC_G1_CFG.actuators["legs"].armature = {
    ".*_hip_pitch_joint": 0.025101925,
    ".*_hip_roll_joint": 0.025101925,
    ".*_hip_yaw_joint": 0.010177520,
    ".*_knee_joint": 0.025101925,
}
GROOT_WBC_G1_CFG.actuators["feet"].stiffness = 28.50
GROOT_WBC_G1_CFG.actuators["feet"].damping = 1.81
GROOT_WBC_G1_CFG.actuators["feet"].effort_limit = 25.0
GROOT_WBC_G1_CFG.actuators["feet"].armature = 0.003609725
GROOT_WBC_G1_CFG.actuators["waist"].stiffness = {
    "waist_yaw_joint": 40.18,
    "waist_roll_joint": 28.50,
    "waist_pitch_joint": 28.50,
}
GROOT_WBC_G1_CFG.actuators["waist"].damping = {
    "waist_yaw_joint": 2.56,
    "waist_roll_joint": 1.81,
    "waist_pitch_joint": 1.81,
}
GROOT_WBC_G1_CFG.actuators["arms"].stiffness = {
    ".*_shoulder_.*_joint": 14.25,
    ".*_elbow_joint": 14.25,
    ".*_wrist_roll_joint": 14.25,
    ".*_wrist_pitch_joint": 16.78,
    ".*_wrist_yaw_joint": 16.78,
}
GROOT_WBC_G1_CFG.actuators["arms"].damping = {
    ".*_shoulder_.*_joint": 0.91,
    ".*_elbow_joint": 0.91,
    ".*_wrist_roll_joint": 0.91,
    ".*_wrist_pitch_joint": 1.07,
    ".*_wrist_yaw_joint": 1.07,
}
GROOT_WBC_G1_CFG.actuators["arms"].effort_limit = {
    ".*_shoulder_.*_joint": 25.0,
    ".*_elbow_joint": 25.0,
    ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0,
    ".*_wrist_yaw_joint": 5.0,
}


@configclass
class GrootWbcClosedLoopSceneCfg(InteractiveSceneCfg):
    """G1, ground, and lighting only; no remotely textured assets."""

    robot: ArticulationCfg = GROOT_WBC_G1_CFG
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Absolute WBC joint-position targets in GR00T policy order."""

    whole_body_joint_target = GrootWholeBodyJointTargetActionCfg(
        asset_name="robot",
        joint_names=G1_29DOF_JOINT_NAMES_ISAACLAB_ORDER,
        preserve_order=True,
        # The WBC already applies action scaling. Avoid changing its dynamics here.
        max_joint_delta_per_step=0.0,
        clip_to_soft_limits=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        root_quat = ObsTerm(func=mdp.root_quat_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class GrootWbcClosedLoopEnvCfg(ManagerBasedRLEnvCfg):
    """Single-environment, 200-Hz PhysX / 50-Hz WBC configuration."""

    scene: GrootWbcClosedLoopSceneCfg = GrootWbcClosedLoopSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards = None
    terminations = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 24 * 60 * 60.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = 4
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)
        # Do not block closed-loop bring-up on remote RTX texture compilation.
        self.wait_for_textures = False
