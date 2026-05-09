# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from copy import deepcopy
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_lower_body_standing import G1LowerBodyStandingRetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_motion_controller_locomotion import (
    G1LowerBodyStandingMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter import (
    G1TriHandUpperBodyMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.zeromq_game_sub_device import ZeroMqGameSubDeviceCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_zeromq_retargeter import (
    G1TriHandUpperBodyZeroMqRetargeterCfg,
)

from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import (
    G1TriHandUpperBodyRetargeterCfg,
)
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path

import copy
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_object_sync import ZmqObjectSyncActionCfg

# ZMQ_SYNC_ROLE = os.environ.get("ZMQ_SYNC_ROLE", "none")
ZMQ_SYNC_ROLE = "publisher"
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.agile_locomotion_observation_cfg import (
    AgileTeacherPolicyObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (  # isort: skip
    G1_UPPER_BODY_IK_ACTION_CFG,
)

FIXED_G1_29DOF_CFG = G1_29DOF_CFG.copy()
FIXED_G1_29DOF_CFG.spawn.articulation_props.fix_root_link = True
FIXED_G1_29DOF_CFG.spawn.rigid_props.disable_gravity = True
# FIXED_G1_29DOF_CFG.init_state.pos = (14.3896, -12.4998, -0.3018)
# FIXED_G1_29DOF_CFG.init_state.rot = (-0.9986, 0.0, 0.0, 0.0523)

REMOTE_FIXED_G1_29DOF_CFG = FIXED_G1_29DOF_CFG.copy()
REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = (0.0, 1.1, 0.75)
REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = (0.7071, 0.0, 0.0, -0.7071)
# REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = (16.1596, -13.0698, -0.4118)
# REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = (0.0, 0.0, 0.0, 1.0)
# REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = (0.65, 0, 0.75)
##
# Scene definition
##
@configclass
class LocomanipulationG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for locomanipulation environment with G1 robot.

    This configuration sets up the G1 humanoid robot for locomanipulation tasks,
    allowing both locomotion and manipulation capabilities. The robot can move its
    base and use its arms for manipulation tasks.
    """

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg() if ZMQ_SYNC_ROLE != "subscriber" else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = FIXED_G1_29DOF_CFG

    remote_robot: ArticulationCfg = REMOTE_FIXED_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/RemoteRobot")

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

    remote_upper_body_ik = copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG)
    remote_upper_body_ik.asset_name = "remote_robot"
    remote_upper_body_ik.controller.articulation_name = "remote_robot"

    object_sync = ZmqObjectSyncActionCfg(asset_name="object", role=ZMQ_SYNC_ROLE,endpoint="tcp://192.168.10.46:15555")

    # lower_body_joint_pos = AgileBasedLowerBodyActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         ".*_hip_.*_joint",
    #         ".*_knee_joint",
    #         ".*_ankle_.*_joint",
    #     ],
    #     policy_output_scale=0.25,
    #     obs_group_name="lower_body_policy",  # need to be the same name as the on in ObservationCfg
    #     policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt",
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    This class is required by the environment configuration but not used in this implementation
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        remote_robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("remote_robot")},
        )
        remote_robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("remote_robot")})
        remote_robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("remote_robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=manip_mdp.get_robot_joint_state, params={"joint_names": [".*_hand.*"]})

        object = ObsTerm(
            func=manip_mdp.object_obs,
            params={"left_eef_link_name": "left_wrist_yaw_link", "right_eef_link_name": "right_wrist_yaw_link"},
        )
        # left_wrist_cam = ObsTerm(
        #     func=base_mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("left_hand_cam"), "data_type": "rgb", "normalize": False},
        # )
        # right_wrist_cam = ObsTerm(
        #     func=base_mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("right_hand_cam"), "data_type": "rgb", "normalize": False},
        # )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    # lower_body_policy: AgileTeacherPolicyObservationsCfg = AgileTeacherPolicyObservationsCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=manip_mdp.task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})


##
# MDP settings
##


@configclass
class LocomanipulationG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 locomanipulation environment.

    This environment is designed for locomanipulation tasks where the G1 humanoid robot
    can perform both locomotion and manipulation simultaneously. The robot can move its
    base and use its arms for manipulation tasks, enabling complex mobile manipulation
    behaviors.
    """

    # Scene settings
    scene: LocomanipulationG1SceneCfg = LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.35),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    xr2: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.35),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 2

        # Set the URDF and mesh paths for the IK controller
        urdf_omniverse_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"  # noqa: E501

        # Retrieve local paths for the URDF and mesh files. Will be cached for call after the first time.
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)
        self.actions.remote_upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

        # For Large-Space 1:1 Tracking mode, both VR devices share the identical world physical space.
        # We explicitly DO NOT bind the XRAnchor to any robot pelvis to prevent double-offsetting.
        # self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        # self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        # self.xr2.anchor_prim_path = "/World/envs/env_0/RemoteRobot/pelvis"
        self.xr2.fixed_anchor_height = True
        # self.xr2.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        # # Added Camera attached to left wrist link
        # self.scene.left_hand_cam = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/cam",
        #     update_period=0.0,
        #     height=256,
        #     width=256,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        #     ),
        #     offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        # )
        # # Added Camera attached to right wrist link
        # self.scene.right_hand_cam = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/cam",
        #     update_period=0.0,
        #     height=256,
        #     width=256,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 10.0)
        #     ),
        #     offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        # )
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyMotionControllerRetargeterCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "motion_controllers": ZeroMqGameSubDeviceCfg(
                    endpoint="tcp://192.168.10.46:14025",
                    topic="state",
                    local_player_id=1,
                    target_remote_player_id=2,
                    auto_start=True,
                    retargeters=[
                        G1TriHandUpperBodyZeroMqRetargeterCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.remote_upper_body_ik.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                ),
            }
        )
