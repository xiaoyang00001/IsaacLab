# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter import (
    G1TriHandUpperBodyMotionControllerRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_zeromq_retargeter import (
    G1TriHandUpperBodyZeroMqRetargeterCfg,
)
from isaaclab.devices.openxr.zeromq_game_sub_device import ZeroMqGameSubDeviceCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.network_cfg import NETWORK_CFG
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.pink_controller_cfg import (
    G1_UPPER_BODY_IK_ACTION_CFG,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_object_sync import ZmqObjectSyncActionCfg
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp

from .mdp import cafe_handover_events as cafe_events
from .mdp import cafe_handover_observations as cafe_obs
from .mdp import cafe_handover_terminations as cafe_terms

from isaaclab_assets.robots.unitree import G1_29DOF_CFG


LOCAL_ROBOT_ASSET_NAME = NETWORK_CFG.local_robot_scene_name
REMOTE_ROBOT_ASSET_NAME = NETWORK_CFG.remote_robot_scene_name
ZMQ_SYNC_ROLE = NETWORK_CFG.zmq_object_sync_role

ROBOT_A_FALLBACK_POS = (0.0, 0.0, 0.75)
ROBOT_A_FALLBACK_QUAT = (1.0, 0.0, 0.0, 0.0)
ROBOT_B_FALLBACK_POS = (1.15, 0.0, 0.75)
ROBOT_B_FALLBACK_QUAT = (0.0, 0.0, 0.0, 1.0)
CUP_SPAWN_FALLBACK_POS = (0.2, 0.42, 0.95)
HANDOVER_ZONE_FALLBACK_POS = (0.62, 0.42, 0.98)
SERVE_ZONE_FALLBACK_POS = (1.0, 0.48, 0.95)
VIEWER_ANCHOR_FALLBACK_POS = HANDOVER_ZONE_FALLBACK_POS

FIXED_G1_29DOF_CFG = G1_29DOF_CFG.copy()
FIXED_G1_29DOF_CFG.spawn.articulation_props.fix_root_link = True
FIXED_G1_29DOF_CFG.spawn.rigid_props.disable_gravity = True
FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_A_FALLBACK_POS
FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_A_FALLBACK_QUAT

REMOTE_FIXED_G1_29DOF_CFG = FIXED_G1_29DOF_CFG.copy()
REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_B_FALLBACK_POS
REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_B_FALLBACK_QUAT


@configclass
class CafeHandoverG1SceneCfg(InteractiveSceneCfg):
    """Placeholder scene cfg for a dual-G1 cafe handover task."""

    counter = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Counter",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.55, -0.3), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    serve_counter = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ServeCounter",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.55, -0.3), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUP_SPAWN_FALLBACK_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.12,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.53, 0.33, 0.12)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=(ZMQ_SYNC_ROLE == "subscriber"),
                disable_gravity=(ZMQ_SYNC_ROLE == "subscriber"),
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        ),
    )

    robot_spawn_a = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotSpawnA",
        init_state=AssetBaseCfg.InitialStateCfg(pos=ROBOT_A_FALLBACK_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    robot_spawn_b = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotSpawnB",
        init_state=AssetBaseCfg.InitialStateCfg(pos=ROBOT_B_FALLBACK_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.15)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    cup_spawn = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CupSpawn",
        init_state=AssetBaseCfg.InitialStateCfg(pos=CUP_SPAWN_FALLBACK_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    handover_zone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HandoverZone",
        init_state=AssetBaseCfg.InitialStateCfg(pos=HANDOVER_ZONE_FALLBACK_POS),
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.16, 0.16),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.3), opacity=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    serve_zone = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ServeZone",
        init_state=AssetBaseCfg.InitialStateCfg(pos=SERVE_ZONE_FALLBACK_POS),
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.16, 0.16),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.25, 0.2), opacity=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    viewer_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ViewerAnchor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=VIEWER_ANCHOR_FALLBACK_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    robot: ArticulationCfg = FIXED_G1_29DOF_CFG
    remote_robot: ArticulationCfg = REMOTE_FIXED_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/RemoteRobot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the dual-G1 cafe handover task."""

    upper_body_ik = copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG)
    upper_body_ik.asset_name = LOCAL_ROBOT_ASSET_NAME
    upper_body_ik.controller.articulation_name = LOCAL_ROBOT_ASSET_NAME

    remote_upper_body_ik = copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG)
    remote_upper_body_ik.asset_name = REMOTE_ROBOT_ASSET_NAME
    remote_upper_body_ik.controller.articulation_name = REMOTE_ROBOT_ASSET_NAME

    cup_sync = ZmqObjectSyncActionCfg(asset_name="cup", role=ZMQ_SYNC_ROLE)


@configclass
class ObservationsCfg:
    """Observation specifications for the cafe handover task."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=manip_mdp.last_action)
        local_robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)},
        )
        remote_robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME)},
        )
        local_robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)},
        )
        remote_robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME)},
        )
        local_links_state = ObsTerm(
            func=manip_mdp.get_all_robot_link_state,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)},
        )
        remote_links_state = ObsTerm(
            func=manip_mdp.get_all_robot_link_state,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME)},
        )
        local_left_eef_pos = ObsTerm(
            func=manip_mdp.get_eef_pos,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "link_name": "left_wrist_yaw_link"},
        )
        local_right_eef_pos = ObsTerm(
            func=manip_mdp.get_eef_pos,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "link_name": "right_wrist_yaw_link"},
        )
        remote_left_eef_pos = ObsTerm(
            func=manip_mdp.get_eef_pos,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME), "link_name": "left_wrist_yaw_link"},
        )
        remote_right_eef_pos = ObsTerm(
            func=manip_mdp.get_eef_pos,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME), "link_name": "right_wrist_yaw_link"},
        )
        local_hand_joint_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "joint_names": [".*_hand.*"]},
        )
        remote_hand_joint_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME), "joint_names": [".*_hand.*"]},
        )
        cup_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cup")})
        cup_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cup")})
        cup_lin_vel = ObsTerm(func=base_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("cup")})
        cup_ang_vel = ObsTerm(func=base_mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("cup")})
        handover_zone_pos = ObsTerm(
            func=cafe_obs.named_prim_pos,
            params={"prim_name": "HandoverZone", "fallback_pos": HANDOVER_ZONE_FALLBACK_POS},
        )
        serve_zone_pos = ObsTerm(
            func=cafe_obs.named_prim_pos,
            params={"prim_name": "ServeZone", "fallback_pos": SERVE_ZONE_FALLBACK_POS},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the cafe handover task."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cup_dropped = DoneTerm(func=cafe_terms.cup_dropped)
    cup_tilt_exceeded = DoneTerm(func=cafe_terms.cup_tilt_exceeded)
    success = DoneTerm(
        func=cafe_terms.cup_in_serve_zone,
        params={
            "serve_zone_prim_name": "ServeZone",
            "fallback_target_pos": SERVE_ZONE_FALLBACK_POS,
        },
    )


@configclass
class EventsCfg:
    """Runtime events for the cafe handover task."""

    report_anchor_status = EventTerm(
        func=cafe_events.report_named_prim_status,
        mode="startup",
    )
    place_robots_startup = EventTerm(
        func=cafe_events.place_robots_from_named_prims,
        mode="startup",
        params={
            "robot_a_name": "robot",
            "robot_b_name": "remote_robot",
            "robot_a_prim_name": "RobotSpawnA",
            "robot_b_prim_name": "RobotSpawnB",
            "fallback_robot_a_pos": ROBOT_A_FALLBACK_POS,
            "fallback_robot_a_quat": ROBOT_A_FALLBACK_QUAT,
            "fallback_robot_b_pos": ROBOT_B_FALLBACK_POS,
            "fallback_robot_b_quat": ROBOT_B_FALLBACK_QUAT,
        },
    )
    place_robots_reset = EventTerm(
        func=cafe_events.place_robots_from_named_prims,
        mode="reset",
        params={
            "robot_a_name": "robot",
            "robot_b_name": "remote_robot",
            "robot_a_prim_name": "RobotSpawnA",
            "robot_b_prim_name": "RobotSpawnB",
            "fallback_robot_a_pos": ROBOT_A_FALLBACK_POS,
            "fallback_robot_a_quat": ROBOT_A_FALLBACK_QUAT,
            "fallback_robot_b_pos": ROBOT_B_FALLBACK_POS,
            "fallback_robot_b_quat": ROBOT_B_FALLBACK_QUAT,
        },
    )
    place_cup_startup = EventTerm(
        func=cafe_events.place_rigid_asset_from_named_prim,
        mode="startup",
        params={
            "asset_name": "cup",
            "anchor_prim_name": "CupSpawn",
            "fallback_pos": CUP_SPAWN_FALLBACK_POS,
        },
    )
    place_cup_reset = EventTerm(
        func=cafe_events.place_rigid_asset_from_named_prim,
        mode="reset",
        params={
            "asset_name": "cup",
            "anchor_prim_name": "CupSpawn",
            "fallback_pos": CUP_SPAWN_FALLBACK_POS,
        },
    )
    align_viewer_startup = EventTerm(
        func=cafe_events.align_viewer_to_named_prim,
        mode="startup",
        params={
            "prim_name": "ViewerAnchor",
            "fallback_target": VIEWER_ANCHOR_FALLBACK_POS,
        },
    )


@configclass
class CafeHandoverG1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a placeholder dual-G1 cafe handover environment."""

    scene: CafeHandoverG1SceneCfg = CafeHandoverG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.35),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )
    xr2: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.35),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        local_robot_asset_name = NETWORK_CFG.local_robot_scene_name
        remote_robot_asset_name = NETWORK_CFG.remote_robot_scene_name

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2

        urdf_omniverse_path = (
            f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
            "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
        )
        self.actions.upper_body_ik.asset_name = local_robot_asset_name
        self.actions.upper_body_ik.controller.articulation_name = local_robot_asset_name
        self.actions.remote_upper_body_ik.asset_name = remote_robot_asset_name
        self.actions.remote_upper_body_ik.controller.articulation_name = remote_robot_asset_name
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)
        self.actions.remote_upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

        self.xr.fixed_anchor_height = True
        self.xr2.fixed_anchor_height = True
        if NETWORK_CFG.xr_anchor_follow_local_robot:
            self.xr.anchor_prim_path = NETWORK_CFG.get_local_robot_body_prim_path(
                env_index=0, body_name=NETWORK_CFG.xr_anchor_body_name
            )
            self.xr2.anchor_prim_path = NETWORK_CFG.get_remote_robot_body_prim_path(
                env_index=0, body_name=NETWORK_CFG.xr_anchor_body_name
            )

        print(
            "[cafe_handover_cfg] "
            f"local_player_id={NETWORK_CFG.local_player_id}, "
            f"target_remote_player_id={NETWORK_CFG.target_remote_player_id}, "
            f"local_robot={local_robot_asset_name}, "
            f"remote_robot={remote_robot_asset_name}, "
            f"cup_sync_role={ZMQ_SYNC_ROLE}"
        )

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
                    topic="state",
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
