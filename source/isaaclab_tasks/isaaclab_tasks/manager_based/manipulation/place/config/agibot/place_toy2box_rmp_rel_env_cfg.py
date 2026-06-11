# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import ObjectTableSceneCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG  # isort: skip
from isaaclab.controllers.config.rmp_flow import AGIBOT_LEFT_ARM_RMPFLOW_CFG, AGIBOT_RIGHT_ARM_RMPFLOW_CFG  # isort: skip


def _resolve_local_asset_path(*relative_paths: str) -> str | None:
    """Resolve an asset from common local Isaac Sim asset mirror locations."""
    candidate_roots = [
        os.getenv("ROBOTYAO_ISAAC_ASSET_ROOT"),
        os.getenv("ISAACSIM_ASSETS_ROOT"),
        os.getenv("ISAAC_SIM_ASSETS_ROOT"),
        r"D:\Omniverse\isaacsim_assets\Assets\Isaac\5.1",
        r"D:\Omniverse\isaacsim_assets\Assets\Isaac\5.1\Isaac",
    ]
    for root in candidate_roots:
        if not root:
            continue
        for relative_path in relative_paths:
            candidate = os.path.join(root, relative_path)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None


def _make_fallback_cuboid(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    rigid_props: RigidBodyPropertiesCfg,
    mass_props: MassPropertiesCfg | None = None,
) -> CuboidCfg:
    """Create a simple physics cuboid when the demonstration USD asset is unavailable."""
    return CuboidCfg(
        size=size,
        collision_props=CollisionPropertiesCfg(),
        rigid_props=rigid_props,
        mass_props=mass_props,
        visual_material=PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
    )

##
# Event settings
##


@configclass
class EventCfgPlaceToy2Box:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    init_toy_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.15, 0.20),
                "y": (-0.3, -0.15),
                "z": (-0.65, -0.65),
                "yaw": (-3.14, 3.14),
            },
            "asset_cfgs": [SceneEntityCfg("toy_truck")],
        },
    )
    init_box_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.35),
                "y": (0.0, 0.10),
                "z": (-0.55, -0.55),
                "yaw": (-3.14, 3.14),
            },
            "asset_cfgs": [SceneEntityCfg("box")],
        },
    )


#
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        toy_truck_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("toy_truck"), "return_key": "pos"},
        )
        toy_truck_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("toy_truck"), "return_key": "quat"},
        )
        box_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame, params={"object_cfg": SceneEntityCfg("box"), "return_key": "pos"}
        )
        box_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("box"), "return_key": "quat"},
        )
        eef_pos = ObsTerm(func=mdp.ee_frame_pose_in_base_frame, params={"return_key": "pos"})
        eef_quat = ObsTerm(func=mdp.ee_frame_pose_in_base_frame, params={"return_key": "quat"})
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp = ObsTerm(
            func=place_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("toy_truck"),
                "diff_threshold": 0.05,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = None
    gripper_action: mdp.BinaryJointPositionActionCfg = None

    left_arm_action: mdp.JointPositionActionCfg = None
    right_arm_action: mdp.JointPositionActionCfg = None
    left_gripper_action: mdp.BinaryJointPositionActionCfg = None
    right_gripper_action: mdp.BinaryJointPositionActionCfg = None


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    toy_truck_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("toy_truck")}
    )

    success = DoneTerm(
        func=place_mdp.object_a_is_into_b,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_a_cfg": SceneEntityCfg("toy_truck"),
            "object_b_cfg": SceneEntityCfg("box"),
            "xy_threshold": 0.10,
            "height_diff": 0.06,
            "height_threshold": 0.04,
        },
    )


@configclass
class PlaceToy2BoxEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""

        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4



"""
Env to Replay Sim2Lab Demonstrations with JointSpaceAction
"""


def spawn_agibot_floating(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils import get_current_stage, find_matching_prim_paths
    from pxr import UsdPhysics, PhysxSchema

    print(f"[AgiBot Spawn Debug] spawn_agibot_floating called with prim_path: {prim_path}", flush=True)
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()

    # Find all matching root_joint paths on the stage
    root_joint_pattern = f"{prim_path}/root_joint"
    root_joint_paths = find_matching_prim_paths(root_joint_pattern)
    print(f"[AgiBot Spawn Debug] Resolved root_joint_paths: {root_joint_paths}", flush=True)

    for r_joint_path in root_joint_paths:
        root_joint_prim = stage.GetPrimAtPath(r_joint_path)
        b_link_path = r_joint_path.replace("/root_joint", "/base_link")
        base_link_prim = stage.GetPrimAtPath(b_link_path)
        
        print(f"[AgiBot Spawn Debug] Processing {r_joint_path}. root_joint valid: {root_joint_prim.IsValid() if root_joint_prim else False}, base_link valid: {base_link_prim.IsValid() if base_link_prim else False}", flush=True)
        
        if root_joint_prim.IsValid() and base_link_prim.IsValid():
            # Apply APIs to base_link
            UsdPhysics.ArticulationRootAPI.Apply(base_link_prim)
            PhysxSchema.PhysxArticulationAPI.Apply(base_link_prim)
            
            # Copy values from root_joint to base_link
            root_joint_art_api = UsdPhysics.ArticulationRootAPI(root_joint_prim)
            base_link_art_api = UsdPhysics.ArticulationRootAPI(base_link_prim)
            for attr_name in root_joint_art_api.GetSchemaAttributeNames():
                attr = root_joint_prim.GetAttribute(attr_name)
                if attr.IsValid() and attr.HasValue():
                    base_link_prim.GetAttribute(attr_name).Set(attr.Get())
                    
            root_joint_physx_api = PhysxSchema.PhysxArticulationAPI(root_joint_prim)
            base_link_physx_api = PhysxSchema.PhysxArticulationAPI(base_link_prim)
            for attr_name in root_joint_physx_api.GetSchemaAttributeNames():
                attr = root_joint_prim.GetAttribute(attr_name)
                if attr.IsValid() and attr.HasValue():
                    base_link_prim.GetAttribute(attr_name).Set(attr.Get())
            
            # Remove APIs from root_joint
            root_joint_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            root_joint_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            
            # Disable root_joint to ensure it is floating
            joint_api = UsdPhysics.Joint(root_joint_prim)
            if joint_api:
                joint_api.GetJointEnabledAttr().Set(False)
                
            print(f"[AgiBot Floating Spawn] Successfully migrated ArticulationRootAPI to base_link and disabled root_joint for {r_joint_path}", flush=True)
            
    return prim


class RmpFlowAgibotPlaceToy2BoxEnvCfg(PlaceToy2BoxEnvCfg):
    """Configuration for the Agibot Place Toy2Box RMP Rel Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events = EventCfgPlaceToy2Box()

        # Set Agibot as robot
        self.scene.robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.func = spawn_agibot_floating
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        self.scene.robot.spawn.articulation_props.fix_root_link = False
        self.scene.robot.init_state.pos = (-0.6, 0.0, -1.04)
        self.scene.plane = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.06]),
            spawn=CuboidCfg(
                size=(20.0, 20.0, 0.02),
                collision_props=CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32), roughness=0.8),
            ),
        )

        # add table
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, -0.70]),
            spawn=CuboidCfg(
                size=(1.45, 0.90, 0.08),
                collision_props=CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.48, 0.50, 0.52), roughness=0.7),
            ),
        )

        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Agibot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_gripper_center",
            controller=AGIBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Enable Parallel Gripper:
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_hand_joint1", "right_.*_Support_Joint"],
            open_command_expr={"right_hand_joint1": 0.994, "right_.*_Support_Joint": 0.994},
            close_command_expr={"right_hand_joint1": 0.20, "right_.*_Support_Joint": 0.20},
        )

        # Concurrent Bimanual Action Terms (using RMPFlow for both arms)
        self.actions.left_arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="gripper_center",
            controller=AGIBOT_LEFT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[0.7071, 0.0, -0.7071, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )
        self.actions.right_arm_action = self.actions.arm_action

        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_hand_joint1", "left_.*_Support_Joint"],
            open_command_expr={"left_hand_joint1": 0.994, "left_.*_Support_Joint": 0.994},
            close_command_expr={"left_hand_joint1": 0.0, "left_.*_Support_Joint": 0.0},
        )
        self.actions.right_gripper_action = self.actions.gripper_action

        # find joint ids for grippers
        self.gripper_joint_names = ["right_hand_joint1", "right_Right_1_Joint"]
        self.gripper_open_val = 0.994
        self.gripper_threshold = 0.2

        # Rigid body properties of toy_truck and box
        toy_truck_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        box_properties = toy_truck_properties.copy()

        # Notes: remember to add Physics/Mass properties to the toy_truck mesh to make grasping successful,
        # then you can use below MassPropertiesCfg to set the mass of the toy_truck
        toy_mass_properties = MassPropertiesCfg(
            mass=0.05,
        )
        toy_truck_usd_path = _resolve_local_asset_path(
            "Isaac/IsaacLab/Objects/ToyTruck/toy_truck.usd",
            "IsaacLab/Objects/ToyTruck/toy_truck.usd",
            "Objects/ToyTruck/toy_truck.usd",
        )
        box_usd_path = _resolve_local_asset_path(
            "Isaac/IsaacLab/Objects/Box/box.usd",
            "IsaacLab/Objects/Box/box.usd",
            "Objects/Box/box.usd",
        )

        self.scene.toy_truck = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ToyTruck",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=(
                UsdFileCfg(
                    usd_path=toy_truck_usd_path,
                    rigid_props=toy_truck_properties,
                    mass_props=toy_mass_properties,
                )
                if toy_truck_usd_path is not None
                else _make_fallback_cuboid(
                    size=(0.18, 0.10, 0.08),
                    color=(0.10, 0.25, 0.90),
                    rigid_props=toy_truck_properties,
                    mass_props=toy_mass_properties,
                )
            ),
        )

        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=(
                UsdFileCfg(
                    usd_path=box_usd_path,
                    rigid_props=box_properties,
                )
                if box_usd_path is not None
                else _make_fallback_cuboid(
                    size=(0.28, 0.28, 0.20),
                    color=(0.95, 0.05, 0.35),
                    rigid_props=box_properties,
                    mass_props=MassPropertiesCfg(mass=0.25),
                )
            ),
        )

        # Listens to the required transforms
        self.marker_cfg = FRAME_MARKER_CFG.copy()
        self.marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_gripper_center",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # add contact force sensor for grasped checking
        self.scene.contact_grasp = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_.*_Pad_Link",
            update_period=0.05,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ToyTruck"],
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0
