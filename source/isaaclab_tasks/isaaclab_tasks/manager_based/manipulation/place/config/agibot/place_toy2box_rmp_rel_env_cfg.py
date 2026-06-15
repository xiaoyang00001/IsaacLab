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
import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.sim.utils import clone

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


_TASK_ROBOT_DEFAULT_POS = (-0.8, 0.0, -1.04)
_TASK_TABLE_POS = (0.5, 0.0, -0.42)
_TASK_TABLE_SIZE = (1.45, 0.90, 0.08)
_TASK_TABLE_TOP_Z = _TASK_TABLE_POS[2] + _TASK_TABLE_SIZE[2] * 0.5
_TASK_CUBE_SIZE = 0.04
_TASK_CUBE_Z = _TASK_TABLE_TOP_Z + _TASK_CUBE_SIZE * 0.5
_TASK_BOX_SIZE = (0.34, 0.70, 0.10)
_TASK_BOX_Z = _TASK_TABLE_TOP_Z + _TASK_BOX_SIZE[2] * 0.5
_TASK_BOX_SUCCESS_X_THRESHOLD = 0.15
_TASK_BOX_SUCCESS_Y_THRESHOLD = 0.31
_TASK_CUBE_DEFAULT_POSES = {
    "cube_1": (0.06, -0.24, _TASK_CUBE_Z),
    "cube_2": (0.06, 0.0, _TASK_CUBE_Z),
    "cube_3": (0.06, 0.24, _TASK_CUBE_Z),
}
_TASK_BOX_DEFAULT_POSE = (0.36, 0.0, _TASK_BOX_Z)


def _fixed_pose_range(x: float, y: float, z: float, yaw: float = 0.0) -> dict[str, tuple[float, float]]:
    return {
        "x": (x, x),
        "y": (y, y),
        "z": (z, z),
        "yaw": (yaw, yaw),
    }


def _configure_symmetric_arm_init_pose(robot_cfg) -> None:
    """Mirror the right-arm reset pose onto the left arm for a symmetric default posture."""
    joint_pos = dict(robot_cfg.init_state.joint_pos)
    for joint_index in range(1, 8):
        right_joint = f"right_arm_joint{joint_index}"
        left_joint = f"left_arm_joint{joint_index}"
        if right_joint in joint_pos and left_joint in joint_pos:
            joint_pos[left_joint] = -float(joint_pos[right_joint])
    robot_cfg.init_state.joint_pos = joint_pos


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


@clone
def spawn_open_container_box(prim_path: str, cfg: CuboidCfg, translation=None, orientation=None, **kwargs):
    """Spawn a rigid open-top container made from five cuboid collision pieces."""

    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation, stage=stage)

    size_x, size_y, size_z = cfg.size
    wall_thickness = min(size_x, size_y) * 0.08
    bottom_thickness = min(size_z * 0.18, wall_thickness)
    half_x = size_x * 0.5
    half_y = size_y * 0.5
    half_z = size_z * 0.5

    common = {
        "collision_props": cfg.collision_props,
        "visual_material": cfg.visual_material,
        "physics_material": cfg.physics_material,
        "semantic_tags": cfg.semantic_tags,
    }
    pieces = (
        ("bottom", (size_x, size_y, bottom_thickness), (0.0, 0.0, -half_z + bottom_thickness * 0.5)),
        ("front_wall", (size_x, wall_thickness, size_z), (0.0, half_y - wall_thickness * 0.5, 0.0)),
        ("back_wall", (size_x, wall_thickness, size_z), (0.0, -half_y + wall_thickness * 0.5, 0.0)),
        ("left_wall", (wall_thickness, size_y, size_z), (-half_x + wall_thickness * 0.5, 0.0, 0.0)),
        ("right_wall", (wall_thickness, size_y, size_z), (half_x - wall_thickness * 0.5, 0.0, 0.0)),
    )
    for name, piece_size, local_pos in pieces:
        piece_cfg = CuboidCfg(size=piece_size, **common)
        piece_cfg.func(f"{prim_path}/{name}", piece_cfg, translation=local_pos)

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    return stage.GetPrimAtPath(prim_path)


##
# Event settings
##


@configclass
class EventCfgPlaceToy2Box:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    init_cube_1_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": _fixed_pose_range(*_TASK_CUBE_DEFAULT_POSES["cube_1"]),
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )
    init_cube_2_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": _fixed_pose_range(*_TASK_CUBE_DEFAULT_POSES["cube_2"]),
            "asset_cfgs": [SceneEntityCfg("cube_2")],
        },
    )
    init_cube_3_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": _fixed_pose_range(*_TASK_CUBE_DEFAULT_POSES["cube_3"]),
            "asset_cfgs": [SceneEntityCfg("cube_3")],
        },
    )
    init_box_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": _fixed_pose_range(*_TASK_BOX_DEFAULT_POSE),
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
        cube_1_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("cube_1"), "return_key": "pos"},
        )
        cube_1_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("cube_1"), "return_key": "quat"},
        )
        cube_2_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("cube_2"), "return_key": "pos"},
        )
        cube_2_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("cube_2"), "return_key": "quat"},
        )
        cube_3_positions = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("cube_3"), "return_key": "pos"},
        )
        cube_3_orientations = ObsTerm(
            func=place_mdp.object_poses_in_base_frame,
            params={"object_cfg": SceneEntityCfg("cube_3"), "return_key": "quat"},
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
                "object_cfg": SceneEntityCfg("cube_1"),
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

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("cube_1")}
    )
    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("cube_2")}
    )
    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("cube_3")}
    )
    box_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.85, "asset_cfg": SceneEntityCfg("box")}
    )

    success = DoneTerm(
        func=place_mdp.objects_are_inside_box,
        params={
            "object_cfgs": (SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")),
            "box_cfg": SceneEntityCfg("box"),
            "x_threshold": _TASK_BOX_SUCCESS_X_THRESHOLD,
            "y_threshold": _TASK_BOX_SUCCESS_Y_THRESHOLD,
            "z_min": -0.09,
            "z_max": 0.08,
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
        self.scene.robot.init_state.pos = _TASK_ROBOT_DEFAULT_POS
        _configure_symmetric_arm_init_pose(self.scene.robot)
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
            init_state=AssetBaseCfg.InitialStateCfg(pos=_TASK_TABLE_POS),
            spawn=CuboidCfg(
                size=_TASK_TABLE_SIZE,
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

        # Rigid body properties of cubes and the target container.
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        box_properties = cube_properties.copy()

        cube_mass_properties = MassPropertiesCfg(
            mass=0.05,
        )
        cube_specs = (
            ("cube_1", "Cube1", (0.10, 0.25, 0.90)),
            ("cube_2", "Cube2", (0.08, 0.42, 1.00)),
            ("cube_3", "Cube3", (0.06, 0.58, 0.95)),
        )
        for scene_name, prim_name, color in cube_specs:
            setattr(
                self.scene,
                scene_name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
                    init_state=RigidObjectCfg.InitialStateCfg(),
                    spawn=_make_fallback_cuboid(
                        size=(_TASK_CUBE_SIZE, _TASK_CUBE_SIZE, _TASK_CUBE_SIZE),
                        color=color,
                        rigid_props=cube_properties,
                        mass_props=cube_mass_properties,
                    ),
                ),
            )

        container_cfg = CuboidCfg(
            func=spawn_open_container_box,
            size=_TASK_BOX_SIZE,
            collision_props=CollisionPropertiesCfg(),
            rigid_props=box_properties,
            mass_props=MassPropertiesCfg(mass=0.8),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.95, 0.12, 0.72), roughness=0.5),
        )
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(),
            spawn=container_cfg,
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
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube1", "{ENV_REGEX_NS}/Cube2", "{ENV_REGEX_NS}/Cube3"],
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
