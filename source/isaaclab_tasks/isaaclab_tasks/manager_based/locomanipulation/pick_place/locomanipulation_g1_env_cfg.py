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
from isaaclab.managers import EventTermCfg as EventTerm
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
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.network_cfg import NETWORK_CFG

LOCAL_ROBOT_ASSET_NAME = NETWORK_CFG.local_robot_scene_name
REMOTE_ROBOT_ASSET_NAME = NETWORK_CFG.remote_robot_scene_name
ZMQ_SYNC_ROLE = NETWORK_CFG.zmq_object_sync_role

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
REMOTE_FIXED_G1_29DOF_CFG = FIXED_G1_29DOF_CFG.copy()

FIXED_G1_29DOF_CFG.init_state.rot = (1,0,0,0)
REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = (1.25, 0, 0.75)
REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = (0.0, 0.0, 0.0, 1.0)

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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-4.35, 0.45, 0.6996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # 测试箱子：落在 ConveyorBelt_A08_06 传送带上，由表面速度驱动。
    # 传送带 y∈[0.98, 3.69]，带面 z≈0.85，中心 x=0.62，沿 -y 方向输送。
    test_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.78886, 1.17033, 0.845],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg() if ZMQ_SYNC_ROLE != "subscriber" else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )
    test_box1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.42787, 1.67696, 0.845],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg() if ZMQ_SYNC_ROLE != "subscriber" else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )
    # 本地仓库背景
    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.68,14.39363, 0], rot=[0.7071, 0.0, 0.0, 0.7071]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse.usd"),
        ),
    )

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = FIXED_G1_29DOF_CFG

    remote_robot: ArticulationCfg = REMOTE_FIXED_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/RemoteRobot")

    # Ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     spawn=GroundPlaneCfg(),
    # )

    # Lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG)
    upper_body_ik.asset_name = LOCAL_ROBOT_ASSET_NAME
    upper_body_ik.controller.articulation_name = LOCAL_ROBOT_ASSET_NAME

    remote_upper_body_ik = copy.deepcopy(G1_UPPER_BODY_IK_ACTION_CFG)
    remote_upper_body_ik.asset_name = REMOTE_ROBOT_ASSET_NAME
    remote_upper_body_ik.controller.articulation_name = REMOTE_ROBOT_ASSET_NAME

    object_sync = ZmqObjectSyncActionCfg(asset_name="test_box", role=ZMQ_SYNC_ROLE)
    object_sync1 = ZmqObjectSyncActionCfg(asset_name="test_box1", role=ZMQ_SYNC_ROLE)



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
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)})
        robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)}
        )
        remote_robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME)},
        )
        remote_robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME)}
        )
        remote_robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg(REMOTE_ROBOT_ASSET_NAME)}
        )
        # object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        # object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(
            func=manip_mdp.get_all_robot_link_state,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)},
        )

        left_eef_pos = ObsTerm(
            func=manip_mdp.get_eef_pos,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "link_name": "left_wrist_yaw_link"},
        )
        left_eef_quat = ObsTerm(
            func=manip_mdp.get_eef_quat,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "link_name": "left_wrist_yaw_link"},
        )
        right_eef_pos = ObsTerm(
            func=manip_mdp.get_eef_pos,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "link_name": "right_wrist_yaw_link"},
        )
        right_eef_quat = ObsTerm(
            func=manip_mdp.get_eef_quat,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "link_name": "right_wrist_yaw_link"},
        )

        hand_joint_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state,
            params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME), "joint_names": [".*_hand.*"]},
        )

        object = ObsTerm(
            func=manip_mdp.object_obs,
            params={
                "asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME),
                "left_eef_link_name": "left_wrist_yaw_link",
                "right_eef_link_name": "right_wrist_yaw_link",
            },
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

    # object_dropping = DoneTerm(
    #     func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    # )

    success = DoneTerm(
        func=manip_mdp.task_done_pick_place,
        params={
            "robot_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME),
            "task_link_name": "right_wrist_yaw_link",
        },
    )


@configclass
class EventsCfg:
    """Runtime events（仓库场景：传送带自动运行 + 回合重置时复位球体）。"""

    # prestartup：在 PhysX 初始化前为 SM_CardBoxD_05.usd 注入刚体物理 API。
    # UsdFileCfg 只会 modify（而非 define）RigidBodyAPI，此事件补上 define 步骤。
    setup_test_box_physics = EventTerm(
        func=locomanip_mdp.setup_usd_rigid_object_physics,
        mode="prestartup",
        params={
            "prim_path_template": "/World/envs/env_{}/TestBox",
            "mass": 0.5,
            "linear_damping": 5.0,
            "angular_damping": 0.1,
        },
    )

    setup_test_box1_physics = EventTerm(
        func=locomanip_mdp.setup_usd_rigid_object_physics,
        mode="prestartup",
        params={
            "prim_path_template": "/World/envs/env_{}/TestBox1",
            "mass": 0.5,
            "linear_damping": 5.0,
            "angular_damping": 0.1,
        },
    )

    # 启动时打印 ConveyorBelt_A08_06 的世界包围盒，用于校准 test_box 坐标。
    print_conveyor_bbox = EventTerm(
        func=locomanip_mdp.print_conveyor_world_bbox,
        mode="startup",
        params={"prim_name": "ConveyorBelt_A08_06"},
    )

    # 用传送带实时 bbox 放置两个机器人，避免 simple7/change6 场景切换后相对传送带错位。
    align_robots_to_conveyor_startup = EventTerm(
        func=locomanip_mdp.place_robots_from_conveyor_bbox,
        mode="startup",
        params={"conveyor_prim_name": "ConveyorBelt_A08_06"},
    )

    align_robots_to_conveyor_reset = EventTerm(
        func=locomanip_mdp.place_robots_from_conveyor_bbox,
        mode="reset",
        params={"conveyor_prim_name": "ConveyorBelt_A08_06"},
    )

    # 启动时把第一视角对齐到当前本机控制的机器人，避免两台电脑都落到同一个机器人视角。
    align_viewer_to_conveyor_startup = EventTerm(
        func=locomanip_mdp.align_viewer_to_conveyor_bbox,
        mode="startup",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "viewer_origin_type": "asset_body" if NETWORK_CFG.viewer_follow_body_name else "asset_root",
            "viewer_asset_name": LOCAL_ROBOT_ASSET_NAME if NETWORK_CFG.viewer_follow_local_robot else None,
            "viewer_body_name": NETWORK_CFG.viewer_follow_body_name,
            "reference_viewer_target_xy": (0.0, 0.0),
            "lock_viewer_to_asset": False,
        },
    )

    # 用传送带实时 bbox 放置两个测试箱子，避免 simple7/change6 场景切换后相对传送带错位。
    align_test_boxes_to_conveyor_startup = EventTerm(
        func=locomanip_mdp.place_test_boxes_from_conveyor_bbox,
        mode="startup",
        params={"conveyor_prim_name": "ConveyorBelt_A08_06"},
    )

    align_test_boxes_to_conveyor_reset = EventTerm(
        func=locomanip_mdp.place_test_boxes_from_conveyor_bbox,
        mode="reset",
        params={"conveyor_prim_name": "ConveyorBelt_A08_06"},
    )

    # prestartup：对三段拼接的主传送带 Rollers 施加 PhysxSurfaceVelocityAPI（-X方向）。
    # 三组传送带（_06, _07, _08）首尾相接形成物料流，箱子沿此路径输送。
    # 掉出传送带后靠阻尼自然停止，不驱动其他传送带避免落入 bin 后仍被推动。
    setup_conveyor_belt_physics = EventTerm(
        func=locomanip_mdp.setup_conveyor_belt_physics,
        mode="prestartup",
        params={
            "velocity": (-0.5, 0.0, 0.0),
            "prim_name_patterns": ("ConveyorBelt_A08_06", "ConveyorBelt_A08_07", "ConveyorBelt_A08_08"),
        },
    )



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
    scene: LocomanipulationG1SceneCfg = LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    events: EventsCfg = EventsCfg()
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
        local_robot_asset_name = NETWORK_CFG.local_robot_scene_name
        remote_robot_asset_name = NETWORK_CFG.remote_robot_scene_name

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 2

        # Set the URDF and mesh paths for the IK controller
        urdf_omniverse_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"  # noqa: E501

        # Retrieve local paths for the URDF and mesh files. Will be cached for call after the first time.
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
            "[locomanip_cfg] "
            f"local_player_id={NETWORK_CFG.local_player_id}, "
            f"target_remote_player_id={NETWORK_CFG.target_remote_player_id}, "
            f"local_robot={local_robot_asset_name}, "
            f"remote_robot={remote_robot_asset_name}, "
            f"object_sync_role={ZMQ_SYNC_ROLE}, "
            f"xr_anchor={self.xr.anchor_prim_path}"
        )

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
