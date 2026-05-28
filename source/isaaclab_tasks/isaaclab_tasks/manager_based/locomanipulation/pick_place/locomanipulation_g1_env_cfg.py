# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.network_runtime_cfg import build_dual_machine_runtime_cfg
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
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_robot_sync import ZmqRobotSyncActionCfg
from .mdp import events as locomanip_events

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

RUNTIME_NET_CFG = build_dual_machine_runtime_cfg()
PRIMARY_ROBOT_ASSET_NAME = "robot"
SECONDARY_ROBOT_ASSET_NAME = "remote_robot"
PRIMARY_ROBOT_PRIM_NAME = "Robot"
SECONDARY_ROBOT_PRIM_NAME = "RemoteRobot"
LOCAL_ROBOT_ASSET_NAME = PRIMARY_ROBOT_ASSET_NAME if int(RUNTIME_NET_CFG.local_player_id) == 1 else SECONDARY_ROBOT_ASSET_NAME
REMOTE_ROBOT_ASSET_NAME = SECONDARY_ROBOT_ASSET_NAME if int(RUNTIME_NET_CFG.local_player_id) == 1 else PRIMARY_ROBOT_ASSET_NAME
LOCAL_ROBOT_PRIM_NAME = PRIMARY_ROBOT_PRIM_NAME if int(RUNTIME_NET_CFG.local_player_id) == 1 else SECONDARY_ROBOT_PRIM_NAME
REMOTE_ROBOT_PRIM_NAME = SECONDARY_ROBOT_PRIM_NAME if int(RUNTIME_NET_CFG.local_player_id) == 1 else PRIMARY_ROBOT_PRIM_NAME
# Keep a deterministic fallback owner for each box while neither side is
# actively grasping it. Runtime ownership handoff is handled inside
# ``ZmqObjectSyncAction`` once one side grabs the object.
TEST_BOX_DEFAULT_OWNER_PLAYER_ID = 2
TEST_BOX1_DEFAULT_OWNER_PLAYER_ID = 1
OBJECT_MIRROR_X_CENTER = -5.6170


def _machine_ip_for_player(player_id: int) -> str:
    return RUNTIME_NET_CFG.local_machine_ip if int(RUNTIME_NET_CFG.local_player_id) == int(player_id) else RUNTIME_NET_CFG.peer_machine_ip


def _default_box_owner_machine_ip(owner_player_id: int) -> str:
    return _machine_ip_for_player(owner_player_id)


def _local_object_sync_endpoint() -> str:
    return f"tcp://{RUNTIME_NET_CFG.local_machine_ip}:{RUNTIME_NET_CFG.object_sync_port}"


def _remote_object_sync_endpoint() -> str:
    return f"tcp://{RUNTIME_NET_CFG.peer_machine_ip}:{RUNTIME_NET_CFG.object_sync_port}"


def _local_conveyor_surface_velocity() -> tuple[float, float, float]:
    # Keep conveyor world motion identical on both machines. The local/remote
    # robot role swap and XR/world alignment determine the user's perceived
    # mirror; reversing the conveyor for player 2 makes the box drift to the
    # wrong side in local view.
    return (-0.5, 0.0, 0.0)

# Match the initial runnable setup: keep the simulation scene at its authored
# height and use a local XR anchor offset near the tracking origin. The current
# warehouse scene lives at a different world origin than the old tabletop scene,
# so we keep the "local anchor only" principle from the initial version while
# allowing a tiny per-player Z calibration.
XR_ANCHOR_Z_OFFSET_PLAYER1_M = float(os.environ.get("ISAACLAB_XR_ANCHOR_Z_OFFSET_PLAYER1", "-0.10"))
XR_ANCHOR_Z_OFFSET_PLAYER2_M = float(os.environ.get("ISAACLAB_XR_ANCHOR_Z_OFFSET_PLAYER2", "-0.35"))
SCENE_HEIGHT_OFFSET_M = 0.0
WAREHOUSE_XR_ANCHOR_XY = (
    float(os.environ.get("ISAACLAB_XR_ANCHOR_WORLD_X", "-6.2475")),
    float(os.environ.get("ISAACLAB_XR_ANCHOR_WORLD_Y", "14.5082")),
)


def _local_xr_anchor_z_offset_m() -> float:
    return XR_ANCHOR_Z_OFFSET_PLAYER1_M if int(RUNTIME_NET_CFG.local_player_id) == 1 else XR_ANCHOR_Z_OFFSET_PLAYER2_M


LOCAL_XR_ANCHOR_POS = (WAREHOUSE_XR_ANCHOR_XY[0], WAREHOUSE_XR_ANCHOR_XY[1], _local_xr_anchor_z_offset_m())

# The initial runnable implementation did not apply an extra wrist translation
# after the controller frame rotation. Keep that behavior by default and allow
# per-player overrides only when calibration is needed.
LOCAL_WRIST_POSITION_OFFSET = (
    float(
        os.environ.get(
            f"ISAACLAB_HAND_WRIST_OFFSET_X_PLAYER{int(RUNTIME_NET_CFG.local_player_id)}",
            "0.0",
        )
    ),
    float(os.environ.get(f"ISAACLAB_HAND_WRIST_OFFSET_Y_PLAYER{int(RUNTIME_NET_CFG.local_player_id)}", "0.0")),
    float(os.environ.get(f"ISAACLAB_HAND_WRIST_OFFSET_Z_PLAYER{int(RUNTIME_NET_CFG.local_player_id)}", "0.0")),
)
REMOTE_WRIST_POSITION_OFFSET = (
    float(os.environ.get(f"ISAACLAB_REMOTE_WRIST_OFFSET_X_PLAYER{int(RUNTIME_NET_CFG.local_player_id)}", "0.0")),
    float(os.environ.get(f"ISAACLAB_REMOTE_WRIST_OFFSET_Y_PLAYER{int(RUNTIME_NET_CFG.local_player_id)}", "0.0")),
    float(os.environ.get(f"ISAACLAB_REMOTE_WRIST_OFFSET_Z_PLAYER{int(RUNTIME_NET_CFG.local_player_id)}", "0.0")),
)


def _with_scene_height_offset(pos_xyz: tuple[float, float, float] | list[float]) -> list[float]:
    """Apply the global SteamVR-driven scene height offset to a 3D position."""
    return [float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]) + SCENE_HEIGHT_OFFSET_M]


def _ensure_valid_urdf_file(local_urdf_path: str) -> str:
    """Validate and repair a URDF file if the cached content was corrupted.

    We observed that ``retrieve_file_path`` may occasionally leave a partially
    duplicated temp file (two ``<?xml ...?>`` headers in one file). Pinocchio
    then crashes during startup. To keep both 40.36 / 40.30 portable, we repair
    that cache locally and always return a valid URDF path.
    """

    path = Path(local_urdf_path)
    raw_text = path.read_text(encoding="utf-8", errors="ignore")

    def _is_valid_xml(text: str) -> bool:
        try:
            ET.fromstring(text)
            return True
        except ET.ParseError:
            return False

    if _is_valid_xml(raw_text):
        return str(path)

    candidate_texts: list[str] = []

    # If the file was duplicated, the last XML document is usually complete.
    xml_split_parts = [part.strip() for part in re.split(r"(?=<\?xml)", raw_text) if part.strip()]
    if len(xml_split_parts) > 1:
        candidate_texts.extend(reversed(xml_split_parts))

    # Fallback: keep only the content through the first </robot>.
    first_robot_close = raw_text.find("</robot>")
    if first_robot_close != -1:
        candidate_texts.append(raw_text[: first_robot_close + len("</robot>")].strip())

    for index, candidate in enumerate(candidate_texts):
        if not candidate:
            continue
        if not _is_valid_xml(candidate):
            continue
        repaired_path = path.with_name(f"{path.stem}.repaired_{index}{path.suffix}")
        repaired_path.write_text(candidate + "\n", encoding="utf-8")
        return str(repaired_path)

    raise ValueError(f"Unable to recover a valid URDF from cached file: {local_urdf_path}")

ROBOT_A_INIT_POS = (0.0, 0.0, 0.75 + SCENE_HEIGHT_OFFSET_M)
ROBOT_A_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOT_B_INIT_POS = (1.25, 0.0, 0.75 + SCENE_HEIGHT_OFFSET_M)
# Keep the dual-machine scene physically face-to-face. Player 2 must remain
# rotated 180 degrees in the world; the teleop fallback path is responsible for
# adapting controller poses into that local robot frame.
ROBOT_B_INIT_ROT = (0.0, 0.0, 0.0, 1.0)

ROBOT_A_REFERENCE_XY = (0.0, 0.0)
ROBOT_B_REFERENCE_XY = (1.25, 0.0)
PUBLISHER_ROBOT_REFERENCE_XY = ROBOT_A_REFERENCE_XY
SUBSCRIBER_ROBOT_REFERENCE_XY = ROBOT_B_REFERENCE_XY

FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_A_INIT_POS
FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_A_INIT_ROT
REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_B_INIT_POS
REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_B_INIT_ROT

if int(RUNTIME_NET_CFG.local_player_id) == 1:
    LOCAL_ROBOT_REFERENCE_XY = ROBOT_A_REFERENCE_XY
    REMOTE_ROBOT_REFERENCE_XY = ROBOT_B_REFERENCE_XY
else:
    LOCAL_ROBOT_REFERENCE_XY = ROBOT_B_REFERENCE_XY
    REMOTE_ROBOT_REFERENCE_XY = ROBOT_A_REFERENCE_XY


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
        init_state=AssetBaseCfg.InitialStateCfg(pos=_with_scene_height_offset((-4.0, 0.55, -0.3)), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_with_scene_height_offset((-4.35, 0.45, 0.6996)), rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # 测试箱子：落在 ConveyorBelt_A08_06 传送带 +y 端（入料端）。
    # 传送带沿 Y 轴延伸，y∈[-11.42, -8.70]，中心 x=15.30，带面 z=-0.0155。
    # 箱子沿 -y 方向流向机器人（robot1 y≈-12.50，robot2 y≈-13.07）。
    test_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_with_scene_height_offset((0.78886, 1.17033, 0.845)),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    test_box1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox1",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_with_scene_height_offset((0.42787, 1.67696, 0.845)),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    # 本地仓库背景
    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_with_scene_height_offset((-4.68, 14.39363, 0.0)),
            rot=[0.7071, 0.0, 0.0, 0.7071],
        ),
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

    object_sync = ZmqObjectSyncActionCfg(
        asset_name="test_box",
        role="peer",
        local_endpoint=_local_object_sync_endpoint(),
        remote_endpoint=_remote_object_sync_endpoint(),
        local_machine_ip=RUNTIME_NET_CFG.local_machine_ip,
        remote_machine_ip=RUNTIME_NET_CFG.peer_machine_ip,
        default_owner_machine_ip=_default_box_owner_machine_ip(TEST_BOX_DEFAULT_OWNER_PLAYER_ID),
        robot_asset_name=LOCAL_ROBOT_ASSET_NAME,
        apply_remote_updates=False,
        mirror_x_center=OBJECT_MIRROR_X_CENTER if int(RUNTIME_NET_CFG.local_player_id) == 2 else None,
    )
    object_sync1 = ZmqObjectSyncActionCfg(
        asset_name="test_box1",
        role="peer",
        local_endpoint=_local_object_sync_endpoint(),
        remote_endpoint=_remote_object_sync_endpoint(),
        local_machine_ip=RUNTIME_NET_CFG.local_machine_ip,
        remote_machine_ip=RUNTIME_NET_CFG.peer_machine_ip,
        default_owner_machine_ip=_default_box_owner_machine_ip(TEST_BOX1_DEFAULT_OWNER_PLAYER_ID),
        robot_asset_name=LOCAL_ROBOT_ASSET_NAME,
        apply_remote_updates=False,
        mirror_x_center=OBJECT_MIRROR_X_CENTER if int(RUNTIME_NET_CFG.local_player_id) == 2 else None,
    )



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
        robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg(LOCAL_ROBOT_ASSET_NAME)}
        )
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

    # object_dropping = DoneTerm(
    #     func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    # )

    success = DoneTerm(func=manip_mdp.task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})


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
            "mass": 3.5,
            "linear_damping": 0.5,
            "angular_damping": 0.1,
        },
    )

    setup_test_box1_physics = EventTerm(
        func=locomanip_mdp.setup_usd_rigid_object_physics,
        mode="prestartup",
        params={
            "prim_path_template": "/World/envs/env_{}/TestBox1",
            "mass": 3.5,
            "linear_damping": 0.5,
            "angular_damping": 0.1,
        },
    )

    # 启动时打印 ConveyorBelt_A08_06 的世界包围盒，用于校准 test_box 坐标。
    print_conveyor_bbox = EventTerm(
        func=locomanip_events.print_conveyor_world_bbox,
        mode="startup",
        params={"prim_name": "ConveyorBelt_A08_06"},
    )

    align_robots_to_conveyor_startup = EventTerm(
        func=locomanip_events.place_robots_from_conveyor_bbox,
        mode="startup",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "robot1_name": LOCAL_ROBOT_ASSET_NAME,
            "robot2_name": REMOTE_ROBOT_ASSET_NAME,
            "local_player_id": RUNTIME_NET_CFG.local_player_id,
            "publisher_robot_reference_xy": PUBLISHER_ROBOT_REFERENCE_XY,
            "subscriber_robot_reference_xy": SUBSCRIBER_ROBOT_REFERENCE_XY,
        },
    )

    align_robots_to_conveyor_reset = EventTerm(
        func=locomanip_events.place_robots_from_conveyor_bbox,
        mode="reset",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "robot1_name": LOCAL_ROBOT_ASSET_NAME,
            "robot2_name": REMOTE_ROBOT_ASSET_NAME,
            "local_player_id": RUNTIME_NET_CFG.local_player_id,
            "publisher_robot_reference_xy": PUBLISHER_ROBOT_REFERENCE_XY,
            "subscriber_robot_reference_xy": SUBSCRIBER_ROBOT_REFERENCE_XY,
        },
    )

    align_viewer_to_conveyor_startup = EventTerm(
        func=locomanip_events.align_viewer_to_conveyor_bbox,
        mode="startup",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "viewer_origin_type": "asset_root",
            "viewer_asset_name": LOCAL_ROBOT_ASSET_NAME,
            "viewer_body_name": None,
            "reference_viewer_target_xy": (0.0, 0.0),
            "lock_viewer_to_asset": False,
        },
    )

    align_test_boxes_to_conveyor_startup = EventTerm(
        func=locomanip_events.place_test_boxes_from_conveyor_bbox,
        mode="startup",
        params={"conveyor_prim_name": "ConveyorBelt_A08_06", "local_player_id": RUNTIME_NET_CFG.local_player_id},
    )

    align_test_boxes_to_conveyor_reset = EventTerm(
        func=locomanip_events.place_test_boxes_from_conveyor_bbox,
        mode="reset",
        params={"conveyor_prim_name": "ConveyorBelt_A08_06", "local_player_id": RUNTIME_NET_CFG.local_player_id},
    )

    setup_conveyor_belt_physics = EventTerm(
        func=locomanip_events.setup_conveyor_belt_physics,
        mode="prestartup",
        params={
            "velocity": _local_conveyor_surface_velocity(),
            "prim_name_patterns": ("ConveyorBelt_A08_06", "ConveyorBelt_A08_07", "ConveyorBelt_A08_08"),
        },
    )

    # 箱子改为主要依靠传送带表面速度/接触摩擦带动前进，不再定时强制写线速度。
    # drive_test_box = EventTerm(
    #     func=locomanip_mdp.drive_object_on_conveyor,
    #     mode="interval",
    #     interval_range_s=(0.05, 0.05),
    #     params={"object_name": "test_box", "velocity_x": 0.0, "velocity_y": -0.5},
    # )

    # drive_test_box1 = EventTerm(
    #     func=locomanip_mdp.drive_object_on_conveyor,
    #     mode="interval",
    #     interval_range_s=(0.05, 0.05),
    #     params={"object_name": "test_box1", "velocity_x": 0.0, "velocity_y": -0.5},
    # )


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
        anchor_pos=LOCAL_XR_ANCHOR_POS,
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    xr2: XrCfg = XrCfg(
        anchor_pos=LOCAL_XR_ANCHOR_POS,
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        """Post initialization."""
        local_robot_asset_name = LOCAL_ROBOT_ASSET_NAME
        remote_robot_asset_name = REMOTE_ROBOT_ASSET_NAME

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 4

        # Set the URDF and mesh paths for the IK controller
        urdf_omniverse_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"  # noqa: E501

        # Retrieve local paths for the URDF and mesh files. Will be cached for call after the first time.
        retrieved_urdf_path = retrieve_file_path(urdf_omniverse_path)
        valid_urdf_path = _ensure_valid_urdf_file(retrieved_urdf_path)
        self.actions.upper_body_ik.asset_name = local_robot_asset_name
        self.actions.upper_body_ik.controller.articulation_name = local_robot_asset_name
        self.actions.remote_upper_body_ik.asset_name = remote_robot_asset_name
        self.actions.remote_upper_body_ik.controller.articulation_name = remote_robot_asset_name
        self.actions.upper_body_ik.controller.urdf_path = valid_urdf_path
        self.actions.remote_upper_body_ik.controller.urdf_path = valid_urdf_path

        # Match the initial runnable large-space setup by keeping the anchor
        # independent from robot pelvis motion, but account for the warehouse
        # scene's shifted world origin. The old tabletop scene lived near world
        # (0, 0), while this scene is centered around the conveyor in the
        # (-6, 14) neighborhood, so the anchor must keep that scene-aware XY.
        self.xr.anchor_prim_path = None
        self.xr.anchor_pos = LOCAL_XR_ANCHOR_POS
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rot = (1.0, 0.0, 0.0, 0.0)
        self.xr.anchor_rotation_mode = None

        self.xr2.anchor_prim_path = None
        self.xr2.anchor_pos = LOCAL_XR_ANCHOR_POS
        self.xr2.fixed_anchor_height = True
        self.xr2.anchor_rot = (1.0, 0.0, 0.0, 0.0)
        self.xr2.anchor_rotation_mode = None
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
        teleop_devices = {
            "handtracking": OpenXRDeviceCfg(
                retargeters=[
                    G1TriHandUpperBodyMotionControllerRetargeterCfg(
                        enable_visualization=True,
                        sim_device=self.sim.device,
                        hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                        wrist_position_offset=LOCAL_WRIST_POSITION_OFFSET,
                    ),
                ],
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            ),
            "motion_controllers": ZeroMqGameSubDeviceCfg(
                endpoint=RUNTIME_NET_CFG.tracking_subscribe_endpoint,
                topic="state",
                local_player_id=RUNTIME_NET_CFG.local_player_id,
                target_remote_player_id=RUNTIME_NET_CFG.remote_player_id,
                auto_start=True,
                retargeters=[
                    G1TriHandUpperBodyZeroMqRetargeterCfg(
                        enable_visualization=True,
                        sim_device=self.sim.device,
                        hand_joint_names=self.actions.remote_upper_body_ik.hand_joint_names,
                        wrist_position_offset=REMOTE_WRIST_POSITION_OFFSET,
                    ),
                ],
                sim_device=self.sim.device,
            ),
        }

        self.teleop_devices = DevicesCfg(devices=teleop_devices)
        print(
            "[locomanip_cfg] dual-machine runtime: "
            f"local_ip={RUNTIME_NET_CFG.local_machine_ip}, "
            f"peer_ip={RUNTIME_NET_CFG.peer_machine_ip}, "
            f"local_player_id={RUNTIME_NET_CFG.local_player_id}, "
            f"remote_player_id={RUNTIME_NET_CFG.remote_player_id}, "
            f"object_sync_role={RUNTIME_NET_CFG.object_sync_role}, "
            f"local_robot={local_robot_asset_name}, "
            f"remote_robot={remote_robot_asset_name}, "
            f"xr_anchor={self.xr.anchor_prim_path}, "
            f"xr_anchor_pos={self.xr.anchor_pos}, "
            f"xr_anchor_rot={self.xr.anchor_rot}, "
            f"local_wrist_offset={LOCAL_WRIST_POSITION_OFFSET}, "
            f"remote_wrist_offset={REMOTE_WRIST_POSITION_OFFSET}, "
            f"conveyor_velocity={_local_conveyor_surface_velocity()}, "
            f"teleop_devices={list(self.teleop_devices.devices.keys())}"
        )
