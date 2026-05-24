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

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    AgileBasedLowerBodyActionCfg,
    AutoWalkActionCfg,
    SONICWholeBodyActionCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import SONIC_G1_29DOF_JOINT_ORDER
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

# 第三个机器人（自动行走）：解除根节点固定，启用物理行走
WALKER_G1_29DOF_CFG = G1_29DOF_CFG.copy()
WALKER_G1_29DOF_CFG.spawn.articulation_props.fix_root_link = False
WALKER_G1_29DOF_CFG.spawn.rigid_props.disable_gravity = False
WALKER_G1_29DOF_CFG.init_state.pos = (-2.0, 0.0, 0.75)
WALKER_G1_29DOF_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)

# 第四个机器人：GEAR-SONIC ONNX 驱动（阶段 3.1：真实 decoder obs + encoder zero-fill）
# init_state.pos 与 walker 同 Y（11.008，来自 align_walker_robot_to_conveyor 事件运行时计算），
# X 错开 3m 便于 GUI 视角同框观察。终极方案应仿照 align_walker_robot_to_conveyor 加一个对齐事件。
#
# 阶段 3.3 E3 D：mocap anchor 时变信号已接，解 fix_root_link 再次物理验证
# 对比 3.1 初次物理验证（立刻摔倒），看 mocap motion 信号是否提供有意义的平衡反馈
SONIC_G1_29DOF_CFG = G1_29DOF_CFG.copy()
SONIC_G1_29DOF_CFG.spawn.articulation_props.fix_root_link = False
SONIC_G1_29DOF_CFG.spawn.rigid_props.disable_gravity = False
SONIC_G1_29DOF_CFG.init_state.pos = (-2.0, 11.008, 0.75)
SONIC_G1_29DOF_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)

# SONIC ONNX 模型路径（由 download_from_hf.py 下载，详见 docs/GR00T_WholeBodyControl_集成计划.md）
SONIC_ENCODER_PATH = r"D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
SONIC_DECODER_PATH = r"D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
# Walking mocap (4MB sample，由 download_from_hf.py --sample 下载)
SONIC_MOCAP_PATH = r"D:/src/Isaac/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl"

# 模拟骨骼数据驱动的全身关节列表（缺失关节会被 AutoWalkAction 自动跳过）
WALKER_WHOLE_BODY_JOINTS = [
    # ── 腿部（12） ───────────────────────────────────────
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # ── 腰部（3） ────────────────────────────────────────
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # ── 手臂（14） ───────────────────────────────────────
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # ── 手部（最多 14，若 USD 中缺失会被自动跳过） ─────────
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
]

RUNTIME_NET_CFG = build_dual_machine_runtime_cfg()


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

ROBOT_A_INIT_POS = (0.0, 0.0, 0.75)
ROBOT_A_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
ROBOT_B_INIT_POS = (1.25, 0.0, 0.75)
ROBOT_B_INIT_ROT = (0.0, 0.0, 0.0, 1.0)

ROBOT_A_REFERENCE_XY = (0.0, 0.0)
ROBOT_B_REFERENCE_XY = (1.25, 0.0)

if RUNTIME_NET_CFG.local_player_id == 1:
    FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_A_INIT_POS
    FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_A_INIT_ROT
    REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_B_INIT_POS
    REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_B_INIT_ROT
    LOCAL_ROBOT_REFERENCE_XY = ROBOT_A_REFERENCE_XY
    REMOTE_ROBOT_REFERENCE_XY = ROBOT_B_REFERENCE_XY
else:
    FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_B_INIT_POS
    FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_B_INIT_ROT
    REMOTE_FIXED_G1_29DOF_CFG.init_state.pos = ROBOT_A_INIT_POS
    REMOTE_FIXED_G1_29DOF_CFG.init_state.rot = ROBOT_A_INIT_ROT
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

    # 测试箱子：落在 ConveyorBelt_A08_06 传送带 +y 端（入料端）。
    # 传送带沿 Y 轴延伸，y∈[-11.42, -8.70]，中心 x=15.30，带面 z=-0.0155。
    # 箱子沿 -y 方向流向机器人（robot1 y≈-12.50，robot2 y≈-13.07）。
    test_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.78886, 1.17033, 0.845],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxD_05.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg()
            if RUNTIME_NET_CFG.object_sync_role != "subscriber"
            else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg()
            if RUNTIME_NET_CFG.object_sync_role != "subscriber"
            else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
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

    # 第三个机器人：点击 Play 后自动行走
    walker_robot: ArticulationCfg = WALKER_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/WalkerRobot")

    # 第四个机器人：GEAR-SONIC ONNX 驱动（最小骨架，仅验证 pipeline）
    sonic_robot: ArticulationCfg = SONIC_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/SONICRobot")

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

    upper_body_ik = G1_UPPER_BODY_IK_ACTION_CFG

    # 第四个机器人：GEAR-SONIC ONNX dual-pass 推理
    # 阶段 3.3 E3：接 walking mocap 替代 self-ref + identity，给 encoder 时变 motion 信号
    sonic_wholebody = SONICWholeBodyActionCfg(
        asset_name="sonic_robot",
        encoder_path=SONIC_ENCODER_PATH,
        decoder_path=SONIC_DECODER_PATH,
        joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
        action_scale=0.2,
        mocap_path=SONIC_MOCAP_PATH,
    )

    # 第三个机器人：模拟全身骨骼数据驱动行走（腿+腰+手臂+手）
    # forward_speed 已在 v3 物理驱动后废弃（脚地接触自然推进），不再传入
    walker_skeletal_walk = AutoWalkActionCfg(
        asset_name="walker_robot",
        joint_names=WALKER_WHOLE_BODY_JOINTS,
        walk_frequency=0.8,
        # 腿部
        hip_pitch_amplitude=0.25,
        knee_amplitude=0.30,
        ankle_pitch_amplitude=0.12,
        # 手臂摆动
        arm_swing_amplitude=0.35,
        elbow_bend_amplitude=0.15,
        # 腰部
        waist_yaw_amplitude=0.06,
        # 手部
        hand_curl_amount=0.10,
    )

    publish_robot_state = ZmqRobotSyncActionCfg(
        asset_name="robot",
        role="publisher",
        endpoint=RUNTIME_NET_CFG.local_robot_sync_endpoint,
        topic="robot_state",
    )

    sync_remote_robot_state = ZmqRobotSyncActionCfg(
        asset_name="remote_robot",
        role="subscriber",
        endpoint=RUNTIME_NET_CFG.remote_robot_sync_endpoint,
        topic="robot_state",
    )

    object_sync = ZmqObjectSyncActionCfg(
        asset_name="test_box", role=RUNTIME_NET_CFG.object_sync_role, endpoint=RUNTIME_NET_CFG.object_sync_endpoint
    )
    object_sync1 = ZmqObjectSyncActionCfg(
        asset_name="test_box1", role=RUNTIME_NET_CFG.object_sync_role, endpoint=RUNTIME_NET_CFG.object_sync_endpoint
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
            "mass": 1.5,
            "linear_damping": 5.0,
            "angular_damping": 0.1,
            "kinematic_enabled": RUNTIME_NET_CFG.object_sync_role == "subscriber",
            "disable_gravity": RUNTIME_NET_CFG.object_sync_role == "subscriber",
        },
    )

    setup_test_box1_physics = EventTerm(
        func=locomanip_mdp.setup_usd_rigid_object_physics,
        mode="prestartup",
        params={
            "prim_path_template": "/World/envs/env_{}/TestBox1",
            "mass": 1.5,
            "linear_damping": 5.0,
            "angular_damping": 0.1,
            "kinematic_enabled": RUNTIME_NET_CFG.object_sync_role == "subscriber",
            "disable_gravity": RUNTIME_NET_CFG.object_sync_role == "subscriber",
        },
    )

    # 启动时打印 ConveyorBelt_A08_06 的世界包围盒，用于校准 test_box 坐标。
    print_conveyor_bbox = EventTerm(
        func=locomanip_mdp.print_conveyor_world_bbox,
        mode="startup",
        params={"prim_name": "ConveyorBelt_A08_06"},
    )

    align_robots_to_conveyor_startup = EventTerm(
        func=locomanip_mdp.place_robots_from_conveyor_bbox,
        mode="startup",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "reference_robot1_xy": LOCAL_ROBOT_REFERENCE_XY,
            "reference_robot2_xy": REMOTE_ROBOT_REFERENCE_XY,
        },
    )

    align_robots_to_conveyor_reset = EventTerm(
        func=locomanip_mdp.place_robots_from_conveyor_bbox,
        mode="reset",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "reference_robot1_xy": LOCAL_ROBOT_REFERENCE_XY,
            "reference_robot2_xy": REMOTE_ROBOT_REFERENCE_XY,
        },
    )

    align_viewer_to_conveyor_startup = EventTerm(
        func=locomanip_mdp.align_viewer_to_conveyor_bbox,
        mode="startup",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "viewer_origin_type": "asset_root",
            "viewer_asset_name": "robot",
            "viewer_body_name": None,
            "reference_viewer_target_xy": (0.0, 0.0),
            "lock_viewer_to_asset": False,
        },
    )

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

    # Walker 机器人：放在 robot1 正后方 3.5m，朝向传送带（+Y）
    align_walker_startup = EventTerm(
        func=locomanip_mdp.align_walker_robot_to_conveyor,
        mode="startup",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "reference_robot1_xy": LOCAL_ROBOT_REFERENCE_XY,
            "walker_robot_name": "walker_robot",
            "walker_y_behind": 3.5,
        },
    )

    align_walker_reset = EventTerm(
        func=locomanip_mdp.align_walker_robot_to_conveyor,
        mode="reset",
        params={
            "conveyor_prim_name": "ConveyorBelt_A08_06",
            "reference_robot1_xy": LOCAL_ROBOT_REFERENCE_XY,
            "walker_robot_name": "walker_robot",
            "walker_y_behind": 3.5,
        },
    )

    setup_conveyor_belt_physics = EventTerm(
        func=locomanip_mdp.setup_conveyor_belt_physics,
        mode="prestartup",
        params={
            "velocity": (-0.5, 0.0, 0.0),
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
            anchor_pos=(0.0, 0.0, -0.82),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    xr2: XrCfg = XrCfg(
            anchor_pos=(0.0, 0.0, -0.82),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        """Post initialization."""
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
        self.actions.upper_body_ik.controller.urdf_path = _ensure_valid_urdf_file(retrieved_urdf_path)

        # Bind XR anchors to the aligned robot pelvis frames so AR/XR starts in
        # the same reference frame as the conveyor-aligned scene.
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        # self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.xr2.anchor_prim_path = "/World/envs/env_0/RemoteRobot/pelvis"
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
                            wrist_position_offset=(-0.16, 0.0, 0.0),
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
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                            wrist_position_offset=(-0.16, 0.0, 0.0),
                        ),
                    ],
                    sim_device=self.sim.device,
                ),
            }
        )
