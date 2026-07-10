# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters import G1GripperMotionControllerRetargeterCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    G1GripperSyncActionCfg,
    MuJoCoG1MirrorActionCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_object_sync import ZmqObjectSyncActionCfg

_ENV_REF_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_env_value(value: str) -> str:
    for _ in range(10):
        expanded = _ENV_REF_RE.sub(lambda match: os.environ.get(match.group(1) or match.group(2), ""), value)
        if expanded == value:
            return expanded
        value = expanded
    return value


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, _expand_env_value(value))


def _load_default_network_config() -> None:
    candidates = []
    for env_name in ("ISAACLAB_G1_NETWORK_CONFIG", "G1_NETWORK_CONFIG"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]).expanduser())
    if os.environ.get("GR00T_WBC_ROOT"):
        candidates.append(Path(os.environ["GR00T_WBC_ROOT"]).expanduser() / "config/g1_udp_network.env")
    candidates.append(Path("F:/ISAACWholeBody/GR00T-WholeBodyControl/config/g1_udp_network.env"))
    candidates.append(Path(__file__).resolve().parents[6] / "scripts/gr00t_wbc/g1_udp_network.env")
    for path in candidates:
        _load_env_file(path)


_load_default_network_config()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _isaac_robot_env(robot_id: int, suffix: str, default: str) -> str:
    return os.environ.get(f"ISAACLAB_G1_{robot_id}_{suffix}", os.environ.get(f"ISAACLAB_G1_{suffix}", default))


def _isaac_robot_env_int(robot_id: int, suffix: str, default: int) -> int:
    try:
        return int(_isaac_robot_env(robot_id, suffix, str(default)))
    except (TypeError, ValueError):
        return default


def _isaac_robot_env_float(robot_id: int, suffix: str, default: float) -> float:
    try:
        return float(_isaac_robot_env(robot_id, suffix, str(default)))
    except (TypeError, ValueError):
        return default


def _isaac_robot_env_bool(robot_id: int, suffix: str, default: bool) -> bool:
    value = _isaac_robot_env(robot_id, suffix, "1" if default else "0")
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _ubuntu_sender_ip(robot_id: int, default: str) -> str:
    return os.environ.get(
        f"UBUNTU_ROBOT_{robot_id}_SENDER_IP",
        os.environ.get(f"G1_{robot_id}_SENDER_IP", default),
    )


def _windows_isaaclab_ip(robot_id: int, default: str) -> str:
    return os.environ.get(
        f"WINDOWS_ROBOT_{robot_id}_ISAACLAB_IP",
        os.environ.get(f"ISAACLAB_G1_{robot_id}_HOST_IP", default),
    )


def _robot_name(robot_id: int) -> str:
    return f"robot_{robot_id}"


def _robot_prim_name(robot_id: int) -> str:
    return f"Robot_{robot_id}"


def _peer_robot_id(robot_id: int) -> int:
    return 2 if robot_id == 1 else 1


ISAACLAB_LOCAL_ROBOT_ID = 2 if _env_int("ISAACLAB_LOCAL_ROBOT_ID", 1) == 2 else 1
ISAACLAB_PEER_ROBOT_ID = _peer_robot_id(ISAACLAB_LOCAL_ROBOT_ID)
ISAACLAB_LOCAL_ROBOT_NAME = _robot_name(ISAACLAB_LOCAL_ROBOT_ID)
ISAACLAB_PEER_ROBOT_NAME = _robot_name(ISAACLAB_PEER_ROBOT_ID)
_object_sync_role = os.environ.get("ISAACLAB_OBJECT_SYNC_ROLE", "auto").strip().lower()
if _object_sync_role == "auto":
    ZMQ_SYNC_ROLE = "publisher" if ISAACLAB_LOCAL_ROBOT_ID == 1 else "subscriber"
elif _object_sync_role in {"publisher", "subscriber", "none"}:
    ZMQ_SYNC_ROLE = _object_sync_role
else:
    ZMQ_SYNC_ROLE = "publisher"
ZMQ_SYNC_ENDPOINT = os.environ.get(
    "ISAACLAB_OBJECT_SYNC_ENDPOINT",
    f"tcp://{_windows_isaaclab_ip(1, '127.0.0.1')}:15555",
)

##
# Scene definition
##


def _find_gr00t_g1_43dof_usd() -> str:
    """Resolve the GR00T G1 43-DoF USD used by the sim2sim viewer."""

    candidates = []
    if "GR00T_WBC_ROOT" in os.environ:
        candidates.append(Path(os.environ["GR00T_WBC_ROOT"]).expanduser())
    candidates.extend(
        [
            Path("F:/ISAACWholeBody/GR00T-WholeBodyControl"),
            Path(__file__).resolve().parents[6] / "GR00T-WholeBodyControl",
            Path.cwd() / "GR00T-WholeBodyControl",
        ]
    )
    for root in candidates:
        for usd_name in (
            "g1_43dof_isaaclab_nomdl.usd",
            "g1_43dof.usd",
            "g1_43dof_isaaclab_no_material.usda",
            "g1_43dof_isaaclab_nomdl.usda",
            "g1_43dof_s3.usda",
        ):
            usd_path = root / "gear_sonic/data/robots/g1" / usd_name
            if usd_path.exists():
                return str(usd_path.resolve())
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate GR00T G1 43-DoF USD. Set GR00T_WBC_ROOT to the GR00T-WholeBodyControl path. "
        f"Searched:\n  {searched}"
    )


G1_43DOF_GR00T_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        usd_path=_find_gr00t_g1_43dof_usd(),
        activate_contact_sensors=False,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.70), roughness=0.55),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "left_elbow_joint": 0.6,
            "right_elbow_joint": 0.6,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.5,
                ".*_hip_roll_joint": 2.5,
                ".*_hip_pitch_joint": 2.5,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
            saturation_effort=180.0,
        ),
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.2,
                ".*_ankle_roll_joint": 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            saturation_effort=80.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            stiffness={
                "waist_yaw_joint": 5000.0,
                "waist_roll_joint": 5000.0,
                "waist_pitch_joint": 5000.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100,
            stiffness=3000.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.001,
                ".*_elbow_.*": 0.001,
                ".*_wrist_.*_joint": 0.001,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*",
                ".*_hand_middle_.*",
                ".*_hand_thumb_.*",
            ],
            effort_limit_sim=60.0,
            velocity_limit_sim=20.0,
            stiffness=80.0,
            damping=4.0,
            armature=0.001,
        ),
    },
)
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -1000.66], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.58, 0.54), roughness=0.65),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, -100.76], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.08, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.4,
                dynamic_friction=1.1,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.32, 0.78), roughness=0.4),
        ),
    )
    # 本地仓库背景
    # background = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Background",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[-3.60667,-0.64341, 0], rot=[0.7071, 0.0, 0.0, 0.7071]),
    #     spawn=UsdFileCfg(
    #         usd_path=os.path.join(os.path.dirname(__file__), "warehouse.usd"),
    #     ),
    # )
    # Humanoid robots from the GR00T sim2sim viewer asset.
    # ID 1 stays at the simulation origin; ID 2 is shifted on +Y to avoid overlap.
    robot_1: ArticulationCfg = G1_43DOF_GR00T_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=G1_43DOF_GR00T_CFG.init_state.replace(pos=(0.0, 0.0, 0.78)),
    )
    robot_2: ArticulationCfg = G1_43DOF_GR00T_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        init_state=G1_43DOF_GR00T_CFG.init_state.replace(pos=(0.0, 1.5, 0.78)),
    )
    test_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.78886, 1.17033, 0.845],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.32, 0.22, 0.24),
            rigid_props=(
                sim_utils.RigidBodyPropertiesCfg()
                if ZMQ_SYNC_ROLE != "subscriber"
                else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.18), roughness=0.7),
        ),
    )

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

    # Body/root streams are consumed independently for both robot IDs as physics-tracked references.
    mujoco_g1_mirror_1 = MuJoCoG1MirrorActionCfg(
        asset_name="robot_1",
        transport=os.environ.get("ISAACLAB_G1_TRANSPORT", "zmq"),
        zmq_host=_ubuntu_sender_ip(1, _isaac_robot_env(1, "ZMQ_HOST", "192.168.10.230")),
        zmq_port=_isaac_robot_env_int(1, "ZMQ_PORT", 5557),
        zmq_topic=_isaac_robot_env(1, "ZMQ_TOPIC", "g1_1_debug"),
        root_zmq_host=_ubuntu_sender_ip(
            1,
            _isaac_robot_env(1, "ROOT_ZMQ_HOST", _isaac_robot_env(1, "ZMQ_HOST", "192.168.10.230")),
        ),
        root_zmq_port=_isaac_robot_env_int(1, "ROOT_ZMQ_PORT", 5558),
        root_zmq_topic=_isaac_robot_env(1, "ROOT_ZMQ_TOPIC", "g1_1_root"),
        udp_bind_host=_isaac_robot_env(1, "UDP_BIND_HOST", "0.0.0.0"),
        udp_port=_isaac_robot_env_int(1, "UDP_PORT", 5557),
        udp_topic=_isaac_robot_env(1, "UDP_TOPIC", "g1_1_debug"),
        udp_rcvbuf=_isaac_robot_env_int(1, "UDP_RCVBUF", 262144),
        root_udp_bind_host=_isaac_robot_env(1, "ROOT_UDP_BIND_HOST", "0.0.0.0"),
        root_udp_port=_isaac_robot_env_int(1, "ROOT_UDP_PORT", 5558),
        root_udp_topic=_isaac_robot_env(1, "ROOT_UDP_TOPIC", "g1_1_root"),
        root_udp_rcvbuf=_isaac_robot_env_int(1, "ROOT_UDP_RCVBUF", 262144),
        zmq_pose_source=_isaac_robot_env(1, "POSE_SOURCE", "target"),
        root_motion_mode="source",
        root_zmq_required=True,
        root_position_mode="relative",
        write_root_state=_isaac_robot_env_bool(1, "WRITE_ROOT_STATE", False),
        write_body_joint_state=_isaac_robot_env_bool(1, "WRITE_BODY_JOINT_STATE", False),
        write_hand_joint_state=_isaac_robot_env_bool(1, "WRITE_HAND_JOINT_STATE", False),
        body_joint_target_max_delta=_isaac_robot_env_float(1, "BODY_JOINT_TARGET_MAX_DELTA", 0.08),
        hand_joint_target_max_delta=_isaac_robot_env_float(1, "HAND_JOINT_TARGET_MAX_DELTA", 0.20),
        use_source_joint_velocity=_isaac_robot_env_bool(1, "USE_SOURCE_JOINT_VELOCITY", True),
        mirror_hands=False,
        controller_gripper_enabled=False,
        ground_lock=_isaac_robot_env_bool(1, "GROUND_LOCK", False),
    )
    mujoco_g1_mirror_2 = MuJoCoG1MirrorActionCfg(
        asset_name="robot_2",
        transport=os.environ.get("ISAACLAB_G1_TRANSPORT", "zmq"),
        zmq_host=_ubuntu_sender_ip(2, _isaac_robot_env(2, "ZMQ_HOST", "192.168.10.231")),
        zmq_port=_isaac_robot_env_int(2, "ZMQ_PORT", 5567),
        zmq_topic=_isaac_robot_env(2, "ZMQ_TOPIC", "g1_2_debug"),
        root_zmq_host=_ubuntu_sender_ip(
            2,
            _isaac_robot_env(2, "ROOT_ZMQ_HOST", _isaac_robot_env(2, "ZMQ_HOST", "192.168.10.231")),
        ),
        root_zmq_port=_isaac_robot_env_int(2, "ROOT_ZMQ_PORT", 5568),
        root_zmq_topic=_isaac_robot_env(2, "ROOT_ZMQ_TOPIC", "g1_2_root"),
        udp_bind_host=_isaac_robot_env(2, "UDP_BIND_HOST", "0.0.0.0"),
        udp_port=_isaac_robot_env_int(2, "UDP_PORT", 5567),
        udp_topic=_isaac_robot_env(2, "UDP_TOPIC", "g1_2_debug"),
        udp_rcvbuf=_isaac_robot_env_int(2, "UDP_RCVBUF", 262144),
        root_udp_bind_host=_isaac_robot_env(2, "ROOT_UDP_BIND_HOST", "0.0.0.0"),
        root_udp_port=_isaac_robot_env_int(2, "ROOT_UDP_PORT", 5568),
        root_udp_topic=_isaac_robot_env(2, "ROOT_UDP_TOPIC", "g1_2_root"),
        root_udp_rcvbuf=_isaac_robot_env_int(2, "ROOT_UDP_RCVBUF", 262144),
        zmq_pose_source=_isaac_robot_env(2, "POSE_SOURCE", "target"),
        root_motion_mode="source",
        root_zmq_required=True,
        root_position_mode="relative",
        write_root_state=_isaac_robot_env_bool(2, "WRITE_ROOT_STATE", False),
        write_body_joint_state=_isaac_robot_env_bool(2, "WRITE_BODY_JOINT_STATE", False),
        write_hand_joint_state=_isaac_robot_env_bool(2, "WRITE_HAND_JOINT_STATE", False),
        body_joint_target_max_delta=_isaac_robot_env_float(2, "BODY_JOINT_TARGET_MAX_DELTA", 0.08),
        hand_joint_target_max_delta=_isaac_robot_env_float(2, "HAND_JOINT_TARGET_MAX_DELTA", 0.20),
        use_source_joint_velocity=_isaac_robot_env_bool(2, "USE_SOURCE_JOINT_VELOCITY", True),
        mirror_hands=False,
        controller_gripper_enabled=False,
        ground_lock=_isaac_robot_env_bool(2, "GROUND_LOCK", False),
    )
    local_gripper = G1GripperSyncActionCfg(
        asset_name=ISAACLAB_LOCAL_ROBOT_NAME,
        mode="local_publish",
        robot_id=ISAACLAB_LOCAL_ROBOT_ID,
        transport="zmq",
        zmq_host=_isaac_robot_env(
            ISAACLAB_LOCAL_ROBOT_ID,
            "GRIPPER_ZMQ_HOST",
            _windows_isaaclab_ip(ISAACLAB_LOCAL_ROBOT_ID, "127.0.0.1"),
        ),
        zmq_port=_isaac_robot_env_int(
            ISAACLAB_LOCAL_ROBOT_ID,
            "GRIPPER_ZMQ_PORT",
            5571 if ISAACLAB_LOCAL_ROBOT_ID == 1 else 5572,
        ),
        zmq_topic=_isaac_robot_env(
            ISAACLAB_LOCAL_ROBOT_ID,
            "GRIPPER_ZMQ_TOPIC",
            f"g1_{ISAACLAB_LOCAL_ROBOT_ID}_gripper",
        ),
        timeout=_env_float("ISAACLAB_G1_GRIPPER_TIMEOUT_S", 0.5),
        controller_gripper_finger_close_angle=1.8,
        controller_gripper_thumb_1_angle=1.1,
        controller_gripper_thumb_2_angle=1.8,
        controller_gripper_action_alpha=1.0,
        controller_gripper_use_soft_limits=False,
        write_joint_state=_env_bool("ISAACLAB_G1_GRIPPER_WRITE_JOINT_STATE", False),
        target_max_delta=_env_float("ISAACLAB_G1_GRIPPER_TARGET_MAX_DELTA", 0.20),
    )
    remote_gripper = G1GripperSyncActionCfg(
        asset_name=ISAACLAB_PEER_ROBOT_NAME,
        mode="remote_subscribe",
        robot_id=ISAACLAB_PEER_ROBOT_ID,
        transport="zmq",
        zmq_host=_isaac_robot_env(
            ISAACLAB_PEER_ROBOT_ID,
            "GRIPPER_ZMQ_HOST",
            _windows_isaaclab_ip(ISAACLAB_PEER_ROBOT_ID, "127.0.0.1"),
        ),
        zmq_port=_isaac_robot_env_int(
            ISAACLAB_PEER_ROBOT_ID,
            "GRIPPER_ZMQ_PORT",
            5571 if ISAACLAB_PEER_ROBOT_ID == 1 else 5572,
        ),
        zmq_topic=_isaac_robot_env(
            ISAACLAB_PEER_ROBOT_ID,
            "GRIPPER_ZMQ_TOPIC",
            f"g1_{ISAACLAB_PEER_ROBOT_ID}_gripper",
        ),
        timeout=_env_float("ISAACLAB_G1_GRIPPER_TIMEOUT_S", 0.5),
        controller_gripper_use_soft_limits=False,
        write_joint_state=_env_bool("ISAACLAB_G1_GRIPPER_WRITE_JOINT_STATE", False),
        target_max_delta=_env_float("ISAACLAB_G1_GRIPPER_TARGET_MAX_DELTA", 0.20),
    )
    object_sync = ZmqObjectSyncActionCfg(asset_name="test_box", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)


@configclass
class ObservationsCfg:
    """Empty observation manager config.

    The scene is used for live robot synchronization, not policy rollout or data recording,
    so no ``policy`` observation group is registered.
    """

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""



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
        anchor_pos=(0.0, 0.0, 0.0),
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
        # The default Isaac Lab GPU PhysX buffers target large batched training scenes.
        # This task is a single-env XR mirror, so smaller buffers avoid VRAM exhaustion on 8 GB GPUs.
        self.sim.physx.gpu_max_rigid_contact_count = 2**20
        self.sim.physx.gpu_max_rigid_patch_count = 2**14
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**16
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
        self.sim.physx.gpu_collision_stack_size = 2**24
        self.sim.physx.gpu_heap_capacity = 2**24
        self.sim.physx.gpu_temp_buffer_capacity = 2**22

        local_robot_prim = _robot_prim_name(ISAACLAB_LOCAL_ROBOT_ID)
        self.xr.anchor_prim_path = f"/World/envs/env_0/{local_robot_prim}/head_link"
        self.xr.anchor_rotation_prim_path = f"/World/envs/env_0/{local_robot_prim}/pelvis"
        self.xr.fixed_anchor_height = False
        # Anchor XR to the robot head position, but use the pelvis as the stable robot yaw reference.
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
        self.xr.recenter_yaw_button = ("/user/hand/right", "b")
        self.xr.recenter_yaw_button_event = "release"
        self.xr.recenter_anchor_forward_axis = (-1.0, 0.0, 0.0)
        self.xr.recenter_headset_forward_axis = (0.0, -1.0, 0.0)
        self.xr.recenter_headset_fallback_axis = (1.0, 0.0, 0.0)

        teleop_device = "cpu"
        self.teleop_devices = DevicesCfg(
            devices={
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=[
                        G1GripperMotionControllerRetargeterCfg(
                            sim_device=teleop_device,
                            use_right_b_button=False,
                        ),
                    ],
                    sim_device=teleop_device,
                    xr_cfg=self.xr,
                ),
            }
        )
