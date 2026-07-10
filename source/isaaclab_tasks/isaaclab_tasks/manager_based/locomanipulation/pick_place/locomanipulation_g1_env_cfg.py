# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UrdfFileCfg, UsdFileCfg
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


def _expand_config_refs(values: dict[str, str]) -> dict[str, str]:
    expanded = dict(values)
    for _ in range(10):
        changed = False
        next_values = {}
        for key, value in expanded.items():
            next_value = _ENV_REF_RE.sub(
                lambda match: expanded.get(match.group(1) or match.group(2), ""),
                value,
            )
            next_values[key] = next_value
            changed |= next_value != value
        expanded = next_values
        if not changed:
            break
    return expanded


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
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
            values[key] = value
    return _expand_config_refs(values)


def _load_default_network_config() -> tuple[dict[str, str], Path | None]:
    candidates = [
        Path(__file__).resolve().parents[6] / "scripts/gr00t_wbc/g1_udp_network.env",
    ]
    for path in candidates:
        values = _load_env_file(path)
        if values:
            print(f"[INFO] IsaacLab G1 config loaded: {path}")
            return values, path
    print("[WARN] IsaacLab G1 config file was not found; using built-in defaults.")
    return {}, None


_G1_NETWORK_CONFIG, _G1_NETWORK_CONFIG_PATH = _load_default_network_config()


def _cfg_value(name: str, default: str | None = None) -> str | None:
    return _G1_NETWORK_CONFIG.get(name, default)


def _cfg_int(name: str, default: int) -> int:
    try:
        return int(_cfg_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def _cfg_float(name: str, default: float) -> float:
    try:
        return float(_cfg_value(name, str(default)))
    except (TypeError, ValueError):
        return default


def _cfg_bool_value(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_bool(name: str, default: bool) -> bool:
    return _cfg_bool_value(_cfg_value(name), default)


def _isaac_robot_cfg_value(robot_id: int, suffix: str) -> str | None:
    robot_key = f"ISAACLAB_G1_{robot_id}_{suffix}"
    generic_key = f"ISAACLAB_G1_{suffix}"
    return _G1_NETWORK_CONFIG.get(robot_key, _G1_NETWORK_CONFIG.get(generic_key))


def _isaac_robot_cfg(robot_id: int, suffix: str, default: str) -> str:
    value = _isaac_robot_cfg_value(robot_id, suffix)
    return value if value is not None else default


def _isaac_robot_cfg_int(robot_id: int, suffix: str, default: int) -> int:
    try:
        return int(_isaac_robot_cfg(robot_id, suffix, str(default)))
    except (TypeError, ValueError):
        return default


def _isaac_robot_cfg_float(robot_id: int, suffix: str, default: float) -> float:
    try:
        return float(_isaac_robot_cfg(robot_id, suffix, str(default)))
    except (TypeError, ValueError):
        return default


def _isaac_robot_cfg_bool(robot_id: int, suffix: str, default: bool) -> bool:
    return _cfg_bool_value(_isaac_robot_cfg_value(robot_id, suffix), default)


def _isaac_robot_sync_mode(robot_id: int) -> str:
    mode = _isaac_robot_cfg(robot_id, "LOCOMOTION_SYNC_MODE", "mirror").strip().lower()
    if mode not in {"mirror", "hybrid", "physics", "custom"}:
        print(f"[WARN] Unsupported ISAACLAB_G1_{robot_id}_LOCOMOTION_SYNC_MODE={mode!r}; using mirror.")
        return "mirror"
    return mode


def _isaac_robot_write_root_state(robot_id: int) -> bool:
    mode = _isaac_robot_sync_mode(robot_id)
    if mode == "custom":
        return _isaac_robot_cfg_bool(robot_id, "WRITE_ROOT_STATE", True)
    if mode in {"mirror", "hybrid"}:
        return True
    return False


def _isaac_robot_write_body_joint_state(robot_id: int) -> bool:
    mode = _isaac_robot_sync_mode(robot_id)
    if mode == "custom":
        return _isaac_robot_cfg_bool(robot_id, "WRITE_BODY_JOINT_STATE", True)
    return mode == "mirror"


def _isaac_robot_write_hand_joint_state(robot_id: int) -> bool:
    return _isaac_robot_cfg_bool(robot_id, "WRITE_HAND_JOINT_STATE", False)


def _ubuntu_sender_ip(robot_id: int, default: str) -> str:
    return _cfg_value(f"UBUNTU_ROBOT_{robot_id}_SENDER_IP", _cfg_value(f"G1_{robot_id}_SENDER_IP", default))


def _windows_isaaclab_ip(robot_id: int, default: str) -> str:
    return _cfg_value(
        f"WINDOWS_ROBOT_{robot_id}_ISAACLAB_IP",
        _cfg_value(f"ISAACLAB_G1_{robot_id}_HOST_IP", default),
    )


def _robot_name(robot_id: int) -> str:
    return f"robot_{robot_id}"


def _robot_prim_name(robot_id: int) -> str:
    return f"Robot_{robot_id}"


def _peer_robot_id(robot_id: int) -> int:
    return 2 if robot_id == 1 else 1


def _grasp_object_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    if ZMQ_SYNC_ROLE == "subscriber":
        return sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        enable_gyroscopic_forces=True,
        linear_damping=_cfg_float("ISAACLAB_GRASP_OBJECT_LINEAR_DAMPING", 0.8),
        angular_damping=_cfg_float("ISAACLAB_GRASP_OBJECT_ANGULAR_DAMPING", 2.0),
        max_linear_velocity=_cfg_float("ISAACLAB_GRASP_OBJECT_MAX_LINEAR_VELOCITY", 3.0),
        max_angular_velocity=_cfg_float("ISAACLAB_GRASP_OBJECT_MAX_ANGULAR_VELOCITY", 180.0),
        solver_position_iteration_count=_cfg_int("ISAACLAB_GRASP_OBJECT_SOLVER_POSITION_ITERATIONS", 12),
        solver_velocity_iteration_count=_cfg_int("ISAACLAB_GRASP_OBJECT_SOLVER_VELOCITY_ITERATIONS", 4),
        max_depenetration_velocity=_cfg_float("ISAACLAB_GRASP_OBJECT_MAX_DEPENETRATION_VELOCITY", 2.0),
        max_contact_impulse=_cfg_float("ISAACLAB_GRASP_OBJECT_MAX_CONTACT_IMPULSE", 20.0),
        sleep_threshold=_cfg_float("ISAACLAB_GRASP_OBJECT_SLEEP_THRESHOLD", 0.005),
        stabilization_threshold=_cfg_float("ISAACLAB_GRASP_OBJECT_STABILIZATION_THRESHOLD", 0.001),
    )


GRASP_BOX_SIZE = (
    _cfg_float("ISAACLAB_GRASP_BOX_SIZE_X", 0.32),
    _cfg_float("ISAACLAB_GRASP_BOX_SIZE_Y", 0.22),
    _cfg_float("ISAACLAB_GRASP_BOX_SIZE_Z", 0.24),
)
GRASP_BOX_X = _cfg_float("ISAACLAB_GRASP_BOX_X", 0)
GRASP_BOX_Y = _cfg_float("ISAACLAB_GRASP_BOX_Y", 0)
GRASP_BOX_GROUND_Z = _cfg_float("ISAACLAB_GRASP_BOX_GROUND_Z", 0.0)
GRASP_BOX_VERTICAL_GAP = _cfg_float("ISAACLAB_GRASP_BOX_VERTICAL_GAP", 0.002)


def _grasp_box_position(stack_index: int) -> list[float]:
    return [
        GRASP_BOX_X,
        GRASP_BOX_Y,
        GRASP_BOX_GROUND_Z + 0.5 * GRASP_BOX_SIZE[2] + stack_index * (GRASP_BOX_SIZE[2] + GRASP_BOX_VERTICAL_GAP),
    ]


def _grasp_box_cfg(prim_name: str, stack_index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_grasp_box_position(stack_index),
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=GRASP_BOX_SIZE,
            rigid_props=_grasp_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=_cfg_float("ISAACLAB_GRASP_OBJECT_MASS", 0.45)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=_cfg_float("ISAACLAB_GRASP_OBJECT_CONTACT_OFFSET", 0.006),
                rest_offset=_cfg_float("ISAACLAB_GRASP_OBJECT_REST_OFFSET", 0.0),
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=_cfg_float("ISAACLAB_GRASP_OBJECT_STATIC_FRICTION", 2.0),
                dynamic_friction=_cfg_float("ISAACLAB_GRASP_OBJECT_DYNAMIC_FRICTION", 1.6),
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.18), roughness=0.7),
        ),
    )


ISAACLAB_LOCAL_ROBOT_ID = 2 if _cfg_int("ISAACLAB_LOCAL_ROBOT_ID", 1) == 2 else 1
ISAACLAB_PEER_ROBOT_ID = _peer_robot_id(ISAACLAB_LOCAL_ROBOT_ID)
ISAACLAB_LOCAL_ROBOT_NAME = _robot_name(ISAACLAB_LOCAL_ROBOT_ID)
ISAACLAB_PEER_ROBOT_NAME = _robot_name(ISAACLAB_PEER_ROBOT_ID)
_object_sync_role = _cfg_value("ISAACLAB_OBJECT_SYNC_ROLE", "auto").strip().lower()
if _object_sync_role == "auto":
    ZMQ_SYNC_ROLE = "publisher" if ISAACLAB_LOCAL_ROBOT_ID == 1 else "subscriber"
elif _object_sync_role in {"publisher", "subscriber", "none"}:
    ZMQ_SYNC_ROLE = _object_sync_role
else:
    ZMQ_SYNC_ROLE = "publisher"
ZMQ_SYNC_ENDPOINT = _cfg_value(
    "ISAACLAB_OBJECT_SYNC_ENDPOINT",
    f"tcp://{_windows_isaaclab_ip(1, '127.0.0.1')}:15555",
)

##
# Scene definition
##


G1_HAND_MODEL = str(_cfg_value("ISAACLAB_G1_HAND_MODEL", "brainco")).strip().lower()
if G1_HAND_MODEL not in {"brainco", "inspire", "trihand"}:
    print(f"[WARN] Unsupported ISAACLAB_G1_HAND_MODEL={G1_HAND_MODEL!r}; using brainco.")
    G1_HAND_MODEL = "brainco"


def _find_gr00t_g1_43dof_usd() -> str:
    """Resolve the GR00T G1 43-DoF USD used by the sim2sim viewer."""

    candidates = []
    gr00t_root = _cfg_value("GR00T_WBC_ROOT")
    if gr00t_root:
        candidates.append(Path(gr00t_root).expanduser())
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
        "Could not locate GR00T G1 43-DoF USD. Set GR00T_WBC_ROOT in g1_udp_network.env "
        "to the GR00T-WholeBodyControl path. "
        f"Searched:\n  {searched}"
    )


def _find_inspire_g1_usd() -> str:
    configured = _cfg_value("ISAACLAB_G1_INSPIRE_USD_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path(
                "D:/Omniverse/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/"
                "configuration/inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_retarget_inspire_white_physics.usd"
            ),
            Path(
                "D:/Omniverse/isaacsim_assets/Assets/IsaacLab/Robots/Unitree/G1/"
                "g1_29dof_inspire_hand.usd"
            ),
        ]
    )
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate G1 Inspire hand USD. Set ISAACLAB_G1_INSPIRE_USD_PATH in g1_udp_network.env. "
        f"Searched:\n  {searched}"
    )


def _find_brainco_g1_urdf() -> str:
    configured = _cfg_value("ISAACLAB_G1_BRAINCO_URDF_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(
        Path("F:/ISAACWholeBody/unitree_ros/robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf")
    )
    for path in candidates:
        if path.exists():
            return str(path.resolve())
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate G1 BrainCo hand URDF. Set ISAACLAB_G1_BRAINCO_URDF_PATH in g1_udp_network.env. "
        f"Searched:\n  {searched}"
    )


def _find_active_g1_usd() -> str:
    if G1_HAND_MODEL == "inspire":
        return _find_inspire_g1_usd()
    return _find_gr00t_g1_43dof_usd()


def _g1_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


def _g1_articulation_props() -> sim_utils.ArticulationRootPropertiesCfg:
    return sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        fix_root_link=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


def _g1_spawn_cfg() -> UrdfFileCfg | UsdFileCfg:
    activate_contact_sensors = _cfg_bool("ISAACLAB_G1_ACTIVATE_CONTACT_SENSORS", G1_HAND_MODEL in {"brainco", "inspire"})
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.70), roughness=0.55)
    if G1_HAND_MODEL == "brainco":
        usd_dir = _cfg_value("ISAACLAB_G1_BRAINCO_USD_DIR")
        usd_file_name = _cfg_value("ISAACLAB_G1_BRAINCO_USD_FILE_NAME")
        return UrdfFileCfg(
            asset_path=_find_brainco_g1_urdf(),
            usd_dir=usd_dir if usd_dir else None,
            usd_file_name=usd_file_name if usd_file_name else None,
            force_usd_conversion=_cfg_bool("ISAACLAB_G1_BRAINCO_FORCE_USD_CONVERSION", False),
            make_instanceable=_cfg_bool("ISAACLAB_G1_BRAINCO_MAKE_INSTANCEABLE", True),
            fix_base=_cfg_bool("ISAACLAB_G1_BRAINCO_FIX_BASE", False),
            merge_fixed_joints=_cfg_bool("ISAACLAB_G1_BRAINCO_MERGE_FIXED_JOINTS", False),
            convert_mimic_joints_to_normal_joints=_cfg_bool(
                "ISAACLAB_G1_BRAINCO_CONVERT_MIMIC_JOINTS", False
            ),
            collision_from_visuals=_cfg_bool("ISAACLAB_G1_BRAINCO_COLLISION_FROM_VISUALS", False),
            collider_type=_cfg_value("ISAACLAB_G1_BRAINCO_COLLIDER_TYPE", "convex_hull"),
            self_collision=_cfg_bool("ISAACLAB_G1_BRAINCO_SELF_COLLISION", False),
            replace_cylinders_with_capsules=_cfg_bool("ISAACLAB_G1_BRAINCO_REPLACE_CYLINDERS_WITH_CAPSULES", False),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            activate_contact_sensors=activate_contact_sensors,
            visual_material=visual_material,
            rigid_props=_g1_rigid_props(),
            articulation_props=_g1_articulation_props(),
        )
    return UsdFileCfg(
        usd_path=_find_active_g1_usd(),
        activate_contact_sensors=activate_contact_sensors,
        visual_material=visual_material,
        rigid_props=_g1_rigid_props(),
        articulation_props=_g1_articulation_props(),
    )


def _g1_initial_joint_pos() -> dict[str, float]:
    joint_pos = {
        ".*_hip_pitch_joint": -0.10,
        ".*_knee_joint": 0.30,
        ".*_ankle_pitch_joint": -0.20,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
    }
    if G1_HAND_MODEL == "inspire":
        joint_pos.update(
            {
                "L_.*_joint": 0.0,
                "R_.*_joint": 0.0,
                "L_thumb_proximal_yaw_joint": -1.57,
                "R_thumb_proximal_yaw_joint": -1.57,
            }
        )
    elif G1_HAND_MODEL == "brainco":
        joint_pos.update(
            {
                ".*_thumb_metacarpal_joint": 0.0,
                ".*_thumb_proximal_joint": 0.0,
                ".*_index_proximal_joint": 0.0,
                ".*_middle_proximal_joint": 0.0,
                ".*_ring_proximal_joint": 0.0,
                ".*_pinky_proximal_joint": 0.0,
            }
        )
    return joint_pos


def _g1_hand_joint_expr() -> list[str]:
    if G1_HAND_MODEL == "brainco":
        return [
            ".*_thumb_metacarpal_joint",
            ".*_thumb_proximal_joint",
            ".*_index_proximal_joint",
            ".*_middle_proximal_joint",
            ".*_ring_proximal_joint",
            ".*_pinky_proximal_joint",
        ]
    if G1_HAND_MODEL == "inspire":
        return [
            ".*_index_.*",
            ".*_middle_.*",
            ".*_thumb_.*",
            ".*_ring_.*",
            ".*_pinky_.*",
        ]
    return [
        ".*_hand_index_.*",
        ".*_hand_middle_.*",
        ".*_hand_thumb_.*",
    ]


G1_43DOF_GR00T_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=_g1_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        rot=(0.7071, 0.0, 0.0, 0.7071),
        joint_pos=_g1_initial_joint_pos(),
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
            joint_names_expr=_g1_hand_joint_expr(),
            effort_limit_sim=_cfg_float("ISAACLAB_G1_HAND_EFFORT_LIMIT", 120.0),
            velocity_limit_sim=_cfg_float("ISAACLAB_G1_HAND_VELOCITY_LIMIT", 16.0),
            stiffness=_cfg_float("ISAACLAB_G1_HAND_STIFFNESS", 260.0),
            damping=_cfg_float("ISAACLAB_G1_HAND_DAMPING", 12.0),
            armature=_cfg_float("ISAACLAB_G1_HAND_ARMATURE", 0.005),
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
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
    test_box = _grasp_box_cfg("TestBox", 0)
    test_box_2 = _grasp_box_cfg("TestBox_2", 1)
    test_box_3 = _grasp_box_cfg("TestBox_3", 2)
    test_box_4 = _grasp_box_cfg("TestBox_4", 3)
    test_box_5 = _grasp_box_cfg("TestBox_5", 4)

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


def _mujoco_g1_mirror_cfg(robot_id: int) -> MuJoCoG1MirrorActionCfg:
    sync_mode = _isaac_robot_sync_mode(robot_id)
    default_sender_ip = "192.168.10.230" if robot_id == 1 else "192.168.10.231"
    default_body_port = 5557 if robot_id == 1 else 5567
    default_root_port = 5558 if robot_id == 1 else 5568
    return MuJoCoG1MirrorActionCfg(
        asset_name=_robot_name(robot_id),
        transport=_cfg_value("ISAACLAB_G1_TRANSPORT", "zmq"),
        zmq_host=_ubuntu_sender_ip(robot_id, _isaac_robot_cfg(robot_id, "ZMQ_HOST", default_sender_ip)),
        zmq_port=_isaac_robot_cfg_int(robot_id, "ZMQ_PORT", default_body_port),
        zmq_topic=_isaac_robot_cfg(robot_id, "ZMQ_TOPIC", f"g1_{robot_id}_debug"),
        zmq_pose_source=_isaac_robot_cfg(robot_id, "POSE_SOURCE", "target"),
        root_zmq_host=_ubuntu_sender_ip(
            robot_id,
            _isaac_robot_cfg(
                robot_id,
                "ROOT_ZMQ_HOST",
                _isaac_robot_cfg(robot_id, "ZMQ_HOST", default_sender_ip),
            ),
        ),
        root_zmq_port=_isaac_robot_cfg_int(robot_id, "ROOT_ZMQ_PORT", default_root_port),
        root_zmq_topic=_isaac_robot_cfg(robot_id, "ROOT_ZMQ_TOPIC", f"g1_{robot_id}_root"),
        udp_bind_host=_isaac_robot_cfg(robot_id, "UDP_BIND_HOST", "0.0.0.0"),
        udp_port=_isaac_robot_cfg_int(robot_id, "UDP_PORT", default_body_port),
        udp_topic=_isaac_robot_cfg(robot_id, "UDP_TOPIC", f"g1_{robot_id}_debug"),
        udp_rcvbuf=_isaac_robot_cfg_int(robot_id, "UDP_RCVBUF", 262144),
        root_udp_bind_host=_isaac_robot_cfg(robot_id, "ROOT_UDP_BIND_HOST", "0.0.0.0"),
        root_udp_port=_isaac_robot_cfg_int(robot_id, "ROOT_UDP_PORT", default_root_port),
        root_udp_topic=_isaac_robot_cfg(robot_id, "ROOT_UDP_TOPIC", f"g1_{robot_id}_root"),
        root_udp_rcvbuf=_isaac_robot_cfg_int(robot_id, "ROOT_UDP_RCVBUF", 262144),
        locomotion_sync_mode=sync_mode,
        write_root_state=_isaac_robot_write_root_state(robot_id),
        write_body_joint_state=_isaac_robot_write_body_joint_state(robot_id),
        write_hand_joint_state=_isaac_robot_write_hand_joint_state(robot_id),
        use_source_joint_velocity=_isaac_robot_cfg_bool(robot_id, "USE_SOURCE_JOINT_VELOCITY", True),
        body_joint_target_max_delta=_isaac_robot_cfg_float(robot_id, "BODY_JOINT_TARGET_MAX_DELTA", 0.08),
        hand_joint_target_max_delta=_isaac_robot_cfg_float(robot_id, "HAND_JOINT_TARGET_MAX_DELTA", 0.20),
        hold_default_until_first_packet=_isaac_robot_cfg_bool(robot_id, "HOLD_DEFAULT_UNTIL_FIRST_PACKET", True),
        no_packet_debug_interval_s=_isaac_robot_cfg_float(robot_id, "NO_PACKET_DEBUG_INTERVAL_S", 1.0),
        root_motion_mode=_isaac_robot_cfg(robot_id, "ROOT_MOTION_MODE", "source"),
        root_zmq_required=_isaac_robot_cfg_bool(robot_id, "ROOT_ZMQ_REQUIRED", True),
        root_position_mode=_isaac_robot_cfg(robot_id, "ROOT_POSITION_MODE", "relative"),
        mirror_hands=_isaac_robot_cfg_bool(robot_id, "MIRROR_HANDS", False),
        controller_gripper_enabled=_isaac_robot_cfg_bool(robot_id, "CONTROLLER_GRIPPER_ENABLED", False),
        controller_gripper_finger_close_angle=_cfg_float("ISAACLAB_G1_GRIPPER_FINGER_CLOSE_ANGLE", 1.8),
        controller_gripper_thumb_yaw_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_ANGLE", 0.5),
        controller_gripper_thumb_1_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_1_ANGLE", 1.1),
        controller_gripper_thumb_2_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_2_ANGLE", 1.8),
        controller_gripper_finger_intermediate_scale=_cfg_float(
            "ISAACLAB_G1_GRIPPER_FINGER_INTERMEDIATE_SCALE", 0.75
        ),
        controller_gripper_ring_close_scale=_cfg_float("ISAACLAB_G1_GRIPPER_RING_CLOSE_SCALE", 1.0),
        controller_gripper_pinky_close_scale=_cfg_float("ISAACLAB_G1_GRIPPER_PINKY_CLOSE_SCALE", 0.85),
        controller_gripper_thumb_yaw_open_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_OPEN_ANGLE", -1.57),
        controller_gripper_thumb_yaw_closed_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_CLOSED_ANGLE", -0.45),
        controller_gripper_action_alpha=_cfg_float("ISAACLAB_G1_GRIPPER_ACTION_ALPHA", 1.0),
        controller_gripper_use_soft_limits=_cfg_bool("ISAACLAB_G1_GRIPPER_USE_SOFT_LIMITS", False),
        controller_gripper_write_joint_state=_cfg_bool("ISAACLAB_G1_GRIPPER_WRITE_JOINT_STATE", False),
        controller_gripper_target_max_delta=_cfg_float("ISAACLAB_G1_GRIPPER_TARGET_MAX_DELTA", 0.20),
        ground_lock=_isaac_robot_cfg_bool(robot_id, "GROUND_LOCK", False),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Body/root streams are mirrored independently for both robot IDs.
    mujoco_g1_mirror_1 = _mujoco_g1_mirror_cfg(1)
    mujoco_g1_mirror_2 = _mujoco_g1_mirror_cfg(2)
    local_gripper = G1GripperSyncActionCfg(
        asset_name=ISAACLAB_LOCAL_ROBOT_NAME,
        mode="local_publish",
        robot_id=ISAACLAB_LOCAL_ROBOT_ID,
        transport=_cfg_value("ISAACLAB_G1_GRIPPER_TRANSPORT", "zmq"),
        zmq_host=_isaac_robot_cfg(
            ISAACLAB_LOCAL_ROBOT_ID,
            "GRIPPER_ZMQ_HOST",
            _windows_isaaclab_ip(ISAACLAB_LOCAL_ROBOT_ID, "127.0.0.1"),
        ),
        zmq_port=_isaac_robot_cfg_int(
            ISAACLAB_LOCAL_ROBOT_ID,
            "GRIPPER_ZMQ_PORT",
            5571 if ISAACLAB_LOCAL_ROBOT_ID == 1 else 5572,
        ),
        zmq_topic=_isaac_robot_cfg(
            ISAACLAB_LOCAL_ROBOT_ID,
            "GRIPPER_ZMQ_TOPIC",
            f"g1_{ISAACLAB_LOCAL_ROBOT_ID}_gripper",
        ),
        timeout=_cfg_float("ISAACLAB_G1_GRIPPER_TIMEOUT_S", 0.5),
        controller_gripper_finger_close_angle=_cfg_float("ISAACLAB_G1_GRIPPER_FINGER_CLOSE_ANGLE", 1.8),
        controller_gripper_thumb_yaw_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_ANGLE", 0.5),
        controller_gripper_thumb_1_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_1_ANGLE", 1.1),
        controller_gripper_thumb_2_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_2_ANGLE", 1.8),
        controller_gripper_finger_intermediate_scale=_cfg_float(
            "ISAACLAB_G1_GRIPPER_FINGER_INTERMEDIATE_SCALE", 0.75
        ),
        controller_gripper_ring_close_scale=_cfg_float("ISAACLAB_G1_GRIPPER_RING_CLOSE_SCALE", 1.0),
        controller_gripper_pinky_close_scale=_cfg_float("ISAACLAB_G1_GRIPPER_PINKY_CLOSE_SCALE", 0.85),
        controller_gripper_thumb_yaw_open_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_OPEN_ANGLE", -1.57),
        controller_gripper_thumb_yaw_closed_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_CLOSED_ANGLE", -0.45),
        controller_gripper_action_alpha=_cfg_float("ISAACLAB_G1_GRIPPER_ACTION_ALPHA", 1.0),
        controller_gripper_use_soft_limits=_cfg_bool("ISAACLAB_G1_GRIPPER_USE_SOFT_LIMITS", False),
        write_joint_state=_cfg_bool("ISAACLAB_G1_GRIPPER_WRITE_JOINT_STATE", False),
        target_max_delta=_cfg_float("ISAACLAB_G1_GRIPPER_TARGET_MAX_DELTA", 0.20),
        publish_interval_s=_cfg_float("ISAACLAB_G1_GRIPPER_PUBLISH_INTERVAL_S", 0.0),
        debug_interval_s=_cfg_float("ISAACLAB_G1_GRIPPER_DEBUG_INTERVAL_S", 0.0),
    )
    remote_gripper = G1GripperSyncActionCfg(
        asset_name=ISAACLAB_PEER_ROBOT_NAME,
        mode="remote_subscribe",
        robot_id=ISAACLAB_PEER_ROBOT_ID,
        transport=_cfg_value("ISAACLAB_G1_GRIPPER_TRANSPORT", "zmq"),
        zmq_host=_isaac_robot_cfg(
            ISAACLAB_PEER_ROBOT_ID,
            "GRIPPER_ZMQ_HOST",
            _windows_isaaclab_ip(ISAACLAB_PEER_ROBOT_ID, "127.0.0.1"),
        ),
        zmq_port=_isaac_robot_cfg_int(
            ISAACLAB_PEER_ROBOT_ID,
            "GRIPPER_ZMQ_PORT",
            5571 if ISAACLAB_PEER_ROBOT_ID == 1 else 5572,
        ),
        zmq_topic=_isaac_robot_cfg(
            ISAACLAB_PEER_ROBOT_ID,
            "GRIPPER_ZMQ_TOPIC",
            f"g1_{ISAACLAB_PEER_ROBOT_ID}_gripper",
        ),
        timeout=_cfg_float("ISAACLAB_G1_GRIPPER_TIMEOUT_S", 0.5),
        controller_gripper_use_soft_limits=_cfg_bool("ISAACLAB_G1_GRIPPER_USE_SOFT_LIMITS", False),
        controller_gripper_finger_intermediate_scale=_cfg_float(
            "ISAACLAB_G1_GRIPPER_FINGER_INTERMEDIATE_SCALE", 0.75
        ),
        controller_gripper_ring_close_scale=_cfg_float("ISAACLAB_G1_GRIPPER_RING_CLOSE_SCALE", 1.0),
        controller_gripper_pinky_close_scale=_cfg_float("ISAACLAB_G1_GRIPPER_PINKY_CLOSE_SCALE", 0.85),
        controller_gripper_thumb_yaw_open_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_OPEN_ANGLE", -1.57),
        controller_gripper_thumb_yaw_closed_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_CLOSED_ANGLE", -0.45),
        write_joint_state=_cfg_bool("ISAACLAB_G1_GRIPPER_WRITE_JOINT_STATE", False),
        target_max_delta=_cfg_float("ISAACLAB_G1_GRIPPER_TARGET_MAX_DELTA", 0.20),
        publish_interval_s=_cfg_float("ISAACLAB_G1_GRIPPER_PUBLISH_INTERVAL_S", 0.0),
        debug_interval_s=_cfg_float("ISAACLAB_G1_GRIPPER_DEBUG_INTERVAL_S", 0.0),
    )
    object_sync = ZmqObjectSyncActionCfg(asset_name="test_box", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    object_sync_2 = ZmqObjectSyncActionCfg(asset_name="test_box_2", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    object_sync_3 = ZmqObjectSyncActionCfg(asset_name="test_box_3", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    object_sync_4 = ZmqObjectSyncActionCfg(asset_name="test_box_4", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    object_sync_5 = ZmqObjectSyncActionCfg(asset_name="test_box_5", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)


@configclass
class ObservationsCfg:
    """Empty observation manager config.

    The scene is used for live robot synchronization, not policy rollout or data recording,
    so no ``policy`` observation group is registered.
    """

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    # )

    # success = DoneTerm(
    #     func=manip_mdp.task_done_pick_place,
    #     params={"task_link_name": "right_wrist_yaw_link", "robot_cfg": SceneEntityCfg(ISAACLAB_LOCAL_ROBOT_NAME)},
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
        xr_anchor_link = _cfg_value("ISAACLAB_XR_ANCHOR_LINK", "mid360_link")
        xr_anchor_rotation_link = _cfg_value("ISAACLAB_XR_ANCHOR_ROTATION_LINK", "pelvis")
        self.xr.anchor_prim_path = f"/World/envs/env_0/{local_robot_prim}/{xr_anchor_link}"
        self.xr.anchor_rotation_prim_path = f"/World/envs/env_0/{local_robot_prim}/{xr_anchor_rotation_link}"
        self.xr.fixed_anchor_height = False
        # Anchor XR to the configured robot sensor link, but use the pelvis as the stable yaw reference by default.
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
