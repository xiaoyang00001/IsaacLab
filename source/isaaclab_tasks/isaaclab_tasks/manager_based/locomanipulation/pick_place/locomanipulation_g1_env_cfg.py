# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import MuJoCoG1MirrorActionCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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


def _cfg_str_list_value(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _cfg_pattern_float_dict_value(value: str | None, default: dict[str, float]) -> dict[str, float]:
    if value is None:
        return dict(default)
    value = value.strip()
    if not value:
        return {}

    result: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            pattern, raw_scale = item.rsplit(":", 1)
        elif "=" in item:
            pattern, raw_scale = item.rsplit("=", 1)
        else:
            print(f"[WARN] Ignoring malformed joint target scale override {item!r}; expected pattern:scale.")
            continue
        pattern = pattern.strip()
        try:
            result[pattern] = float(raw_scale.strip())
        except ValueError:
            print(f"[WARN] Ignoring malformed joint target scale override {item!r}; scale is not a float.")
    return result


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


def _isaac_robot_cfg_str_list(robot_id: int, suffix: str, default: list[str]) -> list[str]:
    return _cfg_str_list_value(_isaac_robot_cfg_value(robot_id, suffix), default)


def _isaac_robot_cfg_pattern_float_dict(robot_id: int, suffix: str, default: dict[str, float]) -> dict[str, float]:
    return _cfg_pattern_float_dict_value(_isaac_robot_cfg_value(robot_id, suffix), default)


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


def _robot_name(robot_id: int) -> str:
    return f"robot_{robot_id}"


GRASP_BOX_USD_PATH = Path(__file__).resolve().parent / "assets/grasp_box.usda"
# The packing-table asset top was at world z=0.6996 when the table was spawned
# at z=-0.3. The table now spawns at z=0, so its top is at z=0.9996. Place the
# 0.24 m tall box 2 mm above that surface.
GRASP_BOX_INITIAL_POS = (0.0, 0.5, 1.1216)


def _grasp_box_cfg(prim_name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=GRASP_BOX_INITIAL_POS,
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(usd_path=str(GRASP_BOX_USD_PATH)),
    )


##
# Scene definition
##


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


G1_BODY_STATE_WRITE_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
]
"""Mirrored joints that are allowed to be hard-written into PhysX for stable walking."""

def _g1_robot_rigid_props() -> sim_utils.RigidBodyPropertiesCfg | None:
    if _cfg_bool("ISAACLAB_G1_USE_USD_RIGID_PROPS", True):
        return None
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=_cfg_float("ISAACLAB_G1_RIGID_MAX_DEPENETRATION_VELOCITY", 1.0),
    )


G1_43DOF_GR00T_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        usd_path=_find_gr00t_g1_43dof_usd(),
        activate_contact_sensors=False,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.72, 0.70), roughness=0.55),
        rigid_props=_g1_robot_rigid_props(),
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
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.3,
            "right_shoulder_roll_joint": -0.3,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.0,
            "right_elbow_joint": 1.0,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
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
            effort_limit_sim=_cfg_float("ISAACLAB_G1_ARM_EFFORT_LIMIT", 80.0),
            velocity_limit_sim=_cfg_float("ISAACLAB_G1_ARM_VELOCITY_LIMIT", 12.0),
            stiffness=_cfg_float("ISAACLAB_G1_ARM_STIFFNESS", 600.0),
            damping=_cfg_float("ISAACLAB_G1_ARM_DAMPING", 30.0),
            armature=_cfg_float("ISAACLAB_G1_ARM_ARMATURE", 0.01),
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*",
                ".*_hand_middle_.*",
                ".*_hand_thumb_.*",
            ],
            effort_limit_sim=_cfg_float("ISAACLAB_G1_HAND_EFFORT_LIMIT", 25.0),
            velocity_limit_sim=_cfg_float("ISAACLAB_G1_HAND_VELOCITY_LIMIT", 6.0),
            stiffness=_cfg_float("ISAACLAB_G1_HAND_STIFFNESS", 80.0),
            damping=_cfg_float("ISAACLAB_G1_HAND_DAMPING", 8.0),
            armature=_cfg_float("ISAACLAB_G1_HAND_ARMATURE", 0.02),
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
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
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
    # Single humanoid from the GR00T sim2sim viewer asset.
    robot_1: ArticulationCfg = G1_43DOF_GR00T_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=G1_43DOF_GR00T_CFG.init_state.replace(pos=(0.0, 0.0, 0.78)),
    )
    test_box = _grasp_box_cfg("TestBox")

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
        state_write_pose_source=_isaac_robot_cfg(robot_id, "STATE_WRITE_POSE_SOURCE", "measured"),
        target_only_pose_source=_isaac_robot_cfg(robot_id, "TARGET_ONLY_POSE_SOURCE", "action"),
        hand_pose_source=_isaac_robot_cfg(robot_id, "HAND_POSE_SOURCE", "target"),
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
        zero_target_only_body_velocity=_isaac_robot_cfg_bool(robot_id, "ZERO_TARGET_ONLY_BODY_VELOCITY", False),
        zero_target_only_hand_velocity=_isaac_robot_cfg_bool(robot_id, "ZERO_TARGET_ONLY_HAND_VELOCITY", False),
        body_joint_target_scale_overrides=_isaac_robot_cfg_pattern_float_dict(
            robot_id,
            "BODY_JOINT_TARGET_SCALE_OVERRIDES",
            {},
        ),
        hand_joint_target_max_delta=_isaac_robot_cfg_float(robot_id, "HAND_JOINT_TARGET_MAX_DELTA", 0.20),
        hold_default_until_first_packet=_isaac_robot_cfg_bool(robot_id, "HOLD_DEFAULT_UNTIL_FIRST_PACKET", True),
        no_packet_debug_interval_s=_isaac_robot_cfg_float(robot_id, "NO_PACKET_DEBUG_INTERVAL_S", 1.0),
        root_motion_mode=_isaac_robot_cfg(robot_id, "ROOT_MOTION_MODE", "source"),
        root_zmq_required=_isaac_robot_cfg_bool(robot_id, "ROOT_ZMQ_REQUIRED", True),
        root_position_mode=_isaac_robot_cfg(robot_id, "ROOT_POSITION_MODE", "relative"),
        body_state_write_joint_names=_isaac_robot_cfg_str_list(
            robot_id,
            "BODY_STATE_WRITE_JOINT_NAMES",
            G1_BODY_STATE_WRITE_JOINT_NAMES,
        ),
        mirror_hands=_isaac_robot_cfg_bool(robot_id, "MIRROR_HANDS", True),
        controller_gripper_enabled=_isaac_robot_cfg_bool(robot_id, "CONTROLLER_GRIPPER_ENABLED", False),
        controller_gripper_finger_close_angle=_cfg_float("ISAACLAB_G1_GRIPPER_FINGER_CLOSE_ANGLE", 1.8),
        controller_gripper_thumb_yaw_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_YAW_ANGLE", 0.5),
        controller_gripper_thumb_1_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_1_ANGLE", 1.1),
        controller_gripper_thumb_2_angle=_cfg_float("ISAACLAB_G1_GRIPPER_THUMB_2_ANGLE", 1.8),
        controller_gripper_action_alpha=_cfg_float("ISAACLAB_G1_GRIPPER_ACTION_ALPHA", 1.0),
        controller_gripper_use_soft_limits=_cfg_bool("ISAACLAB_G1_GRIPPER_USE_SOFT_LIMITS", False),
        controller_gripper_write_joint_state=_cfg_bool("ISAACLAB_G1_GRIPPER_WRITE_JOINT_STATE", False),
        controller_gripper_target_max_delta=_cfg_float("ISAACLAB_G1_GRIPPER_TARGET_MAX_DELTA", 0.20),
        ground_lock=_isaac_robot_cfg_bool(robot_id, "GROUND_LOCK", False),
    )


@configclass
class ActionsCfg:
    """Single-G1 GR00T/MuJoCo mirror with physical Isaac Lab arms and hands."""

    mujoco_g1_mirror_1 = _mujoco_g1_mirror_cfg(1)


@configclass
class ObservationsCfg:
    """Empty observation manager config.

    The scene is used for live robot synchronization, not policy rollout or data recording,
    so no ``policy`` observation group is registered.
    """


@configclass
class TerminationsCfg:
    """No automatic episode termination for the live single-G1 test scene."""


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

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot_1/head_link"
        self.xr.anchor_rotation_prim_path = "/World/envs/env_0/Robot_1/pelvis"
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
                    retargeters=[],
                    sim_device=teleop_device,
                    xr_cfg=self.xr,
                ),
            }
        )
