# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import MuJoCoG1MirrorActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.box_drop_reset import BoxDropResetActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_object_sync import (
    ZmqEnvResetSyncActionCfg,
    ZmqSceneStateSyncActionCfg,
)

from isaaclab_assets.robots.unitree import G1_29DOF_CFG

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


def _runtime_cfg_value(name: str, default: str | None = None) -> str | None:
    """Read a process-local override before falling back to the shared env file."""

    return os.environ.get(name, _cfg_value(name, default))


def _runtime_cfg_int(name: str, default: int) -> int:
    try:
        return int(str(_runtime_cfg_value(name, str(default))).strip())
    except (TypeError, ValueError):
        return default


def _runtime_cfg_float(name: str, default: float) -> float:
    try:
        return float(str(_runtime_cfg_value(name, str(default))).strip())
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


def _local_robot_id() -> int:
    """Return the robot whose head and pelvis should anchor the local XR session."""

    raw_value = _runtime_cfg_value("ISAACLAB_LOCAL_ROBOT_ID", "1")
    try:
        robot_id = int(str(raw_value).strip())
    except (TypeError, ValueError):
        print(f"[WARN] Invalid ISAACLAB_LOCAL_ROBOT_ID={raw_value!r}; using robot 1.")
        return 1
    if robot_id not in {1, 2}:
        print(f"[WARN] Unsupported ISAACLAB_LOCAL_ROBOT_ID={raw_value!r}; using robot 1.")
        robot_id = 1
    return robot_id


def _scene_sync_role() -> str:
    """Resolve the local fixed-scene synchronization role."""

    raw_role = str(
        _runtime_cfg_value(
            "ISAACLAB_SCENE_SYNC_ROLE",
            _runtime_cfg_value("ISAACLAB_OBJECT_SYNC_ROLE", "auto"),
        )
    ).strip().lower()
    if raw_role == "auto":
        return "publisher" if _local_robot_id() == 1 else "subscriber"
    if raw_role in {"publisher", "subscriber", "none"}:
        return raw_role
    print(f"[WARN] Unsupported ISAACLAB_SCENE_SYNC_ROLE={raw_role!r}; disabling scene sync.")
    return "none"


def _scene_sync_endpoint(role: str) -> str:
    """Return the role-specific bind or connect endpoint."""

    if role == "publisher":
        return str(
            _runtime_cfg_value(
                "ISAACLAB_SCENE_SYNC_BIND_ENDPOINT",
                _runtime_cfg_value("ISAACLAB_OBJECT_SYNC_BIND_ENDPOINT", "tcp://0.0.0.0:15555"),
            )
        )
    if role == "subscriber":
        publisher_ip = str(
            _runtime_cfg_value(
                "ISAACLAB_SCENE_SYNC_PUBLISHER_IP",
                _runtime_cfg_value(
                    "ISAACLAB_OBJECT_SYNC_PUBLISHER_IP",
                    _cfg_value("WINDOWS_ROBOT_1_ISAACLAB_IP", "127.0.0.1"),
                ),
            )
        )
        return str(
            _runtime_cfg_value(
                "ISAACLAB_SCENE_SYNC_CONNECT_ENDPOINT",
                _runtime_cfg_value("ISAACLAB_OBJECT_SYNC_CONNECT_ENDPOINT", f"tcp://{publisher_ip}:15555"),
            )
        )
    return ""


_SCENE_SYNC_ROLE = _scene_sync_role()
_SCENE_SYNC_ENDPOINT = _scene_sync_endpoint(_SCENE_SYNC_ROLE)
_SCENE_PHYSICS_AUTHORITY = _SCENE_SYNC_ROLE != "subscriber"
_GR00T_RECEIVER_ENABLED = _SCENE_SYNC_ROLE != "subscriber"


def _scene_state_sync_cfg() -> ZmqSceneStateSyncActionCfg:
    """Create the fixed dual-G1/three-box scene-state synchronization action."""

    return ZmqSceneStateSyncActionCfg(
        asset_name="robot_1",
        role=_SCENE_SYNC_ROLE,
        endpoint=_SCENE_SYNC_ENDPOINT,
        topic=str(_runtime_cfg_value("ISAACLAB_SCENE_SYNC_TOPIC", "scene_state")),
        robot_names=("robot_1", "robot_2"),
        object_names=("small_box_1", "small_box_2", "long_box"),
        send_hwm=_runtime_cfg_int("ISAACLAB_SCENE_SYNC_SEND_HWM", 3),
        receive_hwm=_runtime_cfg_int("ISAACLAB_SCENE_SYNC_RECEIVE_HWM", 3),
        stale_timeout_s=_runtime_cfg_float("ISAACLAB_SCENE_SYNC_STALE_TIMEOUT_S", 0.5),
        stale_log_interval_s=_runtime_cfg_float("ISAACLAB_SCENE_SYNC_STALE_LOG_INTERVAL_S", 2.0),
    )


def _env_reset_sync_cfg() -> ZmqEnvResetSyncActionCfg:
    """Use the shared scene-sync socket for PC1-to-PC2 full reset events."""

    return ZmqEnvResetSyncActionCfg(
        asset_name="robot_1",
        role=_SCENE_SYNC_ROLE,
        endpoint=_SCENE_SYNC_ENDPOINT,
        topic=str(_runtime_cfg_value("ISAACLAB_ENV_RESET_SYNC_TOPIC", "env_reset")),
        repeat_frames=_runtime_cfg_int("ISAACLAB_ENV_RESET_SYNC_REPEAT_FRAMES", 10),
        send_hwm=_runtime_cfg_int("ISAACLAB_SCENE_SYNC_SEND_HWM", 3),
        receive_hwm=_runtime_cfg_int("ISAACLAB_SCENE_SYNC_RECEIVE_HWM", 3),
    )


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


# The packing-table asset is spawned at z=-0.3, placing its top at z=0.6996.
TABLE_TOP_Z = 0.6996
SMALL_BOX_HEIGHT = 0.05
LONG_BOX_HEIGHT = 0.10
SMALL_BOX_INITIAL_Z = TABLE_TOP_Z + 0.5 * SMALL_BOX_HEIGHT + 0.002
LONG_BOX_INITIAL_Z = TABLE_TOP_Z + 0.5 * LONG_BOX_HEIGHT + 0.002

# The three boxes use the poses measured in the reference Isaac Sim screenshots.
# The long side of the rectangular box is along X.
SMALL_BOX_SIZE = (0.05, 0.05, SMALL_BOX_HEIGHT)
LONG_BOX_SIZE = (0.20, 0.05, LONG_BOX_HEIGHT)
BOX_NAMES = ("small_box_1", "small_box_2", "long_box")

# Reset as soon as a box has clearly left the tabletop. This catches a falling
# box well before it can remain on the ground until the episode ends.
BOX_DROP_HEIGHT = TABLE_TOP_Z - 0.10


def _box_cfg(
    prim_name: str,
    size: tuple[float, float, float],
    initial_pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
    physics_authority: bool,
) -> RigidObjectCfg:
    """Create a dynamic authority box or a non-physical synchronized follower."""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=initial_pos,
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=not physics_authority,
                disable_gravity=not physics_authority,
                max_depenetration_velocity=3.0 if physics_authority else 0.0,
            ),
            collision_props=(
                sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.003,
                    rest_offset=0.0,
                )
                if physics_authority
                else sim_utils.CollisionPropertiesCfg(collision_enabled=False)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.70,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        ),
    )


##
# Scene definition
##


G1_BODY_STATE_WRITE_JOINT_NAMES = [
    ".*_hip_.*_joint",
    ".*_knee_joint",
    ".*_ankle_.*_joint",
    "waist_.*_joint",
]
"""Mirrored joints that are allowed to be hard-written into PhysX for stable walking."""


def _g1_29dof_udp_cfg() -> ArticulationCfg:
    """Adapt the official G1_29DOF_CFG for the local scene-sync authority role."""

    if _SCENE_PHYSICS_AUTHORITY:
        rigid_props = G1_29DOF_CFG.spawn.rigid_props
        collision_props = G1_29DOF_CFG.spawn.collision_props
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.0,
        )
        collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

    return G1_29DOF_CFG.replace(
        spawn=G1_29DOF_CFG.spawn.replace(
            rigid_props=rigid_props,
            collision_props=collision_props,
        ),
        init_state=G1_29DOF_CFG.init_state.replace(pos=(0.0, 0.0, 0.75)),
    )


G1_29DOF_UDP_CFG = _g1_29dof_udp_cfg()

@configclass
class LocomanipulationG1SceneCfg(InteractiveSceneCfg):
    """PC1 authoritative or PC2 mirrored fixed dual-G1/three-box scene."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, -0.3], rot=[1.0, 0.0, 0.0, 0.0]),
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
    # Two independent humanoids instantiated from the official Isaac Lab G1 29-DoF asset.
    robot_1: ArticulationCfg = G1_29DOF_UDP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=G1_29DOF_UDP_CFG.init_state.replace(pos=(0.0, 0.0, 0.75)),
    )
    robot_2: ArticulationCfg = G1_29DOF_UDP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        init_state=G1_29DOF_UDP_CFG.init_state.replace(
            pos=(0.0, 1.23134, 0.75),
            rot=(0.70710678, 0.0, 0.0, -0.70710678),
        ),
    )

    # Two 5 cm cubes and one 20 x 5 x 10 cm rectangular box on the table.
    small_box_1 = _box_cfg(
        prim_name="SmallBox1",
        size=SMALL_BOX_SIZE,
        initial_pos=(0.00553, 0.31243, SMALL_BOX_INITIAL_Z),
        mass=0.08,
        color=(0.82, 0.66, 0.36),
        physics_authority=_SCENE_PHYSICS_AUTHORITY,
    )
    small_box_2 = _box_cfg(
        prim_name="SmallBox2",
        size=SMALL_BOX_SIZE,
        initial_pos=(-0.10565, 0.31397, SMALL_BOX_INITIAL_Z),
        mass=0.08,
        color=(0.88, 0.72, 0.40),
        physics_authority=_SCENE_PHYSICS_AUTHORITY,
    )
    long_box = _box_cfg(
        prim_name="LongBox",
        size=LONG_BOX_SIZE,
        initial_pos=(-0.04810, 0.41625, LONG_BOX_INITIAL_Z),
        mass=0.25,
        color=(0.76, 0.56, 0.28),
        physics_authority=_SCENE_PHYSICS_AUTHORITY,
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


def _mujoco_g1_mirror_cfg(robot_id: int) -> MuJoCoG1MirrorActionCfg:
    sync_mode = _isaac_robot_sync_mode(robot_id)
    default_sender_ip = "192.168.10.230" if robot_id == 1 else "192.168.10.231"
    default_body_port = 5557 if robot_id == 1 else 5567
    default_root_port = 5558 if robot_id == 1 else 5568
    return MuJoCoG1MirrorActionCfg(
        asset_name=_robot_name(robot_id),
        enabled=_GR00T_RECEIVER_ENABLED,
        transport=_cfg_value("ISAACLAB_G1_TRANSPORT", "zmq"),
        zmq_host=_ubuntu_sender_ip(robot_id, _isaac_robot_cfg(robot_id, "ZMQ_HOST", default_sender_ip)),
        zmq_port=_isaac_robot_cfg_int(robot_id, "ZMQ_PORT", default_body_port),
        zmq_topic=_isaac_robot_cfg(robot_id, "ZMQ_TOPIC", f"g1_{robot_id}_debug"),
        zmq_joint_order=_isaac_robot_cfg(robot_id, "JOINT_ORDER", "mujoco"),
        zmq_pose_source=_isaac_robot_cfg(robot_id, "POSE_SOURCE", "target"),
        state_write_pose_source=_isaac_robot_cfg(robot_id, "STATE_WRITE_POSE_SOURCE", "measured"),
        target_only_pose_source=_isaac_robot_cfg(robot_id, "TARGET_ONLY_POSE_SOURCE", "measured"),
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
        body_joint_target_max_delta=_isaac_robot_cfg_float(robot_id, "BODY_JOINT_TARGET_MAX_DELTA", 0.20),
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
        root_orientation_mode=_isaac_robot_cfg(robot_id, "ROOT_ORIENTATION_MODE", "relative"),
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
    """PC1 GR00T authority or PC2 fixed-scene follower actions."""

    mujoco_g1_mirror_1 = _mujoco_g1_mirror_cfg(1)
    mujoco_g1_mirror_2 = _mujoco_g1_mirror_cfg(2)
    box_drop_reset = BoxDropResetActionCfg(
        asset_name="small_box_1",
        enabled=_SCENE_SYNC_ROLE == "publisher",
        box_names=BOX_NAMES,
        minimum_height=BOX_DROP_HEIGHT,
    )
    scene_state_sync = _scene_state_sync_cfg()
    env_reset_sync = _env_reset_sync_cfg()


@configclass
class ObservationsCfg:
    """Empty observation manager config.

    The scene is used for live robot synchronization, not policy rollout or data recording,
    so no ``policy`` observation group is registered.
    """


@configclass
class TerminationsCfg:
    """No automatic full-environment termination conditions."""

    pass


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

    # Exposed for the teleoperation runner: only local robot 1 binds VR X to reset.
    local_robot_id: int = 1

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

        local_robot_id = _local_robot_id()
        self.local_robot_id = local_robot_id
        local_robot_prim_path = f"/World/envs/env_0/Robot_{local_robot_id}"
        self.xr.anchor_prim_path = f"{local_robot_prim_path}/head_link"
        self.xr.anchor_rotation_prim_path = f"{local_robot_prim_path}/pelvis"
        print(
            f"[INFO] Isaac Lab local robot ID: {local_robot_id}; "
            f"XR anchor={self.xr.anchor_prim_path}"
        )
        print(
            f"[INFO] Isaac Lab scene sync: role={_SCENE_SYNC_ROLE}, "
            f"endpoint={_SCENE_SYNC_ENDPOINT or 'disabled'}, "
            f"physics_authority={_SCENE_PHYSICS_AUTHORITY}, "
            f"gr00t_receivers={_GR00T_RECEIVER_ENABLED}"
        )
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
