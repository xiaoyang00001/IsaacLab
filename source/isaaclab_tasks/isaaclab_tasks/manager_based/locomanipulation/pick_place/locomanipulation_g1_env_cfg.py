# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
from pathlib import Path

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters import G1GripperMotionControllerRetargeterCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.box_success_reset import BoxSuccessResetActionCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    G1GripperSyncActionCfg,
    MuJoCoG1MirrorActionCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_object_sync import (
    ZmqEnvResetSyncActionCfg,
    ZmqObjectSyncActionCfg,
    ZmqSceneStateSyncActionCfg,
)

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
            # mirror the file config into the process environment (without
            # overriding explicit exports) so runtime modules that read
            # os.environ (contact-freeze / adaptive-lead / hand-friction knobs
            # in mdp/actions.py) see the same configuration as this cfg module
            for key, value in values.items():
                os.environ.setdefault(key, value)
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


def _cfg_optional_float(name: str) -> float | None:
    value = _cfg_value(name)
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _cfg_bool_value(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_bool(name: str, default: bool) -> bool:
    return _cfg_bool_value(_cfg_value(name), default)


def _cfg_choice(name: str, default: str, choices: set[str]) -> str:
    value = (_cfg_value(name, default) or default).strip().lower()
    if value not in choices:
        print(f"[WARN] Unsupported {name}={value!r}; using {default!r}.")
        return default
    return value


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
    """Create the fixed dual-G1 scene-state synchronization action.

    Synced rigid objects: the pushcart boxes (cart_box1/2 + test_box) plus the
    second trailer with its two plastic totes (pushcart_2 + cart2_tote1/2).
    """

    return ZmqSceneStateSyncActionCfg(
        asset_name="robot_1",
        role=_SCENE_SYNC_ROLE,
        endpoint=_SCENE_SYNC_ENDPOINT,
        topic=str(_runtime_cfg_value("ISAACLAB_SCENE_SYNC_TOPIC", "scene_state")),
        robot_names=("robot_1", "robot_2"),
        object_names=("cart_box1", "cart_box2", "test_box", "pushcart_2", "cart2_tote1", "cart2_tote2"),
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
BOX_SIZES = (SMALL_BOX_SIZE, SMALL_BOX_SIZE, LONG_BOX_SIZE)

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


def _grasp_object_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    if ZMQ_SYNC_ROLE == "subscriber":
        return sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        linear_damping=_cfg_float("ISAACLAB_GRASP_OBJECT_LINEAR_DAMPING", 0.05),
        angular_damping=_cfg_float("ISAACLAB_GRASP_OBJECT_ANGULAR_DAMPING", 5.0),
        max_angular_velocity=_cfg_float("ISAACLAB_GRASP_OBJECT_MAX_ANGULAR_VELOCITY", 90.0),
        max_contact_impulse=_cfg_optional_float("ISAACLAB_GRASP_OBJECT_MAX_CONTACT_IMPULSE"),
        enable_gyroscopic_forces=_cfg_bool("ISAACLAB_GRASP_OBJECT_ENABLE_GYROSCOPIC_FORCES", False),
        solver_position_iteration_count=_cfg_int("ISAACLAB_GRASP_OBJECT_SOLVER_POSITION_ITERATIONS", 12),
        solver_velocity_iteration_count=_cfg_int("ISAACLAB_GRASP_OBJECT_SOLVER_VELOCITY_ITERATIONS", 4),
        max_depenetration_velocity=_cfg_float("ISAACLAB_GRASP_OBJECT_MAX_DEPENETRATION_VELOCITY", 2.0),
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
                friction_combine_mode=_cfg_choice(
                    "ISAACLAB_GRASP_OBJECT_FRICTION_COMBINE_MODE",
                    "min",
                    {"average", "min", "multiply", "max"},
                ),
                restitution=_cfg_float("ISAACLAB_GRASP_OBJECT_RESTITUTION", 0.0),
                restitution_combine_mode=_cfg_choice(
                    "ISAACLAB_GRASP_OBJECT_RESTITUTION_COMBINE_MODE",
                    "min",
                    {"average", "min", "multiply", "max"},
                ),
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.36, 0.18), roughness=0.7),
        ),
    )


def _any_box_dropped(
    env,
    box_names: tuple[str, ...],
    minimum_height: float = BOX_DROP_HEIGHT,
) -> torch.Tensor:
    """Return true when any box falls below the tabletop drop threshold."""

    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for box_name in box_names:
        box: RigidObject = env.scene[box_name]
        relative_height = box.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        any_dropped = torch.logical_or(any_dropped, relative_height < minimum_height)
    return any_dropped


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
            "g1_43dof.usd",
            "g1_43dof_isaaclab_no_material.usda",
            "g1_43dof_isaaclab_nomdl.usda",
            "g1_43dof_s3.usda",
        ):
            usd_path = root / "gear_sonic/data/robots/g1" / usd_name
            if usd_path.exists():
                print(f"[locomanipulation_g1_env_cfg] G1 43-DoF USD: {usd_path.resolve()}")
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
    if not _SCENE_PHYSICS_AUTHORITY:
        return sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.0,
        )
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
        # required by the hand ContactSensor driving the adaptive finger lead
        activate_contact_sensors=True,
        rigid_props=_g1_robot_rigid_props(),
        collision_props=(
            None
            if _SCENE_PHYSICS_AUTHORITY
            else sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-3.8, 19.008, 0.78),
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
            effort_limit_sim=_cfg_float("ISAACLAB_G1_ARM_EFFORT_LIMIT", 60.0),
            velocity_limit_sim=_cfg_float("ISAACLAB_G1_ARM_VELOCITY_LIMIT", 12.0),
            stiffness=_cfg_float("ISAACLAB_G1_ARM_STIFFNESS", 700.0),
            damping=_cfg_float("ISAACLAB_G1_ARM_DAMPING", 20.0),
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


def _make_graspable_cart_box_spawn_cfg(syncable: bool = False) -> UsdFileCfg:
    """Create the warehouse cardboard box with rigid physics available at spawn time.

    When ``syncable`` is set, the box switches to kinematic + no-gravity on the ZMQ
    subscriber side so it purely follows the publisher's synced pose instead of
    fighting local physics (same pattern as ``test_box``).
    """

    is_sync_subscriber = syncable and ZMQ_SYNC_ROLE == "subscriber"
    return UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "props", "cart_box_d05_physics.usda"),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=is_sync_subscriber,
            disable_gravity=is_sync_subscriber,
            linear_damping=0.1,
            angular_damping=0.1,
            max_depenetration_velocity=0.5,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    )


def _make_pushcart_spawn_cfg(syncable: bool = False) -> UsdFileCfg:
    """Create the pushcart with rigid physics available at spawn time.

    When ``syncable`` is set, the cart switches to kinematic + no-gravity on the ZMQ
    subscriber side so it purely follows the publisher's synced pose instead of
    fighting local physics.
    """

    is_sync_subscriber = syncable and ZMQ_SYNC_ROLE == "subscriber"
    return UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "props", "pushcart_physics.usda"),
        scale=(0.5, 0.5, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=is_sync_subscriber,
            disable_gravity=is_sync_subscriber,
            solver_position_iteration_count=8,
            max_depenetration_velocity=5.0,
        ),
    )


def _make_cart2_tote_spawn_cfg(syncable: bool = False) -> UsdFileCfg:
    """Create the plastic tote (Tote_B04) with rigid physics available at spawn time.

    When ``syncable`` is set, the tote switches to kinematic + no-gravity on the ZMQ
    subscriber side so it purely follows the publisher's synced pose instead of
    fighting local physics (same pattern as the cart boxes).

    抓取调参照抄 test_box_2~5 的 ISAACLAB_GRASP_OBJECT_* 参数组：刚体阻尼/求解器
    迭代/角速度限幅（_grasp_object_rigid_props）、接触偏移、质量（1.0→0.45 kg）；
    高摩擦材质（2.0/1.6，combine=min）绑定在 tote_b04_physics.usda 内。
    """

    is_sync_subscriber = syncable and ZMQ_SYNC_ROLE == "subscriber"
    return UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "props", "tote_b04_physics.usda"),
        # 原 0.01 → 0.6×0.4×0.3 m；缩小一半到 0.005 → 0.3×0.2×0.15 m（原点仍在筐底面）。
        scale=(0.005, 0.005, 0.005),
        mass_props=sim_utils.MassPropertiesCfg(mass=_cfg_float("ISAACLAB_GRASP_OBJECT_MASS", 0.45)),
        rigid_props=(
            sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            if is_sync_subscriber
            else _grasp_object_rigid_props()
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=_cfg_float("ISAACLAB_GRASP_OBJECT_CONTACT_OFFSET", 0.006),
            rest_offset=_cfg_float("ISAACLAB_GRASP_OBJECT_REST_OFFSET", 0.0),
        ),
    )


# 双机工位 y：流水线三段（背景 USD ConveyorBelt_A08_06/_07/_08）实测世界 y 跨度
#   第一段 15.504~18.222（入料，中心 16.863）
#   第二段 12.788~15.507（中心 14.148）  ← 工位设在这里
#   第三段 10.188~12.907（出料，中心 11.548）
# 机器人原先站在第一段中心（16.7），筐一上带就到工位、几乎看不到流动；挪到第二段
# 中心后，筐从原起点 16.4/17.0 有 2.3~2.9 m（约 9~11 s）行程可跑，到工位再停住。
#
# 配套改动在背景 USD 里（warehouse-simple6_v48.usd，备份 .bak-workstation-shift-20260721）：
# 第二段两侧的 SM_HeavyDutyPackingTable_C02_01/_03 与 blue_sorting_bin_01/02 已沿世界
# +Y 平移 0.6 m（y 中心 14.65~14.84 → 15.25~15.44），否则机器人站到 14.148 会插进桌子
# bbox 里。之所以改 USD 而非用事件运行时挪：sorting bin 带刚体（bin_02 还是非 kinematic
# 动态体），运行时改 USD xform 后 PhysX 不同步；而 prestartup 事件与本场景必须保留的
# replicate_physics=True 互斥。
ROBOT_WORKSTATION_Y = _cfg_float("ISAACLAB_ROBOT_WORKSTATION_Y", 14.148)


@configclass
class LocomanipulationG1SceneCfg(InteractiveSceneCfg):
    """Warehouse 双 G1 场景：PC1 为物理权威，PC2 可经 scene_state 同步镜像（含三小箱任务）。"""

    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.68, 14.39363, 0], rot=[0.7071, 0.0, 0.0, 0.7071]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse-simple6_v48.usd"),
        ),
    )
    
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
    # ------------------------------------------------------------------
    # 从 warehouse-simple6_v48.usd 搬出的道具。
    # 原 prim 已在背景 USD 中停用（备份 .bak/.bak2/.bak3）。
    # CartBox1/CartBox2 使用源分支同款 SM_CardBoxD_05.usd 视觉资产。
    # 物理 schema 放在本地 wrapper 中，避免为 prestartup 关闭 replicate_physics 后影响 G1 稳定性。
    # 位姿 = USD 内位姿 × 背景放置变换，与原场景摆放逐位一致。
    #
    # Pushcart 与两个箱子都是独立 RigidObject（IsaacLab 不支持嵌套刚体）。
    # 箱子 z 抬高到推车 bbox 顶面 (z_max=0.3774) 之上避免初始穿透。
    # ------------------------------------------------------------------
    pushcart = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pushcart",
        # 对齐 ConveyorBelt（场景位置 y=14.39363，绕 Z 90°），再叠加 +90°Z = 180°Z 总旋转。
        # 与塑料筐拖车组对调后在外侧位 x=-6.8。
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-6.8, 19.39363, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步推车（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_pushcart_spawn_cfg(syncable=True),
    )
    cart_box1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CartBox1",
        # 推车顶面 z≈0.377，箱子半高 0.0745 → 中心 z
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-6.8, 19.39363, 0.45], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    )
    cart_box2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CartBox2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-6.8, 19.39363, 0.60], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    )
    # cart_box3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CartBox3",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.75], rot=[0.0, 0.0, 0.0, 1.0]),
    #     # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
    #     spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    # )
    # cart_box4 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CartBox4",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.90], rot=[0.0, 0.0, 0.0, 1.0]),
    #     # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
    #     spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    # )
    # worktable_tote = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/WorkTableTote",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-6.15, 18.19363, 0.0], rot=[0.707107, 0.0, 0.0, 0.707107]),
    #     spawn=UsdFileCfg(
    #         usd_path=os.path.join(os.path.dirname(__file__), "props", "tote_a01_physics.usda"),
    #         scale=(0.01, 0.01, 0.01),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=8,
    #             max_depenetration_velocity=5.0,
    #         ),
    #     ),
    # )
    # 第二台拖车（原 USD /Root/MyCart2，静态视觉件已在背景 USD 中停用，备份 .bak4，由本物理拖车替代），
    # 占工作位 x=-5.4。两个塑料筐已从车上挪到流水线滚轮面（见下方 cart2_tote1/2），此车现为空车。
    # pushcart_2 + cart2_tote1/2 三件仍进 scene_state 同步（订阅端切 kinematic 跟随发布端位姿）。
    pushcart_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pushcart2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=_make_pushcart_spawn_cfg(syncable=True),
    )
    # ------------------------------------------------------------------
    # 流水线（背景 USD /Root/ConveyorBelt，NVIDIA ConveyorBelt_A08 ×3 段）本身只有
    # 视觉、无物理——箱子放上去会直接穿落。这里补一块静态碰撞面让物体能停在滚轮面上。
    #
    # 用不可见 kinematic 碰撞板（同 PackingTable 的静态碰撞思路），顶面对齐滚轮顶
    # z≈0.772（皮带贴花面 0.731、滚轮顶 0.772、侧护栏顶 1.169，物体落在滚轮上）。
    # 覆盖滚轮可用宽度 x∈[-6.07,-5.17]（中心 -5.62，宽 0.90）与整条 y 跨度
    # [10.19,18.22]（中心 14.205，长 8.03）。始终 kinematic，不随 scene_state 同步。
    # 世界坐标由背景放置变换(pos=[-4.68,14.39363,0],rot=90°Z)换算自 USD 内几何。
    # ------------------------------------------------------------------
    # 顶面 0.772、板厚 0.04 → 中心 z = 0.772 - 0.02 = 0.752
    conveyor_collider = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorCollider",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-5.62, 14.205, 0.752],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.90, 8.03, 0.04),
            visible=False,  # 只提供碰撞，视觉沿用背景 USD 的流水线模型
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
        ),
    )
    # 两个塑料筐从 pushcart_2 挪到流水线滚轮面上（缩小一半后 0.3×0.2×0.15 m，X 半宽≈0.15）。
    # 筐原点在底面，滚轮顶 z≈0.772 → base 抬到 0.775（+3 mm 间隙避免初始穿透）。
    # 沿 X 分到滚轮碰撞带 x[-6.07,-5.17] 两侧边缘（各留 ~3 cm 余量不掉出），各走各的车道：
    #   cart2_tote1 → +X 侧 x=-5.35，对 robot_1(x=-4.75)，间距 0.67 m；
    #   cart2_tote2 → -X 侧 x=-5.89，对 robot_2(x=-6.70)，间距 0.86 m。
    # Y 是**起点**而非终点：两筐在第一段（16.4 / 17.0，前后错开 0.6 m）出发，被
    # drive_totes_on_conveyor 沿 -Y 送到第二段工位 ROBOT_WORKSTATION_Y 停住，
    # 各自停在对应机器人正前方。X 车道不同，两筐全程不会互相挡道。
    cart2_tote1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cart2Tote1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.35, 16.4, 0.775], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=_make_cart2_tote_spawn_cfg(syncable=True),
    )
    cart2_tote2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cart2Tote2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.89, 17.0, 0.775], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=_make_cart2_tote_spawn_cfg(syncable=True),
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
    # 双机站位：面对面立在流水线两侧（robot_1 在 +X 侧 x=-4.75 朝 -X，
    # robot_2 在 -X 侧 x=-6.7 朝 +X），y 移到第二段中心 ROBOT_WORKSTATION_Y。
    # 流水线(结构占 x[-6.19,-5.04])比原推车宽，robot_2 拉到 x=-6.7 越过流水线 -X 侧外缘避免重叠；
    # 代价是离箱(x=-5.62)约 1.1 m 够不到，此侧为布局/展示站位（robot_1 侧 x=-4.75 仍贴近 +X 边缘）。
    robot_1: ArticulationCfg = G1_43DOF_GR00T_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_1",
        init_state=G1_43DOF_GR00T_CFG.init_state.replace(
            pos=(-4.75, ROBOT_WORKSTATION_Y, 0.78),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Net contact forces on robot_1's hand links: the only contact signal that
    # works for LIGHT objects (a 0.08 kg box never stalls the fingers, so every
    # kinematic contact cue -- residual/velocity/progress -- is blind to it).
    # Drives the adaptive finger-lead shrink in mdp/actions.py.
    hand_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/.*_hand_.*",
        update_period=0.0,
        history_length=1,
    )
    robot_2: ArticulationCfg = G1_43DOF_GR00T_CFG.replace(
        prim_path="/World/envs/env_.*/Robot_2",
        init_state=G1_43DOF_GR00T_CFG.init_state.replace(
            pos=(-6.7, ROBOT_WORKSTATION_Y, 0.78),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    test_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox",
        # 属性对齐 晓阳全身005 a61191017 的 long_box（尺寸/质量/材质/碰撞参数），
        # 仅位置沿用本场景原 test_box 摆放（cart_box4 上方，x/y 对齐、rot 沿用推车朝向）。
        # x 随纸箱推车组对调到 -6.8。
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-6.8, 19.39363, 1.095],
            rot=[0.0, 0.0, 0.0, 1.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.20, 0.05, 0.10),
            rigid_props=(
                sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=3.0,
                )
                if ZMQ_SYNC_ROLE != "subscriber"
                else sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.76, 0.56, 0.28), roughness=0.70),
        ),
    )

    # 晓阳0007 固定三箱同步任务：PackingTable 台面上的两小方块 + 一长方块。
    # 发布端(PC1)为物理权威；订阅端(PC2)切 kinematic 无碰撞，由 scene_state 帧驱动。
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
    test_box_2 = _grasp_box_cfg("TestBox_2", 1)
    test_box_3 = _grasp_box_cfg("TestBox_3", 2)
    test_box_4 = _grasp_box_cfg("TestBox_4", 3)
    test_box_5 = _grasp_box_cfg("TestBox_5", 4)

    # Ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     spawn=GroundPlaneCfg(),
    # )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 方向光制造明暗面，避免 DomeLight 均匀照明导致的"塑料感"
    sun = AssetBaseCfg(
        prim_path="/World/sunLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9238795, 0.3826834, 0.0, 0.0)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.98, 0.95), intensity=3000.0, angle=0.53),
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
        mirror_hands=_isaac_robot_cfg_bool(robot_id, "MIRROR_HANDS", False),
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
    """Action specifications for the MDP."""

    # Body/root streams are mirrored independently for both robot IDs.
    mujoco_g1_mirror_1 = _mujoco_g1_mirror_cfg(1)
    mujoco_g1_mirror_2 = _mujoco_g1_mirror_cfg(2)
    # 晓阳0007：双机固定场景同步（双 G1 + 推车三箱 + 第二拖车两塑料筐，单帧 scene_state）、
    # PC1→PC2 复位事件、装箱成功检测复位。
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
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    # 晓阳0007：三小箱任一掉离 PackingTable 台面 → 整环境复位（发布端触发后经 env_reset_sync 同步 PC2）。
    # 只监控 small_box_1/2 与 long_box，不涉及 warehouse 推车/纸箱等其他物件。
    box_dropped = DoneTerm(
        func=_any_box_dropped,
        params={"box_names": BOX_NAMES},
    )

    # XR teleop 场景使用下陷布局（PackingTable z=-1000.66），
    # 绝对世界系 minimum_height=0.5 会导致 object(z=-100.76) 每步触发复位。
    # teleop 不需要训练式 episode 终止，因此移除 object_dropping。
    # object_dropping = DoneTerm(
    #     func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    # )

    # success = DoneTerm(
    #     func=manip_mdp.task_done_pick_place,
    #     params={"task_link_name": "right_wrist_yaw_link", "robot_cfg": SceneEntityCfg(ISAACLAB_LOCAL_ROBOT_NAME)},
    # )


# ------------------------------------------------------------------
# 流水线驱动参数。背景 USD 的 ConveyorBelt 只有视觉，物体靠 conveyor_collider
# 那块不可见 kinematic 碰撞板托着，所以"流动"由 drive_totes_on_conveyor 每
# 20 ms 覆写筐的水平速度模拟（同 congxian 分支的 drive_object_on_conveyor 思路）。
#
# 方向 -Y：两筐从第一段起点（16.4 / 17.0）流向第二段工位 ROBOT_WORKSTATION_Y，
# 到工位即停住等机器人取料。速度 0.3 m/s——congxian 用 0.5，这里筐缩了一半又轻
# （0.45 kg），放慢一档既不易滑出滚轮带，也给机器人留出抓取窗口。
#
# 摩擦损耗：筐是在**静止**的碰撞板上被拽着滑行（全程动摩擦 μd=0.6，非真传送带的
# 静摩擦输送），substep 间持续制动，实测平均速度约 0.25 m/s，比设定值低 ~17%。
# 同一原因还会让筐缓慢自转（convexDecomposition 的多块凸包接触点不对称），已知，
# 待后续单独处理。
# ------------------------------------------------------------------
CONVEYOR_TOTE_NAMES = ("cart2_tote1", "cart2_tote2")
CONVEYOR_SPEED = _cfg_float("ISAACLAB_CONVEYOR_SPEED", 0.3)
CONVEYOR_Y_RECYCLE = _cfg_float("ISAACLAB_CONVEYOR_Y_RECYCLE", 10.6)
CONVEYOR_Y_RESPAWN = _cfg_float("ISAACLAB_CONVEYOR_Y_RESPAWN", 18.0)
# 筐流到工位即停住（不再驱动，靠 μd=0.6 的摩擦自然停）。设 <=0 可关掉暂停、退回纯循环流。
CONVEYOR_Y_STOP = _cfg_float("ISAACLAB_CONVEYOR_Y_STOP", ROBOT_WORKSTATION_Y)
# 订阅端的筐是 kinematic、纯跟随发布端 scene_state，本地再驱动会和同步打架。
CONVEYOR_ENABLED = _cfg_bool("ISAACLAB_CONVEYOR_ENABLED", True) and ZMQ_SYNC_ROLE != "subscriber"

# 背景里的分拣料箱：bin_01 在 USD 里已是 kinematic，bin_02 却是动态刚体，开局悬空
# 1.5 cm、起步即下沉落到桌面（z 0.3918→0.3737）且会被机器人撞飞。统一锁成 kinematic
# 钉在原始摆位。用 startup 事件而非改 USD：改动留在代码里，好查也好回退。
BACKGROUND_LOCK_PRIM_NAMES = ("blue_sorting_bin_02",)


@configclass
class EventsCfg:
    """Runtime events：背景料箱锁 kinematic + 流水线送筐到工位停住。"""

    lock_sorting_bins = EventTerm(
        func=locomanip_mdp.lock_background_rigid_bodies,
        mode="startup",
        params={
            "prim_names": BACKGROUND_LOCK_PRIM_NAMES,
            "parent_path": "Background/ConveyorBelt",
            "kinematic": True,
        },
    )

    drive_totes = EventTerm(
        func=locomanip_mdp.drive_totes_on_conveyor,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "object_names": CONVEYOR_TOTE_NAMES,
            "velocity_y": -CONVEYOR_SPEED,
            "enabled": CONVEYOR_ENABLED,
            "y_stop": CONVEYOR_Y_STOP if CONVEYOR_Y_STOP > 0 else None,
            "y_recycle": CONVEYOR_Y_RECYCLE,
            "y_respawn": CONVEYOR_Y_RESPAWN,
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
    scene: LocomanipulationG1SceneCfg = LocomanipulationG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

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
        self.sim.physx.gpu_max_rigid_contact_count = 2**22
        self.sim.physx.gpu_max_rigid_patch_count = 2**16
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**18
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18
        self.sim.physx.gpu_collision_stack_size = 2**26
        self.sim.physx.gpu_heap_capacity = 2**26
        self.sim.physx.gpu_temp_buffer_capacity = 2**24

        local_robot_id = _local_robot_id()
        self.local_robot_id = local_robot_id
        local_robot_prim = _robot_prim_name(local_robot_id)
        self.xr.anchor_prim_path = f"/World/envs/env_0/{local_robot_prim}/head_link"
        self.xr.anchor_rotation_prim_path = f"/World/envs/env_0/{local_robot_prim}/pelvis"
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
