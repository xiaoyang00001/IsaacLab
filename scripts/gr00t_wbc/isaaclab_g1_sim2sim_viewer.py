"""Mirror MuJoCo/SONIC G1 joint states in a minimal Isaac Lab scene.

This script is intentionally a visualizer first: it writes root pose and joint
state directly into an Isaac Lab Articulation, so PhysX/PD tracking differences
do not distort the pose you are trying to inspect.

Example:
    TERM=xterm /home/nolovr/IsaacLab/isaaclab.sh -p \
        scripts/gr00t_wbc/isaaclab_g1_sim2sim_viewer.py

    TERM=xterm /home/nolovr/IsaacLab/isaaclab.sh -p \
        scripts/gr00t_wbc/isaaclab_g1_sim2sim_viewer.py \
        --source zmq --zmq-port 5557 --zmq-topic g1_debug
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher


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
            os.environ.setdefault(key, value)


def _arg_value(argv: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _load_default_network_config() -> None:
    candidates = []
    cli_config = _arg_value(sys.argv[1:], "--network-config")
    if cli_config:
        candidates.append(Path(cli_config).expanduser())
    for env_name in ("ISAACLAB_G1_NETWORK_CONFIG", "G1_NETWORK_CONFIG"):
        if os.environ.get(env_name):
            candidates.append(Path(os.environ[env_name]).expanduser())
    candidates.append(Path(__file__).resolve().with_name("g1_udp_network.env"))
    for path in candidates:
        _load_env_file(path)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


_load_default_network_config()


def _find_gr00t_repo_root() -> Path:
    env_root = os.environ.get("GR00T_WBC_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path("/home/nolovr/GR00T-WholeBodyControl"),
            Path(__file__).resolve().parents[3] / "GR00T-WholeBodyControl",
            Path.cwd(),
        ]
    )
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "gear_sonic/data/robots/g1/g1_43dof.usd").exists():
            return root
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate GR00T-WholeBodyControl. Set GR00T_WBC_ROOT to the repository path. "
        f"Searched:\n  {searched}"
    )


REPO_ROOT = _find_gr00t_repo_root()
DEFAULT_USD = REPO_ROOT / "gear_sonic/data/robots/g1/g1_43dof.usd"
DEFAULT_TRAJECTORY_DIR = REPO_ROOT / "gear_sonic_deploy/reference/example/macarena_001__A545"

# Isaac Lab / training order used by the reference CSV files.
ISAACLAB_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# For an input vector in IsaacLab order, this produces MuJoCo / hardware order.
ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

# MuJoCo / Unitree hardware order used by the deploy ZMQ output and MuJoCo qpos.
MUJOCO_29DOF_JOINT_NAMES = [ISAACLAB_29DOF_JOINT_NAMES[i] for i in ISAACLAB_TO_MUJOCO_DOF]

LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
]
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

DEFAULT_MUJOCO_29DOF_Q = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-config",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_G1_NETWORK_CONFIG", Path(__file__).resolve().with_name("g1_udp_network.env"))),
        help="UDP network config file. It is loaded before argument parsing.",
    )
    parser.add_argument(
        "--source",
        choices=("csv", "zmq", "udp", "sine", "idle"),
        default=os.environ.get("ISAACLAB_G1_VIEWER_SOURCE", "csv"),
        help="State source. csv replays reference/example by default; zmq/udp subscribe real-time targets.",
    )
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_USD, help="G1 USD file to load.")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help="Directory containing joint_pos.csv and optional joint_vel.csv/body_pos.csv/body_quat.csv.",
    )
    parser.add_argument(
        "--csv-joint-order",
        choices=("isaaclab", "mujoco"),
        default="isaaclab",
        help="Order of the first 29 columns in CSV joint_pos/joint_vel. Repository reference CSVs use isaaclab.",
    )
    parser.add_argument("--csv-fps", type=float, default=50.0, help="Frame rate of CSV trajectory data.")
    parser.add_argument("--no-loop", action="store_true", help="Stop CSV replay at the final frame instead of looping.")
    parser.add_argument("--no-follow-root", action="store_true", help="Do not apply root pose from CSV/ZMQ/UDP.")
    parser.add_argument("--root-z-offset", type=float, default=0.0, help="Additive offset applied to replayed root height.")
    parser.add_argument(
        "--root-motion-mode",
        choices=("auto", "source", "stance"),
        default="auto",
        help=(
            "Root translation source. source only uses CSV/ZMQ/UDP root pose; stance dead-reckons from the support foot; "
            "auto uses source root when it moves and otherwise falls back to stance."
        ),
    )
    parser.add_argument(
        "--stance-foot-height-tolerance",
        type=float,
        default=0.045,
        help="Foot-body height tolerance above the standing clearance for support-foot root estimation.",
    )
    parser.add_argument(
        "--stance-foot-switch-margin",
        type=float,
        default=0.015,
        help="Height margin for switching the support foot during stance root estimation.",
    )
    parser.add_argument(
        "--stance-root-max-step",
        type=float,
        default=0.035,
        help="Maximum xy correction per simulation step for stance root estimation. Non-positive disables clamping.",
    )
    parser.add_argument(
        "--source-root-motion-eps",
        type=float,
        default=1.0e-3,
        help="Source root xy displacement threshold used by --root-motion-mode auto.",
    )
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Isaac Lab physics step.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Scale replay time.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N simulation steps. 0 means run forever.")
    parser.add_argument("--print-interval", type=int, default=120, help="Print status every N steps. 0 disables logs.")
    parser.add_argument(
        "--ground-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the lowest foot body at or above the default standing foot clearance.",
    )
    parser.add_argument("--ground-height", type=float, default=0.0, help="World z height of the visual ground.")
    parser.add_argument(
        "--ground-lock-clearance",
        type=float,
        default=-1.0,
        help="Minimum foot-body z above ground. Negative means infer from the USD default standing pose.",
    )
    parser.add_argument(
        "--no-camera-follow",
        dest="no_camera_follow",
        action="store_true",
        default=True,
        help="Keep the camera fixed. This is the default so manual viewport angles are not reset.",
    )
    parser.add_argument(
        "--camera-follow",
        dest="no_camera_follow",
        action="store_false",
        help="Continuously move the camera to follow the robot root.",
    )
    parser.add_argument("--camera-update-interval", type=int, default=20, help="Update follow camera every N steps.")
    parser.add_argument("--zmq-host", default=os.environ.get("ISAACLAB_G1_ZMQ_HOST", "127.0.0.1"), help="ZMQ publisher host.")
    parser.add_argument("--zmq-port", type=int, default=_env_int("ISAACLAB_G1_ZMQ_PORT", 5557), help="ZMQ publisher port.")
    parser.add_argument("--zmq-topic", default=os.environ.get("ISAACLAB_G1_ZMQ_TOPIC", "g1_debug"), help="ZMQ topic prefix.")
    parser.add_argument("--zmq-timeout", type=float, default=_env_float("ISAACLAB_G1_TIMEOUT", 0.5), help="Seconds before warning about stale ZMQ data.")
    parser.add_argument(
        "--zmq-warmup-sec",
        type=float,
        default=1.0,
        help="Wait up to this many seconds at startup for the first ZMQ/UDP packets.",
    )
    parser.add_argument("--udp-bind-host", default=os.environ.get("ISAACLAB_G1_UDP_BIND_HOST", "0.0.0.0"), help="UDP local address to bind for state packets.")
    parser.add_argument("--udp-port", type=int, default=_env_int("ISAACLAB_G1_UDP_PORT", 5557), help="UDP local port for state packets.")
    parser.add_argument("--udp-topic", default=os.environ.get("ISAACLAB_G1_UDP_TOPIC", "g1_debug"), help="UDP topic prefix.")
    parser.add_argument("--udp-timeout", type=float, default=_env_float("ISAACLAB_G1_TIMEOUT", 0.5), help="Seconds before warning about stale UDP data.")
    parser.add_argument(
        "--udp-rcvbuf",
        type=int,
        default=_env_int("ISAACLAB_G1_UDP_RCVBUF", 262144),
        help="UDP receive socket SO_RCVBUF in bytes. The kernel may round or double this value.",
    )
    parser.add_argument(
        "--root-zmq",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ISAACLAB_G1_ROOT_ZMQ", True),
        help="In ZMQ mode, also subscribe to the MuJoCo root-state stream for exact walking translation.",
    )
    parser.add_argument("--root-zmq-host", default=os.environ.get("ISAACLAB_G1_ROOT_ZMQ_HOST", os.environ.get("ISAACLAB_G1_ZMQ_HOST", "127.0.0.1")), help="MuJoCo root-state ZMQ publisher host.")
    parser.add_argument("--root-zmq-port", type=int, default=_env_int("ISAACLAB_G1_ROOT_ZMQ_PORT", 5558), help="MuJoCo root-state ZMQ publisher port.")
    parser.add_argument("--root-zmq-topic", default=os.environ.get("ISAACLAB_G1_ROOT_ZMQ_TOPIC", "g1_root"), help="MuJoCo root-state ZMQ topic prefix.")
    parser.add_argument(
        "--root-udp",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ISAACLAB_G1_ROOT_UDP", True),
        help="In UDP mode, also receive the MuJoCo root-state UDP stream for exact walking translation.",
    )
    parser.add_argument("--root-udp-bind-host", default=os.environ.get("ISAACLAB_G1_ROOT_UDP_BIND_HOST", "0.0.0.0"), help="MuJoCo root-state UDP local bind host.")
    parser.add_argument("--root-udp-port", type=int, default=_env_int("ISAACLAB_G1_ROOT_UDP_PORT", 5558), help="MuJoCo root-state UDP local port.")
    parser.add_argument("--root-udp-topic", default=os.environ.get("ISAACLAB_G1_ROOT_UDP_TOPIC", "g1_root"), help="MuJoCo root-state UDP topic prefix.")
    parser.add_argument(
        "--root-udp-rcvbuf",
        type=int,
        default=_env_int("ISAACLAB_G1_ROOT_UDP_RCVBUF", 262144),
        help="Root-state UDP receive socket SO_RCVBUF in bytes.",
    )
    parser.add_argument(
        "--zmq-joint-order",
        choices=("mujoco", "isaaclab"),
        default="mujoco",
        help="Fallback order for 29-DoF ZMQ body_q fields. Deploy output uses mujoco.",
    )
    parser.add_argument(
        "--zmq-pose-source",
        choices=("measured", "target", "auto"),
        default="measured",
        help="Which ZMQ body/root fields to mirror. measured matches the current MuJoCo/control state.",
    )
    parser.add_argument(
        "--keep-usd-materials",
        action="store_true",
        help="Load the source USD materials directly instead of using the local no-MDL cache.",
    )
    parser.add_argument(
        "--usd-cache-dir",
        type=Path,
        default=Path(os.environ.get("ISAACLAB_G1_VIEWER_USD_CACHE", "~/.cache/isaaclab_gr00t_wbc")).expanduser(),
        help="Directory for sanitized USD cache files.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.assets.articulation import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from pxr import Usd, UsdShade  # noqa: E402


@dataclass
class StateSample:
    joint_pos_mujoco: np.ndarray
    joint_vel_mujoco: np.ndarray | None = None
    left_hand_pos: np.ndarray | None = None
    right_hand_pos: np.ndarray | None = None
    left_hand_vel: np.ndarray | None = None
    right_hand_vel: np.ndarray | None = None
    root_pos_w: np.ndarray | None = None
    root_quat_w: np.ndarray | None = None
    source_frame: int | None = None
    source_time: float | None = None
    fresh: bool = True
    done: bool = False


def _read_csv_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[list[float]] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return np.asarray(rows, dtype=np.float32)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-6 or not math.isfinite(norm):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _body_q_to_mujoco_order(values: np.ndarray, joint_order: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < len(MUJOCO_29DOF_JOINT_NAMES):
        raise ValueError(f"Joint vector has {values.size} values, expected at least 29")
    q29 = values[: len(MUJOCO_29DOF_JOINT_NAMES)]
    if joint_order == "isaaclab":
        return q29[ISAACLAB_TO_MUJOCO_DOF].copy()
    if joint_order == "mujoco":
        return q29.copy()
    raise ValueError(f"Unsupported joint order: {joint_order}")


def _hand_q_from_tail(values: np.ndarray, start: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size >= start + 7:
        return values[start : start + 7].copy()
    return None


class CsvTrajectorySource:
    def __init__(
        self,
        trajectory_dir: Path,
        fps: float,
        loop: bool,
        follow_root: bool,
        root_z_offset: float,
        joint_order: str,
    ):
        self.trajectory_dir = trajectory_dir
        self.fps = fps
        self.loop = loop
        self.follow_root = follow_root
        self.root_z_offset = root_z_offset
        self.joint_order = joint_order
        self.joint_pos = _read_csv_matrix(trajectory_dir / "joint_pos.csv")
        self.joint_vel = None
        joint_vel_path = trajectory_dir / "joint_vel.csv"
        if joint_vel_path.exists():
            self.joint_vel = _read_csv_matrix(joint_vel_path)
        self.root_pos = None
        self.root_quat = None
        body_pos_path = trajectory_dir / "body_pos.csv"
        body_quat_path = trajectory_dir / "body_quat.csv"
        if follow_root and body_pos_path.exists() and body_quat_path.exists():
            body_pos = _read_csv_matrix(body_pos_path)
            body_quat = _read_csv_matrix(body_quat_path)
            self.root_pos = body_pos[:, :3].copy()
            self.root_quat = body_quat[:, :4].copy()
            self.root_pos[:, 2] += root_z_offset
        self.num_frames = int(self.joint_pos.shape[0])
        if self.joint_pos.shape[1] < len(MUJOCO_29DOF_JOINT_NAMES):
            raise ValueError(
                f"{trajectory_dir / 'joint_pos.csv'} has {self.joint_pos.shape[1]} columns, "
                f"expected at least {len(MUJOCO_29DOF_JOINT_NAMES)}"
            )

    def sample(self, sim_time: float) -> StateSample:
        frame_float = max(sim_time * self.fps, 0.0)
        frame = int(frame_float)
        done = False
        if self.loop:
            frame %= self.num_frames
        else:
            if frame >= self.num_frames:
                frame = self.num_frames - 1
                done = True
        row = self.joint_pos[frame]
        q = _body_q_to_mujoco_order(row, self.joint_order)
        dq = None
        if self.joint_vel is not None:
            dq = _body_q_to_mujoco_order(self.joint_vel[frame], self.joint_order)
        left_hand_pos = _hand_q_from_tail(row, 29)
        right_hand_pos = _hand_q_from_tail(row, 36)
        left_hand_vel = _hand_q_from_tail(self.joint_vel[frame], 29) if self.joint_vel is not None else None
        right_hand_vel = _hand_q_from_tail(self.joint_vel[frame], 36) if self.joint_vel is not None else None
        root_pos = self.root_pos[frame] if self.root_pos is not None else None
        root_quat = _normalize_quat_wxyz(self.root_quat[frame]) if self.root_quat is not None else None
        return StateSample(
            joint_pos_mujoco=q,
            joint_vel_mujoco=dq,
            left_hand_pos=left_hand_pos,
            right_hand_pos=right_hand_pos,
            left_hand_vel=left_hand_vel,
            right_hand_vel=right_hand_vel,
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            source_frame=frame,
            source_time=frame / self.fps,
            done=done,
        )


class ZmqStateSource:
    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        timeout: float,
        follow_root: bool,
        root_z_offset: float,
        joint_order: str,
        pose_source: str,
    ):
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.zmq = zmq
        self.topic = topic.encode("utf-8")
        self.timeout = timeout
        self.follow_root = follow_root
        self.root_z_offset = root_z_offset
        self.joint_order = joint_order
        self.pose_source = pose_source
        self.transport_name = "ZMQ"
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.endpoint = f"tcp://{host}:{port}"
        self.socket.connect(self.endpoint)
        self.last_sample = StateSample(
            joint_pos_mujoco=DEFAULT_MUJOCO_29DOF_Q.copy(),
            joint_vel_mujoco=np.zeros(29, dtype=np.float32),
            fresh=False,
        )
        self.last_rx_time = 0.0
        self.first_root_pos: np.ndarray | None = None
        self.root_sample_count = 0
        self.root_static_warning_printed = False
        print(f"[INFO] ZMQ connected: {self.endpoint}/{topic}")

    def close(self) -> None:
        self.socket.close(0)
        self.ctx.term()

    def _decode(self, parts: list[bytes]) -> dict[str, Any] | None:
        if not parts:
            return None
        if len(parts) >= 2 and parts[0] == self.topic:
            payload = parts[-1]
        else:
            raw = parts[0]
            payload = raw[len(self.topic) :] if raw.startswith(self.topic) else raw
        return self.msgpack.unpackb(payload, raw=False)

    def _poll_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                parts = self.socket.recv_multipart(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                return latest
            latest = self._decode(parts)

    @staticmethod
    def _first_array(msg: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
        for key in keys:
            if key in msg:
                arr = np.asarray(msg[key], dtype=np.float32).reshape(-1)
                if arr.size > 0:
                    return arr
        return None

    def _select_body_q(self, msg: dict[str, Any]) -> np.ndarray | None:
        if self.pose_source == "target":
            return self._first_array(msg, ("body_q_target", "joint_pos", "q", "dof_pos"))
        if self.pose_source == "measured":
            return self._first_array(msg, ("body_q_measured", "body_q", "joint_pos", "q", "dof_pos"))
        target = self._first_array(msg, ("body_q_target",))
        if target is not None and float(np.max(np.abs(target[: min(target.size, 29)]))) > 1.0e-4:
            return target
        return self._first_array(msg, ("body_q_measured", "body_q", "joint_pos", "q", "dof_pos"))

    def _select_body_dq(self, msg: dict[str, Any]) -> np.ndarray | None:
        if self.pose_source == "target":
            return self._first_array(msg, ("body_dq_target", "joint_vel", "dq", "dof_vel"))
        return self._first_array(msg, ("body_dq_measured", "body_dq", "joint_vel", "dq", "dof_vel"))

    def _select_hand_q(self, msg: dict[str, Any], side: str) -> np.ndarray | None:
        measured_keys = (f"{side}_hand_q", f"{side}_hand_q_measured")
        target_keys = (f"{side}_hand_q_target", f"last_{side}_hand_action")
        if self.pose_source == "target":
            return self._first_array(msg, target_keys + measured_keys)
        if self.pose_source == "measured":
            return self._first_array(msg, measured_keys + target_keys)

        target = self._first_array(msg, target_keys)
        if target is not None and target.size >= 7 and float(np.max(np.abs(target[:7]))) > 1.0e-4:
            return target
        return self._first_array(msg, measured_keys)

    def _select_hand_dq(self, msg: dict[str, Any], side: str) -> np.ndarray | None:
        if self.pose_source == "target":
            return self._first_array(msg, (f"{side}_hand_dq_target", f"{side}_hand_dq", f"{side}_hand_dq_measured"))
        return self._first_array(msg, (f"{side}_hand_dq", f"{side}_hand_dq_measured", f"{side}_hand_dq_target"))

    def _select_root_pos(self, msg: dict[str, Any]) -> np.ndarray | None:
        if self.pose_source == "target":
            return self._first_array(msg, ("root_pos_w", "base_trans_target", "base_pos", "root_pos"))
        if self.pose_source == "measured":
            return self._first_array(msg, ("root_pos_w", "base_trans_measured", "base_pos", "root_pos"))
        target = self._first_array(msg, ("root_pos_w", "base_trans_target", "base_pos", "root_pos"))
        if target is not None and float(np.linalg.norm(target[: min(target.size, 3)])) > 1.0e-4:
            return target
        return self._first_array(msg, ("base_trans_measured",))

    def _select_root_quat(self, msg: dict[str, Any]) -> np.ndarray | None:
        if self.pose_source == "target":
            return self._first_array(msg, ("root_quat_w", "base_quat_target", "base_quat", "root_quat"))
        if self.pose_source == "measured":
            return self._first_array(msg, ("root_quat_w", "base_quat_measured", "base_quat", "root_quat"))
        return self._first_array(
            msg,
            ("root_quat_w", "base_quat_measured", "base_quat_target", "base_quat", "root_quat"),
        )

    def sample(self, sim_time: float) -> StateSample:
        msg = self._poll_latest()
        if msg is None:
            stale = self.last_rx_time == 0.0 or (time.monotonic() - self.last_rx_time) > self.timeout
            self.last_sample.fresh = not stale
            return self.last_sample

        q = self._select_body_q(msg)
        if q is None:
            q = self.last_sample.joint_pos_mujoco
        dq = self._select_body_dq(msg)
        msg_order = str(msg.get("target_order", msg.get("joint_order", self.joint_order))).lower()
        if msg_order not in {"mujoco", "isaaclab"}:
            msg_order = self.joint_order
        q_mujoco = _body_q_to_mujoco_order(q, msg_order)
        dq_mujoco = _body_q_to_mujoco_order(dq, msg_order) if dq is not None and dq.size >= 29 else None

        left_hand_pos = self._select_hand_q(msg, "left")
        right_hand_pos = self._select_hand_q(msg, "right")
        left_hand_vel = self._select_hand_dq(msg, "left")
        right_hand_vel = self._select_hand_dq(msg, "right")
        if left_hand_pos is not None and left_hand_pos.size < 7:
            left_hand_pos = None
        if right_hand_pos is not None and right_hand_pos.size < 7:
            right_hand_pos = None
        if left_hand_vel is not None and left_hand_vel.size < 7:
            left_hand_vel = None
        if right_hand_vel is not None and right_hand_vel.size < 7:
            right_hand_vel = None

        root_pos = None
        root_quat = None
        if self.follow_root:
            root_pos = self._select_root_pos(msg)
            root_quat = self._select_root_quat(msg)
            if root_pos is not None and root_pos.size >= 3:
                root_pos = root_pos[:3].copy()
                root_pos[2] += self.root_z_offset
                if self.first_root_pos is None:
                    self.first_root_pos = root_pos.copy()
                    self.root_sample_count = 1
                else:
                    self.root_sample_count += 1
                if (
                    not self.root_static_warning_printed
                    and self.root_sample_count >= 120
                    and float(np.linalg.norm(root_pos[:2] - self.first_root_pos[:2])) < 1.0e-4
                ):
                    print(
                        f"[WARN] {self.transport_name} root xy is static. Isaac Lab cannot show walking translation "
                        "unless the publisher sends moving root_pos_w/base_trans_target/base_trans_measured."
                    )
                    self.root_static_warning_printed = True
            else:
                root_pos = None
            if root_quat is not None and root_quat.size >= 4:
                root_quat = _normalize_quat_wxyz(root_quat[:4])
            else:
                root_quat = None
        self.last_rx_time = time.monotonic()
        self.last_sample = StateSample(
            joint_pos_mujoco=q_mujoco,
            joint_vel_mujoco=dq_mujoco,
            left_hand_pos=left_hand_pos[:7].copy() if left_hand_pos is not None else None,
            right_hand_pos=right_hand_pos[:7].copy() if right_hand_pos is not None else None,
            left_hand_vel=left_hand_vel[:7].copy() if left_hand_vel is not None else None,
            right_hand_vel=right_hand_vel[:7].copy() if right_hand_vel is not None else None,
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            source_time=sim_time,
            fresh=True,
        )
        return self.last_sample


class UdpStateSource(ZmqStateSource):
    def __init__(
        self,
        bind_host: str,
        port: int,
        topic: str,
        timeout: float,
        rcvbuf: int,
        follow_root: bool,
        root_z_offset: float,
        joint_order: str,
        pose_source: str,
    ):
        import msgpack

        self.msgpack = msgpack
        self.topic = topic.encode("utf-8")
        self.timeout = timeout
        self.follow_root = follow_root
        self.root_z_offset = root_z_offset
        self.joint_order = joint_order
        self.pose_source = pose_source
        self.transport_name = "UDP"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        self.socket.bind((bind_host, port))
        self.socket.setblocking(False)
        actual_rcvbuf = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.last_sample = StateSample(
            joint_pos_mujoco=DEFAULT_MUJOCO_29DOF_Q.copy(),
            joint_vel_mujoco=np.zeros(29, dtype=np.float32),
            fresh=False,
        )
        self.last_rx_time = 0.0
        self.first_root_pos: np.ndarray | None = None
        self.root_sample_count = 0
        self.root_static_warning_printed = False
        print(
            f"[INFO] UDP listening: udp://{bind_host}:{port}/{topic} "
            f"SO_RCVBUF={actual_rcvbuf}"
        )

    def close(self) -> None:
        self.socket.close()

    def _decode_packet(self, packet: bytes) -> dict[str, Any] | None:
        if not packet:
            return None
        if not packet.startswith(self.topic):
            return None
        payload = packet[len(self.topic) :]
        return self.msgpack.unpackb(payload, raw=False)

    def _poll_latest(self) -> dict[str, Any] | None:
        latest = None
        while True:
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                return latest
            decoded = self._decode_packet(packet)
            if decoded is not None:
                latest = decoded


class RootZmqSource:
    def __init__(self, host: str, port: int, topic: str, root_z_offset: float, timeout: float):
        import msgpack
        import zmq

        self.msgpack = msgpack
        self.zmq = zmq
        self.topic = topic.encode("utf-8")
        self.root_z_offset = root_z_offset
        self.timeout = timeout
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.endpoint = f"tcp://{host}:{port}"
        self.socket.connect(self.endpoint)
        self.last_sample: tuple[np.ndarray, np.ndarray] | None = None
        self.last_rx_time = 0.0
        self.fresh = False
        print(f"[INFO] Root ZMQ connected: {self.endpoint}/{topic}")

    def close(self) -> None:
        self.socket.close(0)
        self.ctx.term()

    def _decode(self, parts: list[bytes]) -> dict[str, Any] | None:
        if not parts:
            return None
        if len(parts) >= 2 and parts[0] == self.topic:
            payload = parts[-1]
        else:
            raw = parts[0]
            payload = raw[len(self.topic) :] if raw.startswith(self.topic) else raw
        return self.msgpack.unpackb(payload, raw=False)

    def sample(self) -> tuple[np.ndarray, np.ndarray] | None:
        latest = None
        while True:
            try:
                parts = self.socket.recv_multipart(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            latest = self._decode(parts)
        if latest is None:
            self.fresh = self.last_sample is not None and (time.monotonic() - self.last_rx_time) <= self.timeout
            return self.last_sample

        root_pos = ZmqStateSource._first_array(latest, ("root_pos_w", "base_pos", "root_pos"))
        root_quat = ZmqStateSource._first_array(latest, ("root_quat_w", "base_quat", "root_quat"))
        if root_pos is None or root_pos.size < 3 or root_quat is None or root_quat.size < 4:
            self.fresh = self.last_sample is not None and (time.monotonic() - self.last_rx_time) <= self.timeout
            return self.last_sample

        root_pos = root_pos[:3].copy()
        root_pos[2] += self.root_z_offset
        root_quat = _normalize_quat_wxyz(root_quat[:4])
        self.last_sample = (root_pos, root_quat)
        self.last_rx_time = time.monotonic()
        self.fresh = True
        return self.last_sample


class RootUdpSource(RootZmqSource):
    def __init__(self, bind_host: str, port: int, topic: str, root_z_offset: float, timeout: float, rcvbuf: int):
        import msgpack

        self.msgpack = msgpack
        self.topic = topic.encode("utf-8")
        self.root_z_offset = root_z_offset
        self.timeout = timeout
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, int(rcvbuf))
        self.socket.bind((bind_host, port))
        self.socket.setblocking(False)
        actual_rcvbuf = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self.last_sample: tuple[np.ndarray, np.ndarray] | None = None
        self.last_rx_time = 0.0
        self.fresh = False
        print(
            f"[INFO] Root UDP listening: udp://{bind_host}:{port}/{topic} "
            f"SO_RCVBUF={actual_rcvbuf}"
        )

    def close(self) -> None:
        self.socket.close()

    def _decode_packet(self, packet: bytes) -> dict[str, Any] | None:
        if not packet:
            return None
        if not packet.startswith(self.topic):
            return None
        payload = packet[len(self.topic) :]
        return self.msgpack.unpackb(payload, raw=False)

    def sample(self) -> tuple[np.ndarray, np.ndarray] | None:
        latest = None
        while True:
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            decoded = self._decode_packet(packet)
            if decoded is not None:
                latest = decoded
        if latest is None:
            self.fresh = self.last_sample is not None and (time.monotonic() - self.last_rx_time) <= self.timeout
            return self.last_sample

        root_pos = ZmqStateSource._first_array(latest, ("root_pos_w", "base_pos", "root_pos"))
        root_quat = ZmqStateSource._first_array(latest, ("root_quat_w", "base_quat", "root_quat"))
        if root_pos is None or root_pos.size < 3 or root_quat is None or root_quat.size < 4:
            self.fresh = self.last_sample is not None and (time.monotonic() - self.last_rx_time) <= self.timeout
            return self.last_sample

        root_pos = root_pos[:3].copy()
        root_pos[2] += self.root_z_offset
        root_quat = _normalize_quat_wxyz(root_quat[:4])
        self.last_sample = (root_pos, root_quat)
        self.last_rx_time = time.monotonic()
        self.fresh = True
        return self.last_sample


class SineSource:
    def __init__(self, follow_root: bool):
        self.follow_root = follow_root

    def sample(self, sim_time: float) -> StateSample:
        q = DEFAULT_MUJOCO_29DOF_Q.copy()
        q[15] += 0.45 * math.sin(sim_time * 2.0)
        q[16] += 0.25 * math.sin(sim_time * 1.3)
        q[18] += 0.35 * math.sin(sim_time * 1.7)
        q[22] += 0.45 * math.sin(sim_time * 2.0 + math.pi)
        q[23] -= 0.25 * math.sin(sim_time * 1.3)
        q[25] += 0.35 * math.sin(sim_time * 1.7 + math.pi)
        root_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32) if self.follow_root else None
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) if self.follow_root else None
        return StateSample(
            joint_pos_mujoco=q,
            joint_vel_mujoco=np.zeros(29, dtype=np.float32),
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            source_time=sim_time,
        )


class IdleSource:
    def __init__(self, follow_root: bool):
        self.follow_root = follow_root

    def sample(self, sim_time: float) -> StateSample:
        root_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32) if self.follow_root else None
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) if self.follow_root else None
        return StateSample(
            joint_pos_mujoco=DEFAULT_MUJOCO_29DOF_Q.copy(),
            joint_vel_mujoco=np.zeros(29, dtype=np.float32),
            root_pos_w=root_pos,
            root_quat_w=root_quat,
        )


def build_source() -> Any:
    follow_root = not args_cli.no_follow_root
    if args_cli.source == "csv":
        return CsvTrajectorySource(
            trajectory_dir=args_cli.trajectory_dir,
            fps=args_cli.csv_fps,
            loop=not args_cli.no_loop,
            follow_root=follow_root,
            root_z_offset=args_cli.root_z_offset,
            joint_order=args_cli.csv_joint_order,
        )
    if args_cli.source == "zmq":
        return ZmqStateSource(
            host=args_cli.zmq_host,
            port=args_cli.zmq_port,
            topic=args_cli.zmq_topic,
            timeout=args_cli.zmq_timeout,
            follow_root=follow_root,
            root_z_offset=args_cli.root_z_offset,
            joint_order=args_cli.zmq_joint_order,
            pose_source=args_cli.zmq_pose_source,
        )
    if args_cli.source == "udp":
        return UdpStateSource(
            bind_host=args_cli.udp_bind_host,
            port=args_cli.udp_port,
            topic=args_cli.udp_topic,
            timeout=args_cli.udp_timeout,
            rcvbuf=args_cli.udp_rcvbuf,
            follow_root=follow_root,
            root_z_offset=args_cli.root_z_offset,
            joint_order=args_cli.zmq_joint_order,
            pose_source=args_cli.zmq_pose_source,
        )
    if args_cli.source == "sine":
        return SineSource(follow_root=follow_root)
    return IdleSource(follow_root=follow_root)


@contextlib.contextmanager
def _suppress_stderr_fd():
    """Suppress noisy USD composition diagnostics while creating the sanitized cache."""

    saved_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _sanitized_usd_cache_path(usd_path: Path) -> Path:
    stat = usd_path.stat()
    cache_name = f"{usd_path.stem}.viewer_nomdl.{stat.st_size}.{stat.st_mtime_ns}.usd"
    return args_cli.usd_cache_dir / cache_name


def prepare_robot_usd(usd_path: Path) -> Path:
    """Create a local USD cache without remote MDL materials or remote room references."""

    usd_path = usd_path.expanduser().resolve()
    if args_cli.keep_usd_materials or usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        return usd_path

    cache_path = _sanitized_usd_cache_path(usd_path)
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    with _suppress_stderr_fd():
        stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open robot USD: {usd_path}")

    removed_props = 0
    remove_paths = set()
    for prim in stage.TraverseAll():
        for prop in list(prim.GetProperties()):
            name = prop.GetName()
            if name.startswith("material:binding") and prim.RemoveProperty(name):
                removed_props += 1
        if str(prim.GetPath()) == "/SimpleRoom":
            remove_paths.add(prim.GetPath())
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader):
            remove_paths.add(prim.GetPath())

    for prim_path in sorted(remove_paths, key=lambda path: len(str(path)), reverse=True):
        stage.RemovePrim(prim_path)

    stage.GetRootLayer().Export(str(tmp_path))
    tmp_path.replace(cache_path)
    print(
        f"[INFO] Created USD cache without remote MDL materials: {cache_path} "
        f"(removed_bindings={removed_props}, removed_prims={len(remove_paths)})"
    )
    return cache_path


def design_scene() -> Articulation:
    ground_cfg = sim_utils.GroundPlaneCfg(size=(100.0, 100.0), color=(0.08, 0.16, 0.24))
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    marker_cfg = sim_utils.SphereCfg(
        radius=0.045,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    marker_cfg.func("/World/com_marker", marker_cfg, translation=(0.1, 0.0, 0.0))

    robot_usd = prepare_robot_usd(args_cli.robot_usd)
    args_cli.robot_usd = robot_usd

    robot_cfg = ArticulationCfg(
        prim_path="/World/G1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.78),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={},
    )
    return Articulation(cfg=robot_cfg)


def build_mujoco_to_isaac_joint_ids(robot: Articulation) -> tuple[list[int], list[int], list[str]]:
    isaac_name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    mujoco_ids: list[int] = []
    isaac_ids: list[int] = []
    missing: list[str] = []
    for mujoco_id, name in enumerate(MUJOCO_29DOF_JOINT_NAMES):
        isaac_id = isaac_name_to_id.get(name)
        if isaac_id is None:
            missing.append(name)
            continue
        mujoco_ids.append(mujoco_id)
        isaac_ids.append(isaac_id)
    if missing:
        raise RuntimeError(
            "The USD does not contain required 29-DoF G1 joints: "
            + ", ".join(missing)
            + f"\nLoaded joints are: {robot.joint_names}"
        )
    print("[INFO] Loaded robot:")
    print(f"  USD: {args_cli.robot_usd}")
    print(f"  bodies={robot.num_bodies} joints={robot.num_joints}")
    print(f"  mapped active MuJoCo joints={len(isaac_ids)}")
    return mujoco_ids, isaac_ids, [MUJOCO_29DOF_JOINT_NAMES[i] for i in mujoco_ids]


def build_optional_isaac_joint_ids(robot: Articulation, joint_names: list[str], label: str) -> list[int]:
    isaac_name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    ids: list[int] = []
    missing: list[str] = []
    for name in joint_names:
        isaac_id = isaac_name_to_id.get(name)
        if isaac_id is None:
            missing.append(name)
        else:
            ids.append(isaac_id)
    if missing:
        print(f"[WARN] Missing {label} joints in USD, hand data for these joints will be ignored: {missing}")
    else:
        print(f"  mapped {label} joints={len(ids)}")
    return ids


def build_required_body_ids(robot: Articulation, body_names: list[str], label: str) -> list[int]:
    body_name_to_id = {name: idx for idx, name in enumerate(robot.body_names)}
    ids: list[int] = []
    missing: list[str] = []
    for name in body_names:
        body_id = body_name_to_id.get(name)
        if body_id is None:
            missing.append(name)
        else:
            ids.append(body_id)
    if missing:
        print(f"[WARN] Missing {label} bodies in USD, ground lock disabled for these bodies: {missing}")
    else:
        print(f"  mapped {label} bodies={len(ids)}")
    return ids


def infer_default_foot_clearance(robot: Articulation, foot_body_ids: list[int], ground_height: float) -> float:
    if not foot_body_ids:
        return 0.0
    foot_z = robot.data.body_pos_w[:, foot_body_ids, 2]
    return max(float(torch.min(foot_z).item()) - ground_height, 0.0)


class RootMotionEstimator:
    """Estimate root xy motion by keeping the current support foot fixed in world space."""

    def __init__(
        self,
        mode: str,
        ground_height: float,
        foot_min_z: float,
        height_tolerance: float,
        switch_margin: float,
        max_step: float,
        source_motion_eps: float,
    ):
        self.mode = mode
        self.ground_height = ground_height
        self.foot_min_z = foot_min_z
        self.height_tolerance = height_tolerance
        self.switch_margin = switch_margin
        self.max_step = max_step
        self.source_motion_eps = source_motion_eps
        self.source_origin_xy: torch.Tensor | None = None
        self.source_root_is_moving = False
        self.stance_slot: int | None = None
        self.anchor_xy: torch.Tensor | None = None
        self.active_mode = "source"

    def _should_estimate(self, root_pose: torch.Tensor, source_has_root: bool) -> bool:
        if self.mode == "source":
            self.active_mode = "source"
            return False
        if self.mode == "stance":
            self.active_mode = "stance"
            return True
        if source_has_root:
            source_xy = root_pose[0, :2].detach().clone()
            if self.source_origin_xy is None:
                self.source_origin_xy = source_xy
            elif torch.linalg.norm(source_xy - self.source_origin_xy).item() > self.source_motion_eps:
                self.source_root_is_moving = True
            if self.source_root_is_moving:
                self.active_mode = "source"
                return False
        self.active_mode = "stance"
        return True

    def correct(
        self,
        sim: SimulationContext,
        robot: Articulation,
        root_pose: torch.Tensor,
        root_velocity: torch.Tensor,
        foot_body_ids: list[int],
        source_has_root: bool,
        source_root_authoritative: bool = False,
    ) -> torch.Tensor:
        if source_root_authoritative and self.mode != "stance":
            self.active_mode = "source"
            self.stance_slot = None
            self.anchor_xy = None
            return root_pose
        if not foot_body_ids or not self._should_estimate(root_pose, source_has_root):
            if self.active_mode == "source":
                self.stance_slot = None
                self.anchor_xy = None
            return root_pose

        sim.forward()
        robot.update(0.0)
        foot_pos = robot.data.body_pos_w[0, foot_body_ids, :3].detach()
        foot_height = foot_pos[:, 2] - (self.ground_height + self.foot_min_z)
        candidate_slot = int(torch.argmin(foot_height).item())

        if self.stance_slot is None or self.anchor_xy is None:
            self.stance_slot = candidate_slot
            self.anchor_xy = foot_pos[candidate_slot, :2].clone()
            return root_pose

        current_height = float(foot_height[self.stance_slot].item())
        candidate_height = float(foot_height[candidate_slot].item())
        current_contact = current_height <= self.height_tolerance
        candidate_contact = candidate_height <= self.height_tolerance
        should_switch = (
            candidate_slot != self.stance_slot
            and candidate_contact
            and ((not current_contact) or candidate_height < current_height - self.switch_margin)
        )
        if should_switch:
            self.stance_slot = candidate_slot
            self.anchor_xy = foot_pos[candidate_slot, :2].clone()
            return root_pose

        delta_xy = self.anchor_xy - foot_pos[self.stance_slot, :2]
        delta_norm = float(torch.linalg.norm(delta_xy).item())
        if self.max_step > 0.0 and delta_norm > self.max_step:
            delta_xy = delta_xy * (self.max_step / max(delta_norm, 1.0e-9))
            self.anchor_xy = foot_pos[self.stance_slot, :2] + delta_xy

        if float(torch.linalg.norm(delta_xy).item()) > 1.0e-7:
            root_pose[:, :2] += delta_xy.unsqueeze(0)
            robot.write_root_link_pose_to_sim(root_pose)
            robot.write_root_link_velocity_to_sim(root_velocity)
            sim.forward()
            robot.update(0.0)
        return root_pose


def apply_sample(
    sim: SimulationContext,
    robot: Articulation,
    sample: StateSample,
    mujoco_ids: list[int],
    isaac_ids: list[int],
    left_hand_ids: list[int],
    right_hand_ids: list[int],
    foot_body_ids: list[int],
    foot_min_z: float,
    root_motion: RootMotionEstimator | None,
    source_root_authoritative: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    q = torch.tensor(sample.joint_pos_mujoco[mujoco_ids], dtype=torch.float32, device=device).unsqueeze(0)
    joint_pos[:, isaac_ids] = q
    if sample.joint_vel_mujoco is not None:
        dq = torch.tensor(sample.joint_vel_mujoco[mujoco_ids], dtype=torch.float32, device=device).unsqueeze(0)
        joint_vel[:, isaac_ids] = dq
    if sample.left_hand_pos is not None and len(left_hand_ids) == len(LEFT_HAND_JOINT_NAMES):
        left_q = torch.tensor(sample.left_hand_pos[:7], dtype=torch.float32, device=device).unsqueeze(0)
        joint_pos[:, left_hand_ids] = left_q
    if sample.right_hand_pos is not None and len(right_hand_ids) == len(RIGHT_HAND_JOINT_NAMES):
        right_q = torch.tensor(sample.right_hand_pos[:7], dtype=torch.float32, device=device).unsqueeze(0)
        joint_pos[:, right_hand_ids] = right_q
    if sample.left_hand_vel is not None and len(left_hand_ids) == len(LEFT_HAND_JOINT_NAMES):
        left_dq = torch.tensor(sample.left_hand_vel[:7], dtype=torch.float32, device=device).unsqueeze(0)
        joint_vel[:, left_hand_ids] = left_dq
    if sample.right_hand_vel is not None and len(right_hand_ids) == len(RIGHT_HAND_JOINT_NAMES):
        right_dq = torch.tensor(sample.right_hand_vel[:7], dtype=torch.float32, device=device).unsqueeze(0)
        joint_vel[:, right_hand_ids] = right_dq
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    root_pose = robot.data.default_root_state[:, :7].clone()
    if sample.root_pos_w is not None:
        root_pose[:, :3] = torch.tensor(sample.root_pos_w, dtype=torch.float32, device=device).unsqueeze(0)
    if sample.root_quat_w is not None:
        root_pose[:, 3:7] = torch.tensor(sample.root_quat_w, dtype=torch.float32, device=device).unsqueeze(0)
    root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
    robot.write_root_link_pose_to_sim(root_pose)
    robot.write_root_link_velocity_to_sim(root_velocity)
    if root_motion is not None:
        root_pose = root_motion.correct(
            sim,
            robot,
            root_pose,
            root_velocity,
            foot_body_ids,
            sample.root_pos_w is not None,
            source_root_authoritative,
        )

    if args_cli.ground_lock and foot_body_ids:
        sim.forward()
        robot.update(0.0)
        target_min_z = args_cli.ground_height + foot_min_z
        current_min_z = float(torch.min(robot.data.body_pos_w[:, foot_body_ids, 2]).item())
        z_correction = max(target_min_z - current_min_z, 0.0)
        if z_correction > 1.0e-5:
            root_pose[:, 2] += z_correction
            robot.write_root_link_pose_to_sim(root_pose)
            robot.write_root_link_velocity_to_sim(root_velocity)
            sim.forward()
            robot.update(0.0)
    return root_pose[0, :3].detach().clone(), root_pose[0, 3:7].detach().clone()


def update_camera(sim: SimulationContext, root_pos: torch.Tensor) -> None:
    root = root_pos.detach().cpu().numpy()
    target = [float(root[0]), float(root[1]), float(root[2] + 0.35)]
    eye = [float(root[0] + 2.4), float(root[1] - 3.2), float(root[2] + 1.35)]
    sim.set_camera_view(eye, target)


def run() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.4, -3.2, 1.8], [0.0, 0.0, 0.85])
    robot = design_scene()

    sim.reset()
    robot.update(sim.get_physics_dt())
    mujoco_ids, isaac_ids, _ = build_mujoco_to_isaac_joint_ids(robot)
    left_hand_ids = build_optional_isaac_joint_ids(robot, LEFT_HAND_JOINT_NAMES, "left hand")
    right_hand_ids = build_optional_isaac_joint_ids(robot, RIGHT_HAND_JOINT_NAMES, "right hand")
    foot_body_ids = build_required_body_ids(robot, FOOT_BODY_NAMES, "foot")
    if args_cli.ground_lock_clearance >= 0.0:
        foot_min_z = args_cli.ground_lock_clearance
    else:
        foot_min_z = infer_default_foot_clearance(robot, foot_body_ids, args_cli.ground_height)
    source = build_source()
    root_source = None
    root_stream_label = "root_stream"
    if args_cli.source == "zmq" and args_cli.root_zmq:
        root_source = RootZmqSource(
            host=args_cli.root_zmq_host,
            port=args_cli.root_zmq_port,
            topic=args_cli.root_zmq_topic,
            root_z_offset=args_cli.root_z_offset,
            timeout=args_cli.zmq_timeout,
        )
        root_stream_label = "root_zmq"
    elif args_cli.source == "udp" and args_cli.root_udp:
        root_source = RootUdpSource(
            bind_host=args_cli.root_udp_bind_host,
            port=args_cli.root_udp_port,
            topic=args_cli.root_udp_topic,
            root_z_offset=args_cli.root_z_offset,
            timeout=args_cli.udp_timeout,
            rcvbuf=args_cli.root_udp_rcvbuf,
        )
        root_stream_label = "root_udp"
    root_motion = RootMotionEstimator(
        mode=args_cli.root_motion_mode,
        ground_height=args_cli.ground_height,
        foot_min_z=foot_min_z,
        height_tolerance=args_cli.stance_foot_height_tolerance,
        switch_margin=args_cli.stance_foot_switch_margin,
        max_step=args_cli.stance_root_max_step,
        source_motion_eps=args_cli.source_root_motion_eps,
    )

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step = 0
    print("[INFO] Setup complete.")
    print(f"[INFO] source={args_cli.source} dt={sim_dt:.4f}s playback_speed={args_cli.playback_speed:.3f}")
    print(
        f"[INFO] ground_lock={args_cli.ground_lock} ground_height={args_cli.ground_height:.3f} "
        f"foot_min_z={foot_min_z:.4f}"
    )
    print(f"[INFO] root_motion_mode={args_cli.root_motion_mode}")
    if args_cli.source == "csv":
        print(f"[INFO] trajectory={args_cli.trajectory_dir}")
        print(f"[INFO] csv_joint_order={args_cli.csv_joint_order}")
    if args_cli.source in {"zmq", "udp"}:
        print(f"[INFO] network_joint_order={args_cli.zmq_joint_order}")
        print(f"[INFO] network_pose_source={args_cli.zmq_pose_source}")
        if root_source is not None:
            if args_cli.source == "zmq":
                print(
                    f"[INFO] root_zmq={args_cli.root_zmq_host}:{args_cli.root_zmq_port}/"
                    f"{args_cli.root_zmq_topic}"
                )
            else:
                print(
                    f"[INFO] root_udp={args_cli.root_udp_bind_host}:{args_cli.root_udp_port}/"
                    f"{args_cli.root_udp_topic}"
                )
        if args_cli.zmq_warmup_sec > 0.0:
            deadline = time.monotonic() + args_cli.zmq_warmup_sec
            warmed_debug = False
            warmed_root = root_source is None
            while time.monotonic() < deadline and not (warmed_debug and warmed_root):
                warm_sample = source.sample(sim_time)
                warmed_debug = warmed_debug or warm_sample.fresh
                if root_source is not None:
                    root_source.sample()
                    warmed_root = warmed_root or root_source.fresh
                time.sleep(0.01)
            print(f"[INFO] network_warmup debug={warmed_debug} root={warmed_root}")

    try:
        while simulation_app.is_running():
            sample = source.sample(sim_time * args_cli.playback_speed)
            root_authoritative = False
            if root_source is not None:
                root_sample = root_source.sample()
                if root_sample is not None and root_source.fresh:
                    sample.root_pos_w, sample.root_quat_w = root_sample
                    root_authoritative = True
            root_pos, _ = apply_sample(
                sim,
                robot,
                sample,
                mujoco_ids,
                isaac_ids,
                left_hand_ids,
                right_hand_ids,
                foot_body_ids,
                foot_min_z,
                root_motion,
                root_authoritative,
                sim.device,
            )
            if not args_cli.no_camera_follow and step % max(args_cli.camera_update_interval, 1) == 0:
                update_camera(sim, root_pos)
            sim.step()
            robot.update(sim_dt)
            if args_cli.print_interval > 0 and step % args_cli.print_interval == 0:
                frame = "-" if sample.source_frame is None else str(sample.source_frame)
                fresh = "fresh" if sample.fresh else "stale"
                q_absmax = float(np.max(np.abs(sample.joint_pos_mujoco[:29])))
                lh_absmax = (
                    float(np.max(np.abs(sample.left_hand_pos[:7]))) if sample.left_hand_pos is not None else float("nan")
                )
                rh_absmax = (
                    float(np.max(np.abs(sample.right_hand_pos[:7]))) if sample.right_hand_pos is not None else float("nan")
                )
                foot_z = (
                    float(torch.min(robot.data.body_pos_w[:, foot_body_ids, 2]).item()) if foot_body_ids else float("nan")
                )
                print(
                    f"[INFO] step={step} sim_t={sim_time:.3f} frame={frame} "
                    f"state={fresh} q_absmax={q_absmax:.3f} "
                    f"lh_absmax={lh_absmax:.3f} rh_absmax={rh_absmax:.3f} "
                    f"root_motion={root_motion.active_mode} {root_stream_label}={'fresh' if root_authoritative else 'none'} "
                    f"root=({float(root_pos[0]):.3f},{float(root_pos[1]):.3f},{float(root_pos[2]):.3f}) "
                    f"foot_z={foot_z:.4f}"
                )
            step += 1
            sim_time += sim_dt
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
            if sample.done:
                break
    finally:
        if hasattr(source, "close"):
            source.close()
        if root_source is not None:
            root_source.close()


def close_app() -> None:
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        return
    except TypeError:
        pass
    simulation_app.close()


if __name__ == "__main__":
    run()
    close_app()
