# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Stream stereo color fisheye images from Isaac Lab over ZMQ.

Usage examples:

    isaaclab.bat -p scripts/robotyao/stream_stereo_fisheye_zmq.py --headless
    isaaclab.bat -p scripts/robotyao/stream_stereo_fisheye_zmq.py --width 960 --height 960 --fps 15

The stream is a PUB multipart message:

    [topic, header_json, left_image_bytes, right_image_bytes]

Default topic: ``robotyao.stereo.fisheye.v1``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import traceback
import time
from fractions import Fraction

_ROBOTYAO_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROBOTYAO_CACHE_ROOT = os.path.join(_ROBOTYAO_REPO_ROOT, ".robotyao_cache")
_ROBOTYAO_WARP_CACHE_PATH = os.environ.setdefault(
    "WARP_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "warp")
)
_ROBOTYAO_OPTIX_CACHE_PATH = os.environ.setdefault(
    "OPTIX_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "optix")
)
_ROBOTYAO_CUDA_CACHE_PATH = os.environ.setdefault(
    "CUDA_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "cuda_cache")
)
_ROBOTYAO_NV_SHADER_CACHE_PATH = os.environ.setdefault(
    "NV_SHADER_DISK_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "nv_shader_cache")
)
for _robotyao_cache_path in (
    _ROBOTYAO_WARP_CACHE_PATH,
    _ROBOTYAO_OPTIX_CACHE_PATH,
    _ROBOTYAO_CUDA_CACHE_PATH,
    _ROBOTYAO_NV_SHADER_CACHE_PATH,
):
    os.makedirs(_robotyao_cache_path, exist_ok=True)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RobotYao stereo fisheye RGB ZMQ streamer.")
parser.add_argument("--endpoint", type=str, default="tcp://*:5556", help="ZMQ PUB bind endpoint.")
parser.add_argument("--topic", type=str, default="robotyao.stereo.fisheye.v1", help="ZMQ topic.")
parser.add_argument("--width", type=int, default=1920, help="Camera image width.")
parser.add_argument("--height", type=int, default=1920, help="Camera image height.")
parser.add_argument("--fps", type=float, default=30.0, help="Target publish FPS. Use 0 for every rendered frame.")
parser.add_argument("--encoding", choices=["h264", "jpg"], default="h264", help="Image payload encoding.")
parser.add_argument("--jpeg_quality", type=int, default=85, help="JPEG quality when --encoding jpg is used.")
parser.add_argument("--h264_bitrate", type=int, default=12_000_000, help="H264 target bitrate per eye.")
parser.add_argument("--h264_gop", type=int, default=30, help="H264 keyframe interval per eye.")
parser.add_argument("--h264_preset", type=str, default="ultrafast", help="libx264 preset.")
parser.add_argument("--h264_profile", type=str, default="baseline", help="libx264 profile.")
parser.add_argument("--baseline", type=float, default=0.064, help="Stereo camera baseline in meters.")
parser.add_argument("--fisheye_fov", type=float, default=180.0, help="Fisheye field of view in degrees.")
parser.add_argument("--show-camera-lenses", action="store_true", help="Show debug lens spheres in the simulated scene.")
parser.add_argument("--warmup_frames", type=int, default=8, help="Frames to render before publishing.")
parser.add_argument("--print_every", type=int, default=60, help="Print status every N published frames.")
parser.add_argument("--max_frames", type=int, default=0, help="Stop after N simulation frames. Use 0 to run forever.")
parser.add_argument(
    "--task-scene",
    action="store_true",
    help="Run the registered Isaac Lab Agibot Toy2Box task scene instead of the built-in simple test scene.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    help="Isaac Lab task id used when --task-scene is enabled.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of task environments when --task-scene is enabled.")
parser.add_argument(
    "--task-use-rmpflow",
    action="store_true",
    help="Use the task's original RMPFlow arm action. Requires Agibot RMPFlow assets to be available.",
)
parser.add_argument(
    "--allow-remote-rmpflow-assets",
    action="store_true",
    help="Allow blocking remote lookup/download of Agibot RMPFlow assets. By default RobotYao falls back if assets are not local.",
)
parser.add_argument(
    "--task-camera-mount",
    choices=["head_link", "root"],
    default="head_link",
    help="Task-scene stereo camera mount mode. head_link parents cameras under the Agibot head link.",
)
parser.add_argument(
    "--task-camera-head-link",
    type=str,
    default="link_pitch_head",
    help="Agibot robot link used as the strict parent for task-scene stereo cameras.",
)
parser.add_argument(
    "--task-camera-head-forward-offset",
    type=float,
    default=0.10,
    help="Head-link local +X offset for the stereo lens centers, in meters.",
)
parser.add_argument(
    "--task-camera-head-up-offset",
    type=float,
    default=-0.03,
    help="Head-link local +Z offset for the stereo lens centers, in meters.",
)
parser.add_argument(
    "--task-camera-head-look-down-deg",
    type=float,
    default=0.0,
    help="Additional local downward pitch applied to the head-mounted stereo cameras. 0 follows the head link.",
)
parser.add_argument(
    "--task-camera-head-rig-x",
    type=float,
    default=-0.25597,
    help="Head-link local X translation for the RobotYaoTaskStereo rig parent, in meters.",
)
parser.add_argument(
    "--task-camera-head-rig-y",
    type=float,
    default=0.15846,
    help="Head-link local Y translation for the RobotYaoTaskStereo rig parent, in meters.",
)
parser.add_argument(
    "--task-camera-head-rig-z",
    type=float,
    default=0.0,
    help="Head-link local Z translation for the RobotYaoTaskStereo rig parent, in meters.",
)
parser.add_argument(
    "--task-camera-head-rig-roll-deg",
    type=float,
    default=-90.0,
    help="Head-link local X rotation for the RobotYaoTaskStereo rig parent, in degrees.",
)
parser.add_argument(
    "--task-camera-head-rig-pitch-deg",
    type=float,
    default=0.0,
    help="Head-link local Y rotation for the RobotYaoTaskStereo rig parent, in degrees.",
)
parser.add_argument(
    "--task-camera-head-rig-yaw-deg",
    type=float,
    default=180.0,
    help="Head-link local Z rotation for the RobotYaoTaskStereo rig parent, in degrees.",
)
parser.add_argument(
    "--task-camera-forward-offset",
    type=float,
    default=0.62,
    help="Root-mount fallback forward offset from Agibot root, in meters.",
)
parser.add_argument(
    "--task-camera-height-offset",
    type=float,
    default=2.10,
    help="Root-mount fallback height offset from Agibot root, in meters.",
)
parser.add_argument(
    "--task-camera-look-down-deg",
    type=float,
    default=65.0,
    help="Root-mount fallback stereo rig downward look angle in degrees. 0 keeps the camera level.",
)
parser.add_argument("--debug-task-loop", action="store_true", help="Print detailed task-loop diagnostics.")
parser.add_argument(
    "--debug-save-frame-dir",
    type=str,
    default="",
    help="Save the first published left/right RGB source frames as PNG files for stereo verification.",
)
parser.add_argument(
    "--no-fast-exit-on-max-frames",
    action="store_true",
    help="Use SimulationApp.close() after a finite --max_frames smoke run instead of immediate process exit.",
)
parser.add_argument(
    "--clean-kit-shutdown",
    action="store_true",
    help="Call SimulationApp.close() on exit. Disabled by default because Isaac Sim syntheticdata shutdown can crash on Windows.",
)
parser.add_argument("--unity-control", action="store_true", help="Receive Unity controller input over ZMQ and drive the robot.")
parser.add_argument("--xr-control", action="store_true", help=argparse.SUPPRESS)
parser.add_argument(
    "--unity-input-endpoint",
    "--xr-endpoint",
    dest="unity_input_endpoint",
    type=str,
    default="tcp://127.0.0.1:5555",
    help="Unity controller-input ZMQ PUB endpoint. Isaac Lab connects as a normal ZMQ SUB client.",
)
parser.add_argument(
    "--unity-input-topic",
    "--xr-topic",
    dest="unity_input_topic",
    type=str,
    default="state",
    help="Unity controller-input ZMQ topic.",
)
parser.add_argument(
    "--unity-follow-mode",
    "--xr-follow-mode",
    dest="unity_follow_mode",
    choices=["toggle", "hold"],
    default="toggle",
    help="Right-hand B/A arm-follow mode. Default: B starts following, A stops. Hold mode follows only while B is held.",
)
parser.add_argument("--max-forward-speed", type=float, default=1.0, help="Left stick Y forward speed in m/s.")
parser.add_argument("--max-lateral-speed", type=float, default=0.6, help="Left stick X lateral walking speed in m/s. Use 0 to disable strafe.")
parser.add_argument("--max-yaw-rate", type=float, default=1.2, help="Right stick X yaw rate in rad/s.")
parser.add_argument("--arm-delta-scale", type=float, default=1.0, help="Scale for incremental controller-to-arm deltas.")
parser.add_argument(
    "--arm-rmpflow-axis-map",
    type=str,
    default="y,-x,z",
    help=(
        "Comma-separated scene-delta axes used for the Agibot RMPFlow xyz action. "
        "Default y,-x,z maps controller/scene +X right to robot/RMPFlow -Y right, "
        "+Y forward to +X forward, and +Z up to +Z up."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.xr_control:
    args_cli.unity_control = True

if args_cli.encoding == "h264" and (args_cli.width % 2 != 0 or args_cli.height % 2 != 0):
    raise ValueError("H264 yuv420p encoding requires even --width and --height values.")

try:
    import zmq
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "pyzmq is required in the active Isaac Lab Python environment. "
        "Install it with: .\\isaaclab.bat -p -m pip install pyzmq"
    ) from exc

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import av
except ModuleNotFoundError:
    av = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

if args_cli.encoding == "h264" and av is None:
    raise ModuleNotFoundError(
        "H264 encoding requires PyAV in the active Isaac Lab Python environment. "
        "Install it with: .\\isaaclab.bat -p -m pip install av"
    )

if args_cli.encoding == "jpg" and cv2 is None and Image is None:
    raise ModuleNotFoundError(
        "JPEG encoding requires either OpenCV or Pillow in the active Isaac Lab Python environment. "
        "Install one with: .\\isaaclab.bat -p -m pip install pillow"
    )

# Camera sensors require this flag; force-enable it so the script is hard to misuse.
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

if hasattr(args_cli, "kit_args"):
    for cache_dir in ("texturecache", "exts", "datastore", "kit_cache", "kit_portable"):
        os.makedirs(os.path.join(_ROBOTYAO_CACHE_ROOT, cache_dir), exist_ok=True)
    _robotyao_cache_arg_path = _ROBOTYAO_CACHE_ROOT.replace("\\", "/")
    _robotyao_existing_kit_args = str(args_cli.kit_args or "").strip()
    _robotyao_extra_kit_args = [
        f"--/rtx-transient/resourcemanager/localTextureCachePath={_robotyao_cache_arg_path}/texturecache",
        f"--/exts/omni.kit.registry.nucleus/cachePath={_robotyao_cache_arg_path}/exts",
        f"--/UJITSO/datastore/GRPCDataStoreServer/cachePath={_robotyao_cache_arg_path}/datastore",
        f"--/app/cachePath={_robotyao_cache_arg_path}/kit_cache",
    ]
    if "--portable-root" not in _robotyao_existing_kit_args:
        _robotyao_extra_kit_args.extend(["--portable-root", f"{_robotyao_cache_arg_path}/kit_portable"])
    args_cli.kit_args = " ".join(filter(None, [_robotyao_existing_kit_args, *_robotyao_extra_kit_args]))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from pxr import Gf, UsdGeom

import omni.usd

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.devices.openxr.robotyao_xr_sub_device import RobotYaoXrSubDevice, RobotYaoXrSubDeviceCfg
from isaaclab.devices.openxr.retargeters.robotyao_wheeled_xr_retargeter import (
    RobotYaoWheeledXrRetargeter,
    RobotYaoWheeledXrRetargeterCfg,
)


def _spawn_shape(path: str, cfg, translation: tuple[float, float, float]):
    cfg.func(path, cfg, translation=translation)


def _preview_material(color: tuple[float, float, float], metallic: float = 0.0, roughness: float = 0.35):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=roughness)


def _parse_axis_map(spec: str) -> list[tuple[int, float, str]]:
    """Parse a comma-separated xyz axis map with optional signs."""
    axis_indices = {"x": 0, "y": 1, "z": 2}
    raw = spec.strip().lower().replace(" ", "")
    tokens = raw.split(",") if "," in raw else list(raw)
    if len(tokens) != 3:
        raise ValueError(
            f"Invalid --arm-rmpflow-axis-map '{spec}'. Expected three axes, for example 'y,-x,z' or 'y,x,z'."
        )

    axis_map: list[tuple[int, float, str]] = []
    for token in tokens:
        if not token:
            raise ValueError(f"Invalid --arm-rmpflow-axis-map '{spec}': empty axis token.")
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token[1:] if token.startswith("-") else token
        if axis not in axis_indices:
            raise ValueError(
                f"Invalid --arm-rmpflow-axis-map '{spec}': axis token '{token}' is not one of x, y, z, -x, -y, -z."
            )
        axis_map.append((axis_indices[axis], sign, f"{'-' if sign < 0 else ''}{axis}"))
    return axis_map


def _format_axis_map(axis_map: list[tuple[int, float, str]]) -> str:
    return ",".join(token for _, _, token in axis_map)


def _apply_axis_map_tensor(delta: torch.Tensor, axis_map: list[tuple[int, float, str]]) -> torch.Tensor:
    return torch.stack([delta[index] * sign for index, sign, _ in axis_map]).to(delta)


def _fisheye_full_frame_poly_b(width: int, height: int, fisheye_fov: float) -> float:
    """Return the linear f-theta coefficient that maps max FOV to the image radius."""
    image_radius_px = max(min(float(width), float(height)) * 0.5, 1.0)
    max_theta_rad = math.radians(float(fisheye_fov) * 0.5)
    return max_theta_rad / image_radius_px


def _design_scene(width: int, height: int, baseline: float, fisheye_fov: float, show_camera_lenses: bool) -> Camera:
    """Create a simple wheeled robot, scene references and two color fisheye cameras."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    dome_light_cfg = sim_utils.DomeLightCfg(intensity=2400.0, color=(0.82, 0.86, 0.92))
    dome_light_cfg.func("/World/Light", dome_light_cfg)

    sim_utils.create_prim("/World/Robot", "Xform", translation=(0.0, 0.0, 0.0))
    _spawn_shape(
        "/World/Robot/body",
        sim_utils.CuboidCfg(size=(1.15, 0.72, 0.28), visual_material=_preview_material((0.12, 0.18, 0.22))),
        (0.0, 0.0, 0.42),
    )
    _spawn_shape(
        "/World/Robot/top_plate",
        sim_utils.CuboidCfg(size=(0.72, 0.48, 0.08), visual_material=_preview_material((0.20, 0.26, 0.30))),
        (0.1, 0.0, 0.64),
    )
    _spawn_shape(
        "/World/Robot/neck",
        sim_utils.CylinderCfg(
            radius=0.055, height=0.32, axis="Z", visual_material=_preview_material((0.26, 0.28, 0.28))
        ),
        (0.38, 0.0, 0.82),
    )
    _spawn_shape(
        "/World/Robot/head",
        sim_utils.CuboidCfg(size=(0.28, 0.26, 0.18), visual_material=_preview_material((0.08, 0.11, 0.13))),
        (0.46, 0.0, 1.03),
    )
    for side_name, side_sign, color in (
        ("left", 1.0, (0.20, 0.42, 0.95)),
        ("right", -1.0, (0.95, 0.28, 0.18)),
    ):
        shoulder_y = 0.38 * side_sign
        hand_y = 0.58 * side_sign
        _spawn_shape(
            f"/World/Robot/{side_name}_shoulder",
            sim_utils.SphereCfg(radius=0.055, visual_material=_preview_material((0.52, 0.55, 0.58), metallic=0.2)),
            (0.24, shoulder_y, 0.72),
        )
        _spawn_shape(
            f"/World/Robot/{side_name}_arm_link",
            sim_utils.CuboidCfg(size=(0.32, 0.045, 0.045), visual_material=_preview_material((0.18, 0.22, 0.26))),
            (0.39, (shoulder_y + hand_y) * 0.5, 0.65),
        )
        _spawn_shape(
            f"/World/Robot/{side_name}_hand_target",
            sim_utils.SphereCfg(radius=0.07, visual_material=_preview_material(color, metallic=0.05)),
            (0.54, hand_y, 0.62),
        )

    wheel_mat = _preview_material((0.015, 0.015, 0.018), roughness=0.55)
    hub_mat = _preview_material((0.55, 0.58, 0.60), metallic=0.2)
    for index, x in enumerate((-0.38, 0.38)):
        for side, y in (("left", 0.43), ("right", -0.43)):
            wheel_path = f"/World/Robot/wheel_{index}_{side}"
            _spawn_shape(
                wheel_path,
                sim_utils.CylinderCfg(radius=0.16, height=0.11, axis="Y", visual_material=wheel_mat),
                (x, y, 0.28),
            )
            _spawn_shape(
                f"{wheel_path}_hub",
                sim_utils.CylinderCfg(radius=0.07, height=0.125, axis="Y", visual_material=hub_mat),
                (x, y, 0.28),
            )

    # Visible references in front of the robot make fisheye alignment easier to inspect.
    sim_utils.create_prim("/World/Targets", "Xform")
    target_specs = [
        ("front_red", (3.0, -0.8, 0.45), (0.85, 0.12, 0.10), "Cube"),
        ("front_green", (3.2, 0.8, 0.45), (0.10, 0.72, 0.22), "Cylinder"),
        ("center_blue", (2.4, 0.0, 0.8), (0.12, 0.22, 0.86), "Cone"),
        ("left_yellow", (1.4, 1.5, 0.6), (0.92, 0.72, 0.12), "Cube"),
        ("right_cyan", (1.4, -1.5, 0.6), (0.10, 0.70, 0.78), "Cube"),
    ]
    for name, pos, color, kind in target_specs:
        common = {"visual_material": _preview_material(color, metallic=0.05), "semantic_tags": [("class", name)]}
        if kind == "Cube":
            cfg = sim_utils.CuboidCfg(size=(0.35, 0.35, 0.35), **common)
        elif kind == "Cylinder":
            cfg = sim_utils.CylinderCfg(radius=0.22, height=0.45, axis="Z", **common)
        else:
            cfg = sim_utils.ConeCfg(radius=0.24, height=0.48, axis="Z", **common)
        _spawn_shape(f"/World/Targets/{name}", cfg, pos)

    for i in range(8):
        x = random.uniform(1.2, 4.0)
        y = random.uniform(-2.2, 2.2)
        z = random.uniform(0.20, 0.85)
        color = (random.random() * 0.8 + 0.1, random.random() * 0.8 + 0.1, random.random() * 0.8 + 0.1)
        cfg = sim_utils.CuboidCfg(
            size=(0.15, 0.15, random.uniform(0.2, 0.7)),
            visual_material=_preview_material(color),
            semantic_tags=[("class", "random_marker")],
        )
        _spawn_shape(f"/World/Targets/random_{i:02d}", cfg, (x, y, z))

    # Parent Xforms carry the stereo baseline. The cameras use world convention:
    # forward +X, up +Z. This matches the robot's forward direction in this scene.
    stereo_x = 0.61
    stereo_z = 1.06
    sim_utils.create_prim("/World/Robot/Stereo", "Xform")
    sim_utils.create_prim("/World/Robot/Stereo/Left", "Xform", translation=(stereo_x, baseline * 0.5, stereo_z))
    sim_utils.create_prim("/World/Robot/Stereo/Right", "Xform", translation=(stereo_x, -baseline * 0.5, stereo_z))

    if show_camera_lenses:
        lens_mat_left = _preview_material((0.15, 0.32, 0.85), metallic=0.1)
        lens_mat_right = _preview_material((0.85, 0.22, 0.16), metallic=0.1)
        _spawn_shape(
            "/World/Robot/left_lens_visual",
            sim_utils.SphereCfg(radius=0.035, visual_material=lens_mat_left),
            (stereo_x + 0.04, baseline * 0.5, stereo_z),
        )
        _spawn_shape(
            "/World/Robot/right_lens_visual",
            sim_utils.SphereCfg(radius=0.035, visual_material=lens_mat_right),
            (stereo_x + 0.04, -baseline * 0.5, stereo_z),
        )

    fisheye_poly_b = _fisheye_full_frame_poly_b(width, height, fisheye_fov)
    camera_cfg = CameraCfg(
        prim_path="/World/Robot/Stereo/.*/Fisheye",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            focal_length=5.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=10.0,
            clipping_range=(0.03, 1.0e5),
            fisheye_nominal_width=float(width),
            fisheye_nominal_height=float(height),
            fisheye_optical_centre_x=float(width) * 0.5,
            fisheye_optical_centre_y=float(height) * 0.5,
            fisheye_max_fov=float(fisheye_fov),
            fisheye_polynomial_a=0.0,
            fisheye_polynomial_b=fisheye_poly_b,
            fisheye_polynomial_c=0.0,
            fisheye_polynomial_d=0.0,
            fisheye_polynomial_e=0.0,
            fisheye_polynomial_f=0.0,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    return Camera(cfg=camera_cfg)


class H264EyeEncoder:
    """Low-latency H264 encoder for one eye."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        bitrate: int,
        gop: int,
        preset: str,
        profile: str,
    ):
        if av is None:
            raise RuntimeError(
                "H264 encoding requires PyAV in the active Isaac Lab Python environment. "
                "Install it with: conda run -n env_isaaclab python -m pip install av"
            )

        rate = Fraction(max(float(fps), 1.0)).limit_denominator(1000)
        self._codec = av.CodecContext.create("libx264", "w")
        self._codec.width = int(width)
        self._codec.height = int(height)
        self._codec.pix_fmt = "yuv420p"
        self._codec.time_base = Fraction(rate.denominator, rate.numerator)
        self._codec.framerate = rate
        self._codec.bit_rate = int(bitrate)
        self._codec.gop_size = max(int(gop), 1)
        self._codec.max_b_frames = 0
        self._codec.options = {
            "preset": preset,
            "tune": "zerolatency",
            "profile": profile,
            "x264-params": (
                f"keyint={max(int(gop), 1)}:"
                f"min-keyint={max(int(gop), 1)}:"
                "scenecut=0:"
                "repeat-headers=1:"
                "bframes=0:"
                "aud=1"
            ),
        }
        self._codec.open()
        self._frame_index = 0

    def encode(self, rgb: np.ndarray) -> bytes:
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self._frame_index
        self._frame_index += 1
        return b"".join(bytes(packet) for packet in self._codec.encode(frame))

    def close(self):
        # Drain delayed packets for clean shutdown. The live stream uses zerolatency settings,
        # so this should normally be empty.
        list(self._codec.encode(None))


class StereoH264Encoder:
    def __init__(self):
        self._left = H264EyeEncoder(
            args_cli.width,
            args_cli.height,
            args_cli.fps,
            args_cli.h264_bitrate,
            args_cli.h264_gop,
            args_cli.h264_preset,
            args_cli.h264_profile,
        )
        self._right = H264EyeEncoder(
            args_cli.width,
            args_cli.height,
            args_cli.fps,
            args_cli.h264_bitrate,
            args_cli.h264_gop,
            args_cli.h264_preset,
            args_cli.h264_profile,
        )

    def encode(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> tuple[bytes, bytes]:
        return self._left.encode(left_rgb), self._right.encode(right_rgb)

    def close(self):
        self._left.close()
        self._right.close()


def _encode_jpeg(rgb: np.ndarray, jpeg_quality: int) -> bytes:
    """Encode one RGB uint8 image as JPEG."""
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return encoded.tobytes()

    if Image is None:
        raise RuntimeError("JPEG encoding requires either cv2 or Pillow in the Isaac Lab Python environment.")

    stream = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(stream, format="JPEG", quality=int(jpeg_quality), optimize=False)
    return stream.getvalue()


def _save_rgb_png(rgb: np.ndarray, output_path: str) -> None:
    """Save one RGB uint8 image as a PNG for local source-frame verification."""
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(output_path, bgr):
            raise RuntimeError(f"cv2.imwrite failed: {output_path}")
        return

    if Image is None:
        raise RuntimeError("Debug PNG saving requires either cv2 or Pillow in the Isaac Lab Python environment.")

    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")


def _save_debug_frame_pair(scene_name: str, frame_id: int, left_rgb: np.ndarray, right_rgb: np.ndarray) -> None:
    """Save one left/right source-frame pair when --debug-save-frame-dir is configured."""
    if not args_cli.debug_save_frame_dir:
        return

    output_dir = os.path.abspath(args_cli.debug_save_frame_dir)
    os.makedirs(output_dir, exist_ok=True)
    left_path = os.path.join(output_dir, f"{scene_name}_frame_{frame_id:06d}_left.png")
    right_path = os.path.join(output_dir, f"{scene_name}_frame_{frame_id:06d}_right.png")
    _save_rgb_png(left_rgb, left_path)
    _save_rgb_png(right_rgb, right_path)
    print(f"[RobotYao] Saved stereo source frames: {left_path} | {right_path}", flush=True)


class StereoZmqPublisher:
    def __init__(self, endpoint: str, topic: str):
        self._topic = topic.encode("utf-8")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.bind(endpoint)

    def close(self):
        self._socket.close(0)
        self._context.term()

    def send(self, header: dict, left_payload: bytes, right_payload: bytes):
        header_payload = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self._socket.send_multipart([self._topic, header_payload, left_payload, right_payload], flags=zmq.NOBLOCK)


def _set_xform_common(path: str, translation: np.ndarray, yaw_rad: float | None = None) -> None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return

    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
    if yaw_rad is not None:
        api.SetRotate(
            Gf.Vec3f(0.0, 0.0, float(math.degrees(yaw_rad))), UsdGeom.XformCommonAPI.RotationOrderXYZ
        )


def _set_xform_common_euler_xyz(
    path: str,
    translation: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
) -> None:
    """Set a USD Xform using the same translate/orient XYZ fields shown in the Inspector."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return

    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
    api.SetRotate(
        Gf.Vec3f(float(rotation_deg[0]), float(rotation_deg[1]), float(rotation_deg[2])),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )


def _yaw_from_quat_wxyz(quat: torch.Tensor | np.ndarray) -> float:
    """Return yaw angle in radians from an Isaac Lab wxyz quaternion."""
    if isinstance(quat, torch.Tensor):
        values = quat.detach().cpu().numpy()
    else:
        values = np.asarray(quat, dtype=np.float32)
    w, x, y, z = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_wxyz_from_yaw(yaw_rad: float, device: torch.device | str) -> torch.Tensor:
    """Create an Isaac Lab wxyz quaternion for a yaw-only root pose."""
    half = 0.5 * float(yaw_rad)
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=torch.float32, device=device)


def _quat_wxyz_from_pitch_tuple(pitch_rad: float) -> tuple[float, float, float, float]:
    """Create a wxyz quaternion tuple for a local Y-axis pitch."""
    half = 0.5 * float(pitch_rad)
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def _quat_wxyz_from_euler_xyz_tuple(
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    """Create a wxyz quaternion tuple from XYZ Euler angles."""
    cr = math.cos(0.5 * float(roll_rad))
    sr = math.sin(0.5 * float(roll_rad))
    cp = math.cos(0.5 * float(pitch_rad))
    sp = math.sin(0.5 * float(pitch_rad))
    cy = math.cos(0.5 * float(yaw_rad))
    sy = math.sin(0.5 * float(yaw_rad))
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _make_task_fisheye_camera_cfg(
    prim_path: str,
    width: int,
    height: int,
    fisheye_fov: float,
    offset_pos: tuple[float, float, float],
    offset_rot: tuple[float, float, float, float],
) -> CameraCfg:
    fisheye_poly_b = _fisheye_full_frame_poly_b(width, height, fisheye_fov)
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            focal_length=5.0,
            focus_distance=400.0,
            f_stop=0.0,
            horizontal_aperture=10.0,
            clipping_range=(0.03, 1.0e5),
            fisheye_nominal_width=float(width),
            fisheye_nominal_height=float(height),
            fisheye_optical_centre_x=float(width) * 0.5,
            fisheye_optical_centre_y=float(height) * 0.5,
            fisheye_max_fov=float(fisheye_fov),
            fisheye_polynomial_a=0.0,
            fisheye_polynomial_b=fisheye_poly_b,
            fisheye_polynomial_c=0.0,
            fisheye_polynomial_d=0.0,
            fisheye_polynomial_e=0.0,
            fisheye_polynomial_f=0.0,
        ),
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="world"),
    )


def _spawn_task_camera_lens_visuals(camera_prim_path: str, color: tuple[float, float, float]) -> None:
    """Spawn debug lens spheres under the camera prim so their centers equal the real camera origin."""
    lens_mat = _preview_material(color, metallic=0.1)
    for prim_path in sim_utils.find_matching_prim_paths(camera_prim_path):
        _spawn_shape(
            f"{prim_path}/LensVisual",
            sim_utils.SphereCfg(radius=0.035, visual_material=lens_mat),
            (0.0, 0.0, 0.0),
        )


def _design_task_scene_stereo_cameras(
    width: int, height: int, fisheye_fov: float, show_camera_lenses: bool
) -> Camera | tuple[Camera, Camera]:
    """Create task-scene stereo fisheye cameras, preferably parented under the Agibot head link."""
    if args_cli.task_camera_mount == "head_link":
        head_link_expr = f"/World/envs/env_.*/Robot/{args_cli.task_camera_head_link}"
        head_link_paths = sim_utils.find_matching_prim_paths(head_link_expr)
        if not head_link_paths:
            raise RuntimeError(
                f"Could not find Agibot head camera parent link with path expression: {head_link_expr}"
            )

        for head_link_path in head_link_paths:
            rig_path = f"{head_link_path}/RobotYaoTaskStereo"
            rig_translation = (
                float(args_cli.task_camera_head_rig_x),
                float(args_cli.task_camera_head_rig_y),
                float(args_cli.task_camera_head_rig_z),
            )
            rig_rotation_deg = (
                float(args_cli.task_camera_head_rig_roll_deg),
                float(args_cli.task_camera_head_rig_pitch_deg),
                float(args_cli.task_camera_head_rig_yaw_deg),
            )
            rig_orientation = _quat_wxyz_from_euler_xyz_tuple(
                math.radians(rig_rotation_deg[0]),
                math.radians(rig_rotation_deg[1]),
                math.radians(rig_rotation_deg[2]),
            )
            sim_utils.create_prim(
                rig_path,
                "Xform",
                translation=rig_translation,
                orientation=rig_orientation,
            )
            print(
                "[RobotYao] Created head-mounted stereo rig parent "
                f"{rig_path} translation={rig_translation} orient_xyz_deg={rig_rotation_deg}",
                flush=True,
            )

        left_camera_prim_path = f"{head_link_expr}/RobotYaoTaskStereo/LeftFisheye"
        right_camera_prim_path = f"{head_link_expr}/RobotYaoTaskStereo/RightFisheye"
        camera_offset_rot = _quat_wxyz_from_pitch_tuple(math.radians(float(args_cli.task_camera_head_look_down_deg)))
        # Visual verification with the current head mount shows +local Y appears on the robot's physical right.
        # Keep the named LeftFisheye on the physical left by assigning it the negative local-Y offset.
        left_lateral_offset = float(-args_cli.baseline * 0.5)
        right_lateral_offset = float(args_cli.baseline * 0.5)
        left_camera = Camera(
            cfg=_make_task_fisheye_camera_cfg(
                left_camera_prim_path,
                width,
                height,
                fisheye_fov,
                (
                    float(args_cli.task_camera_head_forward_offset),
                    left_lateral_offset,
                    float(args_cli.task_camera_head_up_offset),
                ),
                camera_offset_rot,
            )
        )
        right_camera = Camera(
            cfg=_make_task_fisheye_camera_cfg(
                right_camera_prim_path,
                width,
                height,
                fisheye_fov,
                (
                    float(args_cli.task_camera_head_forward_offset),
                    right_lateral_offset,
                    float(args_cli.task_camera_head_up_offset),
                ),
                camera_offset_rot,
            )
        )
        if show_camera_lenses:
            _spawn_task_camera_lens_visuals(left_camera_prim_path, (0.15, 0.32, 0.85))
            _spawn_task_camera_lens_visuals(right_camera_prim_path, (0.85, 0.22, 0.16))
        return left_camera, right_camera
    else:
        sim_utils.create_prim("/World/RobotYaoTaskStereo", "Xform")
        sim_utils.create_prim("/World/RobotYaoTaskStereo/Left", "Xform")
        sim_utils.create_prim("/World/RobotYaoTaskStereo/Right", "Xform")

        if show_camera_lenses:
            sim_utils.create_prim("/World/RobotYaoTaskStereoLensVisuals", "Xform")
            lens_mat_left = _preview_material((0.15, 0.32, 0.85), metallic=0.1)
            lens_mat_right = _preview_material((0.85, 0.22, 0.16), metallic=0.1)
            _spawn_shape(
                "/World/RobotYaoTaskStereoLensVisuals/Left",
                sim_utils.SphereCfg(radius=0.035, visual_material=lens_mat_left),
                (0.0, 0.0, 0.0),
            )
            _spawn_shape(
                "/World/RobotYaoTaskStereoLensVisuals/Right",
                sim_utils.SphereCfg(radius=0.035, visual_material=lens_mat_right),
                (0.0, 0.0, 0.0),
            )

        camera_prim_path = "/World/RobotYaoTaskStereo/(Left|Right)/Fisheye"
        camera_offset_rot = (1.0, 0.0, 0.0, 0.0)

    return Camera(
        cfg=_make_task_fisheye_camera_cfg(
            camera_prim_path,
            width,
            height,
            fisheye_fov,
            (0.0, 0.0, 0.0),
            camera_offset_rot,
        )
    )


def _initialize_late_sensor(sensor) -> None:
    """Initialize a sensor created after the Isaac Lab environment has already started simulation."""
    if not sensor.is_initialized:
        print(f"[RobotYao] Initializing late sensor: {type(sensor).__name__}", flush=True)
        sensor._initialize_impl()
        sensor._is_initialized = True
        print(f"[RobotYao] Late sensor initialized: {type(sensor).__name__}", flush=True)


def _initialize_late_camera_or_pair(camera: Camera | tuple[Camera, Camera]) -> None:
    if isinstance(camera, tuple):
        for eye_camera in camera:
            _initialize_late_sensor(eye_camera)
    else:
        _initialize_late_sensor(camera)


def _reset_camera_or_pair(camera: Camera | tuple[Camera, Camera]) -> None:
    if isinstance(camera, tuple):
        for eye_camera in camera:
            eye_camera.reset()
    else:
        camera.reset()


class RobotYaoSceneController:
    """Applies compact XR retargeter commands to the simple USD robot."""

    def __init__(self):
        self._base_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._base_yaw = 0.0
        self._left_hand_position = np.array([0.54, 0.58, 0.62], dtype=np.float32)
        self._right_hand_position = np.array([0.54, -0.58, 0.62], dtype=np.float32)
        self._arm_min = np.array([0.05, -0.85, 0.25], dtype=np.float32)
        self._arm_max = np.array([0.95, 0.85, 1.15], dtype=np.float32)

    def apply(self, command: torch.Tensor | np.ndarray | None, dt: float) -> None:
        if command is None:
            return
        if isinstance(command, torch.Tensor):
            command_np = command.detach().cpu().numpy()
        else:
            command_np = np.asarray(command, dtype=np.float32)
        if command_np.size < RobotYaoWheeledXrRetargeter.OUTPUT_SIZE:
            return

        forward = float(command_np[0])
        lateral = float(command_np[1])
        yaw_rate = float(command_np[2])
        arm_follow_active = command_np[3] > 0.5

        self._base_yaw += yaw_rate * dt
        local_delta = np.array([forward * dt, lateral * dt, 0.0], dtype=np.float32)
        c = math.cos(self._base_yaw)
        s = math.sin(self._base_yaw)
        world_delta = np.array(
            [c * local_delta[0] - s * local_delta[1], s * local_delta[0] + c * local_delta[1], 0.0],
            dtype=np.float32,
        )
        self._base_position += world_delta

        if arm_follow_active:
            raw_right_delta = command_np[RobotYaoWheeledXrRetargeter.RAW_RIGHT_DELTA_START : RobotYaoWheeledXrRetargeter.RAW_RIGHT_DELTA_START + 3]
            scene_right_delta = command_np[RobotYaoWheeledXrRetargeter.RIGHT_ARM_DELTA_START : RobotYaoWheeledXrRetargeter.RIGHT_ARM_DELTA_START + 3]
            if np.any(raw_right_delta != 0.0):
                print(
                    f"[DEBUG SimpleController] Right Hand Delta - "
                    f"Controller (Isaac xyz): [{raw_right_delta[0]:.6f}, {raw_right_delta[1]:.6f}, {raw_right_delta[2]:.6f}], "
                    f"Scaled scene delta: [{scene_right_delta[0]:.6f}, {scene_right_delta[1]:.6f}, {scene_right_delta[2]:.6f}]",
                    flush=True
                )
            self._left_hand_position = np.clip(self._left_hand_position + command_np[4:7], self._arm_min, self._arm_max)
            self._right_hand_position = np.clip(
                self._right_hand_position + command_np[7:10], self._arm_min, self._arm_max
            )

        _set_xform_common("/World/Robot", self._base_position, self._base_yaw)
        self._set_arm_visuals("left", np.array([0.24, 0.38, 0.72], dtype=np.float32), self._left_hand_position)
        self._set_arm_visuals("right", np.array([0.24, -0.38, 0.72], dtype=np.float32), self._right_hand_position)

    def _set_arm_visuals(self, side: str, shoulder: np.ndarray, hand: np.ndarray) -> None:
        midpoint = (shoulder + hand) * 0.5
        _set_xform_common(f"/World/Robot/{side}_hand_target", hand)
        _set_xform_common(f"/World/Robot/{side}_arm_link", midpoint)


class RobotYaoTaskSceneController:
    """Applies Unity XR commands to the registered Agibot Toy2Box task scene.

    The Agibot A2D task robot does not expose Ridgeback-style dummy base joints.
    For interactive VR streaming, the base is moved by updating the articulation
    root pose. The registered Toy2Box RMPFlow task exposes a right-arm action,
    while the default direct-joint fallback applies incremental controller
    deltas to both arms.
    """

    def __init__(self, env, stereo_camera: Camera | tuple[Camera, Camera] | None = None):
        self._env = env
        self._robot = env.scene["robot"]
        self._stereo_camera = stereo_camera
        self._camera_mount = args_cli.task_camera_mount
        self._show_camera_lenses = bool(args_cli.show_camera_lenses)
        self._last_camera_positions: list[list[float]] = []
        self._last_camera_targets: list[list[float]] = []
        self._env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
        self._root_pos = None
        self._root_yaw = None
        self._zero_root_velocity = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
        self._rmpflow_axis_map = _parse_axis_map(args_cli.arm_rmpflow_axis_map)
        self._previous_right_gripper_pos_w: torch.Tensor | None = None
        self._right_gripper_body_id = None
        self._debug_follow_was_active = False
        self._debug_cum_right_rmpflow_delta = torch.zeros(3, dtype=torch.float32, device=env.device)
        self._debug_cum_right_ee_delta_w = torch.zeros(3, dtype=torch.float32, device=env.device)
        right_gripper_body_ids, right_gripper_body_names = self._robot.find_bodies(["right_gripper_center"])
        if len(right_gripper_body_ids) > 0:
            self._right_gripper_body_id = right_gripper_body_ids[0]
        fisheye_poly_b = _fisheye_full_frame_poly_b(args_cli.width, args_cli.height, args_cli.fisheye_fov)
        if self._camera_mount == "head_link":
            print(
                "[RobotYao] Task stereo rig mount: "
                f"parent=/World/envs/env_*/Robot/{args_cli.task_camera_head_link}, "
                f"rig_translate=({args_cli.task_camera_head_rig_x:.5f}, "
                f"{args_cli.task_camera_head_rig_y:.5f}, {args_cli.task_camera_head_rig_z:.5f}) m, "
                f"rig_orient=({args_cli.task_camera_head_rig_roll_deg:.1f}, "
                f"{args_cli.task_camera_head_rig_pitch_deg:.1f}, {args_cli.task_camera_head_rig_yaw_deg:.1f}) deg, "
                f"local_forward={args_cli.task_camera_head_forward_offset:.3f} m, "
                f"local_up={args_cli.task_camera_head_up_offset:.3f} m, "
                f"baseline={args_cli.baseline:.3f} m, "
                f"left_local_y={-args_cli.baseline * 0.5:.3f} m, "
                f"right_local_y={args_cli.baseline * 0.5:.3f} m, "
                f"local_look_down={args_cli.task_camera_head_look_down_deg:.1f} deg, "
                f"fisheye_poly_b={fisheye_poly_b:.9f}",
                flush=True,
            )
        else:
            print(
                "[RobotYao] Task stereo rig mount: "
                f"root_forward={args_cli.task_camera_forward_offset:.3f} m, "
                f"root_height={args_cli.task_camera_height_offset:.3f} m, "
                f"baseline={args_cli.baseline:.3f} m, "
                f"look_down={args_cli.task_camera_look_down_deg:.1f} deg, "
                f"fisheye_poly_b={fisheye_poly_b:.9f}",
                flush=True,
            )
        print(f"[DEBUG] Articulation root_pos_w at init: {self._robot.data.root_pos_w[0].tolist()}", flush=True)
        print(f"[DEBUG] Articulation root_quat_w at init: {self._robot.data.root_quat_w[0].tolist()}", flush=True)
        print(
            "[RobotYao] Arm RMPFlow delta axis map: "
            f"controller_delta_already_in_isaac_xyz -> rmpflow[{_format_axis_map(self._rmpflow_axis_map)}], "
            f"arm_delta_scale={args_cli.arm_delta_scale:.3f}, "
            f"right_ee_body={right_gripper_body_names[0] if len(right_gripper_body_names) > 0 else 'not_found'}",
            flush=True,
        )
        self._left_arm_joint_ids, self._left_arm_joint_names = self._robot.find_joints(["left_arm_joint.*"])
        self._right_arm_joint_ids, self._right_arm_joint_names = self._robot.find_joints(["right_arm_joint.*"])
        self.update_camera_xforms()

    def apply_before_step(self, command: torch.Tensor | np.ndarray | None, dt: float) -> torch.Tensor:
        """Move the Agibot root and build a task action tensor for env.step()."""
        actions = torch.zeros((self._env.num_envs, self._env.action_manager.total_action_dim), device=self._env.device)
        if actions.shape[1] > 0:
            # The last binary gripper action is positive=open, negative=close.
            actions[:, -1] = 1.0

        if command is None:
            command_tensor = torch.zeros(RobotYaoWheeledXrRetargeter.OUTPUT_SIZE, dtype=torch.float32, device=self._env.device)
        else:
            if isinstance(command, torch.Tensor):
                command_tensor = command.to(device=self._env.device, dtype=torch.float32).flatten()
            else:
                command_tensor = torch.tensor(command, dtype=torch.float32, device=self._env.device).flatten()
            if command_tensor.numel() < RobotYaoWheeledXrRetargeter.OUTPUT_SIZE:
                self.update_camera_xforms()
                return actions

        # Check simulator state sanity
        current_root_pos = self._robot.data.root_pos_w[0]
        current_root_quat = self._robot.data.root_quat_w[0]
        if torch.any(torch.isnan(current_root_pos)) or torch.any(torch.isinf(current_root_pos)) or torch.any(torch.isnan(current_root_quat)) or torch.any(torch.isinf(current_root_quat)):
            print("[WARNING] Simulator root pose has exploded (NaN/Inf)!", flush=True)
            self.update_camera_xforms()
            return actions

        # Check for NaN/Inf in command
        forward = float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_FORWARD])
        lateral = float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_LATERAL])
        yaw_rate = float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_YAW])
        if not math.isfinite(forward) or not math.isfinite(lateral) or not math.isfinite(yaw_rate):
            print(f"[WARNING] Invalid base command (NaN/Inf): forward={forward}, lateral={lateral}, yaw_rate={yaw_rate}", flush=True)
            forward = 0.0
            lateral = 0.0
            yaw_rate = 0.0

        # Clamp command to physical limits (speeds)
        forward = max(-2.0, min(2.0, forward))
        lateral = max(-2.0, min(2.0, lateral))
        yaw_rate = max(-1.57, min(1.57, yaw_rate))

        is_moving = (abs(forward) > 0.01 or abs(lateral) > 0.01 or abs(yaw_rate) > 0.01)
        follow_active = command_tensor[RobotYaoWheeledXrRetargeter.ARM_FOLLOW_ACTIVE] > 0.5
        left_delta_start = RobotYaoWheeledXrRetargeter.LEFT_ARM_DELTA_START
        right_delta_start = RobotYaoWheeledXrRetargeter.RIGHT_ARM_DELTA_START
        left_scene_delta = command_tensor[left_delta_start : left_delta_start + 3]
        right_scene_delta = command_tensor[right_delta_start : right_delta_start + 3]
        raw_right_delta_start = RobotYaoWheeledXrRetargeter.RAW_RIGHT_DELTA_START
        right_controller_delta = command_tensor[raw_right_delta_start : raw_right_delta_start + 3]
        right_rmpflow_delta = _apply_axis_map_tensor(right_scene_delta, self._rmpflow_axis_map)
        if follow_active and torch.any(right_controller_delta != 0.0):
            print(
                f"[DEBUG TaskController] Right Hand Delta - "
                f"Controller (Isaac xyz): [{right_controller_delta[0].item():.6f}, {right_controller_delta[1].item():.6f}, {right_controller_delta[2].item():.6f}], "
                f"Scaled scene delta: [{right_scene_delta[0].item():.6f}, {right_scene_delta[1].item():.6f}, {right_scene_delta[2].item():.6f}], "
                f"RMPFlow Action: [{right_rmpflow_delta[0].item():.6f}, {right_rmpflow_delta[1].item():.6f}, {right_rmpflow_delta[2].item():.6f}]",
                flush=True
            )

        actual_right_ee_delta_w = None
        if self._right_gripper_body_id is not None:
            right_gripper_pos_w = self._robot.data.body_pos_w[0, self._right_gripper_body_id].clone()
            if self._previous_right_gripper_pos_w is not None:
                actual_right_ee_delta_w = right_gripper_pos_w - self._previous_right_gripper_pos_w
            self._previous_right_gripper_pos_w = right_gripper_pos_w
        follow_active_bool = bool(follow_active.item())
        if follow_active_bool:
            if not self._debug_follow_was_active:
                self._debug_cum_right_rmpflow_delta.zero_()
                self._debug_cum_right_ee_delta_w.zero_()
                actual_right_ee_delta_w = None
            self._debug_cum_right_rmpflow_delta += right_rmpflow_delta.detach()
            if actual_right_ee_delta_w is not None:
                self._debug_cum_right_ee_delta_w += actual_right_ee_delta_w.detach()
        else:
            self._debug_cum_right_rmpflow_delta.zero_()
            self._debug_cum_right_ee_delta_w.zero_()
        self._debug_follow_was_active = follow_active_bool

        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1
        if args_cli.debug_task_loop and self._step_count % 10 == 0:
            print(
                f"[DEBUG] Step {self._step_count} - "
                f"base={command_tensor[0:3].tolist()}, "
                f"follow={bool(follow_active.item())}, "
                f"right_controller_delta_isaac={right_controller_delta.tolist()}, "
                f"left_scene_delta={left_scene_delta.tolist()}, "
                f"right_scene_delta={right_scene_delta.tolist()}, "
                f"right_rmpflow_delta={right_rmpflow_delta.tolist()}, "
                f"axis_map={_format_axis_map(self._rmpflow_axis_map)}, "
                f"actual_right_ee_delta_w={None if actual_right_ee_delta_w is None else actual_right_ee_delta_w.tolist()}, "
                f"right_rmpflow_cum={self._debug_cum_right_rmpflow_delta.tolist()}, "
                f"actual_right_ee_cum_w={self._debug_cum_right_ee_delta_w.tolist()}, "
                f"is_moving={is_moving}, root_pos={current_root_pos.tolist()}",
                flush=True,
            )

        # Check if environment was reset
        if hasattr(self._env, "reset_buf") and self._env.reset_buf is not None:
            if self._env.reset_buf[0].item():
                self._root_pos = None
                self._root_yaw = None
                self._previous_right_gripper_pos_w = None
                self._debug_follow_was_active = False
                self._debug_cum_right_rmpflow_delta.zero_()
                self._debug_cum_right_ee_delta_w.zero_()

        if self._root_pos is None or torch.any(torch.isnan(self._root_pos)):
            self._root_pos = current_root_pos.clone()
            self._root_yaw = _yaw_from_quat_wxyz(current_root_quat)
            self._init_root_z = current_root_pos[2].item()

        if is_moving:
            self._root_yaw += yaw_rate * dt
            c = math.cos(self._root_yaw)
            s = math.sin(self._root_yaw)
            self._root_pos[0] += (c * forward - s * lateral) * dt
            self._root_pos[1] += (s * forward + c * lateral) * dt

        # Always enforce the initial Z height to prevent gravity/penetration drift
        self._root_pos[2] = self._init_root_z

        # Always write the target pose to simulator to hold the base kinematically
        self._write_root_pose()

        if follow_active and actions.shape[1] >= 6:
            # Toy2Box task only controls the right arm. Agibot Lula/RMPFlow uses a different axis order.
            actions[:, 0:3] = right_rmpflow_delta.unsqueeze(0)
            self._apply_direct_arm_delta(
                command_tensor, self._left_arm_joint_ids, left_delta_start
            )
        elif follow_active:
            self._apply_direct_arm_delta(
                command_tensor, self._left_arm_joint_ids, left_delta_start
            )
            self._apply_direct_arm_delta(
                command_tensor, self._right_arm_joint_ids, right_delta_start
            )

        if actions.shape[1] > 0:
            left_grip = command_tensor[RobotYaoWheeledXrRetargeter.LEFT_GRIP]
            right_grip = command_tensor[RobotYaoWheeledXrRetargeter.RIGHT_GRIP]
            left_trigger = command_tensor[RobotYaoWheeledXrRetargeter.LEFT_TRIGGER]
            right_trigger = command_tensor[RobotYaoWheeledXrRetargeter.RIGHT_TRIGGER]
            gripper_close = max(left_grip, right_grip, left_trigger, right_trigger) > 0.5
            actions[:, -1] = -1.0 if gripper_close else 1.0

        self.update_camera_xforms()
        return actions

    def update_camera_xforms(self) -> None:
        """Update or cache task-scene stereo camera Xforms for the selected mount mode."""
        if self._camera_mount == "head_link":
            self._refresh_camera_pose_cache_from_sensor()
            return

        positions: list[list[float]] = []
        targets: list[list[float]] = []
        if self._root_pos is None:
            root = self._robot.data.root_pos_w[0].detach().cpu().numpy()
            root_yaw = _yaw_from_quat_wxyz(self._robot.data.root_quat_w[0])
        else:
            root = self._root_pos.detach().cpu().numpy()
            root_yaw = self._root_yaw
        c = math.cos(root_yaw)
        s = math.sin(root_yaw)
        # Agibot task objects are on a table below the eye point. A small downward
        # look angle keeps the task area near the center of the VR180 image.
        look_down_rad = math.radians(float(args_cli.task_camera_look_down_deg))
        forward_horizontal = math.cos(look_down_rad)
        forward = np.array([c * forward_horizontal, s * forward_horizontal, -math.sin(look_down_rad)], dtype=np.float32)

        for eye_name, side_sign in (("Left", 1.0), ("Right", -1.0)):
            local = np.array(
                [
                    args_cli.task_camera_forward_offset,
                    side_sign * args_cli.baseline * 0.5,
                    args_cli.task_camera_height_offset,
                ],
                dtype=np.float32,
            )
            world = root + np.array(
                [c * local[0] - s * local[1], s * local[0] + c * local[1], local[2]], dtype=np.float32
            )
            positions.append([float(world[0]), float(world[1]), float(world[2])])
            target = world + forward
            targets.append([float(target[0]), float(target[1]), float(target[2])])

            if self._stereo_camera is None:
                _set_xform_common(f"/World/envs/env_0/Robot/RobotYao{eye_name}Fisheye", world, root_yaw)

        if isinstance(self._stereo_camera, tuple):
            for index, camera in enumerate(self._stereo_camera):
                if camera.is_initialized:
                    camera.set_world_poses_from_view(
                        eyes=torch.tensor([positions[index]], dtype=torch.float32, device=self._env.device),
                        targets=torch.tensor([targets[index]], dtype=torch.float32, device=self._env.device),
                    )
        elif self._stereo_camera is not None and self._stereo_camera.is_initialized:
            self._stereo_camera.set_world_poses_from_view(
                eyes=torch.tensor(positions, dtype=torch.float32, device=self._env.device),
                targets=torch.tensor(targets, dtype=torch.float32, device=self._env.device),
            )

        self._last_camera_positions = positions
        self._last_camera_targets = targets
        if self._show_camera_lenses:
            for index, eye_name in enumerate(("Left", "Right")):
                _set_xform_common(
                    f"/World/RobotYaoTaskStereoLensVisuals/{eye_name}",
                    np.asarray(positions[index], dtype=np.float32),
                )

    def _refresh_camera_pose_cache_from_sensor(self) -> None:
        """Cache world poses for head-mounted cameras without modifying their USD transforms."""
        if self._stereo_camera is None:
            return

        positions: list[list[float]] = []
        if isinstance(self._stereo_camera, tuple):
            for camera in self._stereo_camera:
                if not camera.is_initialized or camera.data.pos_w is None or camera.data.pos_w.shape[0] < 1:
                    continue
                pos = camera.data.pos_w.detach().cpu().numpy()[0]
                positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
        elif self._stereo_camera.is_initialized and self._stereo_camera.data.pos_w is not None:
            pos_w = self._stereo_camera.data.pos_w.detach().cpu().numpy()
            positions = [[float(pos[0]), float(pos[1]), float(pos[2])] for pos in pos_w]

        if positions:
            self._last_camera_positions = positions
            self._last_camera_targets = []

    @property
    def last_camera_positions(self) -> list[list[float]]:
        return self._last_camera_positions

    @property
    def last_camera_targets(self) -> list[list[float]]:
        return self._last_camera_targets

    def _write_root_pose(self) -> None:
        """Write the interactive root pose to the first task environment."""
        root_pose = torch.zeros((1, 7), dtype=torch.float32, device=self._env.device)
        root_pose[0, 0:3] = self._root_pos
        root_pose[0, 3:7] = _quat_wxyz_from_yaw(self._root_yaw, self._env.device)
        self._robot.write_root_pose_to_sim(root_pose, env_ids=self._env_ids)
        self._robot.write_root_velocity_to_sim(self._zero_root_velocity, env_ids=self._env_ids)

    def _apply_direct_arm_delta(self, command_tensor: torch.Tensor, joint_ids: list[int], delta_start: int) -> None:
        """Apply a lightweight joint-space fallback when RMPFlow assets are not available."""
        if len(joint_ids) == 0:
            return

        hand_delta = command_tensor[delta_start : delta_start + 3]
        if torch.any(torch.isnan(hand_delta)) or torch.any(torch.isinf(hand_delta)):
            print(f"[WARNING] Invalid arm hand_delta (NaN/Inf) at start {delta_start}!", flush=True)
            hand_delta = torch.zeros_like(hand_delta)

        joint_delta = torch.zeros(
            (self._env.num_envs, len(joint_ids)), dtype=torch.float32, device=self._env.device
        )

        # 这里是测试用的轻量增量映射，不伪装成完整 IK；真实双臂遥操作后续应接入双臂 IK/RMPFlow。
        if joint_delta.shape[1] > 0:
            joint_delta[:, 0] = hand_delta[1] * 2.5
        if joint_delta.shape[1] > 1:
            joint_delta[:, 1] = hand_delta[2] * 2.0
        if joint_delta.shape[1] > 3:
            joint_delta[:, 3] = -hand_delta[0] * 2.0

        joint_targets = self._robot.data.joint_pos[:, joint_ids].clone() + joint_delta
        joint_targets = torch.clamp(joint_targets, min=-2.8, max=2.8)
        self._robot.set_joint_position_target(joint_targets, joint_ids=joint_ids)


def _create_unity_control_device() -> RobotYaoXrSubDevice:
    retargeter_cfg = RobotYaoWheeledXrRetargeterCfg(
        sim_device=args_cli.device,
        max_forward_speed=args_cli.max_forward_speed,
        max_lateral_speed=args_cli.max_lateral_speed,
        max_yaw_rate=args_cli.max_yaw_rate,
        arm_delta_scale=args_cli.arm_delta_scale,
        follow_button_mode=args_cli.unity_follow_mode,
    )
    retargeter = RobotYaoWheeledXrRetargeter(retargeter_cfg)
    device_cfg = RobotYaoXrSubDeviceCfg(
        endpoint=args_cli.unity_input_endpoint,
        topic=args_cli.unity_input_topic,
        sim_device=args_cli.device,
        auto_start=True,
    )
    return RobotYaoXrSubDevice(device_cfg, retargeters=[retargeter])


def _create_task_scene_env():
    """Create the registered Isaac Lab Agibot Toy2Box task scene for RobotYao streaming."""
    print(f"[RobotYao] Parsing task config: {args_cli.task}", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    use_task_rmpflow = bool(args_cli.task_use_rmpflow)
    if use_task_rmpflow and "Agibot" in args_cli.task and not args_cli.allow_remote_rmpflow_assets:
        print(
            "[RobotYao] Agibot RMPFlow assets are not bundled locally; using direct right-arm joint fallback. "
            "Add --allow-remote-rmpflow-assets to try the remote RMPFlow asset path.",
            flush=True,
        )
        use_task_rmpflow = False
        args_cli.task_use_rmpflow = False
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.sim.render_interval = 1
    if hasattr(env_cfg, "observations"):
        # Streaming only needs the task scene and action manager. Heavy task-only observations
        # are disabled here to avoid waiting on frame/contact sensor data during startup.
        if hasattr(env_cfg.observations, "subtask_terms"):
            env_cfg.observations.subtask_terms = None
        policy_obs = getattr(env_cfg.observations, "policy", None)
        if policy_obs is not None:
            for term_name in (
                "toy_truck_positions",
                "toy_truck_orientations",
                "box_positions",
                "box_orientations",
                "eef_pos",
                "eef_quat",
                "gripper_pos",
            ):
                if hasattr(policy_obs, term_name):
                    setattr(policy_obs, term_name, None)
    if not use_task_rmpflow and hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "arm_action"):
        env_cfg.actions.arm_action = None
        if hasattr(env_cfg.scene, "ee_frame"):
            env_cfg.scene.ee_frame = None
        if hasattr(env_cfg.scene, "contact_grasp"):
            env_cfg.scene.contact_grasp = None
        print("[RobotYao] Task RMPFlow action disabled; using direct right-arm joint fallback.", flush=True)
    print("[RobotYao] Creating Gym task environment.", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[RobotYao] Resetting Gym task environment.", flush=True)
    try:
        env.reset()
    except FileNotFoundError as exc:
        if args_cli.task_use_rmpflow and "RmpFlowAssets/agibot" in str(exc).replace("\\", "/"):
            print(
                "[RobotYao] Agibot RMPFlow assets were not found; falling back to direct right-arm joint control.",
                flush=True,
            )
            try:
                env.close()
            except Exception as close_exc:
                print(f"[RobotYao] Failed to close partially initialized RMPFlow env: {close_exc}", flush=True)
            args_cli.task_use_rmpflow = False
            return _create_task_scene_env()
        raise
    print("[RobotYao] Gym task environment created.", flush=True)
    return env


def run_task_scene_simulator(
    env,
    camera: Camera | tuple[Camera, Camera],
    control_device: RobotYaoXrSubDevice | None = None,
    task_controller: RobotYaoTaskSceneController | None = None,
):
    """Run the registered task scene while publishing stereo fisheye frames."""
    publisher = StereoZmqPublisher(args_cli.endpoint, args_cli.topic)
    h264_encoder = StereoH264Encoder() if args_cli.encoding == "h264" else None
    publish_interval = 0.0 if args_cli.fps <= 0.0 else 1.0 / args_cli.fps
    next_publish_time = time.perf_counter()
    published = 0
    frame_id = 0
    debug_frame_saved = False
    start_time = time.perf_counter()

    print(
        "[RobotYao] Streaming Agibot task stereo RGB fisheye frames "
        f"task={args_cli.task}, {args_cli.width}x{args_cli.height}, encoding={args_cli.encoding}, "
        f"endpoint={args_cli.endpoint}, topic={args_cli.topic}",
        flush=True,
    )
    if control_device is not None:
        print(
            f"[RobotYao] Unity controller input enabled, endpoint={args_cli.unity_input_endpoint}, "
            f"topic={args_cli.unity_input_topic}",
            flush=True,
        )

    try:
        while simulation_app.is_running():
            dt = float(env.step_dt)
            if args_cli.debug_task_loop and frame_id < 2:
                print(f"[RobotYao] Task loop frame {frame_id + 1}: build action.", flush=True)
            command = control_device.advance() if control_device is not None else None
            actions = (
                task_controller.apply_before_step(command, dt)
                if task_controller is not None
                else torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            )

            if args_cli.debug_task_loop and frame_id < 2:
                print(f"[RobotYao] Task loop frame {frame_id + 1}: env.step.", flush=True)
            with torch.inference_mode():
                env.step(actions)
            if args_cli.debug_task_loop and frame_id < 2:
                print(f"[RobotYao] Task loop frame {frame_id + 1}: update camera xforms.", flush=True)
            if task_controller is not None:
                task_controller.update_camera_xforms()
            if args_cli.debug_task_loop and frame_id < 2:
                print(f"[RobotYao] Task loop frame {frame_id + 1}: camera.update.", flush=True)
            if isinstance(camera, tuple):
                for eye_camera in camera:
                    eye_camera.update(dt=dt)
            else:
                camera.update(dt=dt)
            if task_controller is not None and args_cli.task_camera_mount == "head_link":
                task_controller.update_camera_xforms()
            if args_cli.debug_task_loop and frame_id < 2:
                print(f"[RobotYao] Task loop frame {frame_id + 1}: camera updated.", flush=True)
            frame_id += 1
            should_stop_after_frame = args_cli.max_frames > 0 and frame_id >= args_cli.max_frames

            now = time.perf_counter()
            if frame_id <= args_cli.warmup_frames:
                if should_stop_after_frame:
                    break
                continue
            if publish_interval > 0.0 and now < next_publish_time:
                if should_stop_after_frame:
                    break
                continue
            if publish_interval > 0.0:
                next_publish_time = max(now, next_publish_time + publish_interval)

            if isinstance(camera, tuple):
                left_tensor = camera[0].data.output.get("rgb")
                right_tensor = camera[1].data.output.get("rgb")
                if left_tensor is None or right_tensor is None or left_tensor.shape[0] < 1 or right_tensor.shape[0] < 1:
                    if should_stop_after_frame:
                        break
                    continue
                left_rgb = np.ascontiguousarray(left_tensor.detach().cpu().numpy()[0, :, :, :3], dtype=np.uint8)
                right_rgb = np.ascontiguousarray(right_tensor.detach().cpu().numpy()[0, :, :, :3], dtype=np.uint8)
            else:
                if args_cli.debug_task_loop and frame_id <= 2:
                    print(f"[RobotYao] Task loop frame {frame_id}: read tiled camera rgb.", flush=True)
                rgb_tensor = camera.data.output.get("rgb")
                if rgb_tensor is None or rgb_tensor.shape[0] < 2:
                    if args_cli.debug_task_loop and frame_id <= 2:
                        shape = None if rgb_tensor is None else tuple(rgb_tensor.shape)
                        print(f"[RobotYao] Task loop frame {frame_id}: rgb not ready shape={shape}.", flush=True)
                    if should_stop_after_frame:
                        break
                    continue
                if args_cli.debug_task_loop and frame_id <= 2:
                    print(f"[RobotYao] Task loop frame {frame_id}: copy rgb tensor shape={tuple(rgb_tensor.shape)}.", flush=True)
                rgb_images = rgb_tensor.detach().cpu().numpy()
                left_rgb = np.ascontiguousarray(rgb_images[0, :, :, :3], dtype=np.uint8)
                right_rgb = np.ascontiguousarray(rgb_images[1, :, :, :3], dtype=np.uint8)
            if not debug_frame_saved:
                _save_debug_frame_pair("task_scene", frame_id, left_rgb, right_rgb)
                debug_frame_saved = True
            if h264_encoder is not None:
                left_payload, right_payload = h264_encoder.encode(left_rgb, right_rgb)
                if not left_payload or not right_payload:
                    if should_stop_after_frame:
                        break
                    continue
            else:
                left_payload = _encode_jpeg(left_rgb, args_cli.jpeg_quality)
                right_payload = _encode_jpeg(right_rgb, args_cli.jpeg_quality)

            header = {
                "version": 1,
                "frame_id": int(frame_id),
                "timestamp_ns": int(time.time_ns()),
                "width": int(args_cli.width),
                "height": int(args_cli.height),
                "encoding": args_cli.encoding,
                "pixel_format": "rgb8",
                "quality": int(args_cli.jpeg_quality) if args_cli.encoding == "jpg" else None,
                "h264": {
                    "annex_b": True,
                    "bitrate": int(args_cli.h264_bitrate),
                    "gop": int(args_cli.h264_gop),
                    "profile": args_cli.h264_profile,
                    "preset": args_cli.h264_preset,
                    "source_format": "rgb24",
                    "encoded_format": "yuv420p",
                }
                if args_cli.encoding == "h264"
                else None,
                "eye_order": "left_right",
                "baseline_m": float(args_cli.baseline),
                "fisheye": {
                    "model": "fisheyePolynomial",
                    "fov_deg": float(args_cli.fisheye_fov),
                    "cx": float(args_cli.width) * 0.5,
                    "cy": float(args_cli.height) * 0.5,
                    "radius": min(float(args_cli.width), float(args_cli.height)) * 0.5,
                    "radius_px": min(float(args_cli.width), float(args_cli.height)) * 0.5,
                    "poly_a": 0.0,
                    "poly_b": _fisheye_full_frame_poly_b(args_cli.width, args_cli.height, args_cli.fisheye_fov),
                    "poly_c": 0.0,
                    "poly_d": 0.0,
                    "poly_e": 0.0,
                    "poly_f": 0.0,
                },
                "scene": {
                    "mode": "task",
                    "task": args_cli.task,
                    "robot": "Agibot A2D",
                },
                "camera_mount": {
                    "mode": args_cli.task_camera_mount,
                    "frame": (
                        f"Robot/{args_cli.task_camera_head_link}"
                        if args_cli.task_camera_mount == "head_link"
                        else "agibot_root_yaw"
                    ),
                    "head_link": args_cli.task_camera_head_link
                    if args_cli.task_camera_mount == "head_link"
                    else None,
                    "head_rig_translate_m": [
                        float(args_cli.task_camera_head_rig_x),
                        float(args_cli.task_camera_head_rig_y),
                        float(args_cli.task_camera_head_rig_z),
                    ]
                    if args_cli.task_camera_mount == "head_link"
                    else None,
                    "head_rig_orient_xyz_deg": [
                        float(args_cli.task_camera_head_rig_roll_deg),
                        float(args_cli.task_camera_head_rig_pitch_deg),
                        float(args_cli.task_camera_head_rig_yaw_deg),
                    ]
                    if args_cli.task_camera_mount == "head_link"
                    else None,
                    "head_local_forward_offset_m": float(args_cli.task_camera_head_forward_offset)
                    if args_cli.task_camera_mount == "head_link"
                    else None,
                    "head_local_up_offset_m": float(args_cli.task_camera_head_up_offset)
                    if args_cli.task_camera_mount == "head_link"
                    else None,
                    "head_local_look_down_deg": float(args_cli.task_camera_head_look_down_deg)
                    if args_cli.task_camera_mount == "head_link"
                    else None,
                    "root_forward_offset_m": float(args_cli.task_camera_forward_offset)
                    if args_cli.task_camera_mount == "root"
                    else None,
                    "root_height_offset_m": float(args_cli.task_camera_height_offset)
                    if args_cli.task_camera_mount == "root"
                    else None,
                    "root_look_down_deg": float(args_cli.task_camera_look_down_deg)
                    if args_cli.task_camera_mount == "root"
                    else None,
                    "left_world_pos": task_controller.last_camera_positions[0]
                    if task_controller is not None and len(task_controller.last_camera_positions) > 0
                    else None,
                    "right_world_pos": task_controller.last_camera_positions[1]
                    if task_controller is not None and len(task_controller.last_camera_positions) > 1
                    else None,
                    "left_world_target": task_controller.last_camera_targets[0]
                    if task_controller is not None and len(task_controller.last_camera_targets) > 0
                    else None,
                    "right_world_target": task_controller.last_camera_targets[1]
                    if task_controller is not None and len(task_controller.last_camera_targets) > 1
                    else None,
                },
            }
            publisher.send(header, left_payload, right_payload)
            published += 1

            if args_cli.print_every > 0 and published % args_cli.print_every == 0:
                elapsed = max(time.perf_counter() - start_time, 1.0e-6)
                mb = (len(left_payload) + len(right_payload)) / (1024.0 * 1024.0)
                print(
                    f"[RobotYao] task_scene published={published} fps={published / elapsed:.1f} "
                    f"last_payload={mb:.2f} MiB frame_id={frame_id}",
                    flush=True,
                )

            if should_stop_after_frame:
                break
    finally:
        print(f"[RobotYao] Task scene stopped. frames={frame_id}, published={published}", flush=True)
        if control_device is not None:
            try:
                control_device.stop()
            except Exception as exc:
                print(f"[RobotYao] Failed to stop Unity control device: {exc}", flush=True)
        if h264_encoder is not None:
            try:
                h264_encoder.close()
            except Exception as exc:
                print(f"[RobotYao] Failed to close H264 encoder: {exc}", flush=True)
        try:
            publisher.close()
        except Exception as exc:
            print(f"[RobotYao] Failed to close stereo publisher: {exc}", flush=True)
        try:
            env.close()
        except Exception as exc:
            print(f"[RobotYao] Failed to close task scene env: {exc}", flush=True)


def run_simulator(
    sim: sim_utils.SimulationContext,
    camera: Camera,
    control_device: RobotYaoXrSubDevice | None = None,
    robot_controller: RobotYaoSceneController | None = None,
):
    publisher = StereoZmqPublisher(args_cli.endpoint, args_cli.topic)
    h264_encoder = StereoH264Encoder() if args_cli.encoding == "h264" else None
    publish_interval = 0.0 if args_cli.fps <= 0.0 else 1.0 / args_cli.fps
    next_publish_time = time.perf_counter()
    published = 0
    frame_id = 0
    debug_frame_saved = False
    start_time = time.perf_counter()

    print(
        "[RobotYao] Streaming stereo RGB fisheye frames "
        f"{args_cli.width}x{args_cli.height}, encoding={args_cli.encoding}, endpoint={args_cli.endpoint}, "
        f"topic={args_cli.topic}"
    )
    if control_device is not None:
        print(
            f"[RobotYao] Unity controller input enabled, endpoint={args_cli.unity_input_endpoint}, "
            f"topic={args_cli.unity_input_topic}"
        )

    try:
        while simulation_app.is_running():
            physics_dt = sim.get_physics_dt()
            if control_device is not None and robot_controller is not None:
                robot_controller.apply(control_device.advance(), physics_dt)

            sim.step()
            camera.update(dt=physics_dt)
            frame_id += 1
            should_stop_after_frame = args_cli.max_frames > 0 and frame_id >= args_cli.max_frames

            now = time.perf_counter()
            if frame_id <= args_cli.warmup_frames:
                if should_stop_after_frame:
                    break
                continue
            if publish_interval > 0.0 and now < next_publish_time:
                if should_stop_after_frame:
                    break
                continue
            if publish_interval > 0.0:
                next_publish_time = max(now, next_publish_time + publish_interval)

            rgb_tensor = camera.data.output.get("rgb")
            if rgb_tensor is None or rgb_tensor.shape[0] < 2:
                if should_stop_after_frame:
                    break
                continue

            rgb_images = rgb_tensor.detach().cpu().numpy()
            left_rgb = np.ascontiguousarray(rgb_images[0, :, :, :3], dtype=np.uint8)
            right_rgb = np.ascontiguousarray(rgb_images[1, :, :, :3], dtype=np.uint8)
            if not debug_frame_saved:
                _save_debug_frame_pair("simple_scene", frame_id, left_rgb, right_rgb)
                debug_frame_saved = True
            if h264_encoder is not None:
                left_payload, right_payload = h264_encoder.encode(left_rgb, right_rgb)
                if not left_payload or not right_payload:
                    if should_stop_after_frame:
                        break
                    continue
            else:
                left_payload = _encode_jpeg(left_rgb, args_cli.jpeg_quality)
                right_payload = _encode_jpeg(right_rgb, args_cli.jpeg_quality)

            header = {
                "version": 1,
                "frame_id": int(frame_id),
                "timestamp_ns": int(time.time_ns()),
                "width": int(args_cli.width),
                "height": int(args_cli.height),
                "encoding": args_cli.encoding,
                "pixel_format": "rgb8",
                "quality": int(args_cli.jpeg_quality) if args_cli.encoding == "jpg" else None,
                "h264": {
                    "annex_b": True,
                    "bitrate": int(args_cli.h264_bitrate),
                    "gop": int(args_cli.h264_gop),
                    "profile": args_cli.h264_profile,
                    "preset": args_cli.h264_preset,
                    "source_format": "rgb24",
                    "encoded_format": "yuv420p",
                }
                if args_cli.encoding == "h264"
                else None,
                "eye_order": "left_right",
                "baseline_m": float(args_cli.baseline),
                "fisheye": {
                    "model": "fisheyePolynomial",
                    "fov_deg": float(args_cli.fisheye_fov),
                    "cx": float(args_cli.width) * 0.5,
                    "cy": float(args_cli.height) * 0.5,
                    "radius": min(float(args_cli.width), float(args_cli.height)) * 0.5,
                    "radius_px": min(float(args_cli.width), float(args_cli.height)) * 0.5,
                    "poly_a": 0.0,
                    "poly_b": _fisheye_full_frame_poly_b(args_cli.width, args_cli.height, args_cli.fisheye_fov),
                    "poly_c": 0.0,
                    "poly_d": 0.0,
                    "poly_e": 0.0,
                    "poly_f": 0.0,
                },
            }
            publisher.send(header, left_payload, right_payload)
            published += 1

            if args_cli.print_every > 0 and published % args_cli.print_every == 0:
                elapsed = max(time.perf_counter() - start_time, 1.0e-6)
                mb = (len(left_payload) + len(right_payload)) / (1024.0 * 1024.0)
                print(
                    f"[RobotYao] published={published} fps={published / elapsed:.1f} "
                    f"last_payload={mb:.2f} MiB frame_id={frame_id}"
                )

            if should_stop_after_frame:
                break
    finally:
        print(f"[RobotYao] Simple scene stopped. frames={frame_id}, published={published}")
        if control_device is not None:
            try:
                control_device.stop()
            except Exception as exc:
                print(f"[RobotYao] Failed to stop Unity control device: {exc}", flush=True)
        if h264_encoder is not None:
            try:
                h264_encoder.close()
            except Exception as exc:
                print(f"[RobotYao] Failed to close H264 encoder: {exc}", flush=True)
        try:
            publisher.close()
        except Exception as exc:
            print(f"[RobotYao] Failed to close stereo publisher: {exc}", flush=True)


def main():
    if args_cli.task_scene:
        env = _create_task_scene_env()
        print("[RobotYao] Setting debug camera view.", flush=True)
        env.sim.set_camera_view(eye=[1.5, -1.0, 1.5], target=[0.5, 0.0, 0.0])
        print("[RobotYao] Creating task stereo fisheye cameras.", flush=True)
        camera = _design_task_scene_stereo_cameras(
            args_cli.width, args_cli.height, args_cli.fisheye_fov, args_cli.show_camera_lenses
        )
        _initialize_late_camera_or_pair(camera)
        print("[RobotYao] Creating task scene controller.", flush=True)
        task_controller = RobotYaoTaskSceneController(env, camera)
        print("[RobotYao] Resetting task stereo cameras.", flush=True)
        _reset_camera_or_pair(camera)
        control_device = _create_unity_control_device() if args_cli.unity_control else None
        print("[RobotYao] Task scene setup complete.", flush=True)
        run_task_scene_simulator(env, camera, control_device, task_controller)
        return

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.0, 3.5, 2.2], target=[0.6, 0.0, 0.8])

    camera = _design_scene(
        args_cli.width, args_cli.height, args_cli.baseline, args_cli.fisheye_fov, args_cli.show_camera_lenses
    )
    sim.reset()

    control_device = _create_unity_control_device() if args_cli.unity_control else None
    robot_controller = RobotYaoSceneController() if args_cli.unity_control else None

    print("[RobotYao] Setup complete.")
    run_simulator(sim, camera, control_device, robot_controller)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()

    if args_cli.no_fast_exit_on_max_frames:
        args_cli.clean_kit_shutdown = True

    if not args_cli.clean_kit_shutdown:
        print("[RobotYao] Run complete; fast process exit to avoid Kit shutdown native crashes.", flush=True)
        os._exit(exit_code)

    print("[RobotYao] Closing SimulationApp.", flush=True)
    simulation_app.close(wait_for_replicator=False)
    print("[RobotYao] SimulationApp closed.", flush=True)
