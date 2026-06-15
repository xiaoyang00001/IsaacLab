# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

r"""Benchmark a minimal RobotYao scene without task managers or stereo cameras.

This script spawns only the Agibot A2D robot, a table, a ground slab, and
three small cubes. It is intended to separate base simulation/rendering cost
from the full Toy2Box task and stereo fisheye pipeline.

Example:

.. code-block:: powershell

    .\isaaclab.bat -p .\scripts\robotyao\benchmark_basic_scene_fps.py `
      --width 1920 --height 1080 --print_every 60 `
      --/app/runLoops/main/rateLimitEnabled=false `
      --/app/runLoops/main/manualModeEnabled=true `
      --/app/asyncRendering=true `
      --/app/asyncRenderingLowLatency=true

    .\isaaclab.bat -p .\scripts\robotyao\benchmark_basic_scene_fps.py `
      --width 1920 --height 1080 --print_every 60 `
      --with-stereo-fisheye --camera-width 1920 --camera-height 1920 `
      --/app/runLoops/main/rateLimitEnabled=false `
      --/app/runLoops/main/manualModeEnabled=true `
      --/app/asyncRendering=true `
      --/app/asyncRenderingLowLatency=true
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

_ROBOTYAO_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ROBOTYAO_CACHE_ROOT = os.path.join(_ROBOTYAO_REPO_ROOT, ".robotyao_cache")
for _cache_name in ("warp", "optix", "cuda_cache", "nv_shader_cache"):
    _cache_path = os.path.join(_ROBOTYAO_CACHE_ROOT, _cache_name)
    os.makedirs(_cache_path, exist_ok=True)

os.environ.setdefault("WARP_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "warp"))
os.environ.setdefault("OPTIX_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "optix"))
os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "cuda_cache"))
os.environ.setdefault("NV_SHADER_DISK_CACHE_PATH", os.path.join(_ROBOTYAO_CACHE_ROOT, "nv_shader_cache"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RobotYao minimal scene FPS benchmark.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of benchmark environments.")
parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Physics timestep in seconds.")
parser.add_argument(
    "--with-stereo-fisheye",
    action="store_true",
    help="Create and update head-mounted stereo fisheye cameras, but do not copy, encode, or publish frames.",
)
parser.add_argument("--camera-width", type=int, default=1920, help="Stereo fisheye camera image width.")
parser.add_argument("--camera-height", type=int, default=1920, help="Stereo fisheye camera image height.")
parser.add_argument("--baseline", type=float, default=0.064, help="Stereo fisheye camera baseline in meters.")
parser.add_argument("--fisheye-fov", type=float, default=180.0, help="Stereo fisheye camera field of view in degrees.")
parser.add_argument("--unity-control", action="store_true", help="Receive Unity controller input over ZMQ.")
parser.add_argument(
    "--unity-input-endpoint",
    type=str,
    default="tcp://127.0.0.1:5555",
    help="Unity controller-input ZMQ PUB endpoint. Isaac Lab connects as a normal ZMQ SUB client.",
)
parser.add_argument("--unity-input-topic", type=str, default="state", help="Unity controller-input ZMQ topic.")
parser.add_argument(
    "--unity-follow-mode",
    choices=["toggle", "hold"],
    default="toggle",
    help="Arm-follow mode for left Y/X and right B/A.",
)
parser.add_argument("--max-forward-speed", type=float, default=1.0, help="Left stick Y forward speed in m/s.")
parser.add_argument("--max-lateral-speed", type=float, default=0.6, help="Left stick X lateral speed in m/s.")
parser.add_argument("--max-yaw-rate", type=float, default=1.2, help="Right stick X yaw rate in rad/s.")
parser.add_argument("--arm-delta-scale", type=float, default=1.0, help="Scale for incremental controller arm deltas.")
parser.add_argument(
    "--arm-rotation-delta-scale",
    type=float,
    default=1.0,
    help="Scale for incremental controller orientation deltas, in axis-angle radians.",
)
parser.add_argument(
    "--arm-command-position-deadband",
    type=float,
    default=0.0015,
    help="Ignore arm position delta commands whose norm is below this value in meters.",
)
parser.add_argument(
    "--arm-command-rotation-deadband",
    type=float,
    default=0.006,
    help="Ignore arm rotation delta commands whose norm is below this value in radians.",
)
parser.add_argument(
    "--arm-follow-start-warmup-frames",
    type=int,
    default=5,
    help="Hold arm joints for this many frames after pressing Y/B before accepting arm deltas.",
)
parser.add_argument(
    "--arm-cumulative-target",
    dest="arm_cumulative_target",
    action="store_true",
    default=True,
    help="Track persistent EE targets from Y/B press to X/A release, matching stream_stereo_fisheye_zmq.py.",
)
parser.add_argument(
    "--no-arm-cumulative-target",
    dest="arm_cumulative_target",
    action="store_false",
    help="Disable cumulative EE target tracking.",
)
parser.add_argument(
    "--basic-arm-control-mode",
    choices=["rmpflow", "direct"],
    default="rmpflow",
    help="Arm control implementation for benchmark. rmpflow matches stream_stereo_fisheye_zmq.py task behavior.",
)
parser.add_argument(
    "--arm-rmpflow-axis-map",
    type=str,
    default="y,-x,z",
    help=(
        "Comma-separated scene-delta axes used for the Agibot RMPFlow xyz action. "
        "Matches stream_stereo_fisheye_zmq.py default."
    ),
)
parser.add_argument(
    "--left-arm-rmpflow-axis-map",
    type=str,
    default=None,
    help="Axis mapping for the left arm. If None, defaults to --arm-rmpflow-axis-map.",
)
parser.add_argument(
    "--right-arm-rmpflow-axis-map",
    type=str,
    default=None,
    help="Axis mapping for the right arm. If None, defaults to --arm-rmpflow-axis-map.",
)
parser.add_argument("--debug-task-loop", action="store_true", help="Print basic-controller diagnostics every 10 frames.")
parser.add_argument(
    "--render-every-n-frames",
    type=int,
    default=1,
    help="Render the viewport every N physics frames. Use 0 to benchmark physics without explicit rendering.",
)
parser.add_argument("--print_every", type=int, default=60, help="Print average timing every N frames.")
parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames. 0 runs until the app exits.")
parser.add_argument(
    "--no-fast-exit-on-max-frames",
    action="store_true",
    help="Call SimulationApp.close() after --max_frames instead of exiting the process immediately.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _kit_passthrough_args = parser.parse_known_args()

if _kit_passthrough_args:
    # Keep the direct Kit style usable:
    #   script.py --/app/asyncRendering=true --/app/...
    # The args remain in sys.argv for Kit, and parse_known_args prevents argparse
    # from rejecting them before AppLauncher starts.
    print(f"[RobotYaoBasic] Passing Kit args through: {' '.join(_kit_passthrough_args)}", flush=True)

if args_cli.with_stereo_fisheye and hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

_robotyao_cache_arg_path = _ROBOTYAO_CACHE_ROOT.replace("\\", "/")
_existing_kit_args = str(args_cli.kit_args or "").strip()
_extra_kit_args = [
    f"--/rtx-transient/resourcemanager/localTextureCachePath={_robotyao_cache_arg_path}/texturecache",
    f"--/exts/omni.kit.registry.nucleus/cachePath={_robotyao_cache_arg_path}/exts",
    f"--/UJITSO/datastore/GRPCDataStoreServer/cachePath={_robotyao_cache_arg_path}/datastore",
    f"--/app/cachePath={_robotyao_cache_arg_path}/kit_cache",
]
if "--portable-root" not in _existing_kit_args:
    _extra_kit_args.extend(["--portable-root", f"{_robotyao_cache_arg_path}/kit_portable"])
args_cli.kit_args = " ".join(filter(None, [_existing_kit_args, *_extra_kit_args]))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms, apply_delta_pose

from isaaclab.devices.openxr.robotyao_xr_sub_device import RobotYaoXrSubDevice, RobotYaoXrSubDeviceCfg
from isaaclab.devices.openxr.retargeters.robotyao_wheeled_xr_retargeter import (
    RobotYaoWheeledXrRetargeter,
    RobotYaoWheeledXrRetargeterCfg,
)
from isaaclab_assets.robots.agibot import AGIBOT_A2D_CFG  # isort: skip
from isaaclab_tasks.manager_based.manipulation.place.config.agibot.place_toy2box_rmp_rel_env_cfg import (  # isort: skip
    _TASK_CUBE_DEFAULT_POSES,
    _TASK_CUBE_SIZE,
    _TASK_ROBOT_DEFAULT_POS,
    _TASK_TABLE_POS,
    _TASK_TABLE_SIZE,
    _configure_symmetric_arm_init_pose,
    spawn_agibot_floating,
)


_ROBOTYAO_LEGACY_COMMAND_SIZE = RobotYaoWheeledXrRetargeter.RIGHT_ARM_ROT_DELTA_START + 3


def _make_robot_cfg():
    robot_cfg = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.func = spawn_agibot_floating
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.spawn.articulation_props.fix_root_link = False
    robot_cfg.init_state.pos = _TASK_ROBOT_DEFAULT_POS
    _configure_symmetric_arm_init_pose(robot_cfg)
    return robot_cfg


def _cube_cfg(prim_name: str, pose: tuple[float, float, float], color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pose),
        spawn=sim_utils.CuboidCfg(
            size=(_TASK_CUBE_SIZE, _TASK_CUBE_SIZE, _TASK_CUBE_SIZE),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
        ),
    )


def _fisheye_full_frame_poly_b(width: int, height: int, fisheye_fov: float) -> float:
    """Return the linear f-theta coefficient that maps max FOV to the image radius."""
    image_radius_px = max(min(float(width), float(height)) * 0.5, 1.0)
    max_theta_rad = math.radians(float(fisheye_fov) * 0.5)
    return max_theta_rad / image_radius_px


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


def _make_fisheye_camera_cfg(
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


def _quat_wxyz_from_yaw_tensor(yaw_rad: float, count: int, device: str) -> torch.Tensor:
    half = 0.5 * float(yaw_rad)
    quat = torch.zeros((count, 4), dtype=torch.float32, device=device)
    quat[:, 0] = math.cos(half)
    quat[:, 3] = math.sin(half)
    return quat


def _torch_arm_follow_flags(command_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    follow_active = command_tensor[RobotYaoWheeledXrRetargeter.ARM_FOLLOW_ACTIVE] > 0.5
    if command_tensor.numel() > RobotYaoWheeledXrRetargeter.RIGHT_ARM_FOLLOW_ACTIVE:
        left_follow_active = command_tensor[RobotYaoWheeledXrRetargeter.LEFT_ARM_FOLLOW_ACTIVE] > 0.5
        right_follow_active = command_tensor[RobotYaoWheeledXrRetargeter.RIGHT_ARM_FOLLOW_ACTIVE] > 0.5
    else:
        left_follow_active = follow_active
        right_follow_active = follow_active
    return left_follow_active, right_follow_active, follow_active


def _parse_axis_map(spec: str) -> list[tuple[int, float, str]]:
    """Parse a comma-separated xyz axis map with optional signs."""
    axis_indices = {"x": 0, "y": 1, "z": 2}
    raw = spec.strip().lower().replace(" ", "")
    tokens = raw.split(",") if "," in raw else list(raw)
    if len(tokens) != 3:
        raise ValueError(f"Invalid axis map '{spec}'. Expected three axes, for example 'y,-x,z'.")

    axis_map: list[tuple[int, float, str]] = []
    for token in tokens:
        if not token:
            raise ValueError(f"Invalid axis map '{spec}': empty axis token.")
        sign = -1.0 if token.startswith("-") else 1.0
        axis = token[1:] if token.startswith("-") else token
        if axis not in axis_indices:
            raise ValueError(f"Invalid axis map '{spec}': token '{token}' is not one of x, y, z, -x, -y, -z.")
        axis_map.append((axis_indices[axis], sign, f"{'-' if sign < 0 else ''}{axis}"))
    return axis_map


def _format_axis_map(axis_map: list[tuple[int, float, str]]) -> str:
    return ",".join(token for _, _, token in axis_map)


def _apply_axis_map_tensor(delta: torch.Tensor, axis_map: list[tuple[int, float, str]]) -> torch.Tensor:
    return torch.stack([delta[index] * sign for index, sign, _ in axis_map]).to(delta)


def _create_head_stereo_fisheye_cameras(width: int, height: int, fisheye_fov: float) -> tuple[Camera, Camera]:
    """Create the same two head-mounted fisheye Camera sensors used by the task scene."""
    head_link_expr = "/World/envs/env_.*/Robot/link_pitch_head"
    head_link_paths = sim_utils.find_matching_prim_paths(head_link_expr)
    if not head_link_paths:
        raise RuntimeError(f"Could not find Agibot head link for stereo cameras: {head_link_expr}")

    rig_translation = (-0.25597, 0.15846, 0.0)
    rig_rotation_deg = (-90.0, 0.0, 180.0)
    rig_orientation = _quat_wxyz_from_euler_xyz_tuple(
        math.radians(rig_rotation_deg[0]),
        math.radians(rig_rotation_deg[1]),
        math.radians(rig_rotation_deg[2]),
    )
    for head_link_path in head_link_paths:
        rig_path = f"{head_link_path}/RobotYaoBasicStereo"
        sim_utils.create_prim(rig_path, "Xform", translation=rig_translation, orientation=rig_orientation)
        print(
            "[RobotYaoBasic] Created head-mounted stereo rig parent "
            f"{rig_path} translation={rig_translation} orient_xyz_deg={rig_rotation_deg}",
            flush=True,
        )

    left_camera_prim_path = f"{head_link_expr}/RobotYaoBasicStereo/LeftFisheye"
    right_camera_prim_path = f"{head_link_expr}/RobotYaoBasicStereo/RightFisheye"
    camera_offset_rot = _quat_wxyz_from_pitch_tuple(0.0)
    left_lateral_offset = float(-args_cli.baseline * 0.5)
    right_lateral_offset = float(args_cli.baseline * 0.5)

    left_camera = Camera(
        cfg=_make_fisheye_camera_cfg(
            left_camera_prim_path,
            width,
            height,
            fisheye_fov,
            (0.10, left_lateral_offset, -0.03),
            camera_offset_rot,
        )
    )
    right_camera = Camera(
        cfg=_make_fisheye_camera_cfg(
            right_camera_prim_path,
            width,
            height,
            fisheye_fov,
            (0.10, right_lateral_offset, -0.03),
            camera_offset_rot,
        )
    )
    return left_camera, right_camera


@configclass
class BasicRobotYaoSceneCfg(InteractiveSceneCfg):
    """Minimal scene with the Agibot robot, table, and cubes."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.06]),
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32), roughness=0.8),
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = _make_robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_TASK_TABLE_POS),
        spawn=sim_utils.CuboidCfg(
            size=_TASK_TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.50, 0.52), roughness=0.7),
        ),
    )
    cube_1 = _cube_cfg("Cube1", _TASK_CUBE_DEFAULT_POSES["cube_1"], (0.10, 0.25, 0.90))
    cube_2 = _cube_cfg("Cube2", _TASK_CUBE_DEFAULT_POSES["cube_2"], (0.08, 0.42, 1.00))
    cube_3 = _cube_cfg("Cube3", _TASK_CUBE_DEFAULT_POSES["cube_3"], (0.06, 0.58, 0.95))


class PerfWindow:
    """Small averaging window for benchmark timing."""

    def __init__(self):
        self.count = 0
        self.totals: dict[str, float] = {}

    def add(self, **sections: float) -> None:
        self.count += 1
        for name, value in sections.items():
            self.totals[name] = self.totals.get(name, 0.0) + max(float(value), 0.0)

    def pop_line(self, frame_id: int, fps: float) -> str:
        parts = []
        for name in ("control", "write", "step", "render", "update", "camera", "copy", "total"):
            if name in self.totals:
                parts.append(f"{name}={1000.0 * self.totals[name] / max(self.count, 1):.1f}ms")
        line = f"[RobotYaoBasic] perf samples={self.count} frame_id={frame_id} fps={fps:.1f} " + " ".join(parts)
        self.count = 0
        self.totals.clear()
        return line


def _reset_scene(scene: InteractiveScene) -> torch.Tensor:
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)

    for rigid_object in scene.rigid_objects.values():
        object_root_state = rigid_object.data.default_root_state.clone()
        object_root_state[:, :3] += scene.env_origins
        rigid_object.write_root_state_to_sim(object_root_state)

    scene.reset()
    return joint_pos


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> float:
    w = float(quat[0].item())
    x = float(quat[1].item())
    y = float(quat[2].item())
    z = float(quat[3].item())
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class BasicUnityController:
    """Lightweight Unity controller path for the basic Agibot benchmark scene."""

    def __init__(self, scene: InteractiveScene):
        self._scene = scene
        self._robot = scene["robot"]
        self._device = self._robot.data.default_root_state.device
        self._env_ids = torch.arange(scene.num_envs, dtype=torch.long, device=self._device)
        self._zero_root_velocity = torch.zeros((scene.num_envs, 6), dtype=torch.float32, device=self._device)
        self._left_body_offset_pos = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=self._device)
        self._left_body_offset_rot = torch.tensor(
            [[0.7071, 0.0, -0.7071, 0.0]], dtype=torch.float32, device=self._device
        ).repeat(scene.num_envs, 1)
        self._root_pos: torch.Tensor | None = None
        self._root_yaw = 0.0
        self._init_root_z: torch.Tensor | None = None
        self._joint_targets: torch.Tensor | None = None
        self._lift_joint_id = None
        self._lift_joint_pos: torch.Tensor | None = None
        self._body_lift_mode = False
        self._previous_body_lift_mode = False
        self._left_arm_hold_after_lift = False
        self._right_arm_hold_after_lift = False
        self._left_arm_follow_warmup_frames = 0
        self._right_arm_follow_warmup_frames = 0
        self._previous_left_follow_start_button = False
        self._previous_right_follow_start_button = False
        self._previous_right_grip_pressed = False
        self._reset_requested = False
        self._reset_reason = ""
        self._step_count = 0
        self._left_gripper_body_id = None
        self._right_gripper_body_id = None
        self._left_arm_target_pos_b: torch.Tensor | None = None
        self._right_arm_target_pos_b: torch.Tensor | None = None
        self._rmpflow_controllers: dict[str, object] = {}
        self._rmpflow_joint_ids: dict[str, list[int]] = {}
        left_map_str = args_cli.left_arm_rmpflow_axis_map or args_cli.arm_rmpflow_axis_map
        right_map_str = args_cli.right_arm_rmpflow_axis_map or args_cli.arm_rmpflow_axis_map
        self._left_axis_map = _parse_axis_map(left_map_str)
        self._right_axis_map = _parse_axis_map(right_map_str)
        left_gripper_body_ids, left_gripper_body_names = self._robot.find_bodies(["gripper_center"])
        if len(left_gripper_body_ids) > 0:
            self._left_gripper_body_id = left_gripper_body_ids[0]
        right_gripper_body_ids, right_gripper_body_names = self._robot.find_bodies(["right_gripper_center"])
        if len(right_gripper_body_ids) > 0:
            self._right_gripper_body_id = right_gripper_body_ids[0]
        self._left_arm_joint_ids, self._left_arm_joint_names = self._robot.find_joints(["left_arm_joint.*"])
        self._right_arm_joint_ids, self._right_arm_joint_names = self._robot.find_joints(["right_arm_joint.*"])
        self._left_gripper_joint_ids, self._left_gripper_joint_names = self._robot.find_joints(
            ["left_hand_joint1", "left_.*_Support_Joint"]
        )
        self._right_gripper_joint_ids, self._right_gripper_joint_names = self._robot.find_joints(
            ["right_hand_joint1", "right_.*_Support_Joint"]
        )
        lift_joint_ids, lift_joint_names = self._robot.find_joints(["joint_lift_body"])
        if len(lift_joint_ids) > 0:
            self._lift_joint_id = lift_joint_ids[0]
        print(
            "[RobotYaoBasic] Unity basic controller joints: "
            f"left_arm={self._left_arm_joint_names}, right_arm={self._right_arm_joint_names}, "
            f"left_gripper={self._left_gripper_joint_names}, right_gripper={self._right_gripper_joint_names}, "
            f"lift={lift_joint_names[0] if len(lift_joint_names) > 0 else 'not_found'}, "
            f"left_axis_map={_format_axis_map(self._left_axis_map)}, "
            f"right_axis_map={_format_axis_map(self._right_axis_map)}, "
            f"left_ee_body={left_gripper_body_names[0] if len(left_gripper_body_names) > 0 else 'not_found'}, "
            f"right_ee_body={right_gripper_body_names[0] if len(right_gripper_body_names) > 0 else 'not_found'}, "
            f"arm_control_mode={args_cli.basic_arm_control_mode}, "
            f"warmup={int(args_cli.arm_follow_start_warmup_frames)}",
            flush=True,
        )
        if args_cli.basic_arm_control_mode == "rmpflow":
            self._initialize_rmpflow_controllers()

    def reset(self, *, hold_arms: bool = False) -> None:
        self._root_pos = self._robot.data.root_pos_w.detach().clone()
        self._root_yaw = _yaw_from_quat_wxyz(self._robot.data.root_quat_w[0].detach())
        self._init_root_z = self._root_pos[:, 2].detach().clone()
        self._joint_targets = self._robot.data.joint_pos.detach().clone()
        if self._lift_joint_id is not None:
            self._lift_joint_pos = self._joint_targets[:, [self._lift_joint_id]].clone()
        self._body_lift_mode = False
        self._previous_body_lift_mode = False
        self._left_arm_hold_after_lift = hold_arms
        self._right_arm_hold_after_lift = hold_arms
        self._left_arm_follow_warmup_frames = 0
        self._right_arm_follow_warmup_frames = 0
        self._previous_left_follow_start_button = False
        self._previous_right_follow_start_button = False
        self._previous_right_grip_pressed = False
        self._left_arm_target_pos_b = None
        self._right_arm_target_pos_b = None
        for controller in self._rmpflow_controllers.values():
            try:
                controller.reset_idx()
            except Exception:
                pass

    def _initialize_rmpflow_controllers(self) -> None:
        try:
            from isaaclab.controllers.config.rmp_flow import AGIBOT_LEFT_ARM_RMPFLOW_CFG, AGIBOT_RIGHT_ARM_RMPFLOW_CFG
            from isaaclab.controllers.rmp_flow import RmpFlowController
        except Exception as exc:
            print(f"[RobotYaoBasic] [WARNING] RMPFlow import failed; falling back to direct arm mapping: {exc}", flush=True)
            return

        for side, cfg, ee_body_id in (
            ("left", AGIBOT_LEFT_ARM_RMPFLOW_CFG, self._left_gripper_body_id),
            ("right", AGIBOT_RIGHT_ARM_RMPFLOW_CFG, self._right_gripper_body_id),
        ):
            if ee_body_id is None:
                print(f"[RobotYaoBasic] [WARNING] {side} EE body missing; falling back to direct arm mapping.", flush=True)
                continue
            try:
                controller = RmpFlowController(cfg=cfg, device=self._device)
                controller.initialize("/World/envs/env_.*/Robot")
                joint_ids, _ = self._robot.find_joints(controller.active_dof_names)
                self._rmpflow_controllers[side] = controller
                self._rmpflow_joint_ids[side] = joint_ids
                print(
                    f"[RobotYaoBasic] {side} RMPFlow controller initialized: "
                    f"frame={cfg.frame_name}, joints={controller.active_dof_names}, joint_ids={joint_ids}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[RobotYaoBasic] [WARNING] {side} RMPFlow init failed; falling back to direct mapping: {exc}",
                    flush=True,
                )

    def _get_arm_position_b(self, side: str) -> torch.Tensor | None:
        if side == "left":
            body_id = self._left_gripper_body_id
        elif side == "right":
            body_id = self._right_gripper_body_id
        else:
            raise ValueError(f"Unsupported arm side: {side}")
        if body_id is None:
            return None
        ee_pos_w = self._robot.data.body_pos_w[:, body_id]
        ee_quat_w = self._robot.data.body_quat_w[:, body_id]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b[0].detach().clone()

    def _get_arm_target_pos_b(self, side: str) -> torch.Tensor | None:
        if side == "left":
            return self._left_arm_target_pos_b
        if side == "right":
            return self._right_arm_target_pos_b
        raise ValueError(f"Unsupported arm side: {side}")

    def _set_arm_target_pos_b(self, side: str, target_pos_b: torch.Tensor | None) -> None:
        if side == "left":
            self._left_arm_target_pos_b = target_pos_b
        elif side == "right":
            self._right_arm_target_pos_b = target_pos_b
        else:
            raise ValueError(f"Unsupported arm side: {side}")

    def _reset_arm_position_target(self, side: str) -> None:
        self._set_arm_target_pos_b(side, None)

    def _position_delta_for_cumulative_target(
        self,
        *,
        side: str,
        mapped_delta: torch.Tensor,
        track_active: bool,
    ) -> torch.Tensor:
        if not args_cli.arm_cumulative_target:
            return mapped_delta
        if not track_active:
            self._reset_arm_position_target(side)
            return mapped_delta
        if torch.any(torch.isnan(mapped_delta)) or torch.any(torch.isinf(mapped_delta)):
            self._reset_arm_position_target(side)
            return torch.zeros_like(mapped_delta)

        current_pos_b = self._get_arm_position_b(side)
        if current_pos_b is None:
            return mapped_delta

        target_pos_b = self._get_arm_target_pos_b(side)
        if target_pos_b is None:
            target_pos_b = current_pos_b.clone()
        target_pos_b = target_pos_b + mapped_delta.detach()
        self._set_arm_target_pos_b(side, target_pos_b)
        return target_pos_b - current_pos_b

    def request_reset(self, reason: str) -> None:
        if self._reset_requested:
            return
        self._reset_requested = True
        self._reset_reason = reason

    def consume_reset_request(self) -> str | None:
        if not self._reset_requested:
            return None
        reason = self._reset_reason or "unspecified"
        self._reset_requested = False
        self._reset_reason = ""
        return reason

    def apply(self, command: torch.Tensor | None, dt: float) -> None:
        if self._root_pos is None or self._joint_targets is None:
            self.reset()
        assert self._root_pos is not None
        assert self._joint_targets is not None

        if command is None:
            command_tensor = torch.zeros(
                RobotYaoWheeledXrRetargeter.OUTPUT_SIZE, dtype=torch.float32, device=self._device
            )
        else:
            command_tensor = command.to(device=self._device, dtype=torch.float32).flatten()
            if command_tensor.numel() < _ROBOTYAO_LEGACY_COMMAND_SIZE:
                self._hold_targets()
                return

        forward = float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_FORWARD])
        lateral = float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_LATERAL])
        yaw_rate = float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_YAW])
        height_vel = (
            float(command_tensor[RobotYaoWheeledXrRetargeter.BASE_HEIGHT_VEL])
            if command_tensor.numel() > RobotYaoWheeledXrRetargeter.BASE_HEIGHT_VEL
            else 0.0
        )
        if not math.isfinite(forward):
            forward = 0.0
        if not math.isfinite(lateral):
            lateral = 0.0
        if not math.isfinite(yaw_rate):
            yaw_rate = 0.0
        if not math.isfinite(height_vel):
            height_vel = 0.0

        left_follow_active, right_follow_active, follow_active = _torch_arm_follow_flags(command_tensor)
        left_follow_active_bool = bool(left_follow_active.item())
        right_follow_active_bool = bool(right_follow_active.item())
        follow_active_bool = bool(follow_active.item())
        body_lift_mode = bool(command_tensor[RobotYaoWheeledXrRetargeter.LEFT_GRIP].item() > 0.5)
        self._body_lift_mode = body_lift_mode
        right_grip_pressed = bool(command_tensor[RobotYaoWheeledXrRetargeter.RIGHT_GRIP].item() > 0.5)
        if right_grip_pressed and not self._previous_right_grip_pressed:
            self.request_reset("right Grip pressed")

        left_follow_start_button = bool(command_tensor[RobotYaoWheeledXrRetargeter.LEFT_SECONDARY].item() > 0.5)
        right_follow_start_button = bool(command_tensor[RobotYaoWheeledXrRetargeter.RIGHT_SECONDARY].item() > 0.5)
        left_follow_start_pressed = left_follow_start_button and not self._previous_left_follow_start_button
        right_follow_start_pressed = right_follow_start_button and not self._previous_right_follow_start_button
        if self._previous_body_lift_mode and not body_lift_mode:
            self._left_arm_hold_after_lift = True
            self._right_arm_hold_after_lift = True
            self._left_arm_follow_warmup_frames = 0
            self._right_arm_follow_warmup_frames = 0
            self._reset_arm_position_target("left")
            self._reset_arm_position_target("right")
        start_warmup_frames = max(0, int(args_cli.arm_follow_start_warmup_frames))
        if left_follow_start_pressed:
            self._left_arm_hold_after_lift = False
            self._left_arm_follow_warmup_frames = start_warmup_frames
            self._reset_arm_position_target("left")
        if right_follow_start_pressed:
            self._right_arm_hold_after_lift = False
            self._right_arm_follow_warmup_frames = start_warmup_frames
            self._reset_arm_position_target("right")
        if not left_follow_active_bool:
            self._left_arm_follow_warmup_frames = 0
            self._reset_arm_position_target("left")
        if not right_follow_active_bool:
            self._right_arm_follow_warmup_frames = 0
            self._reset_arm_position_target("right")

        left_delta_start = RobotYaoWheeledXrRetargeter.LEFT_ARM_DELTA_START
        right_delta_start = RobotYaoWheeledXrRetargeter.RIGHT_ARM_DELTA_START
        left_scene_delta = command_tensor[left_delta_start : left_delta_start + 3]
        right_scene_delta = command_tensor[right_delta_start : right_delta_start + 3]
        left_rot_delta_start = RobotYaoWheeledXrRetargeter.LEFT_ARM_ROT_DELTA_START
        right_rot_delta_start = RobotYaoWheeledXrRetargeter.RIGHT_ARM_ROT_DELTA_START
        left_scene_rot_delta = command_tensor[left_rot_delta_start : left_rot_delta_start + 3]
        right_scene_rot_delta = command_tensor[right_rot_delta_start : right_rot_delta_start + 3]
        if body_lift_mode:
            left_scene_delta = torch.zeros_like(left_scene_delta)
            right_scene_delta = torch.zeros_like(right_scene_delta)
            left_scene_rot_delta = torch.zeros_like(left_scene_rot_delta)
            right_scene_rot_delta = torch.zeros_like(right_scene_rot_delta)

        left_mapped_delta = _apply_axis_map_tensor(left_scene_delta, self._left_axis_map)
        right_mapped_delta = _apply_axis_map_tensor(right_scene_delta, self._right_axis_map)
        left_mapped_rot_delta = _apply_axis_map_tensor(left_scene_rot_delta, self._left_axis_map)
        right_mapped_rot_delta = _apply_axis_map_tensor(right_scene_rot_delta, self._right_axis_map)

        arm_position_deadband = max(0.0, float(args_cli.arm_command_position_deadband))
        arm_rotation_deadband = max(0.0, float(args_cli.arm_command_rotation_deadband))
        left_position_command_norm = float(torch.linalg.norm(left_mapped_delta).item())
        right_position_command_norm = float(torch.linalg.norm(right_mapped_delta).item())
        left_rotation_command_norm = float(torch.linalg.norm(left_mapped_rot_delta).item())
        right_rotation_command_norm = float(torch.linalg.norm(right_mapped_rot_delta).item())
        if left_position_command_norm <= arm_position_deadband:
            left_scene_delta = torch.zeros_like(left_scene_delta)
            left_mapped_delta = torch.zeros_like(left_mapped_delta)
        if right_position_command_norm <= arm_position_deadband:
            right_scene_delta = torch.zeros_like(right_scene_delta)
            right_mapped_delta = torch.zeros_like(right_mapped_delta)
        if left_rotation_command_norm <= arm_rotation_deadband:
            left_scene_rot_delta = torch.zeros_like(left_scene_rot_delta)
            left_mapped_rot_delta = torch.zeros_like(left_mapped_rot_delta)
        if right_rotation_command_norm <= arm_rotation_deadband:
            right_scene_rot_delta = torch.zeros_like(right_scene_rot_delta)
            right_mapped_rot_delta = torch.zeros_like(right_mapped_rot_delta)

        left_in_follow_warmup = self._left_arm_follow_warmup_frames > 0
        right_in_follow_warmup = self._right_arm_follow_warmup_frames > 0
        left_target_tracking_active = left_follow_active_bool and not body_lift_mode and not self._left_arm_hold_after_lift
        right_target_tracking_active = (
            right_follow_active_bool and not body_lift_mode and not self._right_arm_hold_after_lift
        )
        left_arm_action_delta = self._position_delta_for_cumulative_target(
            side="left",
            mapped_delta=left_mapped_delta,
            track_active=left_target_tracking_active,
        )
        right_arm_action_delta = self._position_delta_for_cumulative_target(
            side="right",
            mapped_delta=right_mapped_delta,
            track_active=right_target_tracking_active,
        )
        left_position_action_norm = float(torch.linalg.norm(left_arm_action_delta).item())
        right_position_action_norm = float(torch.linalg.norm(right_arm_action_delta).item())
        left_has_arm_command = bool(
            left_position_action_norm > arm_position_deadband
            or float(torch.linalg.norm(left_mapped_rot_delta).item()) > arm_rotation_deadband
        )
        right_has_arm_command = bool(
            right_position_action_norm > arm_position_deadband
            or float(torch.linalg.norm(right_mapped_rot_delta).item()) > arm_rotation_deadband
        )
        drive_left_follow_active_bool = (
            left_follow_active_bool
            and not body_lift_mode
            and not self._left_arm_hold_after_lift
            and not left_in_follow_warmup
            and left_has_arm_command
        )
        drive_right_follow_active_bool = (
            right_follow_active_bool
            and not body_lift_mode
            and not self._right_arm_hold_after_lift
            and not right_in_follow_warmup
            and right_has_arm_command
        )
        if self._left_arm_follow_warmup_frames > 0:
            self._left_arm_follow_warmup_frames -= 1
        if self._right_arm_follow_warmup_frames > 0:
            self._right_arm_follow_warmup_frames -= 1

        self._root_yaw += max(-1.57, min(1.57, yaw_rate)) * dt
        c = math.cos(self._root_yaw)
        s = math.sin(self._root_yaw)
        forward = max(-2.0, min(2.0, forward))
        lateral = max(-2.0, min(2.0, lateral))
        self._root_pos[:, 0] += (c * forward - s * lateral) * dt
        self._root_pos[:, 1] += (s * forward + c * lateral) * dt
        if self._init_root_z is not None:
            self._root_pos[:, 2] = self._init_root_z
        self._write_root_pose()

        if self._lift_joint_id is not None and self._lift_joint_pos is not None:
            self._lift_joint_pos += max(-1.0, min(1.0, height_vel)) * 0.3 * dt
            lift_limits = self._robot.data.soft_joint_pos_limits[:, self._lift_joint_id, :]
            self._lift_joint_pos = torch.clamp(
                self._lift_joint_pos,
                min=lift_limits[:, [0]],
                max=lift_limits[:, [1]],
            )
            self._joint_targets[:, [self._lift_joint_id]] = self._lift_joint_pos

        if drive_left_follow_active_bool:
            self._apply_arm_delta("left", left_arm_action_delta, left_mapped_rot_delta)
        if drive_right_follow_active_bool:
            self._apply_arm_delta("right", right_arm_action_delta, right_mapped_rot_delta)

        self._apply_gripper_target(
            self._left_gripper_joint_ids,
            close_fraction=float(command_tensor[RobotYaoWheeledXrRetargeter.LEFT_TRIGGER]),
            close_value=0.0,
        )
        self._apply_gripper_target(
            self._right_gripper_joint_ids,
            close_fraction=float(command_tensor[RobotYaoWheeledXrRetargeter.RIGHT_TRIGGER]),
            close_value=0.20,
        )
        if args_cli.debug_task_loop:
            self._step_count += 1
            if self._step_count % 10 == 0:
                raw_left_delta_start = RobotYaoWheeledXrRetargeter.RAW_LEFT_DELTA_START
                raw_right_delta_start = RobotYaoWheeledXrRetargeter.RAW_RIGHT_DELTA_START
                raw_left_delta = command_tensor[raw_left_delta_start : raw_left_delta_start + 3]
                raw_right_delta = command_tensor[raw_right_delta_start : raw_right_delta_start + 3]
                print(
                    f"[DEBUG BasicController] step={self._step_count} "
                    f"base={[forward, lateral, yaw_rate]} follow={follow_active_bool} "
                    f"left_follow={left_follow_active_bool} right_follow={right_follow_active_bool} "
                    f"drive_left={drive_left_follow_active_bool} drive_right={drive_right_follow_active_bool} "
                    f"body_lift={body_lift_mode} "
                    f"left_hold_after_lift={self._left_arm_hold_after_lift} "
                    f"right_hold_after_lift={self._right_arm_hold_after_lift} "
                    f"left_warmup={self._left_arm_follow_warmup_frames} "
                    f"right_warmup={self._right_arm_follow_warmup_frames} "
                    f"left_raw_delta={raw_left_delta.tolist()} right_raw_delta={raw_right_delta.tolist()} "
                    f"left_scene_delta={left_scene_delta.tolist()} right_scene_delta={right_scene_delta.tolist()} "
                    f"left_mapped_delta={left_mapped_delta.tolist()} right_mapped_delta={right_mapped_delta.tolist()} "
                    f"left_action_delta={left_arm_action_delta.tolist()} "
                    f"right_action_delta={right_arm_action_delta.tolist()} "
                    f"left_scene_rot={left_scene_rot_delta.tolist()} right_scene_rot={right_scene_rot_delta.tolist()} "
                    f"left_mapped_rot={left_mapped_rot_delta.tolist()} right_mapped_rot={right_mapped_rot_delta.tolist()} "
                    f"left_axis_map={_format_axis_map(self._left_axis_map)} "
                    f"right_axis_map={_format_axis_map(self._right_axis_map)} "
                    f"arm_control_mode={args_cli.basic_arm_control_mode}",
                    flush=True,
                )
        self._previous_body_lift_mode = body_lift_mode
        self._previous_left_follow_start_button = left_follow_start_button
        self._previous_right_follow_start_button = right_follow_start_button
        self._previous_right_grip_pressed = right_grip_pressed
        self._hold_targets()

    def _apply_arm_delta(self, side: str, position_delta: torch.Tensor, rotation_delta: torch.Tensor) -> None:
        if args_cli.basic_arm_control_mode == "rmpflow" and side in self._rmpflow_controllers:
            self._apply_rmpflow_delta(side, position_delta, rotation_delta)
            return
        joint_ids = self._left_arm_joint_ids if side == "left" else self._right_arm_joint_ids
        self._apply_direct_arm_delta(position_delta, rotation_delta, joint_ids)

    def _apply_rmpflow_delta(self, side: str, position_delta: torch.Tensor, rotation_delta: torch.Tensor) -> None:
        if self._joint_targets is None:
            return
        controller = self._rmpflow_controllers.get(side)
        joint_ids = self._rmpflow_joint_ids.get(side, [])
        if controller is None or len(joint_ids) == 0:
            joint_ids = self._left_arm_joint_ids if side == "left" else self._right_arm_joint_ids
            self._apply_direct_arm_delta(position_delta, rotation_delta, joint_ids)
            return

        body_id = self._left_gripper_body_id if side == "left" else self._right_gripper_body_id
        if body_id is None:
            joint_ids = self._left_arm_joint_ids if side == "left" else self._right_arm_joint_ids
            self._apply_direct_arm_delta(position_delta, rotation_delta, joint_ids)
            return
        if torch.any(torch.isnan(position_delta)) or torch.any(torch.isinf(position_delta)):
            position_delta = torch.zeros_like(position_delta)
        if torch.any(torch.isnan(rotation_delta)) or torch.any(torch.isinf(rotation_delta)):
            rotation_delta = torch.zeros_like(rotation_delta)

        ee_pos_w = self._robot.data.body_pos_w[:, body_id]
        ee_quat_w = self._robot.data.body_quat_w[:, body_id]
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        ee_pose_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        if side == "left":
            ee_pose_b, ee_quat_b = combine_frame_transforms(
                ee_pose_b,
                ee_quat_b,
                self._left_body_offset_pos,
                self._left_body_offset_rot,
            )

        delta_action = torch.cat([position_delta, rotation_delta]).unsqueeze(0)
        if delta_action.shape[0] != self._scene.num_envs:
            delta_action = delta_action.repeat(self._scene.num_envs, 1)
        ee_pos_des, ee_quat_des = apply_delta_pose(ee_pose_b, ee_quat_b, delta_action)
        ee_pose_des = torch.cat([ee_pos_des, ee_quat_des], dim=1)

        controller.set_command(ee_pose_des)
        joint_pos_des, joint_vel_des = controller.compute()
        self._joint_targets[:, joint_ids] = joint_pos_des
        self._clamp_joint_targets(joint_ids)
        self._robot.set_joint_velocity_target(joint_vel_des, joint_ids=joint_ids)

    def _write_root_pose(self) -> None:
        assert self._root_pos is not None
        root_pose = torch.zeros((self._scene.num_envs, 7), dtype=torch.float32, device=self._device)
        root_pose[:, 0:3] = self._root_pos
        root_pose[:, 3:7] = _quat_wxyz_from_yaw_tensor(self._root_yaw, self._scene.num_envs, self._device)
        self._robot.write_root_pose_to_sim(root_pose, env_ids=self._env_ids)
        self._robot.write_root_velocity_to_sim(self._zero_root_velocity, env_ids=self._env_ids)

    def _apply_direct_arm_delta(
        self,
        hand_delta: torch.Tensor,
        rotation_delta: torch.Tensor,
        joint_ids: list[int],
    ) -> None:
        if self._joint_targets is None or len(joint_ids) == 0:
            return
        if torch.any(torch.isnan(hand_delta)) or torch.any(torch.isinf(hand_delta)):
            hand_delta = torch.zeros_like(hand_delta)
        if torch.any(torch.isnan(rotation_delta)) or torch.any(torch.isinf(rotation_delta)):
            rotation_delta = torch.zeros_like(rotation_delta)
        joint_delta = torch.zeros(
            (self._scene.num_envs, len(joint_ids)), dtype=torch.float32, device=self._device
        )
        if joint_delta.shape[1] > 0:
            joint_delta[:, 0] = hand_delta[1] * 2.5
        if joint_delta.shape[1] > 1:
            joint_delta[:, 1] = hand_delta[2] * 2.0
        if joint_delta.shape[1] > 3:
            joint_delta[:, 3] = -hand_delta[0] * 2.0
        if joint_delta.shape[1] > 4:
            joint_delta[:, 4] = rotation_delta[0]
        if joint_delta.shape[1] > 5:
            joint_delta[:, 5] = rotation_delta[1]
        if joint_delta.shape[1] > 6:
            joint_delta[:, 6] = rotation_delta[2]
        self._joint_targets[:, joint_ids] += joint_delta
        self._clamp_joint_targets(joint_ids)

    def _apply_gripper_target(self, joint_ids: list[int], close_fraction: float, close_value: float) -> None:
        if self._joint_targets is None or len(joint_ids) == 0:
            return
        close_fraction = max(0.0, min(1.0, close_fraction if math.isfinite(close_fraction) else 0.0))
        value = 0.994 + (float(close_value) - 0.994) * close_fraction
        self._joint_targets[:, joint_ids] = value
        self._clamp_joint_targets(joint_ids)

    def _clamp_joint_targets(self, joint_ids: list[int]) -> None:
        if self._joint_targets is None or len(joint_ids) == 0:
            return
        joint_limits = self._robot.data.soft_joint_pos_limits[:, joint_ids, :]
        self._joint_targets[:, joint_ids] = torch.clamp(
            self._joint_targets[:, joint_ids],
            min=joint_limits[..., 0],
            max=joint_limits[..., 1],
        )

    def _hold_targets(self) -> None:
        if self._joint_targets is None:
            return
        self._robot.set_joint_position_target(self._joint_targets)


def _create_unity_control_device() -> RobotYaoXrSubDevice:
    retargeter_cfg = RobotYaoWheeledXrRetargeterCfg(
        sim_device=args_cli.device,
        max_forward_speed=args_cli.max_forward_speed,
        max_lateral_speed=args_cli.max_lateral_speed,
        max_yaw_rate=args_cli.max_yaw_rate,
        arm_delta_scale=args_cli.arm_delta_scale,
        arm_rotation_delta_scale=args_cli.arm_rotation_delta_scale,
        arm_position_delta_dead_zone=args_cli.arm_command_position_deadband,
        arm_rotation_delta_dead_zone=args_cli.arm_command_rotation_deadband,
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


def run_benchmark(
    sim: SimulationContext,
    scene: InteractiveScene,
    stereo_cameras: tuple[Camera, Camera] | None = None,
    control_device: RobotYaoXrSubDevice | None = None,
    unity_controller: BasicUnityController | None = None,
) -> None:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    render_stride = max(0, int(args_cli.render_every_n_frames))
    print_every = max(1, int(args_cli.print_every))
    default_joint_pos = _reset_scene(scene)
    if unity_controller is not None:
        unity_controller.reset()
    perf = PerfWindow()
    frame_id = 0
    start_time = time.perf_counter()

    print(
        "[RobotYaoBasic] Running minimal scene benchmark: "
        f"num_envs={scene.num_envs}, dt={sim_dt}, render_every_n_frames={render_stride}, "
        f"robot=Agibot A2D, table=True, cubes=3, stereo_cameras={stereo_cameras is not None}, "
        f"unity_control={control_device is not None}, task_managers=False, video_stream=False",
        flush=True,
    )

    while simulation_app.is_running():
        loop_t0 = time.perf_counter()

        if control_device is not None and unity_controller is not None:
            unity_controller.apply(control_device.advance(), sim_dt)
            reset_reason = unity_controller.consume_reset_request()
            if reset_reason is not None:
                print(f"[RobotYaoBasic] Resetting basic scene: {reset_reason}", flush=True)
                default_joint_pos = _reset_scene(scene)
                unity_controller.reset(hold_arms=True)
                if stereo_cameras is not None:
                    for camera in stereo_cameras:
                        camera.reset()
        else:
            robot.set_joint_position_target(default_joint_pos)
        control_t1 = time.perf_counter()
        scene.write_data_to_sim()
        write_t2 = time.perf_counter()
        sim.step(render=False)
        step_t3 = time.perf_counter()
        if render_stride > 0 and frame_id % render_stride == 0:
            sim.render()
        render_t4 = time.perf_counter()
        scene.update(sim_dt)
        update_t5 = time.perf_counter()
        if stereo_cameras is not None:
            for camera in stereo_cameras:
                camera.update(dt=sim_dt)
        camera_t6 = time.perf_counter()

        frame_id += 1
        elapsed = max(time.perf_counter() - start_time, 1.0e-6)
        perf.add(
            control=control_t1 - loop_t0,
            write=write_t2 - control_t1,
            step=step_t3 - write_t2,
            render=render_t4 - step_t3,
            update=update_t5 - render_t4,
            camera=camera_t6 - update_t5,
            copy=0.0,
            total=camera_t6 - loop_t0,
        )

        if frame_id % print_every == 0:
            print(perf.pop_line(frame_id, frame_id / elapsed), flush=True)

        if int(args_cli.max_frames) > 0 and frame_id >= int(args_cli.max_frames):
            break

    print(f"[RobotYaoBasic] stopped frames={frame_id}", flush=True)


def main() -> None:
    sim_cfg = SimulationCfg(dt=float(args_cli.dt), device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.4, -1.3, 1.2], target=[0.2, 0.0, -0.25])

    scene_cfg = BasicRobotYaoSceneCfg(num_envs=int(args_cli.num_envs), env_spacing=3.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)
    stereo_cameras = None
    if args_cli.with_stereo_fisheye:
        print(
            "[RobotYaoBasic] Creating stereo fisheye cameras without video streaming: "
            f"{int(args_cli.camera_width)}x{int(args_cli.camera_height)}, fov={float(args_cli.fisheye_fov):.1f}",
            flush=True,
        )
        stereo_cameras = _create_head_stereo_fisheye_cameras(
            int(args_cli.camera_width),
            int(args_cli.camera_height),
            float(args_cli.fisheye_fov),
        )
    sim.reset()
    if stereo_cameras is not None:
        for camera in stereo_cameras:
            camera.reset()
    control_device = _create_unity_control_device() if args_cli.unity_control else None
    unity_controller = BasicUnityController(scene) if control_device is not None else None
    if control_device is not None:
        print(
            f"[RobotYaoBasic] Unity controller input enabled, endpoint={args_cli.unity_input_endpoint}, "
            f"topic={args_cli.unity_input_topic}",
            flush=True,
        )
    print("[RobotYaoBasic] setup complete.", flush=True)
    try:
        run_benchmark(sim, scene, stereo_cameras, control_device, unity_controller)
    finally:
        if control_device is not None:
            try:
                control_device.stop()
            except Exception as exc:
                print(f"[RobotYaoBasic] Failed to stop Unity control device: {exc}", flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        raise
    finally:
        if args_cli.no_fast_exit_on_max_frames:
            simulation_app.close()
        elif int(args_cli.max_frames) > 0 or exit_code != 0:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        else:
            simulation_app.close()
