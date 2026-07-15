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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    G1GripperSyncActionCfg,
    MuJoCoG1MirrorActionCfg,
    SonicDeployTargetActionCfg,
    SonicRobotStatePublisherActionCfg,
    UnitreeDdsLowCmdActionCfg,
    UnitreeLowStatePublisherActionCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_tasks.manager_based.locomanipulation.pick_place.zmq_object_sync import (
    ZmqObjectSyncActionCfg,
    ZmqRobotSyncActionCfg,
)
from copy import deepcopy

from isaaclab_assets.robots.unitree import G1_29DOF_CFG
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import (
    SONIC_G1_29DOF_DEFAULT_ANGLES,
    SONIC_G1_29DOF_JOINT_ORDER,
)

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


def _isaac_robot_env(robot_id: int, suffix: str, default: str) -> str:
    return os.environ.get(f"ISAACLAB_G1_{robot_id}_{suffix}", os.environ.get(f"ISAACLAB_G1_{suffix}", default))


def _isaac_robot_env_int(robot_id: int, suffix: str, default: int) -> int:
    try:
        return int(_isaac_robot_env(robot_id, suffix, str(default)))
    except (TypeError, ValueError):
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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


SONIC_G1_PHYSICS_MODE = _env_flag("SONIC_G1_PHYSICS_MODE", False)
SONIC_G1_FIX_ROOT = not SONIC_G1_PHYSICS_MODE
SONIC_G1_VISUAL_SERVO_MODE = _env_flag("SONIC_G1_VISUAL_SERVO_MODE", SONIC_G1_FIX_ROOT)
SONIC_G1_SELF_COLLISIONS = SONIC_G1_PHYSICS_MODE and _env_flag("SONIC_G1_SELF_COLLISIONS", False)
ENABLE_WALKER_ROBOT = _env_flag("LOCIMANIP_ENABLE_WALKER_ROBOT", False) and SONIC_G1_PHYSICS_MODE
# banyun 工位机器人开关：默认（1）SONICRobot 顶替 Robot_1 站工位，Robot_1 及其
# mirror/gripper 动作项随之下线；设 LOCOMANIP_SONIC_REPLACE_ROBOT1=0 恢复原
# Robot_1（GR00T 43dof 镜像），SONICRobot 退回南侧 8m 行走通道出生点。
SONIC_REPLACE_ROBOT1 = _env_flag("LOCOMANIP_SONIC_REPLACE_ROBOT1", True)
# 对端镜像机开关：LOCOMANIP_ENABLE_ROBOT2=0 时 Robot_2（43dof）连同其镜像流/
# 夹爪同步整链下线。每台 G1 articulation 约腰斩 env_hz（PhysX 关节解算与它动不动
# 无关），SONIC 闭环单机调试给不出实时余量时先关它；默认 1 保持双机遥操行为不变。
ENABLE_ROBOT2 = _env_flag("LOCOMANIP_ENABLE_ROBOT2", True)


def _robot_asset_present(robot_id: int) -> bool:
    """镜像机器人实体是否在场（robot_1 被 SONIC 顶替 / robot_2 被开关下线时缺席）。"""
    return ENABLE_ROBOT2 if robot_id == 2 else not SONIC_REPLACE_ROBOT1
print(
    "[locomanip_cfg] "
    f"SONIC_G1_FIX_ROOT={SONIC_G1_FIX_ROOT} "
    f"SONIC_G1_PHYSICS_MODE={SONIC_G1_PHYSICS_MODE} "
    f"SONIC_G1_VISUAL_SERVO_MODE={SONIC_G1_VISUAL_SERVO_MODE} "
    f"SONIC_G1_SELF_COLLISIONS={SONIC_G1_SELF_COLLISIONS} "
    f"ENABLE_WALKER_ROBOT={ENABLE_WALKER_ROBOT} "
    f"SONIC_REPLACE_ROBOT1={SONIC_REPLACE_ROBOT1} "
    f"ENABLE_ROBOT2={ENABLE_ROBOT2} "
    f"legacy_SONIC_G1_FIX_ROOT_env={os.environ.get('SONIC_G1_FIX_ROOT', '<unset>')!r}"
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
            Path.home() / "GR00T-WholeBodyControl",
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
        "Could not locate GR00T G1 43-DoF USD. Set GR00T_WBC_ROOT to the GR00T-WholeBodyControl path. "
        f"Searched:\n  {searched}"
    )


G1_43DOF_GR00T_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=UsdFileCfg(
        usd_path=_find_gr00t_g1_43dof_usd(),
        activate_contact_sensors=False,
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
        pos=(-3.8, 19.008, 0.78),
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



# 第四个机器人：GEAR-SONIC ONNX/Deploy 驱动。
#
# 短期目标是先验证 PICO manager -> GR00T/SONIC deploy -> IsaacLab 的目标链路，
# 等价于 MuJoCo run_sim_loop.py 当前承担的“站稳状态源/可视化机器人”角色；因此默认固定
# root 并关闭重力，避免没有完整 LowState/平衡闭环时启动即倒。长期做 IsaacLab Unitree DDS
# 闭环物理验证时，设置 SONIC_G1_PHYSICS_MODE=1 才恢复自由根节点和重力。
# init_state.pos 与 walker 同 Y（11.008，来自 align_walker_robot_to_conveyor 事件运行时计算），
# X 错开 3m 便于 GUI 视角同框观察。终极方案应仿照 align_walker_robot_to_conveyor 加一个对齐事件。
#
# 阶段 3.3 E3 D：mocap anchor 时变信号已接，解 fix_root_link 再次物理验证
# 对比 3.1 初次物理验证（立刻摔倒），看 mocap motion 信号是否提供有意义的平衡反馈
#
# 阶段 A（gr00t-sonic-actuator-match 分支）：用 SONIC 训练同款 ImplicitActuator + PD 配方
# 替换默认 G1_29DOF_CFG 的 DCMotor。参考 gear_sonic/envs/manager_env/robots/g1.py:10-358
# （来自 BeyondMimic / whole_body_tracking）。NATURAL_FREQ=10Hz、DAMPING_RATIO=2.0，
# 各 actuator armature 配 stiffness=armature×NATURAL_FREQ²、damping=2×DAMPING_RATIO×armature×NATURAL_FREQ。
# 注意：不动 G1_29DOF_CFG（robot / walker_robot / remote_robot 仍用 IsaacLab DCMotor）。
# 默认 fixed-root deploy 验证是“目标可视化”，不是完整物理闭环；这里会在定义完训练 PD
# 后按 SONIC_G1_VISUAL_SERVO_MODE 切回 IsaacLab 原始高刚度位置伺服，让手臂更忠实跟随
# GR00T/SONIC deploy 的实际 motor target。设置 SONIC_G1_VISUAL_SERVO_MODE=0 可恢复训练 PD。
from isaaclab.actuators import ImplicitActuatorCfg as _SonicImplicitActuatorCfg

_SONIC_ARMATURE_5020 = 0.003609725
_SONIC_ARMATURE_7520_14 = 0.010177520
_SONIC_ARMATURE_7520_22 = 0.025101925
_SONIC_ARMATURE_4010 = 0.00425
_SONIC_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10Hz
_SONIC_DAMPING_RATIO = 2.0

_S_5020 = _SONIC_ARMATURE_5020 * _SONIC_NATURAL_FREQ**2
_S_7520_14 = _SONIC_ARMATURE_7520_14 * _SONIC_NATURAL_FREQ**2
_S_7520_22 = _SONIC_ARMATURE_7520_22 * _SONIC_NATURAL_FREQ**2
_S_4010 = _SONIC_ARMATURE_4010 * _SONIC_NATURAL_FREQ**2

_D_5020 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_5020 * _SONIC_NATURAL_FREQ
_D_7520_14 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_7520_14 * _SONIC_NATURAL_FREQ
_D_7520_22 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_7520_22 * _SONIC_NATURAL_FREQ
_D_4010 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_4010 * _SONIC_NATURAL_FREQ

SONIC_G1_29DOF_CFG = G1_29DOF_CFG.copy()
SONIC_G1_29DOF_CFG.spawn.activate_contact_sensors = SONIC_G1_PHYSICS_MODE
SONIC_G1_29DOF_CFG.spawn.articulation_props.fix_root_link = SONIC_G1_FIX_ROOT
SONIC_G1_29DOF_CFG.spawn.articulation_props.enabled_self_collisions = SONIC_G1_SELF_COLLISIONS
SONIC_G1_29DOF_CFG.spawn.rigid_props.disable_gravity = _env_flag("SONIC_G1_DISABLE_GRAVITY", SONIC_G1_FIX_ROOT)
if SONIC_G1_PHYSICS_MODE:
    # retain_accelerations=True 让 PhysX 在每步结束后保留 link 加速度，
    # 使 body_com_lin_acc_w 返回真实值（用于 IMU accelerometer 计算）。
    SONIC_G1_29DOF_CFG.spawn.rigid_props.retain_accelerations = True
# Z=0.76：脚底在地面上方约 9mm。lock_root_z=False 的物理模式下 root Z 自由，
# settle 阶段自然落地；若 spawn 时脚穿透地面（如 0.72 → 约 -3cm）会触发 PhysX
# depenetration 向上弹射冲击。宁高勿低。
SONIC_G1_29DOF_CFG.init_state.pos = (-2.0, 11.008, 0.76)
SONIC_G1_29DOF_CFG.init_state.rot = (1.0, 0.0, 0.0, 0.0)
SONIC_G1_29DOF_CFG.init_state.joint_pos = dict(
    zip(SONIC_G1_29DOF_JOINT_ORDER, SONIC_G1_29DOF_DEFAULT_ANGLES, strict=True)
)
# 整体替换 actuators，与 SONIC 训练完全对齐
SONIC_G1_29DOF_CFG.actuators = {
    "legs": _SonicImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 139.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 20.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": _S_7520_22,
            ".*_hip_roll_joint": _S_7520_22,
            ".*_hip_yaw_joint": _S_7520_14,
            ".*_knee_joint": _S_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": _D_7520_22,
            ".*_hip_roll_joint": _D_7520_22,
            ".*_hip_yaw_joint": _D_7520_14,
            ".*_knee_joint": _D_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": _SONIC_ARMATURE_7520_22,
            ".*_hip_roll_joint": _SONIC_ARMATURE_7520_22,
            ".*_hip_yaw_joint": _SONIC_ARMATURE_7520_14,
            ".*_knee_joint": _SONIC_ARMATURE_7520_22,
        },
    ),
    "feet": _SonicImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * _S_5020,
        damping=2.0 * _D_5020,
        armature=2.0 * _SONIC_ARMATURE_5020,
    ),
    "waist": _SonicImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * _S_5020,
        damping=2.0 * _D_5020,
        armature=2.0 * _SONIC_ARMATURE_5020,
    ),
    "waist_yaw": _SonicImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=88.0,
        velocity_limit_sim=32.0,
        stiffness=_S_7520_14,
        damping=_D_7520_14,
        armature=_SONIC_ARMATURE_7520_14,
    ),
    "arms": _SonicImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_pitch_joint": _S_5020,
            ".*_shoulder_roll_joint": _S_5020,
            ".*_shoulder_yaw_joint": _S_5020,
            ".*_elbow_joint": _S_5020,
            ".*_wrist_roll_joint": _S_5020,
            ".*_wrist_pitch_joint": _S_4010,
            ".*_wrist_yaw_joint": _S_4010,
        },
        damping={
            ".*_shoulder_pitch_joint": _D_5020,
            ".*_shoulder_roll_joint": _D_5020,
            ".*_shoulder_yaw_joint": _D_5020,
            ".*_elbow_joint": _D_5020,
            ".*_wrist_roll_joint": _D_5020,
            ".*_wrist_pitch_joint": _D_4010,
            ".*_wrist_yaw_joint": _D_4010,
        },
        armature={
            ".*_shoulder_pitch_joint": _SONIC_ARMATURE_5020,
            ".*_shoulder_roll_joint": _SONIC_ARMATURE_5020,
            ".*_shoulder_yaw_joint": _SONIC_ARMATURE_5020,
            ".*_elbow_joint": _SONIC_ARMATURE_5020,
            ".*_wrist_roll_joint": _SONIC_ARMATURE_5020,
            ".*_wrist_pitch_joint": _SONIC_ARMATURE_4010,
            ".*_wrist_yaw_joint": _SONIC_ARMATURE_4010,
        },
    ),
}
# ---------------------------------------------------------------------------
# MuJoCo 已验证 plant 对齐（2026-07-15 四路源码取证，KB sim2sim 清单第①项）。
# 取证结论：kp/kd 与 deploy policy_parameters.hpp 逐关节完全一致；deploy LowCmd
# 的 dq_target/tau_ff 三态恒 0（g1_deploy_onnx_ref.cpp:3129-3132）——"丢 dq/tau"
# 是伪缺口。真实差异只有下面三处。
#
# 2026-07-15 闭环 A/B 判定（12 轮 BVH + 4 轮键盘站立，健康段隔离指标，方差带判据）：
# - BVH 120s 长窗 ×4：整包开 vs 基线四抖动指标 2v2 分离（手臂高频 −16.7%、
#   腰 −18.3%）——但该结论**不迁移到站立**，见下。
#   30s 短窗 ×8 补充：⚠️单开 NO_ARMATURE 手臂高频恶化 2v2 分离（转子惯量是
#   天然低通）；单开 TORQUE_PARITY 中性偏负——拆包依然禁止。
# - **键盘 Planner IDLE 站立 120s ×4（JITTER_INPUT=keyboard，无 BVH 输入）翻案**：
#   整包开站不住——xy 漂移 {6.0, 13.4}m vs 关 {0.19, 0.48}m（差 12~70 倍）、
#   tilt 均值 {2.88, 3.32}° vs {1.05, 1.63}°，两指标 2v2 完全分离；站立下手臂
#   高频反而更差（0.16~0.24 vs 0.08~0.15）。机器人靠不停踏步纠偏缓慢游走，
#   与用户实测"容易摔、没感到消抖"一致。BVH 轮的"改善"是跟随目标主导的
#   测量偏置，不是平衡质量变好。
# 结论：**整包对齐默认关闭**（2026-07-15 站立协议判定为稳定性回归并回退；
# 设 SONIC_G1_MUJOCO_*=1 可逐项开启复现取证/对照实验）。
# ---------------------------------------------------------------------------
# ① 髋 pitch/roll 力矩上限：被验证的 MuJoCo plant 钳在 ±88（g1_29dof.xml
#    actuatorfrcrange 与 g1_29dof_gear_wbc.yaml motor_effort_limit_list 双源一致；
#    电机换代 7520_14→7520_22 后 hpp 升到 139，但验证 plant 从未跟进）。
#    Isaac 139 = 髋部纠偏力矩比验证 plant 大 1.6×，policy 过纠偏嫌疑。
if _env_flag("SONIC_G1_MUJOCO_TORQUE_PARITY", False):
    SONIC_G1_29DOF_CFG.actuators["legs"].effort_limit_sim[".*_hip_roll_joint"] = 88.0
    SONIC_G1_29DOF_CFG.actuators["legs"].effort_limit_sim[".*_hip_pitch_joint"] = 88.0
# ② armature：MuJoCo joint 无 armature/damping（XML 无 <default> 段，默认 0）；
#    Isaac 带 0.0036~0.025 的转子惯量，关节加速响应比验证 plant 慢。
if _env_flag("SONIC_G1_MUJOCO_NO_ARMATURE", False):
    for _sonic_act in SONIC_G1_29DOF_CFG.actuators.values():
        _sonic_act.armature = 0.0
# ③ 关节速度钳位：MuJoCo 不钳关节速度；Isaac velocity_limit_sim 20~37 rad/s
#    会截断软 PD policy 的快甩纠偏。
if _env_flag("SONIC_G1_MUJOCO_NO_VEL_LIMIT", False):
    for _sonic_act in SONIC_G1_29DOF_CFG.actuators.values():
        _sonic_act.velocity_limit_sim = 1.0e3

if SONIC_G1_VISUAL_SERVO_MODE:
    SONIC_G1_29DOF_CFG.actuators = deepcopy(G1_29DOF_CFG.actuators)

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


@configclass
class LocomanipulationG1SceneCfg(InteractiveSceneCfg):
    """Scene configuration for locomanipulation environment with G1 robot.

    This configuration sets up the G1 humanoid robot for locomanipulation tasks,
    allowing both locomotion and manipulation capabilities. The robot can move its
    base and use its arms for manipulation tasks.
    """

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
        # PackingTable (z=-1000.66, 高 0.08) 顶面 = -1000.62，物体半高 0.06 → 底面贴桌面的中心 z
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, -1000.56], rot=[1, 0, 0, 0]),
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
        # 对齐 ConveyorBelt（场景位置 y=14.39363，绕 Z 90°），再叠加 +90°Z = 180°Z 总旋转
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步推车（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_pushcart_spawn_cfg(syncable=True),
    )
    cart_box1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CartBox1",
        # 推车顶面 z≈0.377，箱子半高 0.0745 → 中心 z
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.45], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    )
    cart_box2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CartBox2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.60], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    )
    cart_box3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CartBox3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.75], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    )
    cart_box4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CartBox4",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-5.4, 19.39363, 0.90], rot=[0.0, 0.0, 0.0, 1.0]),
        # syncable=True：跨机 ZMQ 同步该箱子（订阅端切换为 kinematic，跟随发布端位姿）
        spawn=_make_graspable_cart_box_spawn_cfg(syncable=True),
    )
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
    cart2_tote1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cart2Tote1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-8.94726, 20.14363, 0.3774], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "props", "tote_b04_physics.usda"),
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                max_depenetration_velocity=5.0,
            ),
        ),
    )
    cart2_tote2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cart2Tote2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-8.94726, 20.14363, 0.6774], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "props", "tote_b04_physics.usda"),
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                max_depenetration_velocity=5.0,
            ),
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
    # banyun 工位（ID 1 基准位姿，推车旁，面向 +Y）默认由 SONICRobot 顶替（见
    # 下方 sonic_robot）；LOCOMANIP_SONIC_REPLACE_ROBOT1=0 时恢复 Robot_1 镜像
    # 机器人。Robot_2 保留在工位右侧 1.5 m（避开 x=-5.4 的推车）。
    if not SONIC_REPLACE_ROBOT1:
        robot_1 = G1_43DOF_GR00T_CFG.replace(
            prim_path="/World/envs/env_.*/Robot_1",
        )
    # LOCOMANIP_ENABLE_ROBOT2=0 时对端镜像机下线（SONIC 闭环单机调试省一台 G1 的
    # PhysX 解算，见模块顶部开关注释）
    if ENABLE_ROBOT2:
        robot_2: ArticulationCfg = G1_43DOF_GR00T_CFG.replace(
            prim_path="/World/envs/env_.*/Robot_2",
            init_state=G1_43DOF_GR00T_CFG.init_state.replace(pos=(-2.3, 19.008, 0.78)),
        )

    # SONIC deploy 驱动的 G1（29dof，训练同款 PD/armature）。
    # 默认（LOCOMANIP_SONIC_REPLACE_ROBOT1=1）顶替原 Robot_1 站 banyun 工位
    # （推车旁，面向 +Y）。Z=0.76 沿用 SONIC 标定出生高度（脚底离地 ~9mm，
    # 勿抄 43dof 的 0.78，见 SONIC_G1_29DOF_CFG.init_state 注释）。
    # 开关设 0 时退回南侧 8m 行走通道出生点 (-2.0, 11.008)，与镜像双机/推车互不干扰。
    # 默认固定根+关重力（SONIC_G1_PHYSICS_MODE=1 恢复自由根，闭环物理行走）。
    if SONIC_REPLACE_ROBOT1:
        sonic_robot: ArticulationCfg = SONIC_G1_29DOF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/SONICRobot",
            init_state=SONIC_G1_29DOF_CFG.init_state.replace(
                pos=(-3.8, 19.008, 0.76),
                rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
        )
    else:
        sonic_robot: ArticulationCfg = SONIC_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/SONICRobot")
    test_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TestBox",
        # 叠放在 cart_box4 顶面：cart_box4 中心 z=0.90，箱子半高 0.075 → 顶面 z=0.975；
        # test_box 半高 0.12 → 中心 z=1.095。x/y 对齐 cart_box4，rot 沿用推车朝向。
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-5.4, 19.39363, 1.095],
            rot=[0.0, 0.0, 0.0, 1.0],
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
    # 高摩擦地面 μ=1.0/combine=max，对齐 MuJoCo deploy 参考环境（与 SonicSolo/
    # SonicFullscene 一致）。默认 GroundPlaneCfg 是 μ=0.5/average，SONIC 物理
    # 行走时脚底摩擦减半，蹬地/侧移打滑。
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="max",
            ),
        ),
    )

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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Body/root streams are mirrored independently for both robot IDs.
    # 默认（LOCOMANIP_SONIC_REPLACE_ROBOT1=1）Robot_1 被 SONICRobot 顶替，其镜像流
    # 随之下线（工位关节/根驱动改走下方 sonic_wholebody 的 SONIC deploy 链路）；
    # 开关设 0 时恢复本镜像流驱动 Robot_1。
    if not SONIC_REPLACE_ROBOT1:
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
            root_motion_mode="source",
            root_zmq_required=True,
            root_position_mode="relative",
            mirror_hands=False,
            controller_gripper_enabled=False,
        )
    # Robot_2 的镜像流随开关一并下线（实体不在场时挂载即在 asset 解析阶段崩溃）
    if ENABLE_ROBOT2:
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
            root_motion_mode="source",
            root_zmq_required=True,
            root_position_mode="relative",
            mirror_hands=False,
            controller_gripper_enabled=False,
        )
    # gripper 同步只对在场的镜像机器人挂载：robot_1 被 SONIC 顶替、或 robot_2 被
    # LOCOMANIP_ENABLE_ROBOT2=0 下线时，实体缺席，挂上即在 asset 解析阶段崩溃；
    # 在场判定统一走 _robot_asset_present。
    if _robot_asset_present(ISAACLAB_LOCAL_ROBOT_ID):
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
            write_joint_state=True,
        )
    if _robot_asset_present(ISAACLAB_PEER_ROBOT_ID):
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
            write_joint_state=True,
        )
    object_sync = ZmqObjectSyncActionCfg(asset_name="test_box", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    pushcart_sync = ZmqObjectSyncActionCfg(asset_name="pushcart", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    cart_box1_sync = ZmqObjectSyncActionCfg(asset_name="cart_box1", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    cart_box2_sync = ZmqObjectSyncActionCfg(asset_name="cart_box2", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    cart_box3_sync = ZmqObjectSyncActionCfg(asset_name="cart_box3", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)
    cart_box4_sync = ZmqObjectSyncActionCfg(asset_name="cart_box4", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT)

    # ------------------------------------------------------------------
    # SONIC deploy 驱动（sonic_robot）。本类是两个 SONIC 场景的唯一定义来源：
    # SonicSolo/SonicFullscene 从 ActionsCfg 实例摘取这三项，保证环境变量
    # 行为（transport 选择、发布开关、全部调参）三个任务永远一致。
    #
    # Minimal bridge modes:
    #   SONIC_DEPLOY_TRANSPORT=zmq (default): GR00T deploy publishes debug body_q_target over ZMQ.
    #   SONIC_DEPLOY_TRANSPORT=dds: IsaacLab behaves like a virtual G1, subscribing rt/lowcmd
    #                               and publishing rt/lowstate for GR00T deploy.
    # ------------------------------------------------------------------
    if os.environ.get("SONIC_DEPLOY_TRANSPORT", "zmq").lower() == "dds":
        sonic_wholebody = UnitreeDdsLowCmdActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            domain_id=int(os.environ.get("UNITREE_DDS_DOMAIN_ID", "0")),
            network_interface=os.environ.get("UNITREE_DDS_INTERFACE", ""),
            lowcmd_topic=os.environ.get("UNITREE_LOWCMD_TOPIC", "rt/lowcmd"),
            lowstate_topic=os.environ.get("UNITREE_LOWSTATE_TOPIC", "rt/lowstate"),
            secondary_imu_topic=os.environ.get("UNITREE_SECONDARY_IMU_TOPIC", "rt/secondary_imu"),
            target_order="mujoco",
            target_rate_limit_rad_per_step=0.08,
            stabilize_root_pose=_env_flag("SONIC_DEPLOY_STABILIZE_ROOT", SONIC_G1_FIX_ROOT),
            stale_timeout_s=0.5,
            publish_lowstate_every_apply=True,
            mode_machine=int(os.environ.get("UNITREE_G1_MODE_MACHINE", "5")),
            debug_log_interval=50,
        )
    else:
        sonic_wholebody = SonicDeployTargetActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            endpoint=os.environ.get("SONIC_DEPLOY_ENDPOINT", "tcp://127.0.0.1:5557"),
            topic=os.environ.get("SONIC_DEPLOY_TOPIC", "g1_debug"),
            target_field=os.environ.get("SONIC_DEPLOY_TARGET_FIELD", "last_action"),
            target_order="mujoco",
            target_rate_limit_rad_per_step=float(os.environ.get("SONIC_DEPLOY_TARGET_RATE_LIMIT", "0.16")),
            # 物理模式：解锁后旁路 rate limiter——软增益 policy 靠快甩目标偏置生成
            # 扭矩，slew limiter 在平衡环里是 100-250ms 人为迟滞（实测钉死 0.04 摔倒）
            rate_limit_only_while_root_locked=_env_flag(
                "SONIC_DEPLOY_RATE_LIMIT_ONLY_LOCKED", not SONIC_G1_FIX_ROOT
            ),
            # 释放曲线保持原速（5 步翻倍）、不封顶：2026-07-15 闭环 A/B 实测
            # 放缓+封顶把平衡环勒死（摔倒 6→20 次）。放电由下方 drain 排空解决。
            post_unlock_rate_limit_growth_steps=float(
                os.environ.get("SONIC_DEPLOY_POST_UNLOCK_RATE_LIMIT_GROWTH_STEPS", "5")
            ),
            post_unlock_rate_limit_max_delta=float(
                os.environ.get("SONIC_DEPLOY_POST_UNLOCK_RATE_LIMIT_MAX_DELTA", "0")
            ),
            # 解锁前锁根排空目标积压（drain-then-release）。默认关：2026-07-15
            # 闭环 A/B 实测排空无法收敛（policy 锁根开环振荡数 rad），不优于基线。
            unlock_drain_backlog=_env_flag("SONIC_DEPLOY_UNLOCK_DRAIN", False),
            unlock_drain_rate_limit_rad_per_step=float(
                os.environ.get("SONIC_DEPLOY_UNLOCK_DRAIN_RATE_LIMIT", "0.3")
            ),
            unlock_drain_max_steps=int(os.environ.get("SONIC_DEPLOY_UNLOCK_DRAIN_MAX_STEPS", "100")),
            unlock_drain_backlog_epsilon=float(
                os.environ.get("SONIC_DEPLOY_UNLOCK_DRAIN_EPSILON", "0.05")
            ),
            stabilize_root_pose=_env_flag("SONIC_DEPLOY_STABILIZE_ROOT", SONIC_G1_FIX_ROOT),
            lock_root_z=SONIC_G1_FIX_ROOT,  # 物理模式放 Z 自由，让 PhysX settle 到正确地面高度
            startup_settle_steps=0 if SONIC_G1_FIX_ROOT else 50,  # 物理模式先 settle 再跟 deploy target
            # 物理模式 unlock 渐变释放（按物理步计数）。SONIC_DEPLOY_UNLOCK_BLEND_STEPS=0
            # 可做"瞬时交接"实验（最接近 MuJoCo eval 的自由根起始状态）
            unlock_blend_steps=int(
                os.environ.get("SONIC_DEPLOY_UNLOCK_BLEND_STEPS", "0" if SONIC_G1_FIX_ROOT else "50")
            ),
            hold_after_unlock=_env_flag("SONIC_DEPLOY_HOLD_AFTER_UNLOCK", False),  # 诊断：设1则unlock后保持站立不跟deploy
            # 摔倒自动恢复（对齐 MuJoCo base_sim.check_fall：root 高度 <0.2m 即自动
            # 扶正）。SONIC_DEPLOY_AUTO_RECOVER=0 恢复纯手动（J 键）；settle 后是否
            # 自动重新解锁由 SONIC_DEPLOY_AUTO_UNLOCK_AFTER_RECOVER 控制。
            # 对象同步订阅端（对端镜像机）默认关闭：root 高度来自 sonic_robot_sync
            # 同步流，对端机器人摔倒会让本地 recover 状态机 50Hz 空转刷屏（其写入
            # 随即被同步流覆盖，毫无效果）。显式设环境变量仍可强制打开。
            auto_recover_on_fall=_env_flag("SONIC_DEPLOY_AUTO_RECOVER", ZMQ_SYNC_ROLE != "subscriber"),
            fall_root_height_m=float(os.environ.get("SONIC_DEPLOY_FALL_HEIGHT", "0.2")),
            auto_unlock_after_recover=_env_flag("SONIC_DEPLOY_AUTO_UNLOCK_AFTER_RECOVER", True),
            # 恢复位置默认对齐 MuJoCo mj_resetData（回出生点位姿，清 odom/yaw 漂移）；
            # SONIC_DEPLOY_RECOVER_IN_PLACE=1 切回原地扶正（XR 视点连续，07-06 版行为）
            recover_in_place=_env_flag("SONIC_DEPLOY_RECOVER_IN_PLACE", False),
            stale_timeout_s=0.5,
            fallback_to_last_action=True,
            fallback_to_body_q_target=True,
            reference_target_field=os.environ.get("SONIC_DEPLOY_REFERENCE_TARGET_FIELD", "body_q_target"),
            # 物理模式需要 policy 腿部平衡补偿，默认不 blend 掉
            blend_reference_lower_body=_env_flag("SONIC_DEPLOY_BLEND_REFERENCE_LOWER_BODY", SONIC_G1_FIX_ROOT),
            hold_last_reference_target=_env_flag("SONIC_DEPLOY_HOLD_LAST_REFERENCE", True),
            # 物理模式下 deploy base_trans/quat 非物理真实值，默认不跟随
            follow_base_yaw_target=_env_flag("SONIC_DEPLOY_FOLLOW_BASE_YAW", SONIC_G1_FIX_ROOT),
            follow_base_translation_target=_env_flag("SONIC_DEPLOY_FOLLOW_BASE_TRANSLATION", SONIC_G1_FIX_ROOT),
            base_quat_target_field=os.environ.get("SONIC_DEPLOY_BASE_QUAT_FIELD", "base_quat_target"),
            base_trans_target_field=os.environ.get("SONIC_DEPLOY_BASE_TRANS_FIELD", "base_trans_target"),
            base_yaw_rate_limit_rad_per_step=float(os.environ.get("SONIC_DEPLOY_BASE_YAW_RATE_LIMIT", "0.12")),
            base_translation_rate_limit_m_per_step=float(
                os.environ.get("SONIC_DEPLOY_BASE_TRANSLATION_RATE_LIMIT", "0.08")
            ),
            base_translation_scale=float(os.environ.get("SONIC_DEPLOY_BASE_TRANSLATION_SCALE", "2.0")),
            follow_base_height_target=_env_flag("SONIC_DEPLOY_FOLLOW_BASE_HEIGHT", False),
            base_height_rate_limit_m_per_step=float(os.environ.get("SONIC_DEPLOY_BASE_HEIGHT_RATE_LIMIT", "0.05")),
            base_height_scale=float(os.environ.get("SONIC_DEPLOY_BASE_HEIGHT_SCALE", "1.0")),
            keep_feet_on_ground=_env_flag("SONIC_DEPLOY_KEEP_FEET_ON_GROUND", False),
            foot_ground_scale=float(os.environ.get("SONIC_DEPLOY_FOOT_GROUND_SCALE", "0.35")),
            max_squat_drop_m=float(os.environ.get("SONIC_DEPLOY_MAX_SQUAT_DROP", "0.45")),
            # Synthetic base motion 是固定根可视化功能；物理模式 root 由 PhysX 驱动，默认关
            synthetic_base_motion_from_lower_body=_env_flag(
                "SONIC_DEPLOY_SYNTHETIC_BASE_MOTION", not SONIC_G1_PHYSICS_MODE
            ),
            synthetic_base_motion_gain=float(os.environ.get("SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_GAIN", "0.35")),
            synthetic_base_motion_deadzone=float(
                os.environ.get("SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_DEADZONE", "0.002")
            ),
            synthetic_base_motion_max_step_m=float(
                os.environ.get("SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_MAX_STEP", "0.035")
            ),
            debug_log_interval=50,
        )

    # 默认 ZMQ 链路下，关节由 SonicDeployTargetAction 驱动，但不会回传机器人状态。
    # 设 SONIC_PUBLISH_LOWSTATE=1（或 teleop 的 --publish_lowstate）时额外开一路 DDS，
    # 把 sonic_robot 的 sim 状态发到 rt/lowstate，供 GR00T/SONIC deploy 当状态源。
    # DDS 传输模式（SONIC_DEPLOY_TRANSPORT=dds）已由 UnitreeDdsLowCmdAction 发布 lowstate，
    # 因此这里仅在非 dds 模式下挂载，避免重复发布与 DDS 重复初始化。
    if os.environ.get("SONIC_DEPLOY_TRANSPORT", "zmq").lower() != "dds" and _env_flag(
        "SONIC_PUBLISH_LOWSTATE", False
    ):
        sonic_lowstate_pub = UnitreeLowStatePublisherActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            domain_id=int(os.environ.get("UNITREE_DDS_DOMAIN_ID", "0")),
            network_interface=os.environ.get("UNITREE_DDS_INTERFACE", ""),
            lowstate_topic=os.environ.get("UNITREE_LOWSTATE_TOPIC", "rt/lowstate"),
            secondary_imu_topic=os.environ.get("UNITREE_SECONDARY_IMU_TOPIC", "rt/secondary_imu"),
            publish_secondary_imu=_env_flag("SONIC_PUBLISH_SECONDARY_IMU", True),
            target_order="mujoco",
            mode_machine=int(os.environ.get("UNITREE_G1_MODE_MACHINE", "5")),
            debug_log_interval=100,
        )

    # 真实物理闭环桥：IsaacLab 用简单 ZMQ/msgpack 发布 sonic_robot 真实状态，
    # C++ proxy 再用 Unitree C++ SDK 转成 rt/lowstate，避开 Python DDS 与 C++ deploy 不互通的问题。
    if _env_flag("SONIC_PUBLISH_STATE_ZMQ", False):
        sonic_state_pub = SonicRobotStatePublisherActionCfg(
            asset_name="sonic_robot",
            joint_names=list(SONIC_G1_29DOF_JOINT_ORDER),
            bind_endpoint=os.environ.get("SONIC_STATE_ZMQ_BIND", "tcp://127.0.0.1:5560"),
            topic=os.environ.get("SONIC_STATE_ZMQ_TOPIC", "sonic_state"),
            target_order="mujoco",
            mode_machine=int(os.environ.get("UNITREE_G1_MODE_MACHINE", "5")),
            debug_log_interval=100,
        )

    # SONIC 行走跨机同步：让对端（ISAACLAB_LOCAL_ROBOT_ID=2）也能看到 SONIC
    # 机器人行走。ID=1（SONIC 物理行走机）沿对象同步同一 PUB socket 发布
    # sonic_robot 根位姿+关节角（topic=sonic_robot）；ID=2 订阅并运动学跟随。
    # 必须声明在 sonic_wholebody 之后：订阅端收不到 deploy 包只会锁根站立，
    # 本 term 同一步内后写覆盖其根位姿/关节目标（见 ZmqRobotSyncAction 注释）。
    sonic_robot_sync = ZmqRobotSyncActionCfg(
        asset_name="sonic_robot", role=ZMQ_SYNC_ROLE, endpoint=ZMQ_SYNC_ENDPOINT
    )


@configclass
class ObservationsCfg:
    """Empty observation manager config.

    The scene is used for live robot synchronization, not policy rollout or data recording,
    so no ``policy`` observation group is registered.
    """

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)

    # XR teleop 场景使用下陷布局（PackingTable z=-1000.66），
    # 绝对世界系 minimum_height=0.5 会导致 object(z=-100.76) 每步触发复位。
    # teleop 不需要训练式 episode 终止，因此移除 object_dropping。
    # object_dropping = DoneTerm(
    #     func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    # )

    # 默认开关下本机 ID=1 的主角机器人从 robot_1 换成 sonic_robot（29dof 同样有
    # right_wrist_yaw_link）；本机镜像机器人缺席（被 SONIC 顶替或被
    # LOCOMANIP_ENABLE_ROBOT2=0 下线）时同样回退 sonic_robot，避免解析崩溃
    success = DoneTerm(
        func=manip_mdp.task_done_pick_place,
        params={
            "task_link_name": "right_wrist_yaw_link",
            "robot_cfg": SceneEntityCfg(
                ISAACLAB_LOCAL_ROBOT_NAME
                if _robot_asset_present(ISAACLAB_LOCAL_ROBOT_ID)
                else "sonic_robot"
            ),
        },
    )


@configclass
class EventsCfg:
    """事件配置：默认场景复位 + 仓库地砖摩擦补绑。"""

    # 基类默认 events=DefaultEventManagerCfg() 仅含此项；本类顶替默认后必须
    # 自带，否则 env.reset() 不再恢复实体姿态（Articulation.reset() 只清
    # buffer 不写位姿，见 SonicSolo 同名注释）。
    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    # warehouse-simple6_v48.usd 的 SM_floor* 地砖有碰撞体但未绑物理材质
    # （PhysX 默认 μ=0.5/average），与 μ=1.0/max 的 GroundPlane 在 z=0 共面
    # 叠放；SONIC 工位 (-3.8, 19.008) 脚下正是 SM_floor32——接触分配到哪块
    # 摩擦就跟谁，蹬地/侧移随机打滑。prestartup 按名字补绑对齐（配方同
    # SonicFullscene 的 bind_warehouse_floor_friction，那边地板是
    # CollisionPlane/CollisionMesh 两条固定路径，这边是 24 块引用地砖故按名匹配）。
    bind_warehouse_floor_friction = EventTerm(
        func=locomanip_mdp.bind_floor_physics_material,
        mode="prestartup",
        params={
            "prim_path_templates": (),
            "prim_name_regex": r"^SM_floor\d",
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0,
            "friction_combine_mode": "max",
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
    # replicate_physics=False：EventsCfg 的 prestartup 地砖摩擦补绑要写 USD，
    # 框架要求关闭场景复制（teleop 恒 num_envs=1，无复制收益，同 SonicSolo）。
    scene: LocomanipulationG1SceneCfg = LocomanipulationG1SceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

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
        # 每 env 步渲染一次（=50Hz，同 SonicSolo，勿设 >decimation）。原值 2 是
        # 100Hz 渲染目标，XR 下渲染整个 warehouse 直接把 env_hz 拖离 SONIC
        # 闭环要求的墙钟 50Hz 实时线。
        self.sim.render_interval = 4
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

        if not _robot_asset_present(ISAACLAB_LOCAL_ROBOT_ID):
            # 本机镜像机器人缺席（Robot_1 被 SONICRobot 顶替，或 Robot_2 被
            # LOCOMANIP_ENABLE_ROBOT2=0 下线）→ 锚到 SONICRobot。29dof g1.usd 的
            # head_link 嵌套在 torso_link 下（GR00T 43dof 是根下 head_link，抄错会静默失效）
            self.xr.anchor_prim_path = "/World/envs/env_0/SONICRobot/torso_link/head_link"
            self.xr.anchor_rotation_prim_path = "/World/envs/env_0/SONICRobot/pelvis"
        else:
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
                # 勿加 "handtracking" 键：无 retargeter 的 OpenXRDevice.advance()
                # 返回 raw dict，动作维度非零的配置（本机 ID=2，或
                # LOCOMANIP_SONIC_REPLACE_ROBOT1=0 时 local_gripper 在场）下主循环
                # action.repeat() 直接 AttributeError。teleop_se3_agent 对缺失设备
                # 名会自动回退到 motion_controllers（teleop_se3_agent.py:350），
                # XR 锚点同样生效，且 4 维 gripper retargeter 输出与动作空间匹配。
            }
        )
