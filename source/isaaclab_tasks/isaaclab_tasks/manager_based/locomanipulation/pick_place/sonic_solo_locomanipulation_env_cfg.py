# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环物理调试专用极简场景。

py-spy 实测（2026-06-10）：完整 locomanipulation 场景 env_hz≈18.5（0.37× 实时），
开销构成 渲染 30% + 4 台机器人 actuator python 写入 24% + PhysX 17% + 杂项。
deploy 按墙钟 50Hz 推进步态相位，sim 必须接近实时行走才有意义。

本场景只保留：sonic_robot + 高摩擦 GroundPlane + 天光 + SONIC deploy/发布 action 项，
裁掉仓库 USD、其余 3 台 G1、传送带、pick-place 全套机构。任务 id 含
"Locomanipulation" 以复用 teleop_se3_agent.py 的 deploy_target_mode 与 U 键回调。

SONIC 相关 action 项直接从主配置 `ActionsCfg` 实例摘取，保证两个场景的
环境变量行为（transport 选择、发布开关、全部调参）永远一致，不会漂移。
"""

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

from . import locomanipulation_g1_env_cfg as _main

# 主配置的 ActionsCfg 在类体执行时已按环境变量决定 transport 与发布 term；
# 实例化一份并摘取 SONIC 相关项（dataclass 实例化时 deepcopy 默认值，互不影响）。
_MAIN_ACTIONS = _main.ActionsCfg()


@configclass
class SonicSoloSceneCfg(InteractiveSceneCfg):
    """极简场景：1 台 SONIC G1 + 地面 + 天光。"""

    sonic_robot: ArticulationCfg = _main.SONIC_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/SONICRobot"
    )

    # 与主场景同款高摩擦地面（μ=1.0 对齐 MuJoCo deploy 参考环境）
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

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class SonicSoloActionsCfg:
    """只保留 SONIC deploy target + 状态发布（与主配置同一来源）。"""

    sonic_wholebody = _MAIN_ACTIONS.sonic_wholebody

    if hasattr(_MAIN_ACTIONS, "sonic_state_pub"):
        sonic_state_pub = _MAIN_ACTIONS.sonic_state_pub
    if hasattr(_MAIN_ACTIONS, "sonic_lowstate_pub"):
        sonic_lowstate_pub = _MAIN_ACTIONS.sonic_lowstate_pub


@configclass
class SonicSoloObservationsCfg:
    """最小观测组（ManagerBasedRLEnv 需要 policy 组存在）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("sonic_robot")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SonicSoloTerminationsCfg:
    # teleop_se3_agent.py 会把 time_out 置 None；字段必须存在
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class SonicSoloLocomanipulationEnvCfg(ManagerBasedRLEnvCfg):
    """SONIC 闭环物理调试极简环境。"""

    scene: SonicSoloSceneCfg = SonicSoloSceneCfg(num_envs=1, env_spacing=8.0)
    observations: SonicSoloObservationsCfg = SonicSoloObservationsCfg()
    actions: SonicSoloActionsCfg = SonicSoloActionsCfg()
    terminations: SonicSoloTerminationsCfg = SonicSoloTerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # 物理步频对齐参考 plant：SONIC 训练 = 200Hz/decimation4，MuJoCo deploy
        # sim2sim = 500Hz（默认 timestep 0.002）。此前 100Hz 的接触脉冲/joint_vel
        # 噪声特性与训练分布不同——慢动作时代被时间稀释掩盖，50Hz 实时下全带宽
        # 进入 policy 观测。控制率不变：4 × 1/200 = 0.02s = 50Hz env。
        # ⚠️ GPU pipeline 下 4 物理子步的同步开销 ≈11ms 会掉出实时；
        # 配合启动参数 --device cpu（单机器人 CPU PhysX ~1-2ms，绰绰有余）。
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 4  # 每 env 步渲染一次（时序均匀），勿设 >decimation

        # XR 第一视角锚点：配方移植自 晓阳全身001 分支（head 锚 + 朝向跟随），
        # 视点刚性绑定机器人头部——注意本 g1.usd 里 head_link 嵌套在 torso_link 下
        # （晓阳的 GR00T 43dof USD 是根下 /Robot/head_link，路径不同）。
        # fixed_anchor_height=False 让高度也跟随头部；FOLLOW_PRIM_SMOOTHED 让房间
        # 朝向平滑跟随机器人转身（yaw-only，见 xr_anchor_utils.py）。
        # ⚠️ 前提：SteamVR 驱动侧滤掉 HMD 平移（NOLO 驱动特性）。若 PICO 走标准
        # 串流不滤平移，佩戴者真实身高会叠加在 head_link 之上导致视点偏高，
        # 届时给 anchor_pos 加负 Z 补偿（真机标定）。
        # 只有真正构造出 OpenXRDevice 时锚点才会生效（见 openxr_device.py
        # __init__），所以必须同时挂一个 "handtracking" teleop device；
        # 启动时还需要 --teleop_device handtracking 才会选中它
        # （start_windows_isaaclab_sonic.ps1 -Xr 已联动追加该参数）。
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(1.0, 0.0, 0.0, 0.0),
            anchor_prim_path="/World/envs/env_0/SONICRobot/torso_link/head_link",
            anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
            fixed_anchor_height=False,
        )
        self.teleop_devices = DevicesCfg(devices={"handtracking": OpenXRDeviceCfg(xr_cfg=self.xr)})
