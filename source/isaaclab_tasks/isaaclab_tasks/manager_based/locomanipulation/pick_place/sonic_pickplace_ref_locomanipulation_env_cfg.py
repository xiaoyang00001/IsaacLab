# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""跟踪 origin/0716-校验 的 pickplace 参考场景（打包桌 + 3 箱），机器人栈沿用本
分支主配置（robot_1 43dof 镜像 / robot_2 / sonic_robot 29dof，靠开关切换）。

背景
----
本分支主配置 ``LocomanipulationG1EnvCfg`` 的场景已演进成"仓库 + 推车 + 纸箱堆"
（warehouse-simple6_v48.usd + pushcart + cart_box*），且 packing_table 被挪到
z=-1000 停用、0716 的三个小箱逻辑被删除。而 origin/0716-校验 分支保留的是一个
干净的打包桌 + 3 箱抓取测试场景。本文件把 0716 的那套**场景几何**原样复现成
一个**新任务 id**，这样在本分支上可以直接用 ``--task`` 切换到 0716 参考布局，
而不必回退主配置或依赖本地已被修改的 pickplace 场景。

机器人 / 控制保持不动
--------------------
robot_1（GR00T 43dof 三指手镜像机器人，可抓取）、robot_2（对端镜像）、
sonic_robot（29dof SONIC 走路 deploy）以及它们的切换开关
（``LOCOMANIP_SONIC_REPLACE_ROBOT1`` / ``LOCOMANIP_ENABLE_ROBOT2``）与主配置
逐字一致——直接引用 ``_main`` 里的 ArticulationCfg 与 ActionsCfg 实例，行为不漂移。
唯一裁掉的是跨机物体/机器人同步项（object_sync/pushcart_sync/cart_box*_sync/
sonic_robot_sync）：它们引用的仓库道具已从本场景移除，且会绑定 15555 端口；本
参考场景是单机，故全部去掉。

坐标换算
--------
0716 场景假设机器人在世界原点、rot=(0.7071,0,0,0.7071) 面朝 +Y，桌子/箱子在其
正前方（+Y）。本分支的操作位机器人（sonic_robot 于 REPLACE_ROBOT1=1，或
robot_1 于 =0）同样在 (-3.8, 19.008)、同样 rot=(0.7071,0,0,0.7071) 面朝 +Y。
朝向一致，故物体**只需按机器人基座 (-3.8, 19.008) 平移，无需旋转**，即可复现
0716 的相对布局。桌面世界高度沿用 0716 的 TABLE_TOP_Z=0.9996。打包桌用自包含
Cuboid 实体桌（顶面对齐 0.9996），不用 0716 的 Nucleus packing_table.usd——后者是
外部资产引用，在部分机器上加载不出来（桌子不可见+箱子掉地），详见 packing_table 处注释。

启动
----
    scripts/start_ubuntu_isaaclab_sonic.sh \
        --task Isaac-PickPlaceRef-Locomanipulation-G1-v0 \
        --device cuda:0 --xr --enable-pinocchio     # 43dof 抓取模式（LOCOMANIP_SONIC_REPLACE_ROBOT1=0）
    # SONIC 走路模式则 --device cpu（默认 REPLACE_ROBOT1=1）
任务 id 含 "Locomanipulation-G1"：teleop_se3_agent 会默认 motion_controllers、
强制 enable_pinocchio、XR 恒 active（与 0716 直接启动的行为一致）。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from copy import deepcopy

from . import locomanipulation_g1_env_cfg as _main
from .configs.action_cfg import G1GripperSyncActionCfg

# ---------------------------------------------------------------------------
# 操作位锚点：0716 机器人基座 (0,0) → 本分支操作位机器人基座 (-3.8, 19.008)。
# sonic_robot（REPLACE_ROBOT1=1）与 robot_1（=0）都出生在这里，朝向 +Y 一致。
# 物体 = 0716 相对坐标 + 该锚点（仅 x/y 平移，z 保持世界高度不变）。
# ---------------------------------------------------------------------------
_ANCHOR_X = -3.8
_ANCHOR_Y = 19.008
# 桌+箱整体相对机器人再往正前方(+Y)推远的距离，避免机器人贴着桌子。桌子和 3 个箱子
# 共用这一个偏移，相对摆放不变（箱子仍在桌面上），只调这一个数即可改远近。
_FWD = 0.20

# ---- 以下常量/工厂函数逐字移植自 origin/0716-校验 的 locomanipulation_g1_env_cfg.py ----
# packing_table.usd 顶面：桌子 spawn 在 z=0 时顶面世界高度 = 0.9996。
TABLE_TOP_Z = 0.9996
SMALL_BOX_HEIGHT = 0.05
LONG_BOX_HEIGHT = 0.10
SMALL_BOX_INITIAL_Z = TABLE_TOP_Z + 0.5 * SMALL_BOX_HEIGHT + 0.002  # 1.0266
LONG_BOX_INITIAL_Z = TABLE_TOP_Z + 0.5 * LONG_BOX_HEIGHT + 0.002    # 1.0516

SMALL_BOX_SIZE = (0.05, 0.05, SMALL_BOX_HEIGHT)
LONG_BOX_SIZE = (0.20, 0.05, LONG_BOX_HEIGHT)  # 长边沿 X


def _box_cfg(
    prim_name: str,
    size: tuple[float, float, float],
    initial_pos: tuple[float, float, float],
    mass: float,
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """程序化可抓取箱子（尺寸单位米），物理参数与 0716 参考场景逐字一致。"""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=initial_pos,
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
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


# 主配置的 ActionsCfg 在类体执行时已按环境变量决定 transport / 发布 term / 镜像
# 机在场性；实例化一份并按名摘取（dataclass 实例化 deepcopy 默认值，互不影响），
# 与 SonicSolo 摘取 sonic_wholebody 的手法相同。
_MAIN_ACTIONS = _main.ActionsCfg()


# ---------------------------------------------------------------------------
# SONIC 机器人换 43-DoF USD（带三指手）：SONIC deploy 照常驱动 29 个身体关节
# （关节名与 g1_43dof.usd 逐字一致，pxr 已核对），多出的 14 个手指关节由 sonic_gripper
# 手柄遥操 → 边走边抓。用户选型 B（physics_mode=1 自由根走路 + 抓取）。
# ⚠️ SONIC 走路策略是在 29-DoF stock g1.usd 上训练的；换 43-DoF GR00T 身体（不同资产、
# 质量/惯量可能不同）的走路稳定性未测，实机验证，必要时调 SONIC_* 参数。
# 仅本任务用；不改 _main.SONIC_G1_29DOF_CFG（SonicSolo/Fullscene 仍用 29dof）。
# ---------------------------------------------------------------------------
SONIC_G1_43DOF_CFG = deepcopy(_main.SONIC_G1_29DOF_CFG)
# 换 43dof USD（沿用 _find_gr00t_g1_43dof_usd 已解析好的绝对路径，与 robot_1 同一文件）
SONIC_G1_43DOF_CFG.spawn.usd_path = _main.G1_43DOF_GR00T_CFG.spawn.usd_path
# 身体沿用 SONIC 训练 PD（deepcopy 已带 legs/feet/waist/waist_yaw/arms）；补三指手执行器组
# （配方同 G1_43DOF_GR00T_CFG.hands，否则 14 个手指关节无执行器覆盖，IsaacLab 会报错）
SONIC_G1_43DOF_CFG.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_hand_index_.*", ".*_hand_middle_.*", ".*_hand_thumb_.*"],
    effort_limit_sim=60.0,
    velocity_limit_sim=20.0,
    stiffness=80.0,
    damping=4.0,
    armature=0.001,
)
# 29 身体关节初始角沿用 SONIC 默认；14 个手指补 0.0（张开）
SONIC_G1_43DOF_CFG.init_state.joint_pos = {
    **SONIC_G1_43DOF_CFG.init_state.joint_pos,
    ".*_hand_.*": 0.0,
}


@configclass
class PickPlaceRefSceneCfg(InteractiveSceneCfg):
    """0716 参考布局：机器人栈（沿用主配置）+ 打包桌 + 3 箱 + 高摩擦地面 + 灯光。"""

    # -------------------- 机器人栈：与 _main.LocomanipulationG1SceneCfg 逐字一致 --------------------
    # banyun 工位默认由 SONICRobot 顶替 Robot_1；LOCOMANIP_SONIC_REPLACE_ROBOT1=0
    # 恢复 43dof 镜像 Robot_1（可抓取）。Robot_2 为对端镜像机，工位右侧 1.5m。
    if not _main.SONIC_REPLACE_ROBOT1:
        robot_1: ArticulationCfg = _main.G1_43DOF_GR00T_CFG.replace(
            prim_path="/World/envs/env_.*/Robot_1",
        )
    if _main.ENABLE_ROBOT2:
        robot_2: ArticulationCfg = _main.G1_43DOF_GR00T_CFG.replace(
            prim_path="/World/envs/env_.*/Robot_2",
            init_state=_main.G1_43DOF_GR00T_CFG.init_state.replace(pos=(-2.3, 19.008, 0.78)),
        )
    # sonic_robot 用 43-DoF USD（带三指手，能抓）——见文件顶部 SONIC_G1_43DOF_CFG。
    if _main.SONIC_REPLACE_ROBOT1:
        sonic_robot: ArticulationCfg = SONIC_G1_43DOF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/SONICRobot",
            init_state=SONIC_G1_43DOF_CFG.init_state.replace(
                pos=(-3.8, 19.008, 0.76),
                rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
        )
    else:
        sonic_robot: ArticulationCfg = SONIC_G1_43DOF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/SONICRobot"
        )

    # -------------------- 0716 参考场景几何（按锚点平移，朝向不变） --------------------
    # 打包桌：自包含 Cuboid 实体桌（顶面对齐 0716 的 TABLE_TOP_Z=0.9996）。
    # ⚠️不用 0716 的 Nucleus packing_table.usd——那是外部资产引用，在部分机器上
    # （Nucleus 本地缓存缺该资产/版本不一致）加载不出来，会导致"桌子不可见 + 箱子
    # 掉地上"（本机能加载但另一台 nolovr-MS-7D99 复现该故障）。改用 kinematic 实体
    # Cuboid，自带碰撞、无外部依赖，与 sonic 分支主场景的 packing_table 同款做法。
    # size Z = TABLE_TOP_Z、中心 z = TABLE_TOP_Z/2 → 台体自地面 z=0 顶到 0.9996。
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[_ANCHOR_X + 0.0, _ANCHOR_Y + 0.55 + _FWD, TABLE_TOP_Z / 2.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, TABLE_TOP_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35), roughness=0.6),
        ),
    )

    # 桌面上两个 5cm 立方 + 一个 20×5×10cm 长箱（位姿取自 0716 参考截图标定）
    small_box_1 = _box_cfg(
        prim_name="SmallBox1",
        size=SMALL_BOX_SIZE,
        initial_pos=(_ANCHOR_X + 0.00553, _ANCHOR_Y + 0.31243 + _FWD, SMALL_BOX_INITIAL_Z),
        mass=0.08,
        color=(0.82, 0.66, 0.36),
    )
    small_box_2 = _box_cfg(
        prim_name="SmallBox2",
        size=SMALL_BOX_SIZE,
        initial_pos=(_ANCHOR_X - 0.10565, _ANCHOR_Y + 0.31397 + _FWD, SMALL_BOX_INITIAL_Z),
        mass=0.08,
        color=(0.88, 0.72, 0.40),
    )
    long_box = _box_cfg(
        prim_name="LongBox",
        size=LONG_BOX_SIZE,
        initial_pos=(_ANCHOR_X - 0.04810, _ANCHOR_Y + 0.41625 + _FWD, LONG_BOX_INITIAL_Z),
        mass=0.25,
        color=(0.76, 0.56, 0.28),
    )

    # -------------------- 地面 + 灯光 --------------------
    # 高摩擦地面 μ=1.0/combine=max：SONIC 走路模式脚底摩擦需求（与主配置/ SonicSolo
    # 一致；0716 用默认 μ=0.5 是因为镜像机器人硬写根位姿不靠脚底摩擦）。
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
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # 方向光制造明暗面（视觉增强，非 0716 原有；可删）
    sun = AssetBaseCfg(
        prim_path="/World/sunLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9238795, 0.3826834, 0.0, 0.0)),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.98, 0.95), intensity=3000.0, angle=0.53),
    )


@configclass
class PickPlaceRefActionsCfg:
    """驱动机器人的 action 项（从主 ActionsCfg 实例摘取），裁掉跨机同步项。

    保留：sonic_wholebody（SONIC deploy）、可选状态发布、43dof 镜像 mirror_1/2、
          local/remote 夹爪同步——全部按主配置的在场性开关（hasattr）条件摘取。
    裁掉：object_sync / pushcart_sync / cart_box1..4_sync / sonic_robot_sync——
          对应仓库道具已从本场景移除，且这些项会绑定 ZMQ 15555（本参考场景单机）。
    """

    # SONIC deploy 驱动 sonic_robot（永远在场）
    sonic_wholebody = _MAIN_ACTIONS.sonic_wholebody

    if hasattr(_MAIN_ACTIONS, "sonic_state_pub"):
        sonic_state_pub = _MAIN_ACTIONS.sonic_state_pub
    if hasattr(_MAIN_ACTIONS, "sonic_lowstate_pub"):
        sonic_lowstate_pub = _MAIN_ACTIONS.sonic_lowstate_pub

    # 43dof 镜像机器人（robot_1 仅 REPLACE_ROBOT1=0 时在场；robot_2 仅 ENABLE_ROBOT2=1）
    if hasattr(_MAIN_ACTIONS, "mujoco_g1_mirror_1"):
        mujoco_g1_mirror_1 = _MAIN_ACTIONS.mujoco_g1_mirror_1
    if hasattr(_MAIN_ACTIONS, "mujoco_g1_mirror_2"):
        mujoco_g1_mirror_2 = _MAIN_ACTIONS.mujoco_g1_mirror_2

    # 夹爪同步（仅对在场的镜像机器人挂载，主配置已按 _robot_asset_present 判定）
    if hasattr(_MAIN_ACTIONS, "local_gripper"):
        local_gripper = _MAIN_ACTIONS.local_gripper
    if hasattr(_MAIN_ACTIONS, "remote_gripper"):
        remote_gripper = _MAIN_ACTIONS.remote_gripper

    # SONIC 机器人(43dof)三指手夹爪：手柄扳机 → motion_controllers 的
    # G1GripperMotionControllerRetargeter(4维) → 本项(mode=local_publish → action_dim=4)
    # → 驱动 sonic_robot 手指。仅 REPLACE_ROBOT1=1(sonic 上工位)时挂载；=0 时操作机是
    # robot_1(自带 local_gripper)，本项不挂避免动作维度冲突。ZMQ 端口 5573(避开 deploy
    # 5557/58、state 5560、robot_2 5567/68、镜像夹爪 5571/72)。夹持角度同 robot_1 proven 配方。
    if _main.SONIC_REPLACE_ROBOT1:
        sonic_gripper = G1GripperSyncActionCfg(
            asset_name="sonic_robot",
            mode="local_publish",
            robot_id=1,
            transport="zmq",
            zmq_host="127.0.0.1",
            zmq_port=5573,
            zmq_topic="sonic_gripper",
            controller_gripper_finger_close_angle=1.8,
            controller_gripper_thumb_1_angle=1.1,
            controller_gripper_thumb_2_angle=1.8,
            controller_gripper_action_alpha=1.0,
            controller_gripper_use_soft_limits=False,
            write_joint_state=True,
        )


@configclass
class PickPlaceRefTerminationsCfg:
    """仅 time_out（teleop_se3_agent 会将其置 None，字段须先存在）。

    不复用主配置的 success（task_done_pick_place 默认引用 SceneEntityCfg("object")，
    而本场景不含名为 object 的资产，会在解析时崩溃）；teleop 镜像/走路场景也不需要
    训练式 episode 终止。
    """

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class PickPlaceRefEventsCfg:
    """R 键 env.reset() 复位场景实体（Articulation.reset() 只清 buffer 不写位姿）。"""

    reset_scene_to_default = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class SonicPickPlaceRefLocomanipulationEnvCfg(_main.LocomanipulationG1EnvCfg):
    """0716 pickplace 参考场景 + 本分支机器人栈。

    继承主 ``LocomanipulationG1EnvCfg`` 的 observations（空）、xr 锚点与
    ``__post_init__``（motion_controllers teleop、200Hz/decimation4、XR 锚点按
    在场机器人自适应），仅替换场景与动作（去掉仓库道具/跨机同步），并把终止/事件
    收敛到单机 teleop 所需的最小集。
    """

    scene: PickPlaceRefSceneCfg = PickPlaceRefSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=False
    )
    actions: PickPlaceRefActionsCfg = PickPlaceRefActionsCfg()
    terminations: PickPlaceRefTerminationsCfg = PickPlaceRefTerminationsCfg()
    events: PickPlaceRefEventsCfg = PickPlaceRefEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # sonic_robot 现在是 43-DoF GR00T USD：head_link 在根级
        # (/World/envs/env_0/SONICRobot/head_link)，不是 29dof g1.usd 的
        # torso_link/head_link（父类按 29dof 设的路径在 43dof 上不存在 → XR 锚点静默失效）。
        # 仅 REPLACE_ROBOT1=1(sonic 上工位)时需修；=0 时操作机是 robot_1，父类已锚到
        # /Robot_1/head_link（43dof 同结构），不动。
        if _main.SONIC_REPLACE_ROBOT1:
            self.xr.anchor_prim_path = "/World/envs/env_0/SONICRobot/head_link"
            self.xr.anchor_rotation_prim_path = "/World/envs/env_0/SONICRobot/pelvis"
