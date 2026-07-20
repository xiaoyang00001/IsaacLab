# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime events for the warehouse locomanipulation scene.

流水线驱动：背景 USD 的 ConveyorBelt_A08 三段是纯视觉件（无 PhysX 表面速度、
无滚轮刚体），物体靠场景里的不可见 kinematic 碰撞板 ``conveyor_collider`` 托住。
因此"流动"不能靠物理带动，只能由本模块按固定周期覆写筐的 root 线速度来模拟。

另含背景刚体的 kinematic 锁定，见 ``lock_background_rigid_bodies``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, UsdPhysics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def lock_background_rigid_bodies(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_names: tuple[str, ...] = (),
    parent_path: str = "Background/ConveyorBelt",
    kinematic: bool = True,
):
    """把背景 USD 里的刚体切成 kinematic，钉在原始摆位上。

    背景里的分拣料箱有两种状态：``blue_sorting_bin_01`` 在 USD 里已是 kinematic
    （实测 z 恒 0.4355 不动），而 ``blue_sorting_bin_02`` 是**动态**刚体——开局
    悬空约 1.5 cm，仿真一起步就下沉、回弹后落在 packing table 面上（z 0.3918 →
    0.3737），y 也漂几毫米，而且之后会被机器人撞飞。这里统一锁成 kinematic。

    只改 ``kinematicEnabled``，不动 collision/visual：料箱仍是可碰撞的实体，只是
    不再受重力与外力驱动。
    """

    if not prim_names:
        return

    stage = get_current_stage()
    if stage is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    for env_id in env_ids.tolist():
        for name in prim_names:
            root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{parent_path}/{name}")
            if not (root and root.IsValid()):
                print(f"[locomanip_event] lock_background: prim 不存在 {name}")
                continue

            locked = 0
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                attr = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr()
                if not attr:
                    attr = UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(kinematic)
                attr.Set(kinematic)
                locked += 1

            if locked == 0:
                print(f"[locomanip_event] lock_background: {name} 下没有刚体")
            else:
                print(f"[locomanip_event] lock_background: {name} 锁定 {locked} 个刚体 kinematic={kinematic}")


def drive_totes_on_conveyor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    object_names: tuple[str, ...] = ("cart2_tote1", "cart2_tote2"),
    velocity_y: float = -0.3,
    enabled: bool = True,
    belt_top_z: float = 0.772,
    z_tolerance: float = 0.15,
    x_range: tuple[float, float] = (-6.17, -5.07),
    y_range: tuple[float, float] = (10.19, 18.22),
    y_stop: float | None = None,
    y_recycle: float = 10.6,
    y_respawn: float = 18.0,
    respawn_z: float = 0.775,
):
    """把塑料筐沿流水线 -Y 方向匀速送走，到工位停住（或到出料端传回入料端）。

    每个 interval tick 只对"确实还躺在滚轮面上"的筐覆写水平速度（Z 速度保留给
    重力/接触，避免把筐按在碰撞板里）。判定用带面几何：

    * ``belt_top_z`` ± ``z_tolerance``：筐原点在底面，静止时 z≈0.775。被机器人拎起
      或掉到地上就超出窗口 → 立即停止驱动，不会把抓在手里的筐硬拖走。
    * ``x_range`` / ``y_range``：滚轮可用带面（略放宽于碰撞板 x[-6.07,-5.17]）。

    ``y_stop`` 是机器人工位：筐一旦流到该 y 就不再驱动，靠 μd=0.6 的动摩擦自然
    停住（约 5 mm 滑行）。这里刻意**不写零速度**——每步硬写零会和机器人的抓取动作
    对抗，而摩擦本身足够大，停得又快又稳。设为 None 则不停、一路流到出料端。

    到达 ``y_recycle`` 后瞬移回 ``y_respawn``（保持各自 X 车道、清零速度）。设了
    ``y_stop`` 时筐停在工位、永远到不了出料端，回收逻辑整段跳过。坐标全部是相对
    env origin 的局部系，与场景配置里的数值同一套。

    性能注意：本函数每个物理步都跑，**必须避免任何 GPU→CPU 同步**。早期版本用
    ``if not on_belt.any(): continue`` 之类做提前返回，每步两个筐最多 6 次同步，
    实测让整个 env.step 从 ~120 ms 涨到 ~160 ms（+30%）。现在一律用 ``torch.where``
    做无分支组装、每个筐只发一次写入。判据里只有 Python 标量（``y_stop is None``）
    才允许走分支。
    """

    if not enabled or abs(velocity_y) < 1e-8:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)

    if len(env_ids) == 0:
        return

    origins = env.scene.env_origins[env_ids]

    for object_name in object_names:
        obj = env.scene[object_name]

        pos_local = obj.data.root_pos_w[env_ids] - origins

        on_belt = (
            (pos_local[:, 2] >= belt_top_z - z_tolerance)
            & (pos_local[:, 2] <= belt_top_z + z_tolerance)
            & (pos_local[:, 0] >= x_range[0])
            & (pos_local[:, 0] <= x_range[1])
            & (pos_local[:, 1] >= y_range[0])
            & (pos_local[:, 1] <= y_range[1])
        )

        if y_stop is None:
            # 纯循环模式：到出料端就传回入料端。y_stop 是 Python 标量，走分支不引入同步。
            recycle = on_belt & (pos_local[:, 1] <= y_recycle)
            pose = obj.data.root_state_w[env_ids, :7].clone()
            pose[:, 1] = torch.where(recycle, origins[:, 1] + y_respawn, pose[:, 1])
            pose[:, 2] = torch.where(recycle, origins[:, 2] + respawn_z, pose[:, 2])
            obj.write_root_pose_to_sim(pose, env_ids=env_ids)
            drive = on_belt & ~recycle
        else:
            # 已到工位的筐不再驱动，交给摩擦停住。
            recycle = None
            drive = on_belt & (pos_local[:, 1] > y_stop)

        vel = obj.data.root_vel_w[env_ids].clone()
        if recycle is not None:
            # 回收的筐清零全部 6 个分量，避免带着旧速度回到入料端。
            keep = (~recycle).unsqueeze(-1)
            vel = vel * keep
        vel[:, 0] = torch.where(drive, torch.zeros_like(vel[:, 0]), vel[:, 0])
        vel[:, 1] = torch.where(drive, torch.full_like(vel[:, 1], velocity_y), vel[:, 1])
        obj.write_root_velocity_to_sim(vel, env_ids=env_ids)
