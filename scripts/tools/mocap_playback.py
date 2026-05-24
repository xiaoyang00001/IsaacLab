"""上游验证：kinematic mocap playback，绕过 SONIC 直接把 mocap dof 写到 sonic_robot 关节。

目的：判断 sonic_robot 摔倒 + 不会 walking 的责任在哪——
  - 如果 kinematic 播放出来是合理 walking 步态 → mocap 数据 / 关节映射 / 坐标系都 OK，
    问题在 SONIC action_term（推理 / 应用方式 / scale / 观测构造）
  - 如果 kinematic 播放也乱（关节穿模、动作怪、不像 walking）→ mocap 数据本身 / 关节顺序 /
    单位 / retarget 有问题，下游 SONIC 怎么调都不会对

复用 sonic_verify 的 env 启动，但不调用 env.step（避免 SONIC action_term 写关节），
直接 sim.step + write_joint_state 覆写 sonic_robot 关节。
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mocap kinematic playback validation.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--task", type=str, default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0", help="Task name."
)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = forever)")
parser.add_argument(
    "--mocap",
    type=str,
    default="D:/src/Isaac/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl",
)
parser.add_argument(
    "--frame_step",
    type=float,
    default=0.6,
    help="mocap frame advance per sim step (default 0.6 = 30fps mocap to 50Hz sim)",
)
parser.add_argument(
    "--loop",
    action="store_true",
    default=True,
    help="loop mocap when reaching the end",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import joblib
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import (
    SONIC_G1_29DOF_JOINT_ORDER,
)
from isaaclab_tasks.utils import parse_env_cfg


def _load_mocap(path: str):
    raw = joblib.load(path)
    name = next(iter(raw.keys()))
    m = raw[name]
    dof = m["dof"]  # (T, 29) — MJCF actuator order，与 SONIC_G1_29DOF_JOINT_ORDER 一致
    root_trans = m["root_trans_offset"]  # (T, 3)
    root_rot_xyzw = m["root_rot"]  # (T, 4)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]  # IsaacLab uses wxyz
    fps = int(m.get("fps", 30))
    print(
        f"[playback] mocap {name!r}: T={dof.shape[0]} fps={fps} "
        f"dof absmax={np.abs(dof).max():.3f} root_trans range={np.ptp(root_trans, axis=0)}"
    )
    return {
        "dof": torch.from_numpy(dof).float(),
        "root_trans": torch.from_numpy(root_trans).float(),
        "root_rot_wxyz": torch.from_numpy(root_rot_wxyz).float(),
        "fps": fps,
        "name": name,
    }


def main():
    print(f"[playback] task={args_cli.task} num_envs={args_cli.num_envs}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print("[playback] env reset done", flush=True)

    unwrapped = env.unwrapped
    sim = unwrapped.sim
    asset = unwrapped.scene["sonic_robot"]
    device = unwrapped.device

    # SONIC_G1_29DOF_JOINT_ORDER → articulation 关节索引
    joint_ids: list[int] = []
    art_joint_names = list(asset.data.joint_names)
    for jn in SONIC_G1_29DOF_JOINT_ORDER:
        if jn not in art_joint_names:
            raise RuntimeError(f"joint {jn} not in articulation; available={art_joint_names[:5]}...")
        joint_ids.append(art_joint_names.index(jn))
    joint_ids_t = torch.tensor(joint_ids, device=device, dtype=torch.long)
    print(f"[playback] resolved {len(joint_ids)}/29 joint indices", flush=True)

    # mocap
    mocap_path = Path(args_cli.mocap)
    if not mocap_path.exists():
        raise FileNotFoundError(f"mocap not found: {mocap_path}")
    mocap = _load_mocap(str(mocap_path))
    n_frames = mocap["dof"].shape[0]
    mocap_dof = mocap["dof"].to(device)  # (T, 29)
    mocap_trans = mocap["root_trans"].to(device)  # (T, 3)
    mocap_rot = mocap["root_rot_wxyz"].to(device)  # (T, 4)
    frame_step = float(args_cli.frame_step)

    # 把 sonic_robot 抬到 mocap 起始 root_trans + offset，便于看
    base_offset = torch.tensor([-2.0, 11.008, 0.0], device=device)  # 原 sonic_robot 位置
    init_z = 0.75

    zero_joint_vel = torch.zeros(args_cli.num_envs, 29, device=device)
    zero_root_vel = torch.zeros(args_cli.num_envs, 6, device=device)
    env_ids = torch.arange(args_cli.num_envs, device=device, dtype=torch.long)

    step = 0
    frame_f = 0.0
    while simulation_app.is_running():
        frame_idx = int(frame_f) % n_frames

        # 关节
        dof_frame = mocap_dof[frame_idx].unsqueeze(0).expand(args_cli.num_envs, -1)
        asset.write_joint_state_to_sim(dof_frame, zero_joint_vel, joint_ids=joint_ids_t, env_ids=env_ids)

        # root pose（让机器人按 mocap 平移 / 旋转，y 平移 + z 抬起）
        root_pos = mocap_trans[frame_idx].unsqueeze(0).expand(args_cli.num_envs, -1).clone()
        root_pos = root_pos + base_offset
        root_pos[:, 2] = root_pos[:, 2] + init_z
        root_quat = mocap_rot[frame_idx].unsqueeze(0).expand(args_cli.num_envs, -1)
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(zero_root_vel, env_ids=env_ids)

        # 推进仿真（render=True 才有视觉）
        sim.step(render=True)
        unwrapped.scene.update(dt=sim.get_physics_dt())

        frame_f += frame_step
        step += 1
        if step % 100 == 0:
            print(
                f"[playback] step={step} frame={frame_idx}/{n_frames} "
                f"dof[0]={mocap_dof[frame_idx, 0]:+.2f} dof[3-knee]={mocap_dof[frame_idx, 3]:+.2f}",
                flush=True,
            )
        if args_cli.max_steps and step >= args_cli.max_steps:
            print(f"[playback] reached max_steps={args_cli.max_steps}, exiting", flush=True)
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
