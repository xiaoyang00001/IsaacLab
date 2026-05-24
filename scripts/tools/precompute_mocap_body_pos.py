"""F4: 用 gear_sonic.Humanoid_Batch 给 mocap PKL 预算 14 body in pelvis frame，落 .npy 缓存。

输入：sample mocap PKL（含 pose_aa (T,30,3) axis-angle + root_trans_offset (T,3)）
输出：<pkl_dir>/<pkl_stem>__body_pos14_pelvis.npy，形状 (T, 14, 3) float32

为什么需要：
- SONIC encoder 的 command_multi_future_nonflat (420D) 是 14 body in pelvis frame
- mocap PKL 不预存 body_pos，需要 forward kinematics
- gear_sonic 自带 torch FK（无 pinocchio 依赖），但顶部 import open3d 用于 mesh I/O
- fk_batch 本身不依赖 open3d → 用 stub 绕过 + subclass 跳过 load_mesh

依赖：
- D:/miniconda3/envs/env_isaaclab/python.exe -m pip install lxml （已装）
- 不装 open3d（节省 500 MB，stub 之）
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import torch


def _stub_open3d():
    """fk_batch 不依赖 open3d，但 torch_humanoid_batch 顶部 import 它。
    我们 subclass 时也会跳过 load_mesh，所以 mesh I/O 不会被实际调用——只需要 import 不报错。
    """
    if "open3d" in sys.modules:
        return
    stub = types.ModuleType("open3d")
    io_stub = types.ModuleType("open3d.io")
    io_stub.read_triangle_mesh = lambda *a, **kw: None
    io_stub.write_triangle_mesh = lambda *a, **kw: None
    stub.io = io_stub
    sys.modules["open3d"] = stub
    sys.modules["open3d.io"] = io_stub


def _build_humanoid_batch(mjcf_path: str):
    """配置 + 实例化 Humanoid_Batch，并 monkey-patch load_mesh 为 no-op。"""
    _stub_open3d()
    from omegaconf import OmegaConf

    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    Humanoid_Batch.load_mesh = lambda self: None

    mjcf_full = Path(mjcf_path).resolve()
    cfg = OmegaConf.create(
        {
            "asset": {
                "assetRoot": str(mjcf_full.parent),
                "assetFileName": mjcf_full.name,
                "urdfFileName": "",
            },
            "extend_config": [],
        }
    )
    return Humanoid_Batch(cfg, device=torch.device("cpu"))


SONIC_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def _resolve_sonic_indices(body_names: list[str]) -> list[int]:
    """SONIC_BODY_NAMES → Humanoid_Batch.body_names 内的索引。"""
    out = []
    for name in SONIC_BODY_NAMES:
        if name not in body_names:
            raise KeyError(f"{name!r} not found in Humanoid_Batch body_names: {body_names}")
        out.append(body_names.index(name))
    return out


def _process_pkl(pkl_path: Path, hb, sonic_idx: list[int]) -> np.ndarray:
    data = joblib.load(pkl_path)
    if len(data) != 1:
        print(f"  [warn] multi-motion PKL, processing all {len(data)} motions sequentially")

    out_per_motion = []
    for name, motion in data.items():
        pose_aa = motion["pose_aa"]  # (T, 30, 3) axis-angle
        root_trans = motion["root_trans_offset"]  # (T, 3)
        T = pose_aa.shape[0]

        pose = torch.from_numpy(pose_aa).float().unsqueeze(0)  # (1, T, 30, 3)
        trans = torch.from_numpy(root_trans).float().unsqueeze(0)  # (1, T, 3)
        fps = int(motion.get("fps", 30))

        with torch.no_grad():
            fk_res = hb.fk_batch(pose, trans, fps=fps, interpolate_data=False)

        # fk_batch return: EasyDict with global_translation / global_rotation / ...
        # 形状 (B, T, num_bodies, 3)
        if hasattr(fk_res, "global_translation"):
            wbody = fk_res.global_translation  # (1, T, N, 3)
        elif hasattr(fk_res, "global_translation_extend"):
            wbody = fk_res.global_translation_extend
        else:
            raise RuntimeError(f"fk_batch return missing global_translation, keys={list(fk_res.keys())}")

        wbody = wbody.squeeze(0).cpu().numpy()  # (T, N, 3)
        body14 = wbody[:, sonic_idx, :]  # (T, 14, 3)
        pelvis = body14[:, 0:1, :]  # (T, 1, 3)
        rel = body14 - pelvis  # (T, 14, 3) in pelvis frame (translation only)

        absmax = float(np.abs(rel).max())
        print(f"  [{name}] T={T} fps={fps} pelvis-frame body_pos absmax={absmax:.4f}")
        out_per_motion.append(rel.astype(np.float32))

    return out_per_motion[0] if len(out_per_motion) == 1 else np.concatenate(out_per_motion, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        default="D:/src/Isaac/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl",
        help="mocap PKL path",
    )
    parser.add_argument(
        "--mjcf",
        default="D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml",
        help="G1 MJCF path",
    )
    parser.add_argument("--out", default=None, help="output .npy (default: same dir as pkl)")
    args = parser.parse_args()

    pkl_path = Path(args.pkl).resolve()
    if not pkl_path.exists():
        sys.exit(f"PKL not found: {pkl_path}")

    print(f"[F4] building Humanoid_Batch from {args.mjcf}")
    hb = _build_humanoid_batch(args.mjcf)
    print(f"  body_names ({len(hb.body_names)}): {hb.body_names[:6]} ...")

    sonic_idx = _resolve_sonic_indices(list(hb.body_names))
    print(f"  SONIC_BODY_NAMES → indices: {sonic_idx}")

    print(f"[F4] loading mocap {pkl_path}")
    body_pos14 = _process_pkl(pkl_path, hb, sonic_idx)

    out_path = Path(args.out) if args.out else pkl_path.with_name(pkl_path.stem + "__body_pos14_pelvis.npy")
    np.save(out_path, body_pos14)
    print(f"[F4] wrote {out_path}  shape={body_pos14.shape}  size={out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
