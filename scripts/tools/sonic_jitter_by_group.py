#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""按身体分组和逐关节下钻 SONIC 闭环抖动。

本脚本与 :mod:`sonic_jitter_report` 使用同一套数据规范化与 healthy 口径：
只分析 true-free、非恢复、非失稳且目标未陈旧的数据。旧版 phase-only NPZ
仍可直接使用；根姿态缺失或短帧不会导致下钻分析崩溃。

用法：

    python3 scripts/tools/sonic_jitter_by_group.py run1.npz [run2.npz ...]
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    from sonic_jitter_report import healthy_mask, load_data, per_joint_hf_rms, phase_masks
except ImportError:  # 作为 scripts.tools 包导入时
    from scripts.tools.sonic_jitter_report import healthy_mask, load_data, per_joint_hf_rms, phase_masks


def group_of(name: str) -> str:
    if any(token in name for token in ("shoulder", "elbow", "wrist")):
        return "手臂"
    if "waist" in name:
        return "腰"
    if any(token in name for token in ("hip", "knee", "ankle")):
        return "腿"
    return "其他"


def _duration(wall_t: np.ndarray, mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask & np.isfinite(wall_t))
    if not indices.size:
        return 0.0
    adjacent = np.diff(wall_t[indices])
    adjacent = adjacent[np.isfinite(adjacent) & (adjacent > 0)]
    dt = float(np.median(adjacent)) if adjacent.size else 0.02
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    total = 0.0
    for segment in np.split(indices, breaks):
        if not segment.size:
            continue
        total += max(float(wall_t[segment[-1]] - wall_t[segment[0]]), 0.0) + dt
    return total


def analyze(path: str) -> dict | None:
    data = load_data(path)
    free = phase_masks(data)["free"]
    healthy = healthy_mask(data, free)
    if not free.any():
        print(f"\n=== {path}: 无自由根实验段，跳过 ===")
        return None
    if healthy.sum() < 12:
        print(
            f"\n=== {path}: 自由根 {_duration(data['wall_t'], free):.2f}s，"
            f"但 healthy 仅 {healthy.sum()} 帧，无法可靠下钻 ==="
        )
        return None

    measured = per_joint_hf_rms(data["q"], data["wall_t"], healthy)
    target = per_joint_hf_rms(data["target"], data["wall_t"], healthy)
    names = data["joint_names"]

    print(
        f"\n=== {path} ===\n"
        f"自由根实验 {_duration(data['wall_t'], free):.2f}s；"
        f"healthy {_duration(data['wall_t'], healthy):.2f}s "
        f"({healthy.sum() / max(free.sum(), 1):.1%})\n"
        "口径：排除 blend/recovery、失稳及其前后 0.5s、陈旧目标。"
    )
    print("组别   实测hf(deg)  目标hf(deg)  过滤比")
    group_metrics = {}
    for group in ("手臂", "腰", "腿"):
        indices = [index for index, name in enumerate(names) if group_of(name) == group]
        if not indices:
            continue
        measured_group = float(np.sqrt(np.nanmean(measured[indices] ** 2)))
        target_group = float(np.sqrt(np.nanmean(target[indices] ** 2)))
        group_metrics[group] = {
            "measured_hf_rms_deg": measured_group,
            "target_hf_rms_deg": target_group,
        }
        print(
            f"{group:<4}   {measured_group:10.3f}  {target_group:10.3f}  "
            f"{target_group / max(measured_group, 1e-9):6.1f}x"
        )

    finite_measured = np.flatnonzero(np.isfinite(measured))
    print("\n实测侧最抖 Top10 关节:")
    for index in finite_measured[np.argsort(measured[finite_measured])[::-1][:10]]:
        print(
            f"  {names[index]:<28} 实测 {measured[index]:6.3f}  "
            f"目标 {target[index]:6.3f}  [{group_of(names[index])}]"
        )

    finite_target = np.flatnonzero(np.isfinite(target))
    print("\n目标侧最抖 Top5 关节:")
    for index in finite_target[np.argsort(target[finite_target])[::-1][:5]]:
        print(
            f"  {names[index]:<28} 目标 {target[index]:6.3f}  "
            f"实测 {measured[index]:6.3f}  [{group_of(names[index])}]"
        )
    return {
        "path": path,
        "free_steps": int(free.sum()),
        "healthy_steps": int(healthy.sum()),
        "healthy_seconds": _duration(data["wall_t"], healthy),
        "groups": group_metrics,
        "measured_joint_hf_rms_deg": measured,
        "target_joint_hf_rms_deg": target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="一个或多个 sonic jitter NPZ")
    args = parser.parse_args()
    for path in args.paths:
        analyze(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
