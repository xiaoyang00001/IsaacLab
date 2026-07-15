#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环抖动指标报告器（纯 numpy，不依赖 Isaac）。

输入 sonic_jitter_verify.py 落盘的 .npz（1 个 = 单跑报告，2 个 = A/B 对比）。

指标设计（50Hz 序列，按锁根/自由根分相、按 arms/waist/legs 分组）：
- hf_rms / hf_p95（deg）：关节位置减 100ms 滑动均值后的高频残差——「肉眼可见的颤抖幅度」。
  同时算目标侧（target_hf）：目标平滑而实测抖 = plant/桥问题；目标本身抖 = 源头问题。
- chatter（flips/s）：带死区（1.5mrad/步）的关节速度换向率——高频往复的「机枪感」。
- track_rms（deg）：实测 vs 目标偏差（软 PD 跟随质量，非抖动本身，用于解释）。
- tilt：均值/最大/高频 RMS（deg）——根部倾角摆动是上身抖的放大器输入。
- step_delta：限速后单步目标变化的 mean/p95/max + 钉死占比（钉死=限速器勒死平衡环，
  病理判据见 KB《SONIC闭环日志分析SOP》§1）。
- 存活：自由相 tilt>45° 或 root_z<0.35 即判摔倒（含时刻）。

用法：
    python3 sonic_jitter_report.py baseline.npz
    python3 sonic_jitter_report.py baseline.npz fixed.npz
"""

import json
import sys

import numpy as np

RAD2DEG = 180.0 / np.pi

GROUPS = {
    "arms": ("_shoulder_", "_elbow_", "_wrist_"),
    "waist": ("waist_",),
    "legs": ("_hip_", "_knee_", "_ankle_"),
}


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """按时间轴（axis=0）的居中滑动均值，边缘用有效窗口。"""
    kernel = np.ones(window) / window
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        conv = np.convolve(x[:, j], kernel, mode="same")
        # 修正边缘的窗口缩短偏置
        norm = np.convolve(np.ones(x.shape[0]), kernel, mode="same")
        out[:, j] = conv / norm
    return out[:, 0] if squeeze else out


def _group_indices(joint_names: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for group, tokens in GROUPS.items():
        idx = [i for i, name in enumerate(joint_names) if any(t in name for t in tokens)]
        result[group] = np.asarray(idx, dtype=np.int64)
    return result


def _series_metrics(q: np.ndarray, target: np.ndarray, idx: np.ndarray) -> dict:
    """单分组的抖动指标。q/target: (T, J)。"""
    if len(idx) == 0 or q.shape[0] < 12:
        return {}
    qg = q[:, idx].astype(np.float64)
    tg = target[:, idx].astype(np.float64)

    hf = qg - _moving_average(qg, 5)  # 100ms 高通残差
    hf_t = tg - _moving_average(tg, 5)

    dq = np.diff(qg, axis=0)
    # 带死区的换向率：|Δq|>1.5mrad 才算有效运动
    active = np.abs(dq) > 1.5e-3
    sign = np.sign(dq) * active
    flips = 0
    total_seconds = (qg.shape[0] - 1) / 50.0
    for j in range(sign.shape[1]):
        s = sign[:, j]
        s = s[s != 0]
        if len(s) > 1:
            flips += int(np.sum(s[1:] * s[:-1] < 0))
    chatter = flips / max(total_seconds, 1e-6) / max(len(idx), 1)

    jerk = np.diff(qg, n=2, axis=0)

    return {
        "hf_rms_deg": float(np.sqrt(np.mean(hf**2)) * RAD2DEG),
        "hf_p95_deg": float(np.percentile(np.abs(hf), 95) * RAD2DEG),
        "target_hf_rms_deg": float(np.sqrt(np.mean(hf_t**2)) * RAD2DEG),
        "chatter_flips_per_s": float(chatter),
        "jerk_rms_mdeg": float(np.sqrt(np.mean(jerk**2)) * RAD2DEG * 1000.0),
        "track_rms_deg": float(np.sqrt(np.mean((qg - tg) ** 2)) * RAD2DEG),
    }


def _phase_report(data: dict, phase: int) -> dict:
    mask = data["phase"] == phase
    if mask.sum() < 12:
        return {}
    q = data["q"][mask]
    target = data["target"][mask]
    tilt = data["tilt_deg"][mask].astype(np.float64)
    root_z = data["root_pos"][mask][:, 2].astype(np.float64)
    step_delta = data["step_delta"][mask].astype(np.float64)
    wall_t = data["wall_t"][mask]

    joint_names = [str(n) for n in data["joint_names"]]
    groups = _group_indices(joint_names)

    report = {"steps": int(mask.sum()), "seconds": float(mask.sum() / 50.0)}
    for group, idx in groups.items():
        metrics = _series_metrics(q, target, idx)
        if metrics:
            report[group] = metrics

    tilt_hf = tilt - _moving_average(tilt, 5)
    report["tilt"] = {
        "mean_deg": float(tilt.mean()),
        "max_deg": float(tilt.max()),
        "hf_rms_deg": float(np.sqrt(np.mean(tilt_hf**2))),
    }
    report["root_z"] = {"min": float(root_z.min()), "max": float(root_z.max())}

    # 限速器钉死占比：与相内最大 step_delta 比对（cap 值随配置漂移，用数据自证）
    sd_max = step_delta.max() if step_delta.size else 0.0
    pinned = float(np.mean(np.abs(step_delta - sd_max) < 1e-4)) if sd_max > 1e-5 else 0.0
    report["step_delta"] = {
        "mean": float(step_delta.mean()),
        "p95": float(np.percentile(step_delta, 95)),
        "max": float(sd_max),
        "pinned_at_max_frac": pinned,
    }

    fallen = (tilt > 45.0) | (root_z < 0.35)
    report["fall"] = {
        "fell": bool(fallen.any()),
        "first_fall_s": float(np.argmax(fallen) / 50.0) if fallen.any() else -1.0,
        "fall_frac": float(fallen.mean()),
    }

    # 健康段隔离：抖动指标与摔倒次数强相关（摔倒/恢复段抬高一切高频量），
    # 剔除 tilt≥20° 或 z≤0.6 的失稳段及其前后 0.5s，才是"正常闭环时抖不抖"。
    # 摔倒运气用 fall/healthy_frac 单独衡量，两者不再互相污染。
    unstable = (tilt >= 20.0) | (root_z <= 0.6)
    padded = np.convolve(unstable.astype(int), np.ones(51), "same") > 0
    healthy = ~padded
    report["healthy"] = {"frac": float(healthy.mean()), "steps": int(healthy.sum())}
    if healthy.sum() >= 100:
        for group, idx in groups.items():
            if len(idx) == 0:
                continue
            qg = q[:, idx].astype(np.float64)
            tgt = target[:, idx].astype(np.float64)
            hf = (qg - _moving_average(qg, 5))[healthy]
            hf_t = (tgt - _moving_average(tgt, 5))[healthy]
            report["healthy"][f"{group}_hf_rms_deg"] = float(np.sqrt(np.mean(hf**2)) * RAD2DEG)
            report["healthy"][f"{group}_target_hf_rms_deg"] = float(np.sqrt(np.mean(hf_t**2)) * RAD2DEG)
        tilt_hf = (tilt - _moving_average(tilt, 5))[healthy]
        report["healthy"]["tilt_hf_rms_deg"] = float(np.sqrt(np.mean(tilt_hf**2)))

    if len(wall_t) > 3:
        dts = np.diff(wall_t)
        hz = 1.0 / np.clip(dts, 1e-6, None)
        report["env_hz"] = {"mean": float(hz.mean()), "p5": float(np.percentile(hz, 5))}
    return report


def load_report(path: str) -> dict:
    raw = np.load(path, allow_pickle=False)
    data = {key: raw[key] for key in raw.files}
    meta = json.loads(str(data["meta"]))
    return {
        "path": path,
        "meta": meta,
        "locked": _phase_report(data, 0),
        "free": _phase_report(data, 1),
    }


def _fmt(value, digits=3):
    if isinstance(value, bool):
        return "YES" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _walk(report: dict, prefix=""):
    for key, value in report.items():
        if isinstance(value, dict):
            yield from _walk(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def print_single(report: dict) -> None:
    print(f"\n=== {report['path']} ===")
    meta = report["meta"]
    print(
        f"env_hz={meta['env_hz_mean']:.1f}(p5 {meta['env_hz_p5']:.1f}) "
        f"rate_limit={meta['target_rate_limit']} blend={meta['unlock_blend_steps']} "
        f"post_unlock_cap={meta['post_unlock_cap']} growth={meta['post_unlock_growth']}"
    )
    for phase in ("locked", "free"):
        section = report[phase]
        if not section:
            print(f"[{phase}] <无数据>")
            continue
        print(f"[{phase}] {section['seconds']:.0f}s")
        for key, value in _walk(section):
            if key in ("steps", "seconds"):
                continue
            print(f"  {key:<38} {_fmt(value)}")


def print_compare(a: dict, b: dict) -> None:
    print(f"\n=== A/B 对比 ===\nA(基线) = {a['path']}\nB(修复) = {b['path']}")
    key_metrics = [
        ("free.healthy.frac", "自由根·健康段占比（高=好）", None),
        ("free.healthy.arms_hf_rms_deg", "健康段·手臂高频抖幅 RMS(deg)", True),
        ("free.healthy.waist_hf_rms_deg", "健康段·腰高频抖幅 RMS(deg)", True),
        ("free.healthy.legs_hf_rms_deg", "健康段·腿高频抖幅 RMS(deg)", True),
        ("free.healthy.tilt_hf_rms_deg", "健康段·倾角高频 RMS(deg)", True),
        ("free.arms.hf_rms_deg", "自由根·手臂高频抖幅 RMS(deg)", True),
        ("free.arms.hf_p95_deg", "自由根·手臂高频抖幅 P95(deg)", True),
        ("free.arms.target_hf_rms_deg", "自由根·目标侧高频(deg)（源头对照）", True),
        ("free.arms.chatter_flips_per_s", "自由根·手臂速度换向率(flips/s)", True),
        ("free.waist.hf_rms_deg", "自由根·腰高频抖幅 RMS(deg)", True),
        ("free.legs.hf_rms_deg", "自由根·腿高频抖幅 RMS(deg)", True),
        ("free.tilt.hf_rms_deg", "自由根·倾角高频 RMS(deg)", True),
        ("free.tilt.max_deg", "自由根·最大倾角(deg)", True),
        ("free.step_delta.max", "自由根·step_delta 峰值(rad)", True),
        ("free.step_delta.pinned_at_max_frac", "自由根·限速钉死占比", None),
        ("free.fall.fell", "自由根·是否摔倒", None),
        ("locked.arms.hf_rms_deg", "锁根·手臂高频抖幅 RMS(deg)", True),
        ("locked.arms.chatter_flips_per_s", "锁根·手臂换向率(flips/s)", True),
    ]

    def get(report, dotted):
        node = report
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    print(f"{'指标':<44}{'A 基线':>12}{'B 修复':>12}{'变化':>12}")
    for dotted, label, lower_better in key_metrics:
        va, vb = get(a, dotted), get(b, dotted)
        if va is None or vb is None:
            continue
        if isinstance(va, bool) or lower_better is None:
            delta = ""
        elif abs(va) > 1e-9:
            pct = (vb - va) / abs(va) * 100.0
            arrow = "↓改善" if (pct < 0) == lower_better and abs(pct) > 5 else (
                "↑恶化" if abs(pct) > 5 else "≈持平")
            delta = f"{pct:+.1f}% {arrow}"
        else:
            delta = ""
        print(f"{label:<44}{_fmt(va):>12}{_fmt(vb):>12}{delta:>14}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    reports = [load_report(p) for p in sys.argv[1:3]]
    for report in reports:
        print_single(report)
    if len(reports) == 2:
        print_compare(reports[0], reports[1])


if __name__ == "__main__":
    main()
