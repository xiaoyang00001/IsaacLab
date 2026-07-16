#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SONIC 闭环评估报告器（纯 NumPy，不依赖 Isaac）。

兼容旧版 ``phase=0/1`` NPZ、新版四态 ``control_state`` NPZ，以及根姿态
可选或比关节流短若干帧的 MuJoCo 录制。新版状态编码为：

* 0: locked
* 1: handover/blend
* 2: true free
* 3: recovery

关键原则：

* 时间、首次跌倒和 chatter 均按 ``wall_t`` 计算，不假定固定 50 Hz。
* 自由根实验的生存统计包含恢复过程；healthy 抖动统计只使用 true-free、
  非恢复且姿态稳定的数据。
* ``reference`` 互相关只表示同一录制内部的参考到目标/实测跟随延迟，
  不是发送端到执行端的端到端网络延迟。
* 覆盖率门禁会标出停流、跳号、陈旧目标和非法目标；无效运行不应进入
  参数优劣结论。

用法：

    python3 sonic_jitter_report.py run.npz
    python3 sonic_jitter_report.py baseline.npz fixed.npz
    python3 sonic_jitter_report.py --a a1.npz a2.npz --b b1.npz b2.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

RAD2DEG = 180.0 / np.pi

STATE_UNKNOWN = -1
STATE_LOCKED = 0
STATE_BLEND = 1
STATE_FREE = 2
STATE_RECOVERY = 3

STATE_NAMES = {
    STATE_UNKNOWN: "unknown",
    STATE_LOCKED: "locked",
    STATE_BLEND: "blend",
    STATE_FREE: "free",
    STATE_RECOVERY: "recovery",
}

GROUPS = {
    "arms": ("_shoulder_", "_elbow_", "_wrist_"),
    "waist": ("waist_",),
    "legs": ("_hip_", "_knee_", "_ankle_"),
}

DEFAULT_GATE = {
    "min_update_coverage": 0.80,
    "min_packet_coverage": 0.98,
    "max_update_gap_s": 0.10,
    "max_target_age_s": 0.10,
    "max_stale_fraction": 0.02,
    "hard_target_stale_s": 0.50,
    "max_invalid_target_fraction": 0.01,
}


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """按时间轴（axis=0）的居中滑动均值，边缘使用实际有效窗口。"""
    x = np.asarray(x, dtype=np.float64)
    window = max(1, min(int(window), max(x.shape[0], 1)))
    kernel = np.ones(window, dtype=np.float64) / window
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    out = np.empty_like(x, dtype=np.float64)
    norm = np.convolve(np.ones(x.shape[0]), kernel, mode="same")
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], kernel, mode="same") / norm
    return out[:, 0] if squeeze else out


def _group_indices(joint_names: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for group, tokens in GROUPS.items():
        idx = [i for i, name in enumerate(joint_names) if any(token in name for token in tokens)]
        result[group] = np.asarray(idx, dtype=np.int64)
    return result


def _safe_meta(raw: dict[str, np.ndarray]) -> dict:
    if "meta" not in raw:
        return {}
    try:
        value = raw["meta"]
        if np.asarray(value).size != 1:
            return {}
        parsed = json.loads(str(np.asarray(value).reshape(())))
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _primary_length(raw: dict[str, np.ndarray]) -> int:
    lengths = []
    for key in ("wall_t", "q", "target"):
        value = raw.get(key)
        if value is not None and np.asarray(value).ndim >= 1:
            lengths.append(len(value))
    if not lengths:
        raise ValueError("NPZ 缺少 wall_t/q/target，无法形成评估时间轴")
    return min(lengths)


def _align_rows(
    value: np.ndarray | None,
    length: int,
    *,
    shape_tail: tuple[int, ...] = (),
    fill=np.nan,
    dtype=np.float64,
) -> np.ndarray:
    """把可选数组安全对齐到主时间轴；短数组补值，长数组截断。"""
    result = np.full((length, *shape_tail), fill, dtype=dtype)
    if value is None:
        return result
    array = np.asarray(value)
    expected_ndim = 1 + len(shape_tail)
    if array.ndim != expected_ndim:
        return result
    if shape_tail and array.shape[1:] != shape_tail:
        return result
    count = min(length, len(array))
    if count:
        try:
            result[:count] = array[:count]
        except (TypeError, ValueError):
            pass
    return result


def _decode_control_state(raw: dict[str, np.ndarray], length: int) -> tuple[np.ndarray, bool]:
    """返回规范四态以及是否确实存在新版 control_state。"""
    if "control_state" not in raw:
        phase = _align_rows(raw.get("phase"), length, fill=-1, dtype=np.int16)
        state = np.full(length, STATE_UNKNOWN, dtype=np.int8)
        state[phase == 0] = STATE_LOCKED
        state[phase == 1] = STATE_FREE
        return state, False

    source = np.asarray(raw["control_state"])
    state = np.full(length, STATE_UNKNOWN, dtype=np.int8)
    count = min(length, len(source)) if source.ndim == 1 else 0
    if count == 0:
        return state, True

    if source.dtype.kind in "iufb":
        numeric = np.asarray(source[:count], dtype=np.float64)
        finite = np.isfinite(numeric)
        rounded = np.full(count, STATE_UNKNOWN, dtype=np.int16)
        rounded[finite] = np.rint(numeric[finite]).astype(np.int16)
        valid = finite & (rounded >= STATE_LOCKED) & (rounded <= STATE_RECOVERY)
        valid_indices = np.flatnonzero(valid)
        state[valid_indices] = rounded[valid_indices].astype(np.int8)
        return state, True

    aliases = {
        "0": STATE_LOCKED,
        "locked": STATE_LOCKED,
        "lock": STATE_LOCKED,
        "1": STATE_BLEND,
        "blend": STATE_BLEND,
        "handover": STATE_BLEND,
        "handover_or_blend": STATE_BLEND,
        "unlock_blend": STATE_BLEND,
        "2": STATE_FREE,
        "free": STATE_FREE,
        "true_free": STATE_FREE,
        "unlocked": STATE_FREE,
        "3": STATE_RECOVERY,
        "recovery": STATE_RECOVERY,
        "recover": STATE_RECOVERY,
    }
    for index, value in enumerate(source[:count]):
        state[index] = aliases.get(str(value).strip().lower(), STATE_UNKNOWN)
    return state, True


def load_data(path: str | Path) -> dict:
    """加载并规范化 NPZ；可选短流用 NaN 补齐，不让报告器因 schema 漂移崩溃。"""
    with np.load(path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}

    length = _primary_length(raw)
    q = np.asarray(raw["q"][:length], dtype=np.float64)
    target = np.asarray(raw["target"][:length], dtype=np.float64)
    if q.ndim != 2 or target.ndim != 2:
        raise ValueError("q/target 必须是二维数组 (T, J)")
    joints = min(q.shape[1], target.shape[1])
    if joints == 0:
        raise ValueError("q/target 没有共同关节列")
    q, target = q[:, :joints], target[:, :joints]

    names_raw = np.asarray(raw.get("joint_names", []))
    joint_names = [str(name) for name in names_raw[:joints]]
    if len(joint_names) < joints:
        joint_names.extend(f"joint_{index}" for index in range(len(joint_names), joints))

    wall_t = _align_rows(raw.get("wall_t"), length, fill=np.nan, dtype=np.float64)
    state, has_control_state = _decode_control_state(raw, length)
    phase = _align_rows(raw.get("phase"), length, fill=-1, dtype=np.int8)

    root_pos = _align_rows(raw.get("root_pos"), length, shape_tail=(3,), fill=np.nan, dtype=np.float64)
    root_quat = _align_rows(raw.get("root_quat"), length, shape_tail=(4,), fill=np.nan, dtype=np.float64)
    tilt_deg = _align_rows(raw.get("tilt_deg"), length, fill=np.nan, dtype=np.float64)

    reference = _align_rows(
        raw.get("reference"), length, shape_tail=(joints,), fill=np.nan, dtype=np.float64
    )
    reference_valid = _align_rows(raw.get("reference_valid"), length, fill=False, dtype=np.bool_)
    if "reference_valid" not in raw:
        reference_valid = np.isfinite(reference).all(axis=1)
    base_valid = _align_rows(raw.get("base_valid"), length, fill=False, dtype=np.bool_)
    fall_known = np.zeros(length, dtype=bool)
    fall_raw = raw.get("fall")
    if "base_valid" in raw:
        fall_known = base_valid.copy()
    elif fall_raw is not None and np.asarray(fall_raw).ndim == 1:
        fall_known[: min(length, len(fall_raw))] = True

    normalized = {
        "path": str(path),
        "meta": _safe_meta(raw),
        "length": length,
        "wall_t": wall_t,
        "q": q,
        "target": target,
        "joint_names": joint_names,
        "phase": phase,
        "control_state": state,
        "has_control_state": has_control_state,
        "step_delta": _align_rows(raw.get("step_delta"), length, fill=np.nan, dtype=np.float64),
        "root_pos": root_pos,
        "root_quat": root_quat,
        "base_valid": base_valid,
        "tilt_deg": tilt_deg,
        "fall": _align_rows(raw.get("fall"), length, fill=False, dtype=np.bool_),
        "fall_known": fall_known,
        "reference": reference,
        "reference_valid": reference_valid,
        "packets": _align_rows(raw.get("packets"), length, fill=-1, dtype=np.int64),
        "packet_count": _align_rows(raw.get("packet_count"), length, fill=-1, dtype=np.int64),
        "valid_target_count": _align_rows(raw.get("valid_target_count"), length, fill=-1, dtype=np.int64),
        "invalid_target_count": _align_rows(raw.get("invalid_target_count"), length, fill=-1, dtype=np.int64),
        "recovery_count": _align_rows(raw.get("recovery_count"), length, fill=-1, dtype=np.int64),
        "target_age_s": _align_rows(raw.get("target_age_s"), length, fill=np.nan, dtype=np.float64),
        "source_index": _align_rows(raw.get("source_index"), length, fill=-1, dtype=np.int64),
        "source_timestamp": _align_rows(raw.get("source_timestamp"), length, fill=np.nan, dtype=np.float64),
    }
    return normalized


def _positive_dts(wall_t: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    valid = np.isfinite(wall_t)
    if mask is not None:
        valid &= mask
    adjacent = valid[1:] & valid[:-1]
    dts = np.diff(wall_t)[adjacent]
    return dts[np.isfinite(dts) & (dts > 0.0)]


def _median_dt(wall_t: np.ndarray, mask: np.ndarray | None = None, fallback: float = 0.02) -> float:
    dts = _positive_dts(wall_t, mask)
    return float(np.median(dts)) if dts.size else fallback


def _contiguous_segments(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    return [part for part in np.split(indices, splits) if part.size]


def _mask_duration(wall_t: np.ndarray, mask: np.ndarray) -> float:
    """按墙钟积分 mask 持续时间；每段末样本使用全局中位周期补齐。"""
    dt = _median_dt(wall_t, mask)
    total = 0.0
    for segment in _contiguous_segments(mask & np.isfinite(wall_t)):
        if len(segment) == 1:
            total += dt
        else:
            elapsed = float(wall_t[segment[-1]] - wall_t[segment[0]])
            total += max(elapsed, 0.0) + dt
    return total


def phase_masks(data: dict) -> dict[str, np.ndarray]:
    """构造实验阶段。

    新 schema 的 ``phase`` 只在 true-free 时为 1，因此不能表示恢复期。对
    ``control_state`` 数据，第一次进入 blend/free/recovery 后直到录制结束都
    属于自由根实验；恢复过程由 control_state 单独标记。
    """
    length = data["length"]
    state = data["control_state"]
    if data["has_control_state"] and np.any(state != STATE_UNKNOWN):
        started = np.flatnonzero(np.isin(state, (STATE_BLEND, STATE_FREE, STATE_RECOVERY)))
        if started.size:
            first = int(started[0])
            locked = np.arange(length) < first
            free = np.arange(length) >= first
        else:
            locked = state == STATE_LOCKED
            free = np.zeros(length, dtype=bool)
        return {"locked": locked, "free": free}

    phase = data["phase"]
    return {"locked": phase == 0, "free": phase == 1}


def _pad_events_by_time(events: np.ndarray, wall_t: np.ndarray, padding_s: float) -> np.ndarray:
    padded = events.copy()
    valid_t = np.isfinite(wall_t)
    for segment in _contiguous_segments(events & valid_t):
        start_t = wall_t[segment[0]] - padding_s
        end_t = wall_t[segment[-1]] + padding_s
        padded |= valid_t & (wall_t >= start_t) & (wall_t <= end_t)
    return padded


def _fall_signal(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """返回 fall 与其已知性；根姿态缺失的帧不会被误判为未摔。"""
    length = data["length"]
    fallen = np.zeros(length, dtype=bool)
    known = np.zeros(length, dtype=bool)
    explicit_known = data["fall_known"]
    fallen |= explicit_known & data["fall"]
    known |= explicit_known
    tilt = data["tilt_deg"]
    valid_tilt = np.isfinite(tilt)
    fallen |= valid_tilt & (tilt > 45.0)
    known |= valid_tilt
    root_z = data["root_pos"][:, 2]
    valid_z = np.isfinite(root_z)
    fallen |= valid_z & (root_z < 0.35)
    known |= valid_z
    return fallen, known


def healthy_mask(data: dict, stage_mask: np.ndarray) -> np.ndarray:
    """true-free、非恢复且远离失稳事件的可比闭环健康段。"""
    state = data["control_state"]
    if data["has_control_state"] and np.any(state != STATE_UNKNOWN):
        eligible = stage_mask & (state == STATE_FREE)
    else:
        eligible = stage_mask & (state != STATE_RECOVERY)

    tilt = data["tilt_deg"]
    root_z = data["root_pos"][:, 2]
    fallen, _ = _fall_signal(data)
    unstable = fallen.copy()
    unstable |= np.isfinite(tilt) & (tilt >= 20.0)
    unstable |= np.isfinite(root_z) & (root_z <= 0.6)
    unstable = _pad_events_by_time(unstable, data["wall_t"], padding_s=0.5)

    age = data["target_age_s"]
    gate = _gate_config(data["meta"])
    stale = ~np.isnan(age) & (age > gate["max_target_age_s"])
    finite_core = np.isfinite(data["wall_t"]) & np.isfinite(data["q"]).all(axis=1)
    finite_core &= np.isfinite(data["target"]).all(axis=1)
    return eligible & ~unstable & ~stale & finite_core


def _highpass_window(wall_t: np.ndarray, mask: np.ndarray, seconds: float = 0.10) -> int:
    window = max(3, int(round(seconds / max(_median_dt(wall_t, mask), 1e-6))))
    return window + 1 if window % 2 == 0 else window


def per_joint_hf_rms(series: np.ndarray, wall_t: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """对 mask 的连续片段分别高通，返回逐关节 RMS（deg）。"""
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError("series 必须为 (T, J)")
    residuals = []
    window = _highpass_window(wall_t, mask)
    for segment in _contiguous_segments(mask):
        if len(segment) < max(3, window):
            continue
        values = series[segment]
        valid_rows = np.isfinite(values).all(axis=1)
        if not valid_rows.all():
            for valid_segment in _contiguous_segments(valid_rows):
                if len(valid_segment) >= max(3, window):
                    sub = values[valid_segment]
                    residuals.append(sub - _moving_average(sub, window))
        else:
            residuals.append(values - _moving_average(values, window))
    if not residuals:
        return np.full(series.shape[1], np.nan)
    residual = np.concatenate(residuals, axis=0)
    return np.sqrt(np.mean(residual**2, axis=0)) * RAD2DEG


def _scalar_hf_rms(series: np.ndarray, wall_t: np.ndarray, mask: np.ndarray) -> float:
    """标量原生单位的高频 RMS（tilt 已是 deg，不能再次 rad→deg）。"""
    values = np.asarray(series, dtype=np.float64)
    residuals = []
    window = _highpass_window(wall_t, mask)
    valid = mask & np.isfinite(values) & np.isfinite(wall_t)
    for segment in _contiguous_segments(valid):
        if len(segment) >= max(3, window):
            segment_values = values[segment]
            residuals.append(segment_values - _moving_average(segment_values, window))
    if not residuals:
        return float("nan")
    residual = np.concatenate(residuals)
    return float(np.sqrt(np.mean(residual**2)))


def _chatter_rate(series: np.ndarray, wall_t: np.ndarray, mask: np.ndarray) -> float:
    flips = 0
    seconds = 0.0
    joints = series.shape[1]
    for segment in _contiguous_segments(mask):
        if len(segment) < 3:
            continue
        values = series[segment]
        finite = np.isfinite(values).all(axis=1) & np.isfinite(wall_t[segment])
        for valid_segment in _contiguous_segments(finite):
            if len(valid_segment) < 3:
                continue
            indices = segment[valid_segment]
            delta = np.diff(series[indices], axis=0)
            sign = np.sign(delta) * (np.abs(delta) > 1.5e-3)
            for column in range(joints):
                active_sign = sign[:, column]
                active_sign = active_sign[active_sign != 0]
                if len(active_sign) > 1:
                    flips += int(np.sum(active_sign[1:] * active_sign[:-1] < 0))
            seconds += max(float(wall_t[indices[-1]] - wall_t[indices[0]]), 0.0)
    return float(flips / max(seconds, 1e-9) / max(joints, 1))


def _series_metrics(
    q: np.ndarray,
    target: np.ndarray,
    idx: np.ndarray,
    wall_t: np.ndarray,
    mask: np.ndarray,
) -> dict:
    if len(idx) == 0 or int(mask.sum()) < 12:
        return {}
    qg = q[:, idx]
    tg = target[:, idx]
    finite = mask & np.isfinite(wall_t)
    finite &= np.isfinite(qg).all(axis=1) & np.isfinite(tg).all(axis=1)
    if finite.sum() < 12:
        return {}

    q_hf = per_joint_hf_rms(qg, wall_t, finite)
    target_hf = per_joint_hf_rms(tg, wall_t, finite)
    q_abs_residual = []
    jerk = []
    window = _highpass_window(wall_t, finite)
    for segment in _contiguous_segments(finite):
        if len(segment) >= max(3, window):
            values = qg[segment]
            q_abs_residual.append(np.abs(values - _moving_average(values, window)))
        if len(segment) >= 3:
            jerk.append(np.diff(qg[segment], n=2, axis=0))

    return {
        "hf_rms_deg": float(np.sqrt(np.nanmean(q_hf**2))),
        "hf_p95_deg": (
            float(np.percentile(np.concatenate(q_abs_residual), 95) * RAD2DEG)
            if q_abs_residual
            else float("nan")
        ),
        "target_hf_rms_deg": float(np.sqrt(np.nanmean(target_hf**2))),
        "chatter_flips_per_s": _chatter_rate(qg, wall_t, finite),
        "jerk_rms_mdeg": (
            float(np.sqrt(np.mean(np.concatenate(jerk) ** 2)) * RAD2DEG * 1000.0)
            if jerk
            else float("nan")
        ),
        "track_rms_deg": float(np.sqrt(np.mean((qg[finite] - tg[finite]) ** 2)) * RAD2DEG),
    }


def _event_count(signal: np.ndarray, mask: np.ndarray) -> int:
    count = 0
    for segment in _contiguous_segments(mask):
        values = signal[segment]
        if values.size:
            count += int(values[0])
            count += int(np.sum(values[1:] & ~values[:-1]))
    return count


def _counter_increments(counter: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    increments = 0
    resets = 0
    for segment in _contiguous_segments(mask & (counter >= 0)):
        if len(segment) < 2:
            continue
        delta = np.diff(counter[segment])
        increments += int(np.clip(delta, 0, None).sum())
        resets += int(np.sum(delta < 0))
    return increments, resets


def _gate_config(meta: dict) -> dict[str, float]:
    gate = DEFAULT_GATE.copy()
    configured = meta.get("target_gate", {})
    if not isinstance(configured, dict):
        configured = {}
    aliases = {
        "min_update_coverage": ("min_update_coverage", "min_valid_coverage"),
        "min_packet_coverage": ("min_packet_coverage",),
        "max_update_gap_s": ("max_update_gap_s", "max_gap_s"),
        "max_target_age_s": ("max_target_age_s", "max_target_stale_s", "target_timeout_s"),
        "max_stale_fraction": ("max_stale_fraction",),
        "hard_target_stale_s": ("hard_target_stale_s",),
        "max_invalid_target_fraction": ("max_invalid_target_fraction",),
    }
    for destination, candidates in aliases.items():
        for key in candidates:
            value = configured.get(key, meta.get(key))
            if isinstance(value, (int, float)) and np.isfinite(value):
                gate[destination] = float(value)
                break
    return gate


def _select_counter(data: dict, keys: tuple[str, ...]) -> tuple[str | None, np.ndarray | None]:
    for key in keys:
        values = data[key]
        if np.sum(values >= 0) >= 2:
            return key, values
    return None, None


def _coverage_report(data: dict, mask: np.ndarray) -> dict:
    wall_t = data["wall_t"]
    gate = _gate_config(data["meta"])
    update_name, update_counter = _select_counter(
        data, ("valid_target_count", "packet_count", "packets")
    )
    source_name, source_counter = _select_counter(data, ("source_index",))
    received_name, received_counter = _select_counter(data, ("packet_count", "packets"))
    result: dict[str, object] = {
        "update_counter": update_name or "missing",
        "packet_counter": source_name or "missing",
        "received_counter": received_name or "missing",
        "gate": gate,
        "invalid": False,
        "invalid_reasons": [],
    }
    reasons: list[str] = result["invalid_reasons"]  # type: ignore[assignment]
    recorded_gate = data["meta"].get("target_gate")
    if isinstance(recorded_gate, dict):
        result["recorded_gate"] = recorded_gate
        if recorded_gate.get("enabled", True) and recorded_gate.get("passed") is False:
            reasons.append("recorder_target_gate_failed")
    status = data["meta"].get("status")
    if isinstance(status, str) and status not in ("", "ok"):
        reasons.append(f"recorder_status_{status}")
    if data["meta"].get("stale_abort") is True:
        reasons.append("recorder_packet_stale_abort")

    valid_wall = mask & np.isfinite(wall_t)
    dts = _positive_dts(wall_t, mask)
    if dts.size == 0 or np.any(np.diff(wall_t[valid_wall]) <= 0.0):
        reasons.append("wall_t_non_monotonic_or_too_short")

    if update_counter is None:
        result.update(
            {
                "update_coverage": float("nan"),
                "valid_coverage": float("nan"),
                "max_update_gap_s": float("nan"),
            }
        )
        reasons.append("missing_update_counter")
    else:
        transition_total = 0
        update_total = 0
        positive_delta_total = 0
        reset_total = 0
        max_gap = 0.0
        for segment in _contiguous_segments(valid_wall & (update_counter >= 0)):
            if len(segment) < 2:
                continue
            delta = np.diff(update_counter[segment])
            transition_total += len(delta)
            update_total += int(np.sum(delta > 0))
            positive_delta_total += int(np.clip(delta, 0, None).sum())
            reset_total += int(np.sum(delta < 0))

            update_positions = np.flatnonzero(delta > 0) + 1
            anchors = np.unique(np.concatenate(([0], update_positions, [len(segment) - 1])))
            if len(anchors) > 1:
                gaps = np.diff(wall_t[segment[anchors]])
                finite_gaps = gaps[np.isfinite(gaps)]
                if finite_gaps.size:
                    max_gap = max(max_gap, float(np.max(finite_gaps)))

        update_coverage = update_total / max(transition_total, 1)
        result.update(
            {
                "update_coverage": float(update_coverage),
                "valid_coverage": float(update_coverage),
                "max_update_gap_s": float(max_gap),
                "update_counter_resets": int(reset_total),
                "updates": int(update_total),
                "valid_counter_increments": int(positive_delta_total),
            }
        )
        if update_coverage < gate["min_update_coverage"]:
            reasons.append("low_update_coverage")
        if max_gap > gate["max_update_gap_s"]:
            reasons.append("update_gap_too_large")
        if reset_total:
            reasons.append("update_counter_reset")

    if source_counter is None:
        result["packet_coverage"] = float("nan")
        reasons.append("missing_source_sequence_counter")
    elif received_counter is None:
        result["packet_coverage"] = float("nan")
        reasons.append("missing_received_packet_counter")
    else:
        received_increments = 0
        expected_increments = 0
        source_resets = 0
        received_resets = 0
        max_packet_jump = 0
        for segment in _contiguous_segments(
            valid_wall & (source_counter >= 0) & (received_counter >= 0)
        ):
            if len(segment) < 2:
                continue
            source_delta = np.diff(source_counter[segment])
            received_delta = np.diff(received_counter[segment])
            positive_source = source_delta[source_delta > 0]
            expected_increments += int(positive_source.sum())
            received_increments += int(np.clip(received_delta, 0, None).sum())
            source_resets += int(np.sum(source_delta < 0))
            received_resets += int(np.sum(received_delta < 0))
            if positive_source.size:
                max_packet_jump = max(max_packet_jump, int(np.max(positive_source)))
        packet_coverage = min(received_increments / max(expected_increments, 1), 1.0)
        result["packet_coverage"] = float(packet_coverage)
        result["source_counter_increments"] = int(expected_increments)
        result["received_counter_increments"] = int(received_increments)
        result["packet_counter_resets"] = int(source_resets)
        result["received_counter_resets"] = int(received_resets)
        result["max_packet_jump"] = int(max_packet_jump)
        if packet_coverage < gate["min_packet_coverage"]:
            reasons.append("low_source_packet_coverage")
        if source_resets:
            reasons.append("source_counter_reset")
        if received_resets:
            reasons.append("received_counter_reset")

    age = data["target_age_s"]
    valid_age = mask & ~np.isnan(age)
    if valid_age.any():
        age_values = age[valid_age]
        finite_age = age_values[np.isfinite(age_values)]
        stale = age_values > gate["max_target_age_s"]
        result["target_age_s"] = {
            "p95": float(np.percentile(finite_age, 95)) if finite_age.size else float("inf"),
            "max": float(np.max(age_values)),
            "stale_frac": float(np.mean(stale)),
        }
        if float(np.mean(stale)) > gate["max_stale_fraction"]:
            reasons.append("stale_target_fraction_too_high")
        if (
            gate["hard_target_stale_s"] > 0.0
            and float(np.max(age_values)) > gate["hard_target_stale_s"]
        ):
            reasons.append("hard_target_stale_limit_exceeded")

    invalid_count = data["invalid_target_count"]
    invalid_updates, invalid_resets = _counter_increments(invalid_count, mask)
    valid_count = data["valid_target_count"]
    valid_updates, _ = _counter_increments(valid_count, mask)
    if np.sum(invalid_count >= 0) >= 2:
        fraction = invalid_updates / max(invalid_updates + valid_updates, 1)
        result["invalid_target_updates"] = int(invalid_updates)
        result["invalid_target_fraction"] = float(fraction)
        result["payload_valid_ratio"] = float(1.0 - fraction)
        if fraction > gate["max_invalid_target_fraction"]:
            reasons.append("invalid_target_fraction_too_high")
        if invalid_resets:
            reasons.append("invalid_target_counter_reset")

    result["invalid"] = bool(reasons)
    return result


def _recovery_count(
    data: dict, mask: np.ndarray, fallen: np.ndarray, fall_known: np.ndarray
) -> tuple[int | None, str]:
    if data["has_control_state"] and np.any(data["control_state"] != STATE_UNKNOWN):
        state_recovery = data["control_state"] == STATE_RECOVERY
        state_count = _event_count(state_recovery, mask)
        counter = data["recovery_count"]
        if np.sum(counter >= 0) >= 2:
            counter_count, _ = _counter_increments(counter, mask)
            return max(state_count, counter_count), "control_state+recovery_count"
        return state_count, "control_state"
    counter = data["recovery_count"]
    if np.sum(counter >= 0) >= 2:
        increments, _ = _counter_increments(counter, mask)
        return increments, "recovery_count"
    if not np.any(mask & fall_known):
        return None, "unavailable"

    # 老数据没有恢复状态，只能把 fall true→false 视作一次推断恢复；末尾仍摔倒不计。
    recoveries = 0
    for segment in _contiguous_segments(mask):
        values = fallen[segment]
        if len(values) > 1:
            recoveries += int(np.sum(~values[1:] & values[:-1]))
    return recoveries, "fall_exit_inferred"


def _root_report(data: dict, mask: np.ndarray) -> dict:
    root = data["root_pos"]
    valid = mask & np.isfinite(root).all(axis=1)
    if not valid.any():
        return {}
    indices = np.flatnonzero(valid)
    xy = root[indices, :2]
    origin = xy[0]
    drift = np.linalg.norm(xy - origin, axis=1)

    path_length = 0.0
    for segment in _contiguous_segments(valid):
        if len(segment) > 1:
            path_length += float(np.linalg.norm(np.diff(root[segment, :2], axis=0), axis=1).sum())
    return {
        "z_min": float(np.min(root[indices, 2])),
        "z_max": float(np.max(root[indices, 2])),
        "xy_max_drift_m": float(np.max(drift)),
        "xy_path_length_m": float(path_length),
        "valid_frac": float(valid.sum() / max(mask.sum(), 1)),
    }


def _state_breakdown(data: dict, mask: np.ndarray) -> dict:
    result = {}
    denominator = max(_mask_duration(data["wall_t"], mask), 1e-12)
    for value, name in STATE_NAMES.items():
        if value == STATE_UNKNOWN and not np.any(mask & (data["control_state"] == value)):
            continue
        state_mask = mask & (data["control_state"] == value)
        if state_mask.any():
            seconds = _mask_duration(data["wall_t"], state_mask)
            result[name] = {
                "steps": int(state_mask.sum()),
                "seconds": float(seconds),
                "frac": float(seconds / denominator),
            }
    return result


def _lagged_pairs(
    reference: np.ndarray,
    signal: np.ndarray,
    mask: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        ref_indices = np.arange(0, len(reference) - lag)
        signal_indices = ref_indices + lag
    else:
        signal_indices = np.arange(0, len(signal) + lag)
        ref_indices = signal_indices - lag
    valid = mask[ref_indices] & mask[signal_indices]
    return reference[ref_indices[valid]], signal[signal_indices[valid]]


def _pair_correlation(reference: np.ndarray, signal: np.ndarray) -> tuple[float, int]:
    if reference.size == 0 or signal.size == 0:
        return float("nan"), 0
    columns = min(reference.shape[1], signal.shape[1])
    correlations = []
    sample_count = 0
    for column in range(columns):
        ref = reference[:, column]
        observed = signal[:, column]
        finite = np.isfinite(ref) & np.isfinite(observed)
        if finite.sum() < 20:
            continue
        ref = ref[finite] - np.mean(ref[finite])
        observed = observed[finite] - np.mean(observed[finite])
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(observed))
        if denominator <= 1e-12:
            continue
        correlations.append(float(np.dot(ref, observed) / denominator))
        sample_count += int(finite.sum())
    if not correlations:
        return float("nan"), 0
    return float(np.mean(correlations)), sample_count


def internal_following_lag(
    reference: np.ndarray,
    signal: np.ndarray,
    wall_t: np.ndarray,
    mask: np.ndarray,
    *,
    max_lag_s: float = 0.50,
) -> dict:
    """互相关估计内部跟随延迟；正 lag 表示 signal 落后 reference。"""
    dt = _median_dt(wall_t, mask)
    max_lag = max(1, int(round(max_lag_s / max(dt, 1e-6))))
    best: tuple[float, int, int] | None = None
    for lag in range(-max_lag, max_lag + 1):
        ref_aligned, signal_aligned = _lagged_pairs(reference, signal, mask, lag)
        correlation, samples = _pair_correlation(ref_aligned, signal_aligned)
        if not np.isfinite(correlation):
            continue
        candidate = (correlation, -abs(lag), lag)
        if best is None or candidate > (best[0], -abs(best[2]), best[2]):
            best = (correlation, samples, lag)
    if best is None:
        return {}
    correlation, samples, lag = best
    return {
        "lag_steps": int(lag),
        "lag_s": float(lag * dt),
        "abs_lag_s": float(abs(lag * dt)),
        "corr": float(correlation),
        "samples": int(samples),
        "meaning": "positive lag means observed signal follows reference; internal recording only",
    }


def _reference_report(data: dict, mask: np.ndarray) -> dict:
    valid = mask & data["reference_valid"]
    valid &= np.isfinite(data["reference"]).all(axis=1)
    if valid.sum() < 30:
        return {}
    target = internal_following_lag(data["reference"], data["target"], data["wall_t"], valid)
    q = internal_following_lag(data["reference"], data["q"], data["wall_t"], valid)
    if not target and not q:
        return {}
    return {
        "scope": "内部跟随延迟，不是端到端网络延迟",
        "reference_to_target": target,
        "reference_to_q": q,
    }


def _phase_report(data: dict, mask: np.ndarray, phase_name: str) -> dict:
    if mask.sum() < 2:
        return {}
    wall_t = data["wall_t"]
    q = data["q"]
    target = data["target"]
    joint_names = data["joint_names"]
    groups = _group_indices(joint_names)

    report: dict[str, object] = {
        "steps": int(mask.sum()),
        "seconds": float(_mask_duration(wall_t, mask)),
        "state": _state_breakdown(data, mask),
        "coverage": _coverage_report(data, mask),
    }
    for group, idx in groups.items():
        metrics = _series_metrics(q, target, idx, wall_t, mask)
        if metrics:
            report[group] = metrics

    tilt = data["tilt_deg"]
    valid_tilt = mask & np.isfinite(tilt)
    if valid_tilt.any():
        tilt_hf = _scalar_hf_rms(tilt, wall_t, valid_tilt)
        report["tilt"] = {
            "mean_deg": float(np.mean(tilt[valid_tilt])),
            "max_deg": float(np.max(tilt[valid_tilt])),
            "hf_rms_deg": float(tilt_hf),
            "valid_frac": float(valid_tilt.sum() / max(mask.sum(), 1)),
        }

    root = _root_report(data, mask)
    if root:
        report["root"] = root
        # 保留旧消费者使用的 root_z 键。
        report["root_z"] = {"min": root["z_min"], "max": root["z_max"]}

    step_delta = data["step_delta"]
    valid_delta = mask & np.isfinite(step_delta)
    if valid_delta.any():
        values = step_delta[valid_delta]
        maximum = float(np.max(values))
        pinned = float(np.mean(np.abs(values - maximum) < 1e-4)) if maximum > 1e-5 else 0.0
        report["step_delta"] = {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": maximum,
            "pinned_at_max_frac": pinned,
        }

    fallen, fall_known = _fall_signal(data)
    fall_mask = mask & fall_known
    recovery_count, recovery_source = _recovery_count(data, mask, fallen, fall_known)
    fall_report: dict[str, object] = {
        "known_frac": float(fall_mask.sum() / max(mask.sum(), 1)),
        "event_count": None,
        "fell": None,
        "first_fall_s": None,
        "survival_s": None,
        "recovery_count": recovery_count,
        "recovery_source": recovery_source,
    }
    if fall_mask.any():
        fallen_known = fallen & fall_mask
        fall_report["event_count"] = _event_count(fallen, fall_mask)
        fall_report["fell"] = bool(fallen_known.any())
        fall_report["fall_frac"] = float(fallen_known.sum() / fall_mask.sum())
        if fallen_known.any():
            stage_indices = np.flatnonzero(mask & np.isfinite(wall_t))
            fall_indices = np.flatnonzero(fallen_known & np.isfinite(wall_t))
            if stage_indices.size and fall_indices.size:
                first_fall_s = float(wall_t[fall_indices[0]] - wall_t[stage_indices[0]])
                fall_report["first_fall_s"] = first_fall_s
                fall_report["survival_s"] = first_fall_s
        else:
            fall_report["first_fall_s"] = -1.0
            fall_report["survival_s"] = float(report["seconds"])
    report["fall"] = fall_report

    healthy = healthy_mask(data, mask)
    healthy_report: dict[str, object] = {
        "frac": float(healthy.sum() / max(mask.sum(), 1)),
        "steps": int(healthy.sum()),
        "seconds": float(_mask_duration(wall_t, healthy)),
        "excludes": "blend/recovery, tilt>=20deg, root_z<=0.6m, fall, +/-0.5s, stale targets",
    }
    if healthy.sum() >= 12:
        for group, idx in groups.items():
            metrics = _series_metrics(q, target, idx, wall_t, healthy)
            if not metrics:
                continue
            healthy_report[f"{group}_hf_rms_deg"] = metrics["hf_rms_deg"]
            healthy_report[f"{group}_target_hf_rms_deg"] = metrics["target_hf_rms_deg"]
            healthy_report[f"{group}_chatter_flips_per_s"] = metrics["chatter_flips_per_s"]
        if np.sum(healthy & np.isfinite(tilt)) >= 12:
            healthy_report["tilt_hf_rms_deg"] = _scalar_hf_rms(
                tilt, wall_t, healthy & np.isfinite(tilt)
            )
    report["healthy"] = healthy_report

    reference_mask = healthy if phase_name == "free" and healthy.sum() >= 30 else mask
    reference_report = _reference_report(data, reference_mask)
    if reference_report:
        report["internal_following_lag"] = reference_report

    dts = _positive_dts(wall_t, mask)
    if dts.size:
        hz = 1.0 / dts
        report["env_hz"] = {
            "mean": float(np.mean(hz)),
            "p5": float(np.percentile(hz, 5)),
            "wall_dt_p95_s": float(np.percentile(dts, 95)),
        }
    return report


def build_report(data: dict) -> dict:
    masks = phase_masks(data)
    return {
        "path": data["path"],
        "meta": data["meta"],
        "locked": _phase_report(data, masks["locked"], "locked"),
        "free": _phase_report(data, masks["free"], "free"),
    }


def load_report(path: str | Path) -> dict:
    return build_report(load_data(path))


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "YES" if value else "no"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def _walk(report: dict, prefix: str = "") -> Iterable[tuple[str, object]]:
    for key, value in report.items():
        if isinstance(value, dict):
            yield from _walk(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _get(report: dict, dotted: str):
    node = report
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def print_single(report: dict) -> None:
    print(f"\n=== {report['path']} ===")
    meta = report["meta"]
    summary = []
    for key, label in (
        ("task", "task"),
        ("target_rate_limit", "rate_limit"),
        ("unlock_blend_steps", "blend_steps"),
        ("step_dt", "nominal_dt"),
    ):
        if key in meta:
            summary.append(f"{label}={_fmt(meta[key])}")
    if summary:
        print(" ".join(summary))
    for phase in ("locked", "free"):
        section = report[phase]
        if not section:
            print(f"[{phase}] <无数据>")
            continue
        invalid = _get(section, "coverage.invalid")
        validity = "INVALID" if invalid else "valid"
        print(f"[{phase}] {section['seconds']:.3f}s ({validity})")
        for key, value in _walk(section):
            if key in ("steps", "seconds") or key.endswith(".meaning") or key.endswith(".scope"):
                continue
            print(f"  {key:<54} {_fmt(value)}")


KEY_METRICS = [
    ("free.coverage.invalid", "自由根·运行无效", None),
    ("free.coverage.update_coverage", "自由根·有效目标更新覆盖率", False),
    ("free.coverage.packet_coverage", "自由根·包序列覆盖率", False),
    ("free.coverage.max_update_gap_s", "自由根·最大更新空窗(s)", True),
    ("free.fall.event_count", "自由根·跌倒事件数", True),
    ("free.fall.recovery_count", "自由根·恢复次数", True),
    ("free.fall.survival_s", "自由根·生存时长(s)", False),
    ("free.healthy.frac", "自由根·健康段占比", False),
    ("free.healthy.arms_hf_rms_deg", "健康段·手臂高频 RMS(deg)", True),
    ("free.healthy.waist_hf_rms_deg", "健康段·腰高频 RMS(deg)", True),
    ("free.healthy.legs_hf_rms_deg", "健康段·腿高频 RMS(deg)", True),
    ("free.healthy.tilt_hf_rms_deg", "健康段·倾角高频 RMS(deg)", True),
    ("free.root.xy_max_drift_m", "自由根·XY 最大漂移(m)", True),
    ("free.root.xy_path_length_m", "自由根·XY 路径长度(m)", True),
    (
        "free.internal_following_lag.reference_to_target.abs_lag_s",
        "内部 reference→target |lag|(s)",
        True,
    ),
    (
        "free.internal_following_lag.reference_to_target.corr",
        "内部 reference→target corr",
        False,
    ),
    (
        "free.internal_following_lag.reference_to_q.abs_lag_s",
        "内部 reference→q |lag|(s)",
        True,
    ),
    ("free.internal_following_lag.reference_to_q.corr", "内部 reference→q corr", False),
    ("free.arms.hf_rms_deg", "自由根·手臂高频 RMS(deg)", True),
    ("free.arms.chatter_flips_per_s", "自由根·手臂换向率(flips/s)", True),
    ("locked.arms.hf_rms_deg", "锁根·手臂高频 RMS(deg)", True),
]


def print_compare(a: dict, b: dict) -> None:
    """保留传统单 A/单 B 对比输出。"""
    print(f"\n=== A/B 对比 ===\nA(基线) = {a['path']}\nB(候选) = {b['path']}")
    comparison_valid = _run_is_valid(a) and _run_is_valid(b)
    if not comparison_valid:
        print(
            "WARNING: 至少一轮被数据门禁判为 INVALID；"
            "仅并列显示数值，不给改善/恶化结论。"
        )
    print(f"{'指标':<43}{'A 基线':>12}{'B 候选':>12}{'变化':>17}")
    for dotted, label, lower_better in KEY_METRICS:
        va, vb = _get(a, dotted), _get(b, dotted)
        if va is None or vb is None:
            continue
        delta = ""
        if (
            comparison_valid
            and lower_better is not None
            and not isinstance(va, (bool, np.bool_))
            and np.isfinite(float(va))
            and np.isfinite(float(vb))
            and abs(float(va)) > 1e-9
        ):
            pct = (float(vb) - float(va)) / abs(float(va)) * 100.0
            if abs(pct) <= 5:
                judgement = "≈持平"
            elif (pct < 0) == lower_better:
                judgement = "改善"
            else:
                judgement = "恶化"
            delta = f"{pct:+.1f}% {judgement}"
        print(f"{label:<43}{_fmt(va):>12}{_fmt(vb):>12}{delta:>17}")


def _numeric_metric(report: dict, dotted: str) -> float | None:
    value = _get(report, dotted)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    return None


def _run_is_valid(report: dict) -> bool:
    return _get(report, "free.coverage.invalid") is not True


_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def t_critical_95(sample_count: int) -> float:
    """Two-sided 95% Student-t critical value for a sample mean."""
    degrees_of_freedom = max(int(sample_count) - 1, 1)
    return _T_CRITICAL_975.get(degrees_of_freedom, 1.96)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) >= 2:
        sd = float(np.std(array, ddof=1))
        half = t_critical_95(len(array)) * sd / np.sqrt(len(array))
    else:
        sd = 0.0
        half = 0.0
    return {
        "n": len(array),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def aggregate_reports(reports: list[dict]) -> dict[str, dict]:
    aggregate = {}
    for dotted, _, _ in KEY_METRICS:
        candidates = reports if dotted == "free.coverage.invalid" else [
            report for report in reports if _run_is_valid(report)
        ]
        values = [
            value
            for report in candidates
            if (value := _numeric_metric(report, dotted)) is not None
        ]
        if values:
            aggregate[dotted] = _summary(values)
    return aggregate


def paired_effects(a_reports: list[dict], b_reports: list[dict]) -> dict[str, dict]:
    """按命令行顺序配对，输出 B-A、95% CI 和配对标准化效应 dz。"""
    if len(a_reports) != len(b_reports):
        return {}
    effects = {}
    for dotted, _, _ in KEY_METRICS:
        pairs = []
        for a_report, b_report in zip(a_reports, b_reports):
            if dotted != "free.coverage.invalid" and not (
                _run_is_valid(a_report) and _run_is_valid(b_report)
            ):
                continue
            a_value = _numeric_metric(a_report, dotted)
            b_value = _numeric_metric(b_report, dotted)
            if a_value is not None and b_value is not None:
                pairs.append(b_value - a_value)
        if not pairs:
            continue
        summary = _summary(pairs)
        # n<3 的标准化效应极不稳定（两点几乎等差会产生荒谬的大 dz）。
        if len(pairs) >= 3 and summary["sd"] > 1e-12:
            summary["paired_dz"] = summary["mean"] / summary["sd"]
        else:
            summary["paired_dz"] = float("nan")
        effects[dotted] = summary
    return effects


def print_aggregate(a_reports: list[dict], b_reports: list[dict]) -> None:
    a_aggregate = aggregate_reports(a_reports)
    b_aggregate = aggregate_reports(b_reports)
    paired = paired_effects(a_reports, b_reports)
    pairing = "paired" if len(a_reports) == len(b_reports) else "unpaired (仅组均值)"
    a_valid = sum(_run_is_valid(report) for report in a_reports)
    b_valid = sum(_run_is_valid(report) for report in b_reports)
    print(
        f"\n=== 运行级聚合 A(valid/total={a_valid}/{len(a_reports)}) / "
        f"B(valid/total={b_valid}/{len(b_reports)}) [{pairing}] ==="
    )
    print(f"{'指标':<38}  {'A mean±SD':>23}  {'B mean±SD':>23}  {'B-A [95%CI], dz':>31}")
    for dotted, label, _ in KEY_METRICS:
        a = a_aggregate.get(dotted)
        b = b_aggregate.get(dotted)
        if not a or not b:
            continue
        effect = paired.get(dotted)
        effect_text = ""
        if effect:
            effect_text = (
                f"{effect['mean']:+.3f} "
                f"[{effect['ci95_low']:+.3f},{effect['ci95_high']:+.3f}] "
                f"dz={_fmt(effect['paired_dz'], 2)} n={effect['n']}"
            )
        a_text = f"{a['mean']:.3f}±{a['sd']:.3f}(n={a['n']})"
        b_text = f"{b['mean']:.3f}±{b['sd']:.3f}(n={b['n']})"
        print(f"{label:<38}  {a_text:>23}  {b_text:>23}  {effect_text:>31}")
    print(
        "注：除“运行无效”外，INVALID 运行自动排除；95% CI 为运行级均值的"
        "正态近似；配对按 --a/--b 文件顺序。"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="兼容模式：1 个文件单报告，2 个文件单次 A/B")
    parser.add_argument("--a", nargs="+", metavar="NPZ", help="A 组运行文件，可给多个")
    parser.add_argument("--b", nargs="+", metavar="NPZ", help="B 组运行文件，可给多个")
    args = parser.parse_args()
    if args.a is not None or args.b is not None:
        if not args.a or not args.b or args.paths:
            parser.error("--a 与 --b 必须同时提供，且不能再混用位置参数")
    elif len(args.paths) not in (1, 2):
        parser.error("位置参数只兼容 1/2 个文件；多运行请使用 --a ... --b ...")
    return args


def main() -> None:
    args = _parse_args()
    if args.a is not None:
        a_reports = [load_report(path) for path in args.a]
        b_reports = [load_report(path) for path in args.b]
        for report in [*a_reports, *b_reports]:
            print_single(report)
        print_aggregate(a_reports, b_reports)
        return

    reports = [load_report(path) for path in args.paths]
    for report in reports:
        print_single(report)
    if len(reports) == 2:
        print_compare(reports[0], reports[1])


if __name__ == "__main__":
    main()
