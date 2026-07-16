#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate a SONIC evaluation matrix and select the best valid candidate.

Selection is deliberately lexicographic rather than a hidden weighted score:

1. valid/reproducible run fraction
2. no-fall run fraction
3. normalized survival time
4. healthy true-free fraction
5. internal reference-to-q following lag
6. healthy-body jitter and XY drift

The output labels a winner ``confirmed`` only when the matrix used a complete
position-balanced design and every expected run for that candidate is valid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import tempfile
from collections import defaultdict
from typing import Any

import numpy as np

try:
    from . import sonic_jitter_report as jitter_report
except ImportError:
    import sonic_jitter_report as jitter_report


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def _get(node: dict, dotted: str, default=None):
    value: Any = node
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _finite(value) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    return None


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size >= 2 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _runtime_valid(report: dict, repo_root: pathlib.Path) -> list[str]:
    reasons: list[str] = []
    meta = report.get("meta", {})
    if meta.get("status") != "ok":
        reasons.append(f"meta_status={meta.get('status', '<missing>')}")
    free = report.get("free", {})
    if not free:
        reasons.append("missing_free_phase")
    if _get(free, "coverage.invalid", True):
        reasons.append("coverage_invalid")

    manifest = meta.get("run_manifest")
    if not isinstance(manifest, dict):
        reasons.append("missing_run_manifest")
    else:
        manifest_root = _get(manifest, "repositories.isaaclab.realpath")
        if manifest_root != str(repo_root):
            reasons.append(f"manifest_worktree_mismatch={manifest_root!r}")

    for key in ("isaaclab_tasks_file", "actions_module_file"):
        raw_path = meta.get(key)
        if not isinstance(raw_path, str):
            reasons.append(f"missing_{key}")
            continue
        try:
            pathlib.Path(raw_path).resolve().relative_to(repo_root)
        except (OSError, ValueError):
            reasons.append(f"{key}_outside_worktree={raw_path}")
    return reasons


def _run_metrics(report: dict) -> dict[str, float | None]:
    free = report["free"]
    seconds = _finite(free.get("seconds")) or 0.0
    fall_events = _finite(_get(free, "fall.event_count")) or 0.0
    survival = _finite(_get(free, "fall.survival_s"))
    survival_fraction = (
        min(max(survival / seconds, 0.0), 1.0)
        if survival is not None and seconds > 0.0
        else None
    )
    lag = _finite(_get(free, "internal_following_lag.reference_to_q.lag_s"))
    lag_corr = _finite(_get(free, "internal_following_lag.reference_to_q.corr"))
    if lag_corr is None or lag_corr < 0.50 or lag is None or lag < 0.0:
        lag = None

    jitter_values = [
        value
        for dotted in (
            "healthy.arms_hf_rms_deg",
            "healthy.waist_hf_rms_deg",
            "healthy.legs_hf_rms_deg",
            "healthy.tilt_hf_rms_deg",
        )
        if (value := _finite(_get(free, dotted))) is not None
    ]
    return {
        "fall_events": fall_events,
        "no_fall": 1.0 if fall_events == 0.0 else 0.0,
        "survival_fraction": survival_fraction,
        "healthy_fraction": _finite(_get(free, "healthy.frac")),
        "reference_to_q_lag_s": lag,
        "reference_to_q_corr": lag_corr,
        "healthy_jitter_mean_deg": float(np.mean(jitter_values)) if jitter_values else None,
        "xy_max_drift_m": _finite(_get(free, "root.xy_max_drift_m")),
        "update_coverage": _finite(_get(free, "coverage.update_coverage")),
        "packet_coverage": _finite(_get(free, "coverage.packet_coverage")),
    }


def _candidate_rank(candidate: dict) -> tuple:
    metrics = candidate["metrics"]

    def higher(name: str, missing: float = -math.inf) -> float:
        summary = metrics.get(name)
        return float(summary["mean"]) if summary else missing

    def lower(name: str, missing: float = math.inf) -> float:
        summary = metrics.get(name)
        return -float(summary["mean"]) if summary else -missing

    return (
        float(candidate["valid_fraction"]),
        higher("no_fall"),
        higher("survival_fraction"),
        higher("healthy_fraction"),
        lower("reference_to_q_lag_s"),
        lower("healthy_jitter_mean_deg"),
        lower("xy_max_drift_m"),
    )


def build_summary(results_path: pathlib.Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    plan_path = pathlib.Path(results["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo_root = pathlib.Path(plan["repo_root"]).resolve()
    expected_per_candidate = int(plan["repeats"])

    runs_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for run in results.get("runs", []):
        runs_by_candidate[str(run.get("candidate"))].append(run)

    candidate_summaries = []
    for definition in plan["candidates"]:
        name = definition["name"]
        run_rows = runs_by_candidate.get(name, [])
        valid_metrics: list[dict] = []
        run_summaries = []
        for row in run_rows:
            reasons = []
            npz_path = row.get("npz")
            if row.get("returncode") != 0:
                reasons.append(f"returncode={row.get('returncode')}")
            if not isinstance(npz_path, str) or not pathlib.Path(npz_path).is_file():
                reasons.append("missing_npz")

            report = None
            if not reasons:
                try:
                    report = jitter_report.load_report(npz_path)
                    reasons.extend(_runtime_valid(report, repo_root))
                except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    reasons.append(f"report_error={exc}")
            metrics = _run_metrics(report) if report is not None and not reasons else None
            if metrics is not None:
                valid_metrics.append(metrics)
            run_summaries.append(
                {
                    "sequence": row.get("sequence"),
                    "block": row.get("block"),
                    "npz": npz_path,
                    "valid": not reasons,
                    "invalid_reasons": reasons,
                    "metrics": metrics,
                }
            )

        metric_names = (
            "fall_events",
            "no_fall",
            "survival_fraction",
            "healthy_fraction",
            "reference_to_q_lag_s",
            "reference_to_q_corr",
            "healthy_jitter_mean_deg",
            "xy_max_drift_m",
            "update_coverage",
            "packet_coverage",
        )
        aggregate = {
            metric: summary
            for metric in metric_names
            if (
                summary := _summary(
                    [
                        value
                        for run_metric in valid_metrics
                        if (value := run_metric.get(metric)) is not None
                    ]
                )
            )
            is not None
        }
        total = len(run_rows)
        valid = len(valid_metrics)
        candidate_summaries.append(
            {
                **definition,
                "expected_runs": expected_per_candidate,
                "observed_runs": total,
                "valid_runs": valid,
                "valid_fraction": valid / max(expected_per_candidate, 1),
                "complete": total == expected_per_candidate and valid == expected_per_candidate,
                "metrics": aggregate,
                "runs": run_summaries,
            }
        )

    eligible = [candidate for candidate in candidate_summaries if candidate["valid_runs"] > 0]
    ranking = sorted(eligible, key=_candidate_rank, reverse=True)
    winner = ranking[0]["name"] if ranking else None
    design_complete = bool(_get(plan, "design.complete_position_balance", False))
    winner_row = ranking[0] if ranking else None
    confidence = "insufficient_valid_data"
    if winner_row is not None:
        confidence = (
            "confirmed"
            if design_complete and winner_row["complete"] and expected_per_candidate >= 2
            else "provisional_screening"
        )

    return {
        "schema_version": 1,
        "matrix_results": str(results_path.resolve()),
        "matrix_plan": str(plan_path.resolve()),
        "scenario": plan["scenario"],
        "selection_rule": [
            "valid run fraction (higher)",
            "no-fall run fraction (higher)",
            "normalized survival time (higher)",
            "healthy true-free fraction (higher)",
            "reference-to-q internal lag with corr>=0.50 (lower)",
            "healthy jitter mean across arms/waist/legs/tilt (lower)",
            "XY max drift (lower)",
        ],
        "winner": winner,
        "confidence": confidence,
        "ranking": [candidate["name"] for candidate in ranking],
        "candidates": candidate_summaries,
    }


def _fmt_metric(candidate: dict, name: str, digits: int = 3) -> str:
    summary = candidate["metrics"].get(name)
    return "n/a" if not summary else f"{summary['mean']:.{digits}f}"


def print_summary(summary: dict) -> None:
    print(
        f"\n=== SONIC matrix summary: {summary['scenario']} ===\n"
        f"winner={summary['winner'] or '<none>'} confidence={summary['confidence']}"
    )
    print(
        f"{'candidate':<20}{'valid':>9}{'no-fall':>10}{'survival':>11}"
        f"{'healthy':>10}{'lag_q(s)':>11}{'jitter':>10}{'xy(m)':>9}"
    )
    by_name = {candidate["name"]: candidate for candidate in summary["candidates"]}
    ordered = summary["ranking"] + [
        name for name in by_name if name not in set(summary["ranking"])
    ]
    for name in ordered:
        candidate = by_name[name]
        print(
            f"{name:<20}"
            f"{candidate['valid_runs']}/{candidate['expected_runs']:>7}"
            f"{_fmt_metric(candidate, 'no_fall'):>10}"
            f"{_fmt_metric(candidate, 'survival_fraction'):>11}"
            f"{_fmt_metric(candidate, 'healthy_fraction'):>10}"
            f"{_fmt_metric(candidate, 'reference_to_q_lag_s'):>11}"
            f"{_fmt_metric(candidate, 'healthy_jitter_mean_deg'):>10}"
            f"{_fmt_metric(candidate, 'xy_max_drift_m'):>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=pathlib.Path, help="matrix_results.json")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    results_path = args.results.resolve()
    output = args.out.resolve() if args.out else results_path.with_name("matrix_summary.json")
    summary = build_summary(results_path)
    _atomic_json(output, summary)
    print_summary(summary)
    print(f"SONIC_EVAL_MATRIX_SUMMARY={output}")
    return 0 if summary["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
