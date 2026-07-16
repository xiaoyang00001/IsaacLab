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
5. healthy-body jitter and XY drift
6. internal reference-to-q following lag
7. target receive age (diagnostic tie-breaker after physical outcomes)

The output labels a winner ``confirmed`` only when the observed plan/results
really form a complete position-balanced design, every candidate has complete
valid blocks, and the winner beats every alternative with a one-sided exact
sign-test p-value <= 0.05.
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
    sd = float(np.std(array, ddof=1)) if array.size >= 2 else 0.0
    half = (
        jitter_report.t_critical_95(int(array.size)) * sd / math.sqrt(float(array.size))
        if array.size >= 2
        else 0.0
    )
    mean = float(np.mean(array))
    return {
        "n": int(array.size),
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _runtime_valid(
    report: dict, repo_root: pathlib.Path, *, expected_free_seconds: float
) -> list[str]:
    reasons: list[str] = []
    meta = report.get("meta", {})
    schema_version = _finite(meta.get("schema_version"))
    if schema_version is None or schema_version < 2:
        reasons.append(f"unsupported_schema_version={meta.get('schema_version', '<missing>')}")
    if meta.get("status") != "ok":
        reasons.append(f"meta_status={meta.get('status', '<missing>')}")
    if meta.get("unlocked") is not True:
        reasons.append("runner_did_not_unlock")
    free = report.get("free", {})
    if not free:
        reasons.append("missing_free_phase")
    if _get(free, "coverage.invalid", True):
        reasons.append("coverage_invalid")
    free_seconds = _finite(free.get("seconds"))
    step_dt = _finite(meta.get("step_dt")) or 0.02
    duration_tolerance = max(0.10, 3.0 * step_dt)
    if free_seconds is None or free_seconds + duration_tolerance < expected_free_seconds:
        reasons.append(
            f"incomplete_free_duration={free_seconds!r}<{expected_free_seconds:.3f}"
        )

    fall_known_fraction = _finite(_get(free, "fall.known_frac"))
    if fall_known_fraction is None or fall_known_fraction < 0.98:
        reasons.append(f"insufficient_fall_observability={fall_known_fraction!r}")
    if _finite(_get(free, "fall.event_count")) is None:
        reasons.append("missing_fall_event_count")
    if _finite(_get(free, "fall.survival_s")) is None:
        reasons.append("missing_survival_time")
    recovery_count = _finite(_get(free, "fall.recovery_count"))
    if recovery_count is None:
        reasons.append("missing_recovery_count")
    elif recovery_count != 0.0:
        reasons.append(f"recovery_contamination={recovery_count:g}")
    if _finite(_get(free, "coverage.target_age_s.p95")) is None:
        reasons.append("missing_target_age")

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
    seconds = _finite(free.get("seconds"))
    fall_events = _finite(_get(free, "fall.event_count"))
    survival = _finite(_get(free, "fall.survival_s"))
    survival_fraction = (
        min(max(survival / seconds, 0.0), 1.0)
        if survival is not None and seconds is not None and seconds > 0.0
        else None
    )
    signed_lag = _finite(_get(free, "internal_following_lag.reference_to_q.lag_s"))
    lag = _finite(_get(free, "internal_following_lag.reference_to_q.abs_lag_s"))
    lag_corr = _finite(_get(free, "internal_following_lag.reference_to_q.corr"))
    if lag_corr is None or lag_corr < 0.50 or lag is None:
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
        "no_fall": None if fall_events is None else 1.0 if fall_events == 0.0 else 0.0,
        "survival_fraction": survival_fraction,
        "healthy_fraction": _finite(_get(free, "healthy.frac")),
        "reference_to_q_abs_lag_s": lag,
        "reference_to_q_signed_lag_s": signed_lag,
        "reference_to_q_corr": lag_corr,
        "target_age_p95_s": _finite(_get(free, "coverage.target_age_s.p95")),
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
        lower("healthy_jitter_mean_deg"),
        lower("xy_max_drift_m"),
        lower("reference_to_q_abs_lag_s"),
        lower("target_age_p95_s"),
    )


def _single_run_rank(metrics: dict) -> tuple:
    def higher(name: str, missing: float = -math.inf) -> float:
        value = metrics.get(name)
        return float(value) if value is not None else missing

    def lower(name: str, missing: float = math.inf) -> float:
        value = metrics.get(name)
        return -float(value) if value is not None else -missing

    return (
        higher("no_fall"),
        higher("survival_fraction"),
        higher("healthy_fraction"),
        lower("healthy_jitter_mean_deg"),
        lower("xy_max_drift_m"),
        lower("reference_to_q_abs_lag_s"),
        lower("target_age_p95_s"),
    )


def _exact_sign_p_one_sided(winner_better: int, paired_blocks: int) -> float:
    """P(X >= winner_better), X~Binomial(p=.5), conservatively counting ties as non-wins."""
    if paired_blocks <= 0:
        return 1.0
    return float(
        sum(math.comb(paired_blocks, count) for count in range(winner_better, paired_blocks + 1))
        / (2**paired_blocks)
    )


def _validate_design(plan: dict, results: dict) -> dict:
    """Derive balance/completeness from observed rows instead of trusting plan flags."""
    reasons: list[str] = []
    definitions = plan.get("candidates", [])
    names = [
        definition.get("name")
        for definition in definitions
        if isinstance(definition, dict) and isinstance(definition.get("name"), str)
    ]
    if len(names) != len(definitions) or len(set(names)) != len(names) or not names:
        reasons.append("invalid_or_duplicate_candidate_definitions")
    try:
        repeats = int(plan.get("repeats"))
    except (TypeError, ValueError):
        repeats = 0
    if repeats < 1:
        reasons.append("invalid_repeats")

    order = plan.get("order")
    expected_total = repeats * len(names)
    expected_by_sequence: dict[int, tuple[int, str]] = {}
    position_counts = {name: [0] * len(names) for name in names}
    if not isinstance(order, list) or len(order) != expected_total:
        reasons.append("plan_order_length_mismatch")
    else:
        parsed_order: list[tuple[int, int, str]] = []
        for row in order:
            try:
                sequence = int(row["sequence"])
                block = int(row["block"])
                candidate = str(row["candidate"])
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_plan_order_row")
                continue
            parsed_order.append((sequence, block, candidate))
        parsed_order.sort()
        if [sequence for sequence, _, _ in parsed_order] != list(range(1, expected_total + 1)):
            reasons.append("plan_sequence_not_contiguous")
        for sequence, block, candidate in parsed_order:
            if sequence in expected_by_sequence:
                reasons.append("duplicate_plan_sequence")
            expected_by_sequence[sequence] = (block, candidate)
        for block in range(1, repeats + 1):
            block_rows = [
                (sequence, candidate)
                for sequence, row_block, candidate in parsed_order
                if row_block == block
            ]
            block_rows.sort()
            if len(block_rows) != len(names) or {name for _, name in block_rows} != set(names):
                reasons.append(f"block_{block}_candidate_set_invalid")
                continue
            for position, (_, candidate) in enumerate(block_rows):
                position_counts[candidate][position] += 1

    position_balance = False
    if names and repeats > 0 and repeats % len(names) == 0 and not any(
        reason.startswith("block_") or reason.startswith("plan_") or reason.startswith("invalid_plan")
        for reason in reasons
    ):
        expected_position_count = repeats // len(names)
        position_balance = all(
            counts == [expected_position_count] * len(names)
            for counts in position_counts.values()
        )
        if not position_balance:
            reasons.append("observed_plan_not_position_balanced")

    result_rows = results.get("runs", [])
    seen_sequences: set[int] = set()
    if not isinstance(result_rows, list) or len(result_rows) != expected_total:
        reasons.append("result_row_count_mismatch")
    if isinstance(result_rows, list):
        for row in result_rows:
            try:
                sequence = int(row["sequence"])
                block = int(row["block"])
                candidate = str(row["candidate"])
            except (KeyError, TypeError, ValueError):
                reasons.append("invalid_result_row")
                continue
            if sequence in seen_sequences:
                reasons.append("duplicate_result_sequence")
            seen_sequences.add(sequence)
            if expected_by_sequence.get(sequence) != (block, candidate):
                reasons.append(f"result_plan_mismatch_sequence_{sequence}")
        if seen_sequences != set(expected_by_sequence):
            reasons.append("result_sequence_set_mismatch")

    return {
        "valid": not reasons,
        "position_balanced": position_balance,
        "reasons": sorted(set(reasons)),
        "expected_total_runs": expected_total,
        "observed_total_runs": len(result_rows) if isinstance(result_rows, list) else 0,
        "position_counts": position_counts,
    }


def _paired_comparison(winner: dict, alternative: dict, expected_blocks: int) -> dict:
    winner_by_block = {
        run["block"]: run
        for run in winner["runs"]
        if run["valid"] and run["metrics"] is not None
    }
    alternative_by_block = {
        run["block"]: run
        for run in alternative["runs"]
        if run["valid"] and run["metrics"] is not None
    }
    common_blocks = sorted(set(winner_by_block) & set(alternative_by_block))
    better = 0
    tied = 0
    for block in common_blocks:
        winner_rank = _single_run_rank(winner_by_block[block]["metrics"])
        alternative_rank = _single_run_rank(alternative_by_block[block]["metrics"])
        if winner_rank > alternative_rank:
            better += 1
        elif winner_rank == alternative_rank:
            tied += 1

    higher_better = ("no_fall", "survival_fraction", "healthy_fraction")
    lower_better = (
        "healthy_jitter_mean_deg",
        "xy_max_drift_m",
        "reference_to_q_abs_lag_s",
        "target_age_p95_s",
    )
    paired_improvements = {}
    for metric in (*higher_better, *lower_better):
        values = []
        for block in common_blocks:
            winner_value = winner_by_block[block]["metrics"].get(metric)
            alternative_value = alternative_by_block[block]["metrics"].get(metric)
            if winner_value is None or alternative_value is None:
                continue
            delta = (
                float(winner_value) - float(alternative_value)
                if metric in higher_better
                else float(alternative_value) - float(winner_value)
            )
            values.append(delta)
        summary = _summary(values)
        if summary is not None:
            paired_improvements[metric] = {
                **summary,
                "orientation": "positive_means_winner_better",
            }

    p_value = _exact_sign_p_one_sided(better, len(common_blocks))
    return {
        "winner": winner["name"],
        "alternative": alternative["name"],
        "paired_blocks": len(common_blocks),
        "winner_better_blocks": better,
        "tied_blocks": tied,
        "winner_worse_blocks": len(common_blocks) - better - tied,
        "direction_fraction": better / max(len(common_blocks), 1),
        "exact_sign_p_one_sided": p_value,
        "complete": len(common_blocks) == expected_blocks,
        "directional_75pct": (
            len(common_blocks) == expected_blocks
            and better >= math.ceil(0.75 * expected_blocks)
        ),
        "statistically_confirmed": (
            len(common_blocks) == expected_blocks
            and expected_blocks >= 8
            and p_value <= 0.05
        ),
        "paired_metric_improvements": paired_improvements,
    }


def build_summary(results_path: pathlib.Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    plan_path = pathlib.Path(results["plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    repo_root = pathlib.Path(plan["repo_root"]).resolve()
    expected_per_candidate = int(plan["repeats"])
    expected_free_seconds = float(plan["free_seconds"])
    design_validation = _validate_design(plan, results)

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
                    reasons.extend(
                        _runtime_valid(
                            report,
                            repo_root,
                            expected_free_seconds=expected_free_seconds,
                        )
                    )
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
            "reference_to_q_abs_lag_s",
            "reference_to_q_signed_lag_s",
            "reference_to_q_corr",
            "target_age_p95_s",
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
        observed_blocks = [
            int(run["block"])
            for run in run_summaries
            if isinstance(run.get("block"), (int, np.integer))
        ]
        expected_blocks = set(range(1, expected_per_candidate + 1))
        block_complete = (
            len(observed_blocks) == expected_per_candidate
            and set(observed_blocks) == expected_blocks
            and len(set(observed_blocks)) == expected_per_candidate
        )
        candidate_summaries.append(
            {
                **definition,
                "expected_runs": expected_per_candidate,
                "observed_runs": total,
                "valid_runs": valid,
                "valid_fraction": valid / max(expected_per_candidate, 1),
                "observed_blocks": sorted(observed_blocks),
                "complete": (
                    total == expected_per_candidate
                    and valid == expected_per_candidate
                    and block_complete
                ),
                "metrics": aggregate,
                "runs": run_summaries,
            }
        )

    eligible = [candidate for candidate in candidate_summaries if candidate["valid_runs"] > 0]
    ranking = sorted(eligible, key=_candidate_rank, reverse=True)
    winner = ranking[0]["name"] if ranking else None
    winner_row = ranking[0] if ranking else None
    all_candidates_complete = bool(candidate_summaries) and all(
        candidate["complete"] for candidate in candidate_summaries
    )
    pairwise_comparisons = []
    if winner_row is not None:
        pairwise_comparisons = [
            _paired_comparison(winner_row, alternative, expected_per_candidate)
            for alternative in candidate_summaries
            if alternative["name"] != winner_row["name"]
        ]
    all_directional = bool(pairwise_comparisons) and all(
        comparison["directional_75pct"] for comparison in pairwise_comparisons
    )
    all_statistically_confirmed = bool(pairwise_comparisons) and all(
        comparison["statistically_confirmed"] for comparison in pairwise_comparisons
    )
    confidence = "insufficient_valid_data"
    if winner_row is not None:
        complete_balanced = (
            design_validation["valid"]
            and design_validation["position_balanced"]
            and all_candidates_complete
        )
        if complete_balanced and all_statistically_confirmed:
            confidence = "confirmed"
        elif complete_balanced and expected_per_candidate >= 4 and all_directional:
            confidence = "balanced_directional_evidence"
        elif complete_balanced:
            confidence = "balanced_inconclusive"
        else:
            confidence = "provisional_screening"

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
            "healthy jitter mean across arms/waist/legs/tilt (lower)",
            "XY max drift (lower)",
            "reference-to-q absolute internal lag with corr>=0.50 (lower)",
            "target receive age p95 (lower; diagnostic tie-breaker)",
        ],
        "winner": winner,
        "confidence": confidence,
        "confirmation": {
            "design_validation": design_validation,
            "all_candidates_complete": all_candidates_complete,
            "minimum_confirmatory_blocks": 8,
            "directional_evidence_rule": (
                "winner better than every alternative in at least 75% of paired blocks"
            ),
            "confirmation_rule": (
                "complete balanced design and one-sided exact sign p<=0.05 versus every alternative"
            ),
            "all_directional": all_directional,
            "all_statistically_confirmed": all_statistically_confirmed,
            "pairwise_comparisons": pairwise_comparisons,
        },
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
        f"{'healthy':>10}{'age95(s)':>11}{'|lag|(s)':>11}{'jitter':>10}{'xy(m)':>9}"
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
            f"{_fmt_metric(candidate, 'target_age_p95_s'):>11}"
            f"{_fmt_metric(candidate, 'reference_to_q_abs_lag_s'):>11}"
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
