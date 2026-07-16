# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for balanced SONIC matrices and deterministic selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tools import run_sonic_eval_matrix as matrix
from scripts.tools import sonic_eval_matrix_report as matrix_report


class TestSonicEvalMatrix(unittest.TestCase):
    def test_williams_blocks_balance_each_position(self):
        names = ["a", "b", "c", "d"]
        order = matrix._balanced_order(names, repeats=len(names))
        rows = [
            [name for _, name in order[index * len(names) : (index + 1) * len(names)]]
            for index in range(len(names))
        ]
        for name in names:
            positions = [row.index(name) for row in rows]
            self.assertEqual(sorted(positions), list(range(len(names))))

    @staticmethod
    def _write_run(
        path: Path, repo_root: Path, *, fell: bool, substep_consume: bool
    ) -> None:
        length = 160
        wall_t = np.arange(length, dtype=np.float64) * 0.02
        joint_names = np.asarray(
            [
                "left_hip_pitch_joint",
                "waist_yaw_joint",
                "left_shoulder_pitch_joint",
            ]
        )
        q = np.zeros((length, len(joint_names)), dtype=np.float32)
        target = np.zeros_like(q)
        root = np.zeros((length, 3), dtype=np.float32)
        root[:, 2] = 0.79
        if fell:
            root[80:, 2] = 0.25
        meta = {
            "schema_version": 2,
            "status": "ok",
            "unlocked": True,
            "step_dt": 0.02,
            "run_manifest": {
                "repositories": {"isaaclab": {"realpath": str(repo_root)}},
                "run": {"policy_dir": "policy/release"},
                "artifacts": {
                    "deploy_binary": {
                        "realpath": "/tmp/test-sonic-deploy",
                        "sha256": "test-deploy-sha256",
                    }
                },
            },
            "isaaclab_tasks_file": str(repo_root / "source/isaaclab_tasks/isaaclab_tasks/__init__.py"),
            "actions_module_file": str(
                repo_root
                / "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                "locomanipulation/pick_place/mdp/actions.py"
            ),
            "sonic_env": {
                "SONIC_DEPLOY_AUTO_RECOVER": "0",
                "SONIC_DEPLOY_ELASTIC_BAND": "0",
                "SONIC_G1_MUJOCO_TORQUE_PARITY": "0",
                "SONIC_G1_MUJOCO_NO_ARMATURE": "0",
                "SONIC_G1_MUJOCO_NO_VEL_LIMIT": "0",
                "SONIC_DEPLOY_SUBSTEP_CONSUME": "1" if substep_consume else "0",
            },
        }
        np.savez_compressed(
            path,
            wall_t=wall_t,
            control_state=np.full(length, 2, dtype=np.int8),
            phase=np.ones(length, dtype=np.int8),
            q=q,
            target=target,
            step_delta=np.zeros(length, dtype=np.float32),
            root_pos=root,
            tilt_deg=np.zeros(length, dtype=np.float32),
            valid_target_count=np.arange(length, dtype=np.int64),
            packet_count=np.arange(length, dtype=np.int64),
            source_index=np.arange(length, dtype=np.int64),
            invalid_target_count=np.zeros(length, dtype=np.int64),
            recovery_count=np.zeros(length, dtype=np.int64),
            target_age_s=np.full(length, 0.005, dtype=np.float64),
            joint_names=joint_names,
            meta=np.asarray(json.dumps(meta)),
        )

    def _build_stability_matrix(
        self, root: Path, repo_root: Path, *, repeats: int
    ) -> tuple[Path, dict]:
        plan_path = root / "matrix_plan.json"
        results_path = root / "matrix_results.json"
        names = ["unstable", "stable"]
        order = matrix._balanced_order(names, repeats=repeats)
        plan = {
            "repo_root": str(repo_root),
            "scenario": "v3_bvh",
            "free_seconds": 3.2,
            "repeats": repeats,
            "deploy_binary": {
                "realpath": "/tmp/test-sonic-deploy",
                "sha256": "test-deploy-sha256",
            },
            "pinned_env": {
                "SONIC_DEPLOY_AUTO_RECOVER": "0",
                "SONIC_DEPLOY_ELASTIC_BAND": "0",
                "SONIC_G1_MUJOCO_TORQUE_PARITY": "0",
                "SONIC_G1_MUJOCO_NO_ARMATURE": "0",
                "SONIC_G1_MUJOCO_NO_VEL_LIMIT": "0",
            },
            "design": {"complete_position_balance": repeats % len(names) == 0},
            "candidates": [
                {"name": "unstable", "policy_dir": "policy/release", "substep_consume": False},
                {"name": "stable", "policy_dir": "policy/release", "substep_consume": True},
            ],
            "order": [
                {"sequence": sequence, "block": block, "candidate": candidate}
                for sequence, (block, candidate) in enumerate(order, start=1)
            ],
        }
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        runs = []
        for sequence, (block, candidate) in enumerate(order, start=1):
            npz = root / f"{sequence:02d}_{candidate}_{block}.npz"
            self._write_run(
                npz,
                repo_root,
                fell=candidate == "unstable",
                substep_consume=candidate == "stable",
            )
            runs.append(
                {
                    "sequence": sequence,
                    "block": block,
                    "candidate": candidate,
                    "returncode": 0,
                    "npz": str(npz),
                }
            )
        results = {"plan": str(plan_path), "runs": runs}
        results_path.write_text(json.dumps(results), encoding="utf-8")
        return results_path, results

    def test_selector_prefers_stable_candidate_and_confirms_eight_balanced_repeats(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path, _ = self._build_stability_matrix(root, repo_root, repeats=8)
            summary = matrix_report.build_summary(results_path)

        self.assertEqual(summary["winner"], "stable")
        self.assertEqual(summary["confidence"], "confirmed")
        self.assertEqual(summary["ranking"][0], "stable")
        comparison = summary["confirmation"]["pairwise_comparisons"][0]
        self.assertEqual(comparison["winner_better_blocks"], 8)
        self.assertLessEqual(comparison["exact_sign_p_one_sided"], 0.05)

    def test_four_balanced_repeats_are_directional_not_confirmed(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            results_path, _ = self._build_stability_matrix(
                Path(directory), repo_root, repeats=4
            )
            summary = matrix_report.build_summary(results_path)

        self.assertEqual(summary["winner"], "stable")
        self.assertEqual(summary["confidence"], "balanced_directional_evidence")
        self.assertEqual(
            summary["confirmation"]["pairwise_comparisons"][0]["winner_better_blocks"], 4
        )

    def test_result_plan_mismatch_cannot_be_confirmed(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results_path, results = self._build_stability_matrix(root, repo_root, repeats=8)
            results["runs"][0]["block"] = 2
            results_path.write_text(json.dumps(results), encoding="utf-8")
            summary = matrix_report.build_summary(results_path)

        self.assertEqual(summary["confidence"], "provisional_screening")
        self.assertFalse(summary["confirmation"]["design_validation"]["valid"])
        self.assertIn(
            "result_plan_mismatch_sequence_1",
            summary["confirmation"]["design_validation"]["reasons"],
        )

    def test_physical_quality_precedes_target_age_tie_breaker(self):
        def candidate(name: str, *, age: float, lag: float, jitter: float, drift: float):
            return {
                "name": name,
                "valid_fraction": 1.0,
                "metrics": {
                    "no_fall": {"mean": 1.0},
                    "survival_fraction": {"mean": 1.0},
                    "healthy_fraction": {"mean": 1.0},
                    "target_age_p95_s": {"mean": age},
                    "reference_to_q_abs_lag_s": {"mean": lag},
                    "healthy_jitter_mean_deg": {"mean": jitter},
                    "xy_max_drift_m": {"mean": drift},
                },
            }

        physically_good = candidate(
            "good", age=0.020, lag=0.020, jitter=0.10, drift=0.10
        )
        physically_bad = candidate(
            "bad", age=0.001, lag=0.400, jitter=2.00, drift=5.00
        )
        self.assertGreater(
            matrix_report._candidate_rank(physically_good),
            matrix_report._candidate_rank(physically_bad),
        )

    def test_selector_uses_absolute_lag_and_accepts_negative_signed_lag(self):
        report = {
            "free": {
                "seconds": 10.0,
                "fall": {"event_count": 0, "survival_s": 10.0},
                "healthy": {"frac": 1.0},
                "coverage": {"target_age_s": {"p95": 0.004}},
                "internal_following_lag": {
                    "reference_to_q": {
                        "lag_s": -0.02,
                        "abs_lag_s": 0.02,
                        "corr": 0.9,
                    }
                },
            }
        }
        metrics = matrix_report._run_metrics(report)
        self.assertEqual(metrics["reference_to_q_signed_lag_s"], -0.02)
        self.assertEqual(metrics["reference_to_q_abs_lag_s"], 0.02)

    def test_missing_fall_signal_is_not_treated_as_no_fall(self):
        report = {
            "free": {
                "seconds": 10.0,
                "fall": {"event_count": None, "survival_s": None},
                "healthy": {"frac": 1.0},
                "coverage": {"target_age_s": {"p95": 0.004}},
                "internal_following_lag": {},
            }
        }
        metrics = matrix_report._run_metrics(report)
        self.assertIsNone(metrics["fall_events"])
        self.assertIsNone(metrics["no_fall"])
        self.assertIsNone(metrics["survival_fraction"])


if __name__ == "__main__":
    unittest.main()
