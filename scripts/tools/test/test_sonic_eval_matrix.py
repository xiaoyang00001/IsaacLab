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
        path: Path,
        repo_root: Path,
        *,
        fell: bool,
        substep_consume: bool,
        seed: int | None = None,
        deploy_realpath: str = "/tmp/test-sonic-deploy",
        deploy_sha256: str = "test-deploy-sha256",
        deploy_root: str | None = None,
        deploy_setup_realpath: str | None = None,
        deploy_setup_sha256: str | None = None,
        deploy_runtime_repository: dict | None = None,
        length: int = 160,
        fall_index: int = 80,
        planned_free_seconds: float | None = None,
        termination_reason: str | None = None,
        fall_grace_s: float = 0.0,
    ) -> None:
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
            root[fall_index:, 2] = 0.25
        manifest = {
            "schema_version": 2 if deploy_root is not None else 1,
            "repositories": {"isaaclab": {"realpath": str(repo_root)}},
            "run": {"policy_dir": "policy/release"},
            "artifacts": {
                "deploy_binary": {
                    "realpath": deploy_realpath,
                    "sha256": deploy_sha256,
                }
            },
        }
        if seed is not None:
            manifest["run"]["seed"] = seed
        if deploy_root is not None:
            manifest["run"]["deploy_root"] = deploy_root
        if deploy_setup_realpath is not None:
            manifest["artifacts"]["deploy_setup_env"] = {
                "realpath": deploy_setup_realpath,
                "sha256": deploy_setup_sha256,
            }
        if deploy_runtime_repository is not None:
            manifest["repositories"]["deploy_runtime"] = deploy_runtime_repository
        meta = {
            "schema_version": 3 if seed is not None or termination_reason else 2,
            "status": "ok",
            "unlocked": True,
            "step_dt": 0.02,
            "run_manifest": manifest,
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
        if seed is not None:
            meta["seed_requested"] = seed
            meta["seed_actual"] = seed
        if planned_free_seconds is not None:
            meta["planned_free_seconds"] = planned_free_seconds
        if termination_reason is not None:
            meta["termination_reason"] = termination_reason
            meta["stop_after_fall_grace_s"] = fall_grace_s
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

    def test_block_seed_is_shared_within_block_and_retry_reuses_it(self):
        args = type(
            "Args",
            (),
            {
                "locked_seconds": 15.0,
                "free_seconds": 55.56,
                "seed_base": 700,
                "stop_after_fall_grace_s": 1.0,
            },
        )()
        self.assertEqual(matrix._block_seed(700, 1), 700)
        self.assertEqual(matrix._block_seed(700, 4), 703)
        first_attempt = matrix._runner_extra_args(args, block=3)
        retry_attempt = matrix._runner_extra_args(args, block=3)
        self.assertEqual(first_attempt, retry_attempt)
        self.assertEqual(first_attempt[first_attempt.index("--seed") + 1], "702")

    @staticmethod
    def _runtime_bundle(name: str) -> dict:
        root = f"/tmp/{name}-runtime"
        return {
            "root": root,
            "realpath": root,
            "repository": {
                "realpath": root,
                "commit": f"{name}-commit",
                "tracked_diff_sha256": f"{name}-diff",
            },
            "setup_env": {
                "realpath": f"{root}/scripts/setup_env.sh",
                "sha256": f"{name}-setup",
            },
        }

    def _build_candidate_bundle_matrix(
        self,
        root: Path,
        repo_root: Path,
        *,
        runtime_mismatch: bool = False,
        seed_mismatch: bool = False,
    ) -> Path:
        names = ["bundle_a", "bundle_b"]
        repeats = 2
        order = matrix._balanced_order(names, repeats)
        definitions = []
        for name in names:
            definitions.append(
                {
                    "name": name,
                    "policy_dir": "policy/release",
                    "substep_consume": name == "bundle_b",
                    "deploy_binary": {
                        "realpath": f"/tmp/{name}-deploy",
                        "sha256": f"{name}-binary",
                    },
                    "deploy_runtime": self._runtime_bundle(name),
                }
            )
        plan_path = root / "matrix_plan.json"
        results_path = root / "matrix_results.json"
        plan = {
            "repo_root": str(repo_root),
            "scenario": "v3_bvh",
            "free_seconds": 3.2,
            "repeats": repeats,
            "pinned_env": {
                "SONIC_DEPLOY_AUTO_RECOVER": "0",
                "SONIC_DEPLOY_ELASTIC_BAND": "0",
                "SONIC_G1_MUJOCO_TORQUE_PARITY": "0",
                "SONIC_G1_MUJOCO_NO_ARMATURE": "0",
                "SONIC_G1_MUJOCO_NO_VEL_LIMIT": "0",
            },
            "seed_policy": {
                "type": "paired_by_block",
                "distinct_across_blocks": True,
            },
            "candidates": definitions,
            "order": [
                {
                    "sequence": sequence,
                    "block": block,
                    "candidate": candidate,
                    "seed": 900 + block,
                }
                for sequence, (block, candidate) in enumerate(order, start=1)
            ],
        }
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        by_name = {definition["name"]: definition for definition in definitions}
        runs = []
        for sequence, (block, candidate) in enumerate(order, start=1):
            definition = by_name[candidate]
            bundle = definition["deploy_runtime"]
            seed = 900 + block
            actual_seed = seed + 1 if seed_mismatch and sequence == 1 else seed
            repository = dict(bundle["repository"])
            if runtime_mismatch and sequence == 1:
                repository["commit"] = "wrong-runtime-commit"
            npz = root / f"{sequence:02d}_{candidate}.npz"
            self._write_run(
                npz,
                repo_root,
                fell=candidate == "bundle_a",
                substep_consume=definition["substep_consume"],
                seed=actual_seed,
                deploy_realpath=definition["deploy_binary"]["realpath"],
                deploy_sha256=definition["deploy_binary"]["sha256"],
                deploy_root=bundle["realpath"],
                deploy_setup_realpath=bundle["setup_env"]["realpath"],
                deploy_setup_sha256=bundle["setup_env"]["sha256"],
                deploy_runtime_repository=repository,
                planned_free_seconds=3.2,
                termination_reason="planned_duration_complete",
            )
            runs.append(
                {
                    "sequence": sequence,
                    "block": block,
                    "candidate": candidate,
                    "seed": seed,
                    "returncode": 0,
                    "npz": str(npz),
                }
            )
        results_path.write_text(
            json.dumps({"plan": str(plan_path), "runs": runs}),
            encoding="utf-8",
        )
        return results_path

    def test_candidate_specific_runtime_bundles_are_valid(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            summary = matrix_report.build_summary(
                self._build_candidate_bundle_matrix(
                    Path(directory), repo_root
                )
            )
        self.assertEqual(summary["winner"], "bundle_b")
        self.assertTrue(all(candidate["complete"] for candidate in summary["candidates"]))
        self.assertTrue(summary["confirmation"]["design_validation"]["paired_seed_valid"])
        self.assertTrue(summary["confirmation"]["pairing_context"]["valid"])

    def test_seed_or_runtime_mismatch_invalidates_run(self):
        repo_root = Path(__file__).resolve().parents[3]
        for keyword, expected_reason in (
            ({"seed_mismatch": True}, "seed_requested_mismatch"),
            ({"runtime_mismatch": True}, "deploy_runtime_commit_mismatch"),
        ):
            with self.subTest(expected_reason=expected_reason), tempfile.TemporaryDirectory() as directory:
                summary = matrix_report.build_summary(
                    self._build_candidate_bundle_matrix(
                        Path(directory), repo_root, **keyword
                    )
                )
                invalid = [
                    reason
                    for candidate in summary["candidates"]
                    for run in candidate["runs"]
                    for reason in run["invalid_reasons"]
                ]
                self.assertTrue(any(reason.startswith(expected_reason) for reason in invalid))
                self.assertEqual(summary["confidence"], "provisional_screening")

    def test_fall_early_stop_is_valid_and_survival_uses_planned_horizon(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "matrix_plan.json"
            results_path = root / "matrix_results.json"
            npz = root / "fallen_early.npz"
            self._write_run(
                npz,
                repo_root,
                fell=True,
                substep_consume=False,
                length=60,
                fall_index=20,
                planned_free_seconds=3.2,
                termination_reason="fall_observed",
                fall_grace_s=0.5,
            )
            plan = {
                "repo_root": str(repo_root),
                "scenario": "v3_bvh",
                "free_seconds": 3.2,
                "repeats": 1,
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
                "candidates": [
                    {
                        "name": "fallen",
                        "policy_dir": "policy/release",
                        "substep_consume": False,
                    }
                ],
                "order": [{"sequence": 1, "block": 1, "candidate": "fallen"}],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            results_path.write_text(
                json.dumps(
                    {
                        "plan": str(plan_path),
                        "runs": [
                            {
                                "sequence": 1,
                                "block": 1,
                                "candidate": "fallen",
                                "returncode": 0,
                                "npz": str(npz),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = matrix_report.build_summary(results_path)

        run = summary["candidates"][0]["runs"][0]
        self.assertTrue(run["valid"], run["invalid_reasons"])
        self.assertAlmostEqual(run["metrics"]["survival_fraction"], 0.4 / 3.2, places=2)

    def test_motion_fidelity_precedes_absolute_jitter(self):
        base = {
            "valid_fraction": 1.0,
            "metrics": {
                "no_fall": {"mean": 1.0},
                "survival_fraction": {"mean": 1.0},
                "healthy_fraction": {"mean": 1.0},
                "reference_to_q_abs_lag_s": {"mean": 0.02},
                "xy_max_drift_m": {"mean": 0.2},
                "target_age_p95_s": {"mean": 0.005},
            },
        }
        faithful = {
            **base,
            "name": "faithful",
            "metrics": {
                **base["metrics"],
                "reference_to_q_corr": {"mean": 0.95},
                "healthy_tracking_error_hf_mean_deg": {"mean": 0.2},
                "healthy_track_rms_mean_deg": {"mean": 1.0},
                "healthy_jitter_mean_deg": {"mean": 0.8},
            },
        }
        frozen_smooth = {
            **base,
            "name": "frozen_smooth",
            "metrics": {
                **base["metrics"],
                "reference_to_q_corr": {"mean": 0.10},
                "healthy_tracking_error_hf_mean_deg": {"mean": 2.0},
                "healthy_track_rms_mean_deg": {"mean": 8.0},
                "healthy_jitter_mean_deg": {"mean": 0.1},
            },
        }
        self.assertGreater(
            matrix_report._candidate_rank(faithful),
            matrix_report._candidate_rank(frozen_smooth),
        )


if __name__ == "__main__":
    unittest.main()
