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
    def _write_run(path: Path, repo_root: Path, *, fell: bool) -> None:
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
            "status": "ok",
            "step_dt": 0.02,
            "run_manifest": {"repositories": {"isaaclab": {"realpath": str(repo_root)}}},
            "isaaclab_tasks_file": str(repo_root / "source/isaaclab_tasks/isaaclab_tasks/__init__.py"),
            "actions_module_file": str(
                repo_root
                / "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
                "locomanipulation/pick_place/mdp/actions.py"
            ),
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
            target_age_s=np.full(length, 0.005, dtype=np.float64),
            joint_names=joint_names,
            meta=np.asarray(json.dumps(meta)),
        )

    def test_selector_prefers_stable_candidate_and_confirms_balanced_repeats(self):
        repo_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "matrix_plan.json"
            results_path = root / "matrix_results.json"
            plan = {
                "repo_root": str(repo_root),
                "scenario": "v3_bvh",
                "repeats": 2,
                "design": {"complete_position_balance": True},
                "candidates": [
                    {"name": "unstable", "policy_dir": "policy/release", "substep_consume": False},
                    {"name": "stable", "policy_dir": "policy/release", "substep_consume": True},
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            runs = []
            sequence = 0
            for block in (1, 2):
                for candidate, fell in (("unstable", True), ("stable", False)):
                    sequence += 1
                    npz = root / f"{candidate}_{block}.npz"
                    self._write_run(npz, repo_root, fell=fell)
                    runs.append(
                        {
                            "sequence": sequence,
                            "block": block,
                            "candidate": candidate,
                            "returncode": 0,
                            "npz": str(npz),
                        }
                    )
            results_path.write_text(
                json.dumps({"plan": str(plan_path), "runs": runs}), encoding="utf-8"
            )
            summary = matrix_report.build_summary(results_path)

        self.assertEqual(summary["winner"], "stable")
        self.assertEqual(summary["confidence"], "confirmed")
        self.assertEqual(summary["ranking"][0], "stable")


if __name__ == "__main__":
    unittest.main()
