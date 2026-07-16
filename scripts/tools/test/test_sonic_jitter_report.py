# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure NumPy regression tests for the SONIC evaluation report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tools import sonic_jitter_report as report


JOINT_NAMES = np.asarray(
    [
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
    ]
)


def synthetic_npz(
    path: Path,
    *,
    length: int = 200,
    control_state: np.ndarray | None = None,
    root_length: int | None = None,
    packets: np.ndarray | None = None,
    packet_count: np.ndarray | None = None,
    valid_target_count: np.ndarray | None = None,
    source_index: np.ndarray | None = None,
    root_z: np.ndarray | None = None,
    target_age_s: np.ndarray | None = None,
    target_gate: dict | None = None,
    dt: float = 0.02,
    include_root: bool = True,
) -> None:
    wall_t = 10.0 + np.arange(length, dtype=np.float64) * dt
    base = np.sin(np.arange(length)[:, None] * 0.07 + np.arange(len(JOINT_NAMES))[None, :])
    q = (0.02 * base).astype(np.float32)
    target = (q + 0.002 * np.cos(np.arange(length)[:, None] * 0.11)).astype(np.float32)
    if root_length is None:
        root_length = length
    root = np.zeros((root_length, 3), dtype=np.float32)
    root[:, 2] = 0.79
    if root_z is not None:
        root[: min(root_length, len(root_z)), 2] = root_z[:root_length]
    tilt = np.ones(root_length, dtype=np.float32)

    values = {
        "wall_t": wall_t,
        "phase": np.ones(length, dtype=np.int8),
        "q": q,
        "target": target,
        "step_delta": np.zeros(length, dtype=np.float32),
        "packets": np.arange(length, dtype=np.int64) if packets is None else packets,
        "joint_names": JOINT_NAMES,
        "meta": np.asarray(json.dumps({"step_dt": dt, "target_gate": target_gate or {}})),
    }
    if include_root:
        values["root_pos"] = root
        values["tilt_deg"] = tilt
    if control_state is not None:
        values["control_state"] = control_state
    if valid_target_count is not None:
        values["valid_target_count"] = valid_target_count
    if source_index is not None:
        values["source_index"] = source_index
    if packet_count is not None:
        values["packet_count"] = packet_count
    if target_age_s is not None:
        values["target_age_s"] = target_age_s
    np.savez_compressed(path, **values)


class TestSonicJitterReport(unittest.TestCase):
    def test_known_internal_following_lag(self):
        rng = np.random.default_rng(7)
        length = 600
        reference = rng.standard_normal((length, 5))
        target = np.zeros_like(reference)
        measured = np.zeros_like(reference)
        target[3:] = reference[:-3]
        measured[7:] = reference[:-7]
        wall_t = np.arange(length, dtype=np.float64) * 0.02
        mask = np.ones(length, dtype=bool)

        target_result = report.internal_following_lag(reference, target, wall_t, mask)
        measured_result = report.internal_following_lag(reference, measured, wall_t, mask)

        self.assertEqual(target_result["lag_steps"], 3)
        self.assertAlmostEqual(target_result["lag_s"], 0.06, places=8)
        self.assertAlmostEqual(target_result["abs_lag_s"], 0.06, places=8)
        self.assertGreater(target_result["corr"], 0.99)
        self.assertEqual(measured_result["lag_steps"], 7)
        self.assertAlmostEqual(measured_result["lag_s"], 0.14, places=8)
        self.assertGreater(measured_result["corr"], 0.99)

    def test_chatter_rate_uses_wall_time(self):
        length = 101
        q = (np.arange(length) % 2).astype(np.float64)[:, None] * 0.01
        mask = np.ones(length, dtype=bool)
        fast = report._chatter_rate(q, np.arange(length) * 0.02, mask)
        slow = report._chatter_rate(q, np.arange(length) * 0.04, mask)
        self.assertAlmostEqual(fast / slow, 2.0, places=8)

    def test_fall_event_and_recovery_count_use_wall_time(self):
        length = 220
        state = np.full(length, report.STATE_FREE, dtype=np.int8)
        state[60:75] = report.STATE_RECOVERY
        state[145:160] = report.STATE_RECOVERY
        root_z = np.full(length, 0.79, dtype=np.float32)
        root_z[50:60] = 0.30
        root_z[135:145] = 0.30

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "falls.npz"
            synthetic_npz(path, length=length, control_state=state, root_z=root_z, dt=0.03)
            result = report.load_report(path)["free"]

        self.assertEqual(result["fall"]["event_count"], 2)
        self.assertEqual(result["fall"]["recovery_count"], 2)
        self.assertEqual(result["fall"]["recovery_source"], "control_state")
        self.assertAlmostEqual(result["fall"]["first_fall_s"], 1.5, places=8)
        self.assertAlmostEqual(result["fall"]["survival_s"], 1.5, places=8)
        self.assertLess(result["healthy"]["steps"], length - 30)

    def test_optional_root_stream_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_root.npz"
            synthetic_npz(path, length=160, include_root=False)
            result = report.load_report(path)["free"]

        self.assertNotIn("root", result)
        self.assertEqual(result["fall"]["known_frac"], 0.0)
        self.assertIsNone(result["fall"]["event_count"])
        self.assertIsNone(result["fall"]["recovery_count"])

    def test_root_stream_one_frame_short_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short_root.npz"
            synthetic_npz(path, length=160, root_length=159)
            result = report.load_report(path)["free"]

        self.assertEqual(result["steps"], 160)
        self.assertAlmostEqual(result["root"]["valid_frac"], 159 / 160)
        self.assertEqual(result["fall"]["event_count"], 0)
        self.assertAlmostEqual(result["fall"]["known_frac"], 159 / 160)

    def test_packet_dropout_marks_run_invalid(self):
        length = 160
        packets = np.empty(length, dtype=np.int64)
        packets[:50] = np.arange(50)
        packets[50:75] = 49
        packets[75:] = np.arange(50, 50 + length - 75)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dropout.npz"
            synthetic_npz(
                path,
                length=length,
                packets=packets,
                packet_count=packets,
                valid_target_count=packets,
                source_index=packets,
            )
            coverage = report.load_report(path)["free"]["coverage"]

        self.assertTrue(coverage["invalid"])
        self.assertEqual(coverage["update_counter"], "valid_target_count")
        self.assertEqual(coverage["packet_counter"], "source_index")
        self.assertIn("update_gap_too_large", coverage["invalid_reasons"])
        self.assertGreater(coverage["max_update_gap_s"], 0.40)
        self.assertLess(coverage["update_coverage"], 0.90)

    def test_single_transient_stale_sample_does_not_invalidate_run(self):
        length = 200
        target_age = np.full(length, 0.005, dtype=np.float64)
        target_age[75] = 0.106
        counts = np.arange(length, dtype=np.int64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transient_stale.npz"
            synthetic_npz(
                path,
                length=length,
                target_age_s=target_age,
                packets=counts,
                packet_count=counts,
                valid_target_count=counts,
                source_index=counts,
                target_gate={
                    "enabled": True,
                    "passed": True,
                    "max_target_stale_s": 0.10,
                    "max_stale_fraction": 0.02,
                    "hard_target_stale_s": 0.50,
                },
            )
            coverage = report.load_report(path)["free"]["coverage"]

        self.assertFalse(coverage["invalid"])
        self.assertAlmostEqual(coverage["target_age_s"]["stale_frac"], 1 / length)
        self.assertGreater(coverage["target_age_s"]["max"], 0.10)

    def test_substep_multi_consume_is_not_misclassified_as_packet_loss(self):
        length = 160
        counts = np.arange(length, dtype=np.int64) * 2

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "substep.npz"
            synthetic_npz(
                path,
                length=length,
                packets=counts,
                packet_count=counts,
                valid_target_count=counts,
                source_index=counts,
                target_age_s=np.full(length, 0.003, dtype=np.float64),
            )
            coverage = report.load_report(path)["free"]["coverage"]

        self.assertFalse(coverage["invalid"])
        self.assertEqual(coverage["packet_coverage"], 1.0)
        self.assertEqual(coverage["max_packet_jump"], 2)

    def test_latest_only_conflation_is_diagnostic_not_invalid_when_stream_is_fresh(self):
        length = 200
        source = np.arange(length, dtype=np.int64)
        received = np.arange(length, dtype=np.int64) - (
            np.arange(length, dtype=np.int64) // 20
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflated_latest.npz"
            synthetic_npz(
                path,
                length=length,
                packets=received,
                packet_count=received,
                valid_target_count=received,
                source_index=source,
                target_age_s=np.full(length, 0.020, dtype=np.float64),
            )
            coverage = report.load_report(path)["free"]["coverage"]

        self.assertFalse(coverage["invalid"])
        self.assertGreater(coverage["packet_coverage"], 0.90)
        self.assertLess(coverage["packet_coverage"], 0.98)
        self.assertLessEqual(coverage["max_update_gap_s"], 0.041)

    def test_run_level_paired_effect(self):
        a = [
            {"free": {"healthy": {"arms_hf_rms_deg": 2.0}}},
            {"free": {"healthy": {"arms_hf_rms_deg": 4.0}}},
            {"free": {"healthy": {"arms_hf_rms_deg": 5.0}}},
        ]
        b = [
            {"free": {"healthy": {"arms_hf_rms_deg": 1.0}}},
            {"free": {"healthy": {"arms_hf_rms_deg": 2.0}}},
            {"free": {"healthy": {"arms_hf_rms_deg": 3.0}}},
        ]
        effect = report.paired_effects(a, b)["free.healthy.arms_hf_rms_deg"]
        self.assertAlmostEqual(effect["mean"], -5 / 3)
        self.assertEqual(effect["n"], 3)
        self.assertTrue(np.isfinite(effect["paired_dz"]))
        normal_half = 1.96 * effect["sd"] / np.sqrt(effect["n"])
        self.assertGreater(effect["ci95_high"] - effect["mean"], normal_half)

    def test_invalid_runs_are_excluded_from_metric_aggregate(self):
        valid = {
            "free": {
                "coverage": {"invalid": False},
                "healthy": {"arms_hf_rms_deg": 1.0},
            }
        }
        invalid = {
            "free": {
                "coverage": {"invalid": True},
                "healthy": {"arms_hf_rms_deg": 100.0},
            }
        }
        aggregate = report.aggregate_reports([valid, invalid])
        metric = aggregate["free.healthy.arms_hf_rms_deg"]
        self.assertEqual(metric["n"], 1)
        self.assertEqual(metric["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
