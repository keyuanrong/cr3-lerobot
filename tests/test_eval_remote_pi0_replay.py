from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluation" / "eval_remote_pi0_replay.py"
SPEC = importlib.util.spec_from_file_location("eval_remote_pi0_replay", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestRemotePi0Replay(unittest.TestCase):
    def test_request_frame_indices_match_live_request_frequency(self) -> None:
        self.assertEqual(MODULE.request_frame_indices(total_frames=45, fps=30.0, request_hz=1.5), [0, 20, 40])

    def test_latency_compensation_skips_stale_actions_and_enforces_minimum(self) -> None:
        actions = np.arange(50 * 7, dtype=np.float32).reshape(50, 7)

        aligned, skipped = MODULE.apply_latency_compensation(
            actions, roundtrip_ms=300.0, control_hz=30.0, min_remaining_actions=15
        )

        self.assertEqual(skipped, 9)
        self.assertTrue(np.array_equal(aligned, actions[9:]))

        rejected, skipped = MODULE.apply_latency_compensation(
            actions[:20], roundtrip_ms=300.0, control_hz=30.0, min_remaining_actions=15
        )

        self.assertEqual(skipped, 9)
        self.assertIsNone(rejected)

    def test_gripper_metrics_track_open_and_close_recalls(self) -> None:
        metrics = MODULE.GripperMetrics(semantics="close_high")
        metrics.update(target=0.0, prediction=0.0)
        metrics.update(target=0.0, prediction=1.0)
        metrics.update(target=1.0, prediction=1.0)
        metrics.update(target=1.0, prediction=0.0)

        summary = metrics.summary()

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["open_recall"], 0.5)
        self.assertEqual(summary["close_recall"], 0.5)
        self.assertEqual(summary["target_open_frames"], 2)
        self.assertEqual(summary["target_close_frames"], 2)


if __name__ == "__main__":
    unittest.main()
