import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_processing" / "recover_relaxed_phase_boundaries_with_qwen.py"
SPEC = importlib.util.spec_from_file_location("recover_relaxed_phase_boundaries", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestRelaxedPhaseRecovery(unittest.TestCase):
    def test_selects_only_guardrail_rejections(self):
        rows = [
            {"decision": "uncertain", "reason": "no_candidate_frames_after_phase_guardrails", "episode": "a"},
            {"decision": "uncertain", "reason": "qwen_request_failed: timeout", "episode": "b"},
            {"decision": "accept", "reason": "", "episode": "c"},
        ]

        selected = MODULE.select_guardrail_rejections(rows)

        self.assertEqual(selected, [rows[0]])

    def test_matches_historical_pipe_delimited_report_keys(self):
        action = {
            "episode": "/tmp/full/drag_episode_1",
            "color": "yellow",
            "coarse": {phase: (index * 10, (index + 1) * 10) for index, phase in enumerate(("approach", "grasp", "carry", "place"))},
        }
        report = [
            {
                "key": "/tmp/full/drag_episode_1|yellow",
                "episode": action["episode"],
                "color": "yellow",
                "decision": "uncertain",
                "reason": "no_candidate_frames_after_phase_guardrails",
            }
        ]

        matched = MODULE.match_rejections_to_actions(report, {MODULE.action_key(action): action})

        self.assertEqual(matched, [action])

    def test_accepts_ordered_boundaries_with_minimum_phase_lengths(self):
        boundaries = {
            "approach_end": 100,
            "grasp_end": 160,
            "carry_end": 230,
            "place_end": 300,
        }
        candidates = list(range(0, 401, 5))

        accepted, reason = MODULE.validate_relaxed_boundaries(
            boundaries,
            candidates,
            task_start=0,
            task_end=360,
            min_phase_frames=20,
            snap_frames=5,
        )

        self.assertIsNone(reason)
        self.assertEqual(accepted, boundaries)

    def test_rejects_a_too_short_carry_phase(self):
        boundaries = {
            "approach_end": 100,
            "grasp_end": 160,
            "carry_end": 170,
            "place_end": 300,
        }

        accepted, reason = MODULE.validate_relaxed_boundaries(
            boundaries,
            list(range(0, 401, 5)),
            task_start=0,
            task_end=360,
            min_phase_frames=20,
            snap_frames=5,
        )

        self.assertIsNone(accepted)
        self.assertEqual(reason, "phase_too_short:carry")


if __name__ == "__main__":
    unittest.main()
