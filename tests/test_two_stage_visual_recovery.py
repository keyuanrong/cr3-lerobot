import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data_processing" / "recover_drag_phase_boundaries_two_stage_qwen.py"
SPEC = importlib.util.spec_from_file_location("recover_drag_phase_boundaries_two_stage_qwen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestTwoStageVisualRecovery(unittest.TestCase):
    def setUp(self) -> None:
        self.minimums = {"approach": 30, "grasp": 30, "carry": 30, "place": 30}

    def test_rejects_missing_lift_boundary(self) -> None:
        answer = {
            "decision": "accept",
            "boundaries": {"approach_end": 100, "grasp_end": None, "carry_end": 300, "place_end": 400},
            "confidence": {"approach_end": 0.9, "grasp_end": 0.0, "carry_end": 0.9, "place_end": 0.9},
        }

        self.assertIsNone(MODULE.validate_visual_boundaries(answer, 0, 500, self.minimums, 0.75))

    def test_rejects_low_confidence_grasp_even_when_a_frame_is_present(self) -> None:
        answer = {
            "decision": "accept",
            "boundaries": {"approach_end": 100, "grasp_end": 200, "carry_end": 300, "place_end": 400},
            "confidence": {"approach_end": 0.9, "grasp_end": 0.5, "carry_end": 0.9, "place_end": 0.9},
        }

        self.assertIsNone(MODULE.validate_visual_boundaries(answer, 0, 500, self.minimums, 0.75))

    def test_accepts_four_clear_ordered_boundaries(self) -> None:
        answer = {
            "decision": "accept",
            "boundaries": {"approach_end": 100, "grasp_end": 200, "carry_end": 300, "place_end": 400},
            "confidence": {"approach_end": 0.9, "grasp_end": 0.9, "carry_end": 0.9, "place_end": 0.9},
        }

        self.assertEqual(
            MODULE.validate_visual_boundaries(answer, 0, 500, self.minimums, 0.75),
            {"approach_end": 100, "grasp_end": 200, "carry_end": 300, "place_end": 400},
        )

    def test_rejects_a_too_short_carry_phase(self) -> None:
        answer = {
            "decision": "accept",
            "boundaries": {"approach_end": 100, "grasp_end": 200, "carry_end": 220, "place_end": 400},
            "confidence": {"approach_end": 0.9, "grasp_end": 0.9, "carry_end": 0.9, "place_end": 0.9},
        }

        self.assertIsNone(MODULE.validate_visual_boundaries(answer, 0, 500, self.minimums, 0.75))

    def test_parser_uses_dashscope_base_url_environment_variable(self) -> None:
        with patch.dict(os.environ, {"DASHSCOPE_BASE_URL": "https://example.test/compatible-mode/v1"}, clear=False):
            with patch.object(sys, "argv", ["recover.py", "--source-report", "source.jsonl", "--output-manifest", "out.jsonl", "--report", "report.jsonl"]):
                args = MODULE.parse_args()

        self.assertEqual(args.base_url, "https://example.test/compatible-mode/v1")

    def test_parser_provides_wrist_candidate_count_for_local_refinement(self) -> None:
        with patch.object(sys, "argv", ["recover.py", "--source-report", "source.jsonl", "--output-manifest", "out.jsonl", "--report", "report.jsonl"]):
            args = MODULE.parse_args()

        self.assertEqual(args.approach_wrist_sample_frames, args.local_sample_frames)

    def test_global_boundaries_snap_to_nearest_sparse_candidate_without_local_distance_limit(self) -> None:
        answer = {
            "boundaries": {"approach_end": 655, "grasp_end": 705, "carry_end": 731, "place_end": 781},
            "confidence": {"approach_end": 0.95, "grasp_end": 0.95, "carry_end": 0.95, "place_end": 0.95},
            "decision": "accept",
        }

        snapped = MODULE.snap_global_boundaries(answer, [640, 704, 736, 800])

        self.assertEqual(
            snapped["boundaries"],
            {"approach_end": 640, "grasp_end": 704, "carry_end": 736, "place_end": 800},
        )

    def test_global_anchors_accept_short_coarse_intervals(self) -> None:
        answer = {
            "decision": "accept",
            "anchors": {"lift": 705, "arrival": 731, "release": 781},
            "confidence": {"lift": 0.95, "arrival": 0.95, "release": 0.95},
        }

        self.assertEqual(
            MODULE.validate_global_anchors(answer, 0, 900, 0.75),
            {"lift": 705, "arrival": 731, "release": 781},
        )

    def test_global_anchors_reject_missing_lift(self) -> None:
        answer = {
            "decision": "accept",
            "anchors": {"lift": None, "arrival": 731, "release": 781},
            "confidence": {"lift": 0.0, "arrival": 0.95, "release": 0.95},
        }

        self.assertIsNone(MODULE.validate_global_anchors(answer, 0, 900, 0.75))

    def test_parser_uses_shorter_final_phase_minimums(self) -> None:
        with patch.object(sys, "argv", ["recover.py", "--source-report", "source.jsonl", "--output-manifest", "out.jsonl", "--report", "report.jsonl"]):
            args = MODULE.parse_args()

        self.assertEqual(
            {
                "approach": args.min_approach_frames,
                "grasp": args.min_grasp_frames,
                "carry": args.min_carry_frames,
                "place": args.min_place_frames,
            },
            {"approach": 20, "grasp": 20, "carry": 20, "place": 15},
        )

    def test_grasp_clip_ranges_cover_the_local_window_in_order(self) -> None:
        self.assertEqual(
            MODULE.grasp_clip_ranges(center=100, frame_count=240, window_frames=20, clips=4),
            [(80, 89), (90, 99), (100, 109), (110, 120)],
        )

    def test_parser_provides_default_grasp_clip_configuration(self) -> None:
        with patch.object(sys, "argv", ["recover.py", "--source-report", "source.jsonl", "--output-manifest", "out.jsonl", "--report", "report.jsonl"]):
            args = MODULE.parse_args()

        self.assertEqual((args.grasp_clip_count, args.grasp_frames_per_clip), (8, 8))


if __name__ == "__main__":
    unittest.main()
