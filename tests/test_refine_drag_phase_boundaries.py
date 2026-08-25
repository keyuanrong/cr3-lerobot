"""Tests for the dense visual phase-boundary refinement helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
import base64
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "data_processing" / "refine_drag_phase_boundaries_with_qwen.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("refine_drag_phase_boundaries_with_qwen", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestDenseBoundaryRefinement(unittest.TestCase):
    def test_visual_fallback_repairs_empty_strict_candidates_inside_gripper_bounds(self) -> None:
        action = {"episode": "/tmp/episode", "color": "red", "coarse": {}}
        rows = [{"gripper": "100"}] * 20
        strict_bounds = {name: (10, 10) for name in MODULE.BOUNDARY_NAMES}
        args = SimpleNamespace(visual_fallback_on_empty_guardrails=True)
        expected_messages = {"approach_end": [{"role": "user", "content": []}]}
        expected_allowed = {"approach_end": [5]}

        with patch.object(
            MODULE,
            "build_messages",
            side_effect=[ValueError("no_candidate_frames_after_phase_guardrails"), (expected_messages, expected_allowed)],
        ) as build:
            messages, allowed, effective_bounds, mode = MODULE.build_initial_messages(
                action, rows, args, strict_bounds, full_cycle=None
            )

        self.assertEqual(messages, expected_messages)
        self.assertEqual(allowed, expected_allowed)
        self.assertEqual(effective_bounds, strict_bounds)
        self.assertEqual(mode, "event_anchored_visual_fallback")
        self.assertEqual(build.call_args_list[0].kwargs["candidate_bounds"], strict_bounds)
        self.assertEqual(build.call_args_list[1].kwargs["candidate_bounds"], strict_bounds)
        self.assertTrue(build.call_args_list[1].kwargs["repair_empty_candidates"])

    def test_visual_recovery_without_gripper_events_uses_coarse_visual_candidates(self) -> None:
        action = {"episode": "/tmp/episode", "color": "red", "coarse": {}}
        rows = [{"gripper": "100"}] * 20
        strict_bounds = {name: (10, 10) for name in MODULE.BOUNDARY_NAMES}
        args = SimpleNamespace(
            visual_fallback_on_empty_guardrails=True,
            visual_recovery_no_gripper_events=True,
        )
        expected_messages = {"approach_end": [{"role": "user", "content": []}]}
        expected_allowed = {"approach_end": [5]}

        with patch.object(
            MODULE,
            "build_messages",
            side_effect=[ValueError("no_candidate_frames_after_phase_guardrails"), (expected_messages, expected_allowed)],
        ) as build:
            _, _, effective_bounds, mode = MODULE.build_initial_messages(action, rows, args, strict_bounds, full_cycle=None)

        self.assertIsNone(effective_bounds)
        self.assertEqual(mode, "visual_recovery_no_gripper_events")
        self.assertIsNone(build.call_args_list[1].kwargs.get("candidate_bounds"))
        self.assertFalse(build.call_args_list[1].kwargs.get("repair_empty_candidates", False))

    def test_visual_fallback_does_not_replace_nonempty_strict_candidates(self) -> None:
        action = {"episode": "/tmp/episode", "color": "red", "coarse": {}}
        rows = [{"gripper": "100"}] * 20
        strict_bounds = {name: (10, 20) for name in MODULE.BOUNDARY_NAMES}
        args = SimpleNamespace(visual_fallback_on_empty_guardrails=True)
        expected_messages = {"approach_end": [{"role": "user", "content": []}]}
        expected_allowed = {"approach_end": [15]}

        with patch.object(MODULE, "build_messages", return_value=(expected_messages, expected_allowed)) as build:
            _, _, effective_bounds, mode = MODULE.build_initial_messages(action, rows, args, strict_bounds, full_cycle=None)

        self.assertEqual(effective_bounds, strict_bounds)
        self.assertEqual(mode, "strict_gripper_guardrails")
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["candidate_bounds"], strict_bounds)

    def test_empty_strict_candidates_raise_when_visual_fallback_is_disabled(self) -> None:
        action = {"episode": "/tmp/episode", "color": "red", "coarse": {}}
        rows = [{"gripper": "100"}] * 20
        strict_bounds = {name: (10, 10) for name in MODULE.BOUNDARY_NAMES}
        args = SimpleNamespace(visual_fallback_on_empty_guardrails=False)

        with patch.object(MODULE, "build_messages", side_effect=ValueError("no_candidate_frames_after_phase_guardrails")):
            with self.assertRaisesRegex(ValueError, "no_candidate_frames_after_phase_guardrails"):
                MODULE.build_initial_messages(action, rows, args, strict_bounds, full_cycle=None)

    def test_empty_guardrail_record_is_selected_for_targeted_retry(self) -> None:
        self.assertTrue(
            MODULE.is_empty_guardrail_record({"decision": "uncertain", "reason": "no_candidate_frames_after_phase_guardrails"})
        )
        self.assertFalse(MODULE.is_empty_guardrail_record({"decision": "accept", "reason": "no_candidate_frames_after_phase_guardrails"}))
        self.assertFalse(MODULE.is_empty_guardrail_record({"decision": "uncertain", "reason": "qwen_request_failed: timeout"}))

    def test_targeted_retry_filters_actions_before_a_smoke_test_limit(self) -> None:
        actions = [
            {"episode": "/tmp/a", "color": "red", "coarse": {"approach": (0, 10)}},
            {"episode": "/tmp/b", "color": "green", "coarse": {"approach": (0, 10)}},
        ]
        report = [
            {
                "key": MODULE.action_key(actions[1]),
                "decision": "uncertain",
                "reason": "no_candidate_frames_after_phase_guardrails",
            }
        ]

        selected = MODULE.actions_for_empty_guardrail_retry(actions, report)

        self.assertEqual(selected, [actions[1]])

    def test_default_initial_search_window_is_six_seconds(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "refine_drag_phase_boundaries_with_qwen.py",
                "--input-manifest",
                "input.jsonl",
                "--output-manifest",
                "output.jsonl",
                "--report",
                "report.jsonl",
            ],
        ):
            args = MODULE.parse_args()

        self.assertEqual(args.window_frames, 180)

    def test_request_failure_is_retryable_but_visual_uncertainty_is_not(self) -> None:
        self.assertTrue(MODULE.is_retryable_record({"errors": ["request_or_parse_failed"]}))
        self.assertTrue(MODULE.is_retryable_record({"errors": ["missing_visual_boundary"]}))
        self.assertFalse(
            MODULE.is_retryable_record({"errors": ["missing_visual_boundary"], "refinement_version": MODULE.REFINEMENT_VERSION})
        )

    def test_combine_boundary_answers_builds_one_strict_answer(self) -> None:
        answers = {
            "approach_end": {"decision": "accept", "boundary": 100, "confidence": 0.9, "reason": "a"},
            "grasp_end": {"decision": "accept", "boundary": 200, "confidence": 0.9, "reason": "b"},
            "carry_end": {"decision": "accept", "boundary": 300, "confidence": 0.9, "reason": "c"},
            "place_end": {"decision": "accept", "boundary": 400, "confidence": 0.9, "reason": "d"},
        }

        combined = MODULE.combine_boundary_answers(answers)

        self.assertEqual(combined["decision"], "accept")
        self.assertEqual(combined["boundaries"]["grasp_end"], 200)

    def test_combine_boundary_answers_marks_one_uncertain_boundary(self) -> None:
        answers = {
            name: {"decision": "accept", "boundary": index * 100, "confidence": 0.9, "reason": "ok"}
            for index, name in enumerate(MODULE.BOUNDARY_NAMES, 1)
        }
        answers["carry_end"] = {"decision": "uncertain", "boundary": None, "confidence": 0.2, "reason": "unclear"}

        combined = MODULE.combine_boundary_answers(answers)

        self.assertEqual(combined["decision"], "uncertain")
        self.assertIsNone(combined["boundaries"]["carry_end"])

    def test_uncertain_numeric_boundary_is_accepted_within_five_frames(self) -> None:
        answer = {"decision": "uncertain", "boundary": 162, "confidence": 0.45, "reason": "approximate"}

        self.assertTrue(MODULE.is_usable_boundary_answer(answer, [159, 163, 167], max_distance=5))
        normalized = MODULE.normalize_boundary_answer(answer, [159, 163, 167], max_distance=5)
        self.assertEqual(normalized["boundary"], 163)

    def test_only_uncertainty_without_a_frame_gets_a_second_boundary_attempt(self) -> None:
        self.assertTrue(MODULE.should_retry_visual_boundary({"decision": "uncertain", "boundary": None}))
        self.assertFalse(MODULE.should_retry_visual_boundary({"decision": "reject"}))
        self.assertFalse(MODULE.should_retry_visual_boundary({"decision": "accept"}))

    def test_snap_boundary_to_nearest_displayed_frame(self) -> None:
        snapped = MODULE.snap_boundary_to_displayed(162, [159, 163, 167], max_distance=4)

        self.assertEqual(snapped, 163)
        self.assertIsNone(MODULE.snap_boundary_to_displayed(170, [159, 163], max_distance=4))

    def test_validate_workers_requires_a_positive_count(self) -> None:
        self.assertEqual(MODULE.validate_workers(2), 2)
        with self.assertRaises(ValueError):
            MODULE.validate_workers(0)

    def test_first_nonincreasing_boundary_is_identified(self) -> None:
        self.assertEqual(MODULE.first_nonincreasing_boundary([100, 200, 200, 300]), "carry_end")
        self.assertIsNone(MODULE.first_nonincreasing_boundary([100, 200, 300, 400]))

    def test_gripper_guardrails_keep_grasp_and_carry_before_release(self) -> None:
        rows = [{"gripper": "100"} for _ in range(300)]
        rows[100:] = [{"gripper": "0"} for _ in range(200)]
        rows[200:] = [{"gripper": "100"} for _ in range(100)]
        action = {
            "coarse": {
                "approach": (0, 110),
                "grasp": (110, 160),
                "carry": (160, 190),
                "place": (190, 230),
            }
        }

        bounds = MODULE.gripper_phase_bounds(
            action, rows, threshold=50, debounce=5, event_grace_frames=15, min_carry_frames=30
        )

        self.assertEqual(bounds["approach_end"], (None, 115))
        self.assertEqual(bounds["grasp_end"], (100, 170))
        self.assertEqual(bounds["carry_end"], (130, 200))
        self.assertEqual(bounds["place_end"], (200, None))

    def test_repair_candidates_include_release_boundary_when_fixed_samples_miss_it(self) -> None:
        candidates = MODULE.repair_candidate_indices(
            sampled=[286, 290, 294, 298, 302, 306, 310, 314, 317],
            bounds=(249, 319),
            after_frame=318,
            frame_count=500,
            count=48,
        )

        self.assertEqual(candidates, [319])

    def test_grasp_fallback_keeps_the_original_local_window(self) -> None:
        args = type("Args", (), {"window_frames": 90, "fallback_window_frames": 180})()

        self.assertEqual(MODULE.fallback_window_for_boundary("grasp_end", args), 90)
        self.assertEqual(MODULE.fallback_window_for_boundary("carry_end", args), 90)
        self.assertEqual(MODULE.fallback_window_for_boundary("place_end", args), 180)

    def test_full_task_approach_ends_no_later_than_gripper_close(self) -> None:
        rows = [{"gripper": "100"} for _ in range(300)]
        rows[100:] = [{"gripper": "0"} for _ in range(200)]
        rows[200:] = [{"gripper": "100"} for _ in range(100)]
        action = {"coarse": {"approach": (0, 110), "grasp": (110, 160), "carry": (160, 190), "place": (190, 230)}}

        bounds = MODULE.gripper_phase_bounds(
            action, rows, threshold=50, debounce=5, event_grace_frames=15, min_carry_frames=30, full_task=True
        )

        self.assertEqual(bounds["approach_end"], (None, 100))

    def test_move_end_prompt_uses_first_entry_into_open_gripper_space(self) -> None:
        definition = MODULE.boundary_definition("approach_end", full_task=True)

        self.assertIn("FIRST", definition)
        self.assertIn("first enters the still-open space between the two gripper fingers", definition)

    def test_full_move_end_message_uses_wrist_only_precise_alignment_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "full" / "drag_episode_test"
            episode.mkdir(parents=True)
            cv2.imwrite(str(episode / "front.jpg"), np.full((12, 16, 3), (0, 0, 255), dtype=np.uint8))
            cv2.imwrite(str(episode / "wrist.jpg"), np.full((12, 16, 3), (0, 255, 0), dtype=np.uint8))
            rows = [{"front_rgb": "front.jpg", "wrist_rgb": "wrist.jpg"} for _ in range(400)]
            action = {
                "episode": str(episode),
                "color": "green",
                "coarse": {
                    "approach": (0, 100),
                    "grasp": (100, 200),
                    "carry": (200, 300),
                    "place": (300, 350),
                },
            }
            args = type(
                "Args",
                (),
                {
                    "window_frames": 90,
                    "sample_frames_per_boundary": 4,
                    "approach_wrist_sample_frames": 9,
                    "full_pre_grasp_window_frames": 90,
                    "full_post_release_window_frames": 90,
                    "full_event_context_frames": 15,
                    "panel_width": 32,
                    "jpeg_quality": 95,
                    "min_approach_frames": 30,
                    "min_grasp_frames": 30,
                    "min_carry_frames": 30,
                    "min_place_frames": 30,
                },
            )()

            messages, allowed = MODULE.build_messages(
                action,
                rows,
                args,
                boundary_names=("approach_end",),
                full_cycle=(120, 300),
            )

        content = messages["approach_end"][0]["content"]
        prompt = content[0]["text"]
        image_url = next(item["image_url"]["url"] for item in content if item["type"] == "image_url")
        encoded = base64.b64decode(image_url.split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIn("first enters the still-open space between the two gripper fingers", prompt)
        self.assertNotIn("last clear pre-contact frame before the final descent", prompt)
        self.assertEqual(len(allowed["approach_end"]), 12)
        self.assertEqual(image.shape[1], 32)
        self.assertGreater(int(image[6, 16, 1]), int(image[6, 16, 2]))

    def test_full_task_cycle_maps_green_to_the_second_close_open_pair(self) -> None:
        rows = ([{"gripper": "100"}] * 10 + [{"gripper": "0"}] * 10 + [{"gripper": "100"}] * 10) * 3
        action = {"episode": "/tmp/full/episode", "color": "green"}

        cycle = MODULE.full_task_cycle(action, rows, threshold=50, debounce=1)

        self.assertEqual(cycle, (40, 50))

    def test_full_approach_candidates_include_close_context(self) -> None:
        indices = MODULE.full_boundary_candidate_indices(
            "approach_end",
            close=300,
            release=450,
            frame_count=1000,
            pre_grasp_window=180,
            post_release_window=180,
            event_context_frames=15,
            min_grasp_frames=30,
            min_carry_frames=30,
            count=48,
        )

        self.assertEqual(indices[0], 120)
        self.assertIn(300, indices)
        self.assertIn(295, indices)
        self.assertIn(290, indices)
        self.assertIn(285, indices)

    def test_first_too_short_phase_rejects_six_frame_carry(self) -> None:
        boundaries = [768, 974, 980, 1032]

        phase = MODULE.first_too_short_phase(
            prior_end=614,
            boundaries=boundaries,
            minimum_frames={"approach": 30, "grasp": 30, "carry": 30, "place": 30},
        )

        self.assertEqual(phase, "carry")

    def test_make_segments_contiguous_assigns_between_adjacent_colors_to_next_approach(self) -> None:
        red = [
            {"episode": "/tmp/e", "color": "red", "phase": "approach", "start": 0, "end": 10},
            {"episode": "/tmp/e", "color": "red", "phase": "grasp", "start": 10, "end": 20},
            {"episode": "/tmp/e", "color": "red", "phase": "carry", "start": 20, "end": 30},
            {"episode": "/tmp/e", "color": "red", "phase": "place", "start": 30, "end": 40},
        ]
        green = [
            {"episode": "/tmp/e", "color": "green", "phase": "approach", "start": 55, "end": 70},
            {"episode": "/tmp/e", "color": "green", "phase": "grasp", "start": 70, "end": 80},
            {"episode": "/tmp/e", "color": "green", "phase": "carry", "start": 80, "end": 90},
            {"episode": "/tmp/e", "color": "green", "phase": "place", "start": 90, "end": 100},
        ]

        contiguous = MODULE.make_segments_contiguous([red, green])

        self.assertEqual(contiguous[4]["start"], 40)
        self.assertTrue(contiguous[4]["continuity_adjusted_start"])

    def test_post_place_tail_extends_full_task_and_adds_return_home(self) -> None:
        red = [
            {"episode": "/tmp/full/episode", "color": "red", "phase": "approach", "start": 0, "end": 10},
            {"episode": "/tmp/full/episode", "color": "red", "phase": "grasp", "start": 10, "end": 20},
            {"episode": "/tmp/full/episode", "color": "red", "phase": "carry", "start": 20, "end": 30},
            {"episode": "/tmp/full/episode", "color": "red", "phase": "place", "start": 30, "end": 40},
        ]
        green = [
            {"episode": "/tmp/full/episode", "color": "green", "phase": "approach", "start": 55, "end": 130},
            {"episode": "/tmp/full/episode", "color": "green", "phase": "grasp", "start": 130, "end": 140},
            {"episode": "/tmp/full/episode", "color": "green", "phase": "carry", "start": 140, "end": 150},
            {"episode": "/tmp/full/episode", "color": "green", "phase": "place", "start": 150, "end": 160},
        ]
        yellow = [
            {"episode": "/tmp/full/episode", "color": "yellow", "phase": "approach", "start": 175, "end": 250},
            {"episode": "/tmp/full/episode", "color": "yellow", "phase": "grasp", "start": 250, "end": 260},
            {"episode": "/tmp/full/episode", "color": "yellow", "phase": "carry", "start": 260, "end": 270},
            {"episode": "/tmp/full/episode", "color": "yellow", "phase": "place", "start": 270, "end": 280},
        ]

        contiguous = MODULE.make_segments_contiguous(
            [red, green, yellow],
            frame_counts={"/tmp/full/episode": 400},
            post_place_tail_frames=60,
        )

        self.assertEqual(contiguous[3]["end"], 40)
        self.assertEqual(contiguous[4]["start"], 40)
        self.assertEqual(contiguous[7]["end"], 160)
        self.assertEqual(contiguous[8]["start"], 160)
        self.assertEqual(contiguous[11]["end"], 310)
        self.assertEqual(contiguous[12]["phase"], "return_home")
        self.assertEqual(contiguous[12]["task"], "return the gripper to the home pose after completing the task")
        self.assertEqual((contiguous[12]["start"], contiguous[12]["end"]), (310, 400))

    def test_post_place_tail_caps_at_remaining_standalone_frames(self) -> None:
        red = [
            {"episode": "/tmp/red/episode", "color": "red", "phase": "approach", "start": 0, "end": 10},
            {"episode": "/tmp/red/episode", "color": "red", "phase": "grasp", "start": 10, "end": 20},
            {"episode": "/tmp/red/episode", "color": "red", "phase": "carry", "start": 20, "end": 30},
            {"episode": "/tmp/red/episode", "color": "red", "phase": "place", "start": 30, "end": 40},
        ]

        contiguous = MODULE.make_segments_contiguous(
            [red],
            frame_counts={"/tmp/red/episode": 75},
            post_place_tail_frames=60,
        )

        self.assertEqual(contiguous[-1]["end"], 75)
        self.assertEqual(len(contiguous), 4)

    def test_rebuild_restores_qwen_boundaries_before_adding_a_tail(self) -> None:
        segments = [
            {"episode": "/tmp/red/episode", "color": "red", "phase": "approach", "start": 0, "end": 10},
            {"episode": "/tmp/red/episode", "color": "red", "phase": "grasp", "start": 10, "end": 20},
            {"episode": "/tmp/red/episode", "color": "red", "phase": "carry", "start": 20, "end": 30},
            {
                "episode": "/tmp/red/episode",
                "color": "red",
                "phase": "place",
                "start": 30,
                "end": 160,
                "post_place_tail_frames": 60,
            },
        ]
        report = [
            {
                "decision": "accept",
                "episode": "/tmp/red/episode",
                "color": "red",
                "coarse": {"approach": (0, 10)},
                "answer": {"boundaries": {"approach_end": 10, "grasp_end": 20, "carry_end": 30, "place_end": 40}},
                "segments": segments,
            }
        ]

        rebuilt = MODULE.rebuild_contiguous_report(
            report,
            frame_counts={"/tmp/red/episode": 300},
            post_place_tail_frames=60,
        )

        self.assertEqual(rebuilt[-1]["end"], 100)
        self.assertEqual(report[0]["segments"][-1]["end"], 100)

    def test_rebuild_contiguous_report_restores_original_start_after_missing_color(self) -> None:
        red = [
            {"episode": "/tmp/full", "color": "red", "phase": "approach", "start": 0, "end": 10},
            {"episode": "/tmp/full", "color": "red", "phase": "grasp", "start": 10, "end": 20},
            {"episode": "/tmp/full", "color": "red", "phase": "carry", "start": 20, "end": 30},
            {"episode": "/tmp/full", "color": "red", "phase": "place", "start": 30, "end": 40},
        ]
        yellow = [
            {"episode": "/tmp/full", "color": "yellow", "phase": "approach", "start": 40, "end": 70},
            {"episode": "/tmp/full", "color": "yellow", "phase": "grasp", "start": 70, "end": 80},
            {"episode": "/tmp/full", "color": "yellow", "phase": "carry", "start": 80, "end": 90},
            {"episode": "/tmp/full", "color": "yellow", "phase": "place", "start": 90, "end": 100},
        ]
        report = [
            {"decision": "accept", "episode": "/tmp/full", "color": "red", "coarse": {"approach": (0, 10)}, "segments": red},
            {
                "decision": "accept",
                "episode": "/tmp/full",
                "color": "yellow",
                "coarse": {"approach": (55, 70)},
                "segments": yellow,
            },
        ]

        rebuilt = MODULE.rebuild_contiguous_report(report)

        self.assertEqual(rebuilt[4]["start"], 55)
        self.assertEqual(report[1]["segments"][0]["start"], 55)

    def test_dense_indices_cover_window_and_requested_count(self) -> None:
        indices = MODULE.dense_indices(center=100, frame_count=500, window_frames=90, count=48)

        self.assertEqual(indices[0], 10)
        self.assertEqual(indices[-1], 190)
        self.assertEqual(len(indices), 48)
        self.assertEqual(indices, sorted(set(indices)))


    def test_make_refined_segments_requires_strict_boundaries(self) -> None:
        coarse = {"approach": (0, 100), "grasp": (100, 200), "carry": (200, 300), "place": (300, 400)}
        answer = {
            "decision": "accept",
            "boundaries": {"approach_end": 95, "grasp_end": 205, "carry_end": 310, "place_end": 390},
        }

        segments, errors = MODULE.make_refined_segments(
            episode=Path("/tmp/episode"), color="red", frame_count=500, prior_end=0, coarse=coarse, answer=answer
        )

        self.assertFalse(errors)
        self.assertEqual(
            [(segment["phase"], segment["start"], segment["end"]) for segment in segments],
            [("approach", 0, 95), ("grasp", 95, 205), ("carry", 205, 310), ("place", 310, 390)],
        )


    def test_make_refined_segments_rejects_out_of_order_boundaries(self) -> None:
        coarse = {"approach": (0, 100), "grasp": (100, 200), "carry": (200, 300), "place": (300, 400)}
        answer = {
            "decision": "accept",
            "boundaries": {"approach_end": 100, "grasp_end": 90, "carry_end": 300, "place_end": 400},
        }

        segments, errors = MODULE.make_refined_segments(
            episode=Path("/tmp/episode"), color="red", frame_count=500, prior_end=0, coarse=coarse, answer=answer
        )

        self.assertFalse(segments)
        self.assertEqual(errors, ["invalid_boundary_order:[100, 90, 300, 400]"])

    def test_actions_from_report_keeps_equal_coarse_boundaries_for_refinement(self) -> None:
        report = {
            "episode": "/tmp/episode",
            "answer": {
                "segments": [
                    {
                        "color": "green",
                        "approach_end": 100,
                        "grasp_end": 150,
                        "carry_end": 150,
                        "place_end": 220,
                    }
                ]
            },
        }

        actions = MODULE.actions_from_report([report])

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["coarse"]["carry"], (150, 150))
