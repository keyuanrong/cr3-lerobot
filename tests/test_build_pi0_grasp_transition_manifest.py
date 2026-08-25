from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "datasets" / "build_pi0_grasp_transition_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_pi0_grasp_transition_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_phase_rows(source: Path, color: str = "red") -> list[dict[str, object]]:
    boundaries = {
        "approach_end": 100,
        "grasp_end": 220,
        "carry_end": 260,
        "place_end": 320,
    }
    return [
        {
            "episode": str(source),
            "color": color,
            "phase": phase,
            "start": start,
            "end": end,
            "refined_boundaries": boundaries,
        }
        for phase, start, end in (
            ("approach", 0, 100),
            ("grasp", 100, 220),
            ("carry", 220, 260),
            ("place", 260, 330),
        )
    ]


def gripper_rows(*, initial_open: bool, open_frame: int | None, close_frame: int) -> list[dict[str, str]]:
    rows = []
    for frame in range(400):
        is_open = initial_open
        if open_frame is not None and frame >= open_frame:
            is_open = True
        if frame >= close_frame:
            is_open = False
        rows.append({"gripper": "100" if is_open else "0"})
    return rows


class TestGraspTransitionManifestBuilder(unittest.TestCase):
    def test_keeps_closed_open_close_lift_transition(self) -> None:
        source = Path("/tmp/data/red/drag_episode_red")
        rows, rejected = MODULE.build_transition_manifest(
            [source],
            make_phase_rows(source),
            {source: 400},
            {source: gripper_rows(initial_open=False, open_frame=120, close_frame=160)},
            context_pre_frames=60,
            open_pre_frames=30,
            close_pre_frames=60,
            lift_post_frames=10,
            threshold=50.0,
            debounce=3,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["start"], row["end"]), (90, 230))
        self.assertEqual(row["task"], MODULE.TOTAL_GOALS["red"])
        self.assertEqual(row["sample_type"], "grasp_transition_event")
        self.assertEqual(row["anchors"]["transition_type"], "closed_open_close_lift")
        self.assertEqual(row["anchors"]["stable_open_frame"], 120)
        self.assertEqual(row["anchors"]["stable_close_frame"], 160)
        self.assertFalse(rejected)

    def test_keeps_open_close_lift_transition(self) -> None:
        source = Path("/tmp/data/green/drag_episode_green")
        rows, rejected = MODULE.build_transition_manifest(
            [source],
            make_phase_rows(source, "green"),
            {source: 400},
            {source: gripper_rows(initial_open=True, open_frame=None, close_frame=160)},
            context_pre_frames=60,
            open_pre_frames=30,
            close_pre_frames=60,
            lift_post_frames=10,
            threshold=50.0,
            debounce=3,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["start"], row["end"]), (100, 230))
        self.assertEqual(row["anchors"]["transition_type"], "open_close_lift")
        self.assertEqual(row["anchors"]["stable_close_frame"], 160)
        self.assertFalse(rejected)

    def test_rejects_closed_context_that_never_opens_before_closing(self) -> None:
        source = Path("/tmp/data/yellow/drag_episode_yellow")
        rows, rejected = MODULE.build_transition_manifest(
            [source],
            make_phase_rows(source, "yellow"),
            {source: 400},
            {source: gripper_rows(initial_open=False, open_frame=None, close_frame=160)},
            context_pre_frames=60,
            open_pre_frames=30,
            close_pre_frames=60,
            lift_post_frames=10,
            threshold=50.0,
            debounce=3,
        )

        self.assertEqual(rows, [])
        self.assertEqual(rejected["missing_stable_open_after_closed"], 1)


if __name__ == "__main__":
    unittest.main()
