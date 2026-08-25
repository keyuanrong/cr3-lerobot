from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "datasets" / "build_pi0_key_event_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_pi0_key_event_manifests", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_phase_rows(source: Path, color: str) -> list[dict[str, object]]:
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


def gripper_rows(*, close_frame: int | None = 150, open_frame: int | None = 300) -> list[dict[str, str]]:
    rows = []
    for frame in range(400):
        value = 100.0
        if close_frame is not None and frame >= close_frame:
            value = 0.0
        if open_frame is not None and frame >= open_frame:
            value = 100.0
        rows.append({"gripper": str(value)})
    return rows


class TestKeyEventManifestBuilder(unittest.TestCase):
    def test_generates_grasp_lift_from_close_and_visual_lift(self) -> None:
        source = Path("/tmp/data/red/drag_episode_red")
        manifests, rejected = MODULE.build_event_manifests(
            [source],
            make_phase_rows(source, "red"),
            {source: 400},
            {source: gripper_rows()},
            close_pre_frames=10,
            lift_post_frames=10,
            open_pre_frames=10,
            release_post_frames=10,
            threshold=50.0,
            debounce=3,
        )

        row = manifests["grasp_lift_event"][0]
        self.assertEqual((row["start"], row["end"]), (140, 230))
        self.assertEqual(row["anchors"]["stable_close_frame"], 150)
        self.assertEqual(row["anchors"]["refined_grasp_end"], 220)
        self.assertEqual(row["task"], MODULE.TOTAL_GOALS["red"])
        self.assertFalse(rejected)

    def test_full_release_stops_at_place_end(self) -> None:
        source = Path("/tmp/data/full/drag_episode_full")
        manifests, rejected = MODULE.build_event_manifests(
            [source],
            make_phase_rows(source, "red"),
            {source: 400},
            {source: gripper_rows()},
            close_pre_frames=10,
            lift_post_frames=10,
            open_pre_frames=10,
            release_post_frames=10,
            threshold=50.0,
            debounce=3,
        )

        row = manifests["release_event"][0]
        self.assertEqual((row["start"], row["end"]), (290, 320))
        self.assertEqual(row["anchors"]["stable_open_frame"], 300)
        self.assertEqual(row["anchors"]["refined_place_end"], 320)
        self.assertFalse(rejected)

    def test_rejects_a_task_without_a_stable_close(self) -> None:
        source = Path("/tmp/data/green/drag_episode_green")
        manifests, rejected = MODULE.build_event_manifests(
            [source],
            make_phase_rows(source, "green"),
            {source: 400},
            {source: gripper_rows(close_frame=None, open_frame=None)},
            close_pre_frames=10,
            lift_post_frames=10,
            open_pre_frames=10,
            release_post_frames=10,
            threshold=50.0,
            debounce=3,
        )

        self.assertEqual(manifests["grasp_lift_event"], [])
        self.assertEqual(manifests["release_event"], [])
        self.assertEqual(rejected["missing_stable_close"], 1)


if __name__ == "__main__":
    unittest.main()
