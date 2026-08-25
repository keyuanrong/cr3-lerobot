from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "datasets" / "build_pi0_unified_goal_manifests.py"
SPEC = importlib.util.spec_from_file_location("build_pi0_unified_goal_manifests", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestUnifiedGoalManifestBuilder(unittest.TestCase):
    def test_events_use_total_goal_and_full_is_excluded_from_atomic_assist(self) -> None:
        red = Path("/tmp/data/red/drag_episode_red")
        full = Path("/tmp/data/full/drag_episode_full")
        phase_rows = [
            {"episode": str(red), "color": "red", "phase": phase, "start": start, "end": end, "task": phase}
            for phase, start, end in (("approach", 0, 100), ("grasp", 100, 200), ("carry", 200, 260), ("place", 260, 320))
        ] + [
            {"episode": str(full), "color": "green", "phase": phase, "start": start, "end": end, "task": phase}
            for phase, start, end in (("approach", 0, 100), ("grasp", 100, 200), ("carry", 200, 260), ("place", 260, 320))
        ]

        manifests = MODULE.build_manifests(
            [red, full], phase_rows, {red: 400, full: 400}, 60, 45, 60, 60
        )

        red_grasp = next(row for row in manifests["goal_grasp_event"] if Path(row["episode"]) == red)
        full_place = next(row for row in manifests["goal_place_event"] if Path(row["episode"]) == full)
        self.assertEqual(red_grasp["task"], MODULE.TOTAL_GOALS["red"])
        self.assertEqual(full_place["task"], MODULE.TOTAL_GOALS["full"])
        self.assertTrue(all(Path(row["episode"]).parent.name != "full" for row in manifests["atomic_assist"]))

    def test_all_generated_rows_are_bounded_by_the_source_episode(self) -> None:
        red = Path("/tmp/data/red/drag_episode_red")
        phase_rows = [
            {"episode": str(red), "color": "red", "phase": phase, "start": start, "end": end, "task": phase}
            for phase, start, end in (("approach", 0, 10), ("grasp", 10, 20), ("carry", 20, 25), ("place", 25, 30))
        ]

        manifests = MODULE.build_manifests([red], phase_rows, {red: 35}, 60, 45, 60, 60)

        for rows in manifests.values():
            for row in rows:
                self.assertGreaterEqual(row["start"], 0)
                self.assertLess(row["start"], row["end"])
                self.assertLessEqual(row["end"], 35)


if __name__ == "__main__":
    unittest.main()
