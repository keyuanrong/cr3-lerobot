from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluation" / "review_grasp_transition_comparison.py"
SPEC = importlib.util.spec_from_file_location("review_grasp_transition_comparison", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestGraspTransitionComparison(unittest.TestCase):
    def test_selects_the_containing_wide_grasp_clip_for_a_transition(self) -> None:
        transition = {
            "episode": "/tmp/data/red/drag_episode_red",
            "color": "red",
            "start": 120,
            "end": 230,
            "anchors": {"transition_type": "closed_open_close_lift"},
        }
        wide = {
            "episode": "/tmp/data/red/drag_episode_red",
            "color": "red",
            "start": 40,
            "end": 250,
        }
        too_short = wide | {"start": 130, "end": 240}

        selected = MODULE.match_wide_segment(transition, [too_short, wide])

        self.assertEqual(selected, wide)

    def test_rejects_a_transition_without_a_containing_wide_clip(self) -> None:
        transition = {
            "episode": "/tmp/data/red/drag_episode_red",
            "color": "red",
            "start": 120,
            "end": 230,
            "anchors": {"transition_type": "closed_open_close_lift"},
        }
        wide = transition | {"start": 130, "end": 240}

        with self.assertRaisesRegex(ValueError, "No containing goal_grasp_event"):
            MODULE.match_wide_segment(transition, [wide])


if __name__ == "__main__":
    unittest.main()
