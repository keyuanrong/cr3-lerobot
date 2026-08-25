from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "datasets" / "export_pi0_unified_goal_review_samples.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("export_pi0_unified_goal_review_samples", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExportPi0UnifiedGoalReviewSamples(unittest.TestCase):
    def test_groups_rows_by_phase_and_target_with_distinct_episodes(self) -> None:
        module = load_script_module()
        rows = {
            "complete_goal": [
                {"episode": "/raw/red/one", "start": 0, "end": 20, "task": "put the red block into the black frame"},
                {"episode": "/raw/red/two", "start": 0, "end": 20, "task": "put the red block into the black frame"},
            ],
            "goal_grasp_event": [
                {
                    "episode": "/raw/full/one",
                    "start": 20,
                    "end": 40,
                    "task": "put the red block into the black frame, then put the green block into the black frame, then put the yellow block into the black frame",
                },
                {
                    "episode": "/raw/full/two",
                    "start": 40,
                    "end": 60,
                    "task": "put the red block into the black frame, then put the green block into the black frame, then put the yellow block into the black frame",
                },
            ],
        }

        selected = module.select_review_rows(rows, samples_per_bucket=2, seed=7)

        self.assertEqual([row["review_bucket"] for row in selected["complete/red"]], ["complete/red"] * 2)
        self.assertEqual([row["review_bucket"] for row in selected["grasp/full"]], ["grasp/full"] * 2)
        self.assertEqual(
            len({row["episode"] for row in selected["grasp/full"]}),
            2,
        )


if __name__ == "__main__":
    unittest.main()
