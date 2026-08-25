from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluation" / "audit_pi0_unified_goal_sampling.py"
SPEC = importlib.util.spec_from_file_location("audit_pi0_unified_goal_sampling", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestUnifiedGoalSamplingAuditDefaults(unittest.TestCase):
    def test_defaults_follow_the_grasp_transition_training_order(self) -> None:
        self.assertEqual(
            MODULE.DEFAULT_REPO_IDS,
            [
                "local/cr3_pi0_unified_goal_v1_complete_goal",
                "local/cr3_pi0_unified_goal_v3_grasp_transition_event",
                "local/cr3_pi0_unified_goal_v1_goal_grasp_event",
                "local/cr3_pi0_unified_goal_v2_release_event",
                "local/cr3_pi0_unified_goal_v1_goal_place_event",
                "local/cr3_pi0_unified_goal_v1_atomic_assist",
            ],
        )


if __name__ == "__main__":
    unittest.main()
