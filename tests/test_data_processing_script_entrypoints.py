import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DIRECT_SCRIPTS = (
    "scripts/data_processing/filter_drag_dataset_with_qwen.py",
    "scripts/data_processing/segment_drag_trajectories_with_qwen.py",
    "scripts/data_processing/refine_drag_phase_boundaries_with_qwen.py",
    "scripts/data_processing/recover_relaxed_phase_boundaries_with_qwen.py",
    "scripts/data_processing/recover_drag_phase_boundaries_two_stage_qwen.py",
)
MODULE_NAMES = (
    "scripts.data_processing.filter_drag_dataset_with_qwen",
    "scripts.data_processing.segment_drag_trajectories_with_qwen",
    "scripts.data_processing.refine_drag_phase_boundaries_with_qwen",
    "scripts.data_processing.recover_relaxed_phase_boundaries_with_qwen",
    "scripts.data_processing.recover_drag_phase_boundaries_two_stage_qwen",
)


class TestDataProcessingScriptEntrypoints(unittest.TestCase):
    def run_help(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [PYTHON, *args, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_direct_script_help_works_for_packaged_entrypoints(self) -> None:
        failures = []
        for script in DIRECT_SCRIPTS:
            result = self.run_help(str(ROOT / script))
            if result.returncode != 0:
                failures.append(f"{script}: rc={result.returncode}, stderr={result.stderr.strip()}")
        self.assertFalse(failures, "\n".join(failures))

    def test_module_help_works_for_packaged_entrypoints(self) -> None:
        failures = []
        for module in MODULE_NAMES:
            result = self.run_help("-m", module)
            if result.returncode != 0:
                failures.append(f"{module}: rc={result.returncode}, stderr={result.stderr.strip()}")
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
