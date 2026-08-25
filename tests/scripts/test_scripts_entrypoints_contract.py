"""Contract tests for relocated scripts package entrypoints."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptEntrypointContractTest(unittest.TestCase):
    def run_python(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        pythonpath_entries = [str(REPO_ROOT / "src")]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        return subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_relocated_entrypoints_support_module_and_script_help(self) -> None:
        """Catch relocation regressions where CLI bootstrap breaks but imports still work."""
        cases = [
            (
                "scripts.collection.record_drag_dataset",
                REPO_ROOT / "scripts" / "collection" / "record_drag_dataset.py",
            ),
            (
                "scripts.inference.run_act_policy",
                REPO_ROOT / "scripts" / "inference" / "run_act_policy.py",
            ),
            (
                "scripts.inference.run_remote_pi0_policy",
                REPO_ROOT / "scripts" / "inference" / "run_remote_pi0_policy.py",
            ),
            (
                "scripts.inference.pi0_remote_policy_server",
                REPO_ROOT / "scripts" / "inference" / "pi0_remote_policy_server.py",
            ),
            (
                "scripts.diagnostics.diagnose_cr3_servoj",
                REPO_ROOT / "scripts" / "diagnostics" / "diagnose_cr3_servoj.py",
            ),
        ]
        for module_name, script_path in cases:
            with self.subTest(mode="module", entrypoint=module_name):
                result = self.run_python("-m", module_name, "--help")
                self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
                self.assertIn("usage:", result.stdout.lower())
            with self.subTest(mode="script", entrypoint=str(script_path)):
                result = self.run_python(str(script_path), "--help")
                self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
                self.assertIn("usage:", result.stdout.lower())

    def test_relocated_modules_import_without_running_main(self) -> None:
        for module_name in (
            "scripts.inference.pi0_remote_policy_server",
            "scripts.inference.run_act_policy",
            "scripts.inference.run_remote_pi0_policy",
            "scripts.diagnostics.diagnose_cr3_servoj",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module.__name__, module_name)


if __name__ == "__main__":
    unittest.main()
