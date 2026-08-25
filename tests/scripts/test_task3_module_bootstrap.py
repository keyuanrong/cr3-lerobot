from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTask3ModuleBootstrap(unittest.TestCase):
    def run_isolated_import(self, module_name: str) -> subprocess.CompletedProcess[str]:
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        code = f"""
import importlib
import os
import pathlib
import sys

repo_root = pathlib.Path.cwd().resolve()
repo_src = (repo_root / "src").resolve()
cleaned = []
for entry in sys.path:
    base = repo_root if entry == "" else pathlib.Path(entry)
    try:
        resolved = base.resolve()
    except OSError:
        cleaned.append(entry)
        continue
    if resolved in {{repo_root, repo_src}}:
        continue
    cleaned.append(entry)
sys.path[:] = cleaned
sys.path.insert(0, os.getcwd())
importlib.import_module({module_name!r})
src = str(repo_src)
normalized = []
for entry in sys.path:
    base = repo_root if entry == "" else pathlib.Path(entry)
    try:
        normalized.append(str(base.resolve()))
    except OSError:
        normalized.append(str(base))
print(src in normalized)
"""
        return subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_modules_add_repo_src_without_pythonpath(self) -> None:
        failures: list[str] = []
        for module_name in (
            "scripts.datasets.convert_drag_to_lerobot",
            "scripts.evaluation.audit_pi0_unified_goal_sampling",
            "scripts.evaluation.eval_remote_pi0_replay",
        ):
            result = self.run_isolated_import(module_name)
            if result.returncode != 0 or result.stdout.strip() != "True":
                failures.append(
                    f"{module_name}: rc={result.returncode}, stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
                )
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
