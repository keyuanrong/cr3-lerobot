from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "datasets" / "convert_drag_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("convert_drag_to_lerobot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Dataset:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved = False

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved = True


class TestConvertEpisode(unittest.TestCase):
    def test_returns_the_converted_segment_length_not_the_raw_episode_length(self) -> None:
        dataset = _Dataset()
        rows = [{} for _ in range(10)]

        with (
            patch.object(MODULE, "read_rows", return_value=rows),
            patch.object(MODULE, "row_state", return_value=[0.0]),
            patch.object(MODULE, "row_action", return_value=[0.0]),
            patch.object(MODULE, "clean_task_text", return_value="task"),
        ):
            converted = MODULE.convert_episode(
                cv2=None,
                dataset=dataset,
                episode_dir=Path("/tmp/episode"),
                image_keys={},
                task="task",
                task_source="argument",
                drop_last_frame=False,
                gripper_threshold=50.0,
                gripper_action_semantics="close_high",
                image_columns=[],
                start_frame=3,
                end_frame=7,
            )

        self.assertEqual(converted, 4)
        self.assertEqual(len(dataset.frames), 4)
        self.assertTrue(dataset.saved)


if __name__ == "__main__":
    unittest.main()
