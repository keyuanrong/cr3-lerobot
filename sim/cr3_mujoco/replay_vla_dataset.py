#!/usr/bin/env python

"""Replay CR3 + LMG-90 LeRobotDataset recordings.

Controls:
  Space: pause / resume
  Left/Right or A/D: step backward / forward while paused
  B/N: previous / next episode
  -/=: slower / faster playback
  S: save current combined frame as PNG
  Q or Esc: quit
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(LEROBOT_ROOT))

os.environ.setdefault("HF_HOME", str(Path("/tmp") / "lerobot_hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path("/tmp") / "lerobot_hf_datasets_cache"))

from lerobot.datasets import LeRobotDataset


DEFAULT_REPO_ID = "local/cr3_lmg90_stack_blocks_vla_teleop_clean"
DEFAULT_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla_teleop_clean"
FRONT_KEY = "observation.images.front_rgb"
WRIST_KEY = "observation.images.wrist_rgb"


def image_to_bgr_uint8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_label(image: np.ndarray, text: str, y: int = 30) -> None:
    cv2.rectangle(image, (0, y - 24), (image.shape[1], y + 8), (0, 0, 0), -1)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_info(image: np.ndarray, lines: list[str]) -> None:
    line_h = 26
    height = line_h * len(lines) + 14
    cv2.rectangle(image, (0, image.shape[0] - height), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
    y = image.shape[0] - height + 30
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
        y += line_h


def episode_rows(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata found under {root / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).sort_values("episode_index")


def task_text(dataset: LeRobotDataset, sample: dict) -> str:
    if "task" in sample:
        return str(sample["task"])
    task_index = int(sample.get("task_index", 0))
    try:
        tasks = dataset.meta.tasks
        if hasattr(tasks, "to_pandas"):
            table = tasks.to_pandas()
            if "task" in table.columns:
                row = table[table["task_index"] == task_index]
                if len(row):
                    return str(row.iloc[0]["task"])
    except Exception:
        pass
    return f"task_index={task_index}"


def make_panel(sample: dict, *, episode: int, frame_in_episode: int, episode_len: int, fps: int, paused: bool, speed: float) -> np.ndarray:
    front = image_to_bgr_uint8(sample[FRONT_KEY])
    wrist = image_to_bgr_uint8(sample[WRIST_KEY])
    wrist = fit_height(wrist, front.shape[0])

    draw_label(front, "front")
    draw_label(wrist, "wrist")
    panel = np.concatenate([front, wrist], axis=1)

    state = np.asarray(sample["observation.state"], dtype=np.float32)
    action = np.asarray(sample["action"], dtype=np.float32)
    task = str(sample.get("task", ""))
    if len(task) > 105:
        task = task[:102] + "..."
    info = [
        f"episode {episode} | frame {frame_in_episode + 1}/{episode_len} | dataset index {int(sample['index'])} | fps {fps} | speed {speed:.2f}x | {'PAUSED' if paused else 'PLAY'}",
        f"state gripper={state[-1]:.3f} | action gripper={action[-1]:.3f} | time={float(sample['timestamp']):.3f}s",
        f"task: {task}",
        "keys: Space pause | A/D or Left/Right step | B/N episode | -/= speed | S screenshot | Q quit",
    ]
    draw_info(panel, info)
    return panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--window-name", default="CR3 VLA Dataset Replay")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    episodes = episode_rows(args.root)
    episode_indices = [int(x) for x in episodes["episode_index"].tolist()]
    if args.episode not in episode_indices:
        raise ValueError(f"Episode {args.episode} not found. Available episodes: {episode_indices}")

    episode_cursor = episode_indices.index(args.episode)
    frame_in_episode = max(0, int(args.start_frame))
    paused = False
    speed = max(float(args.speed), 0.05)
    last_frame_time = 0.0

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    print("CR3 VLA dataset replay")
    print(f"root: {args.root}")
    print(f"episodes: {episode_indices}")
    print("keys: Space pause | A/D step | B/N episode | -/= speed | S screenshot | Q quit")

    while True:
        row = episodes.iloc[episode_cursor]
        episode = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        episode_len = end - start
        frame_in_episode = int(np.clip(frame_in_episode, 0, max(episode_len - 1, 0)))
        sample = dataset[start + frame_in_episode]
        if "task" not in sample:
            sample["task"] = task_text(dataset, sample)

        panel = make_panel(
            sample,
            episode=episode,
            frame_in_episode=frame_in_episode,
            episode_len=episode_len,
            fps=int(dataset.fps),
            paused=paused,
            speed=speed,
        )
        cv2.imshow(args.window_name, panel)

        delay_ms = max(1, int(round(1000.0 / max(float(dataset.fps) * speed, 1e-6))))
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("d"), 83):
            frame_in_episode += 1
            paused = True
        elif key in (ord("a"), 81):
            frame_in_episode -= 1
            paused = True
        elif key == ord("n"):
            episode_cursor = min(episode_cursor + 1, len(episode_indices) - 1)
            frame_in_episode = 0
        elif key == ord("b"):
            episode_cursor = max(episode_cursor - 1, 0)
            frame_in_episode = 0
        elif key in (ord("-"), ord("_")):
            speed = max(0.1, speed / 1.25)
        elif key in (ord("="), ord("+")):
            speed = min(8.0, speed * 1.25)
        elif key == ord("s"):
            out = args.root / f"replay_episode_{episode:06d}_frame_{frame_in_episode:06d}.png"
            cv2.imwrite(str(out), panel)
            print(f"saved screenshot: {out}")
        elif not paused:
            now = time.perf_counter()
            if now >= last_frame_time:
                frame_in_episode += 1
                last_frame_time = now

        if frame_in_episode >= episode_len:
            frame_in_episode = episode_len - 1
            paused = True
        elif frame_in_episode < 0:
            frame_in_episode = 0
            paused = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
