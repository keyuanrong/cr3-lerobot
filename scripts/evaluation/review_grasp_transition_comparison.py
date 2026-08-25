#!/usr/bin/env python
"""Play a wide goal-grasp clip beside its matching grasp-transition crop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def match_wide_segment(
    transition: dict[str, Any], wide_segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return the shortest same-task wide grasp clip that contains ``transition``."""
    candidates = [
        segment
        for segment in wide_segments
        if segment.get("episode") == transition.get("episode")
        and segment.get("color") == transition.get("color")
        and int(segment["start"]) <= int(transition["start"])
        and int(transition["end"]) <= int(segment["end"])
    ]
    if not candidates:
        raise ValueError(
            "No containing goal_grasp_event for "
            f"{transition.get('episode')} / {transition.get('color')} "
            f"[{transition.get('start')}, {transition.get('end')})."
        )
    return min(candidates, key=lambda segment: int(segment["end"]) - int(segment["start"]))


def select_a_transition(transitions: list[dict[str, Any]], number: int) -> dict[str, Any]:
    a_rows = [
        segment
        for segment in transitions
        if segment.get("anchors", {}).get("transition_type") == "closed_open_close_lift"
    ]
    if not a_rows:
        raise ValueError("No A-class closed_open_close_lift clips were found.")
    a_rows.sort(key=lambda segment: (str(segment["episode"]), int(segment["start"])))
    if not 0 <= number < len(a_rows):
        raise ValueError(f"--number must be in [0, {len(a_rows) - 1}] for A-class clips.")
    return a_rows[number]


def load_image(cv2: Any, episode: Path, row: dict[str, str], column: str) -> np.ndarray:
    image = cv2.imread(str(episode / row[column]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(episode / row[column])
    return image


def draw_panel(
    cv2: Any,
    episode: Path,
    rows: list[dict[str, str]],
    segment: dict[str, Any],
    offset: int,
    title: str,
    scale: float,
) -> np.ndarray:
    start, end = int(segment["start"]), int(segment["end"])
    raw_frame = min(start + offset, end - 1)
    front = load_image(cv2, episode, rows[raw_frame], "front_rgb")
    wrist = load_image(cv2, episode, rows[raw_frame], "wrist_rgb")
    height = min(front.shape[0], wrist.shape[0])
    front = cv2.resize(front, (round(front.shape[1] * height / front.shape[0]), height))
    wrist = cv2.resize(wrist, (round(wrist.shape[1] * height / wrist.shape[0]), height))
    images = np.hstack([front, wrist])

    header = np.full((76, images.shape[1], 3), 20, dtype=np.uint8)
    relative = min(offset + 1, end - start)
    cv2.putText(header, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
    cv2.putText(
        header,
        f"raw frame {raw_frame} | clip {relative}/{end - start}",
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (205, 205, 205),
        1,
    )
    panel = np.vstack([header, images])
    if scale != 1.0:
        panel = cv2.resize(panel, None, fx=scale, fy=scale)
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one A-class transition clip with its wide grasp clip.")
    parser.add_argument("--transition-manifest", type=Path, required=True)
    parser.add_argument("--wide-manifest", type=Path, required=True)
    parser.add_argument("--number", type=int, default=0, help="Zero-based A-class result number.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2

    transition = select_a_transition(load_jsonl(args.transition_manifest), args.number)
    wide = match_wide_segment(transition, load_jsonl(args.wide_manifest))
    episode = Path(transition["episode"])
    with (episode / "data.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print("A-class grasp_transition_event:")
    print(json.dumps(transition, ensure_ascii=False, indent=2))
    print("Matching goal_grasp_event:")
    print(json.dumps(wide, ensure_ascii=False, indent=2))
    print("Controls: Space pause/resume, R restart, Q or ESC quit.")

    max_frames = max(int(wide["end"]) - int(wide["start"]), int(transition["end"]) - int(transition["start"]))
    offset, paused = 0, False
    window = "A grasp transition (right) vs wide goal_grasp_event (left)"
    while offset < max_frames:
        left = draw_panel(cv2, episode, rows, wide, offset, "goal_grasp_event: wide context", args.scale)
        right = draw_panel(cv2, episode, rows, transition, offset, "A: closed -> open -> close -> lift", args.scale)
        cv2.imshow(window, np.hstack([left, right]))
        key = cv2.waitKey(0 if paused else max(1, round(1000 / args.fps))) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("r"):
            offset = 0
            paused = False
            continue
        if not paused:
            offset += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
