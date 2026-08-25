#!/usr/bin/env python
"""Export one phase segment from a JSONL phase manifest as a labelled review video."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--color", choices=["red", "green", "yellow"])
    parser.add_argument("--phase", choices=["approach", "grasp", "carry", "place"])
    parser.add_argument("--number", type=int, default=0, help="Zero-based result number after filtering.")
    parser.add_argument("--output-dir", default="outputs/phase_segment_videos")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=0.7)
    return parser.parse_args()


def load_segments(path: Path, color: str | None, phase: str | None) -> list[dict]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        segment = json.loads(line)
        if color and segment["color"] != color:
            continue
        if phase and segment["phase"] != phase:
            continue
        segments.append(segment)
    return segments


def load_image(cv2, episode: Path, row: dict, column: str):
    image = cv2.imread(str(episode / row[column]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(episode / row[column])
    return image


def draw_frame(cv2, episode: Path, row: dict, segment: dict, relative_index: int, total: int, scale: float):
    front = load_image(cv2, episode, row, "front_rgb")
    wrist = load_image(cv2, episode, row, "wrist_rgb")
    height = min(front.shape[0], wrist.shape[0])
    front = cv2.resize(front, (round(front.shape[1] * height / front.shape[0]), height))
    wrist = cv2.resize(wrist, (round(wrist.shape[1] * height / wrist.shape[0]), height))
    panels = np.hstack([front, wrist])
    header = np.full((72, panels.shape[1], 3), 20, dtype=np.uint8)
    title = f"{segment['color']} | {segment['phase']} | {segment['task']}"
    progress = f"{episode.name}  {relative_index + 1}/{total}  raw frame {segment['start'] + relative_index}"
    cv2.putText(header, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    cv2.putText(header, progress, (12, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (205, 205, 205), 1)
    frame = np.vstack([header, panels])
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame


def main():
    args = parse_args()
    import cv2

    segments = load_segments(Path(args.manifest), args.color, args.phase)
    if not segments:
        raise SystemExit("No segment matched the requested filters.")
    if not 0 <= args.number < len(segments):
        raise SystemExit(f"--number must be in [0, {len(segments) - 1}]")

    segment = segments[args.number]
    episode = Path(segment["episode"])
    with (episode / "data.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected_rows = rows[segment["start"] : segment["end"]]
    if not selected_rows:
        raise SystemExit("Segment contains no frames.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{episode.name}_{segment['color']}_{segment['phase']}_{args.number:04d}.mp4"
    first = draw_frame(cv2, episode, selected_rows[0], segment, 0, len(selected_rows), args.scale)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (first.shape[1], first.shape[0])
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        for index, row in enumerate(selected_rows):
            writer.write(draw_frame(cv2, episode, row, segment, index, len(selected_rows), args.scale))
    finally:
        writer.release()
    print(f"wrote: {output_path}")
    print(json.dumps(segment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
