#!/usr/bin/env python

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export raw drag episode image sequences to review videos.")
    parser.add_argument("input", help="A drag_episode_* directory or a directory containing drag_episode_* folders.")
    parser.add_argument("--output-dir", default="outputs/drag_episode_videos")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640, help="Width of each camera panel.")
    parser.add_argument("--height", type=int, default=480, help="Height of each camera panel.")
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_episodes(path: Path) -> list[Path]:
    if (path / "data.csv").is_file():
        return [path]
    return sorted(p for p in path.rglob("drag_episode_*") if (p / "data.csv").is_file())


def read_rows(episode: Path) -> list[dict[str, str]]:
    with (episode / "data.csv").open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_image(cv2, episode: Path, rel_path: str, *, is_depth: bool = False):
    if not rel_path:
        return None
    path = episode / rel_path
    flag = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        return None
    if is_depth and len(image.shape) == 2:
        image = cv2.convertScaleAbs(image, alpha=255.0 / max(float(image.max()), 1.0))
        image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    elif len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def make_panel(cv2, image, title: str, width: int, height: int):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if image is None:
        cv2.putText(canvas, "missing", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        src_h, src_w = image.shape[:2]
        scale = min(width / src_w, (height - 30) / src_h)
        resized = cv2.resize(image, (max(1, int(src_w * scale)), max(1, int(src_h * scale))))
        dst_h, dst_w = resized.shape[:2]
        x = (width - dst_w) // 2
        y = 30 + (height - 30 - dst_h) // 2
        canvas[y : y + dst_h, x : x + dst_w] = resized
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    return canvas


def make_frame(cv2, episode: Path, row: dict[str, str], frame_i: int, total: int, args):
    panels = [
        make_panel(cv2, read_image(cv2, episode, row.get("front_rgb", "")), "front_rgb", args.width, args.height),
        make_panel(cv2, read_image(cv2, episode, row.get("wrist_rgb", "")), "wrist_rgb", args.width, args.height),
    ]
    if args.include_depth:
        panels.append(
            make_panel(
                cv2,
                read_image(cv2, episode, row.get("wrist_depth", ""), is_depth=True),
                "wrist_depth",
                args.width,
                args.height,
            )
        )
    image_row = np.hstack(panels)
    info = np.full((70, image_row.shape[1], 3), 255, dtype=np.uint8)
    task = row.get("task", "")
    cv2.putText(info, f"{episode.name}  frame {frame_i + 1}/{total}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(info, task[:160], (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1)
    return np.vstack([info, image_row])


def export_episode(cv2, episode: Path, output_dir: Path, args) -> Path | None:
    rows = read_rows(episode)
    if not rows:
        print(f"skip empty episode: {episode}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{episode.name}.mp4"
    if output_path.exists() and not args.overwrite:
        print(f"skip existing: {output_path}")
        return output_path

    first_frame = make_frame(cv2, episode, rows[0], 0, len(rows), args)
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        writer.write(first_frame)
        for frame_i, row in enumerate(rows[1:], start=1):
            writer.write(make_frame(cv2, episode, row, frame_i, len(rows), args))
    finally:
        writer.release()

    print(f"wrote {output_path} ({len(rows)} frames)")
    return output_path


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed in this environment.") from exc

    episodes = discover_episodes(Path(args.input))
    if not episodes:
        raise SystemExit("No drag_episode_* directories with data.csv found.")

    output_dir = Path(args.output_dir)
    for episode in episodes:
        export_episode(cv2, episode, output_dir, args)


if __name__ == "__main__":
    main()
