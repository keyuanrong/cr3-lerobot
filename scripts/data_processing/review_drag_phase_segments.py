#!/usr/bin/env python
"""Interactively review visually segmented CR3 task phases."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-good", required=True, help="Reviewed-good segment JSONL.")
    parser.add_argument("--output-reject", required=True, help="Reviewed-reject segment JSONL.")
    parser.add_argument(
        "--episode",
        help="Only review one raw episode. Accepts its full path or unique directory name.",
    )
    parser.add_argument("--start", type=int, default=1, help="One-based segment index.")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("approach", "grasp", "carry", "place"),
        help="Only review the selected task phases.",
    )
    parser.add_argument("--fps", type=float, default=90.0)
    parser.add_argument("--scale", type=float, default=0.55)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--auto-next",
        action="store_true",
        help="Advance to the next segment when playback ends without recording a good/bad decision.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def segment_id(segment: dict) -> tuple:
    return (segment["episode"], int(segment["start"]), int(segment["end"]), segment["task"])


def write_jsonl(path: Path, segments: dict[tuple, dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(segments.values(), key=lambda item: (item["episode"], item["start"], item["end"]))
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")


def read_rows(episode: Path) -> list[dict]:
    with (episode / "data.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_panel(cv2, image, title: str, width: int, height: int):
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    if image is not None:
        src_h, src_w = image.shape[:2]
        ratio = min(width / src_w, (height - 28) / src_h)
        resized = cv2.resize(image, (round(src_w * ratio), round(src_h * ratio)))
        y = 28 + (height - 28 - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return panel


def make_frame(
    cv2,
    segment: dict,
    rows: list[dict],
    frame: int,
    segment_index: int,
    total: int,
    scale: float,
    deciding: bool,
    auto_next: bool,
):
    episode = Path(segment["episode"])
    row = rows[frame]
    def read(column):
        return cv2.imread(str(episode / row[column]), cv2.IMREAD_COLOR)

    width, height = max(240, round(640 * scale)), max(180, round(480 * scale))
    panels = np.hstack([image_panel(cv2, read("front_rgb"), "front_rgb", width, height), image_panel(cv2, read("wrist_rgb"), "wrist_rgb", width, height)])
    info = np.full((102, panels.shape[1], 3), 255, dtype=np.uint8)
    title = f"{segment_index + 1}/{total}  {segment['color']} | {segment['phase']}"
    raw_frame = f"raw frame {frame}   segment [{segment['start']}, {segment['end']})"
    task = f"task: {segment['task']}"
    controls = (
        "ended: auto next | G good | B bad | R replay | Q quit"
        if deciding and auto_next
        else ("ended: G good | B bad | R replay | Q quit" if deciding else "Space pause | +/- speed | A/D prev/next | J/L step | G good | B bad | R replay | Q quit")
    )
    for line, y, size, color in ((title, 25, 0.65, (0, 0, 0)), (task, 55, 0.58, (20, 20, 20)), (raw_frame + "    " + controls, 84, 0.48, (70, 70, 70))):
        cv2.putText(info, line[:190], (8, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1 if y != 25 else 2)
    return np.vstack([info, panels])


def main():
    args = parse_args()
    import cv2

    segments = load_jsonl(Path(args.manifest))
    if args.episode:
        needle = args.episode.rstrip("/")
        segments = [
            segment
            for segment in segments
            if segment.get("episode", "").rstrip("/") == needle or Path(segment.get("episode", "")).name == needle
        ]
    if args.phases:
        allowed_phases = set(args.phases)
        segments = [segment for segment in segments if segment.get("phase") in allowed_phases]
    if not segments:
        raise SystemExit("Manifest is empty after phase filtering.")
    good_path, reject_path = Path(args.output_good), Path(args.output_reject)
    good = {} if args.fresh else {segment_id(item): item for item in load_jsonl(good_path)}
    reject = {} if args.fresh else {segment_id(item): item for item in load_jsonl(reject_path)}
    index = max(0, min(args.start - 1, len(segments) - 1))
    rows = read_rows(Path(segments[index]["episode"]))
    frame = int(segments[index]["start"])
    paused = False
    deciding = False
    cv2.namedWindow("drag phase review", cv2.WINDOW_NORMAL)

    def switch(next_index: int):
        nonlocal index, rows, frame, paused, deciding
        index = max(0, min(next_index, len(segments) - 1))
        rows = read_rows(Path(segments[index]["episode"]))
        frame = int(segments[index]["start"])
        paused = deciding = False

    while True:
        segment = segments[index]
        frame = max(int(segment["start"]), min(frame, int(segment["end"]) - 1))
        cv2.imshow(
            "drag phase review",
            make_frame(cv2, segment, rows, frame, index, len(segments), args.scale, deciding, args.auto_next),
        )
        key = cv2.waitKey(0 if paused or deciding else max(1, round(1000 / args.fps))) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            frame, paused, deciding = int(segment["start"]), False, False
        elif key in (ord("-"), ord("_")):
            args.fps = max(1.0, args.fps / 1.5)
        elif key in (ord("="), ord("+")):
            args.fps = min(240.0, args.fps * 1.5)
        elif key == ord("j"):
            frame, paused = max(int(segment["start"]), frame - 1), True
        elif key == ord("l"):
            frame, paused = min(int(segment["end"]) - 1, frame + 1), True
        elif key == ord("a"):
            switch(index - 1)
        elif key == ord("d"):
            switch(index + 1)
        elif key in (ord("g"), ord("b")):
            target, other = (good, reject) if key == ord("g") else (reject, good)
            target[segment_id(segment)] = segment
            other.pop(segment_id(segment), None)
            write_jsonl(good_path, good)
            write_jsonl(reject_path, reject)
            print(("GOOD" if key == ord("g") else "REJECT"), segment["phase"], segment["episode"])
            switch(index + 1)
        elif not paused:
            frame += 1
            if frame >= int(segment["end"]):
                if args.auto_next and index < len(segments) - 1:
                    switch(index + 1)
                else:
                    frame, paused, deciding = int(segment["end"]) - 1, True, True

    write_jsonl(good_path, good)
    write_jsonl(reject_path, reject)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
