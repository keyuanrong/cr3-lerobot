#!/usr/bin/env python

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Review raw drag episodes recorded by record_drag_dataset.py.")
    parser.add_argument("--data-root", default="data", help="Directory containing drag_episode_* folders.")
    parser.add_argument("--episode-list", help="Optional text file with one episode path per line.")
    parser.add_argument("--episode", type=Path, help="Review one specific episode directory.")
    parser.add_argument("--start", type=int, default=1, help="1-based episode index to start from.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument("--scale", type=float, default=0.55, help="Display scale for each camera panel.")
    parser.add_argument("--output-good", default="data/episode_lists/review_good.txt")
    parser.add_argument("--output-reject", default="data/episode_lists/review_reject.txt")
    parser.add_argument("--fresh", action="store_true", help="Start with empty good/reject files.")
    parser.add_argument(
        "--no-prompt-after-episode",
        action="store_true",
        help="Keep auto-advancing instead of requiring G/B after each episode.",
    )
    return parser.parse_args()


def load_episodes(data_root: Path, episode_list: str | None) -> list[Path]:
    if episode_list:
        paths = []
        for line in Path(episode_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(Path(line))
        return paths
    return sorted(data_root.glob("drag_episode_*"))


def read_episode_rows(episode: Path) -> list[dict[str, str]]:
    csv_path = episode / "data.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def depth_to_bgr(cv2, depth):
    if depth is None:
        return None
    if len(depth.shape) == 3:
        return depth
    depth_u8 = cv2.convertScaleAbs(depth, alpha=255.0 / max(float(depth.max()), 1.0))
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def read_image(cv2, episode: Path, rel_path: str, is_depth: bool = False):
    if not rel_path:
        return None
    path = episode / rel_path
    flag = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        return None
    if is_depth:
        img = depth_to_bgr(cv2, img)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def panel(cv2, img, title: str, width: int, height: int):
    import numpy as np

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if img is not None:
        h, w = img.shape[:2]
        scale = min(width / w, (height - 28) / h)
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        rh, rw = resized.shape[:2]
        x = (width - rw) // 2
        y = 28 + (height - 28 - rh) // 2
        canvas[y : y + rh, x : x + rw] = resized
    else:
        cv2.putText(canvas, "missing", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    return canvas


def make_frame(
    cv2,
    episode: Path,
    rows: list[dict[str, str]],
    frame_i: int,
    episode_i: int,
    total: int,
    scale: float,
    awaiting_decision: bool,
):
    import numpy as np

    row = rows[frame_i]
    task = row.get("task", "").strip()
    front = read_image(cv2, episode, row.get("front_rgb", ""))
    wrist = read_image(cv2, episode, row.get("wrist_rgb", ""))
    depth = read_image(cv2, episode, row.get("wrist_depth", ""), is_depth=True)

    panel_w = max(240, int(640 * scale))
    panel_h = max(180, int(480 * scale))
    top = np.hstack(
        [
            panel(cv2, front, "front_rgb", panel_w, panel_h),
            panel(cv2, wrist, "wrist_rgb", panel_w, panel_h),
            panel(cv2, depth, "wrist_depth", panel_w, panel_h),
        ]
    )
    info_h = 100
    info = np.full((info_h, top.shape[1], 3), 255, dtype=np.uint8)
    duration_s = 0.0
    try:
        duration_s = float(rows[-1].get("timestamp", 0.0)) - float(rows[0].get("timestamp", 0.0))
    except ValueError:
        pass
    text1 = f"episode {episode_i + 1}/{total}  frame {frame_i + 1}/{len(rows)}  {duration_s:.1f}s"
    text_episode = episode.name
    text_task = f"task: {task}" if task else "task: <empty>"
    if awaiting_decision:
        text2 = "episode ended: press G good or B bad | R replay | A/D prev/next | Q quit"
    else:
        text2 = "keys: Space pause/play | -/+ speed | A/D prev/next | J/L step | G good | B bad | R replay | Q quit"
    cv2.putText(info, text1, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(info, text_episode, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
    cv2.putText(info, text_task[:180], (8, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1)
    cv2.putText(info, text2, (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1)
    return np.vstack([info, top])


def write_marks(good: set[Path], reject: set[Path], good_path: Path, reject_path: Path):
    good_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    good_path.write_text("\n".join(str(p) for p in sorted(good)) + ("\n" if good else ""), encoding="utf-8")
    reject_path.write_text(
        "\n".join(str(p) for p in sorted(reject)) + ("\n" if reject else ""), encoding="utf-8"
    )


def load_marks(path: Path) -> set[Path]:
    if not path.exists():
        return set()
    marks = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        marks.add(Path(line))
    return marks


def main():
    args = parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed in this environment. Activate your lerobot env first.") from exc

    episodes = [args.episode] if args.episode else load_episodes(Path(args.data_root), args.episode_list)
    if not episodes:
        raise SystemExit("No episodes found.")
    if args.episode and not args.episode.is_dir():
        raise SystemExit(f"Episode directory does not exist: {args.episode}")

    good_path = Path(args.output_good)
    reject_path = Path(args.output_reject)
    if args.fresh:
        good: set[Path] = set()
        reject: set[Path] = set()
    else:
        good = load_marks(good_path)
        reject = load_marks(reject_path)
        good -= reject
        reject -= good
        print(f"Loaded existing marks: {len(good)} good, {len(reject)} reject")

    episode_i = max(0, min(args.start - 1, len(episodes) - 1))
    paused = False
    awaiting_decision = False
    frame_i = 0
    rows = read_episode_rows(episodes[episode_i])

    cv2.namedWindow("drag episode review", cv2.WINDOW_NORMAL)

    def advance_episode():
        nonlocal episode_i, rows, frame_i, paused, awaiting_decision
        if episode_i < len(episodes) - 1:
            episode_i += 1
            rows = read_episode_rows(episodes[episode_i])
            frame_i = 0
            paused = False
            awaiting_decision = False
        else:
            paused = True
            awaiting_decision = True

    while True:
        if not rows:
            print(f"Skipping empty episode: {episodes[episode_i]}")
            episode_i = min(episode_i + 1, len(episodes) - 1)
            rows = read_episode_rows(episodes[episode_i])
            frame_i = 0
            continue

        frame_i = max(0, min(frame_i, len(rows) - 1))
        cv2.imshow(
            "drag episode review",
            make_frame(
                cv2,
                episodes[episode_i],
                rows,
                frame_i,
                episode_i,
                len(episodes),
                args.scale,
                awaiting_decision,
            ),
        )

        delay_ms = max(1, int(1000 / args.fps))
        key = cv2.waitKey(0 if paused or awaiting_decision else delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            frame_i = 0
            paused = False
            awaiting_decision = False
        elif key in (ord("-"), ord("_")):
            args.fps = max(1.0, args.fps / 1.5)
            print(f"review fps: {args.fps:.1f}")
        elif key in (ord("="), ord("+")):
            args.fps = min(240.0, args.fps * 1.5)
            print(f"review fps: {args.fps:.1f}")
        elif key == ord("j"):
            frame_i = max(0, frame_i - 1)
            paused = True
        elif key == ord("l"):
            frame_i = min(len(rows) - 1, frame_i + 1)
            paused = True
        elif key == ord("a"):
            episode_i = max(0, episode_i - 1)
            rows = read_episode_rows(episodes[episode_i])
            frame_i = 0
            paused = False
            awaiting_decision = False
        elif key == ord("d"):
            episode_i = min(len(episodes) - 1, episode_i + 1)
            rows = read_episode_rows(episodes[episode_i])
            frame_i = 0
            paused = False
            awaiting_decision = False
        elif key == ord("g"):
            good.add(episodes[episode_i])
            reject.discard(episodes[episode_i])
            write_marks(good, reject, good_path, reject_path)
            print(f"GOOD   {episodes[episode_i]}")
            advance_episode()
        elif key == ord("b"):
            reject.add(episodes[episode_i])
            good.discard(episodes[episode_i])
            write_marks(good, reject, good_path, reject_path)
            print(f"REJECT {episodes[episode_i]}")
            advance_episode()
        elif not paused:
            frame_i += 1
            if frame_i >= len(rows):
                if args.no_prompt_after_episode:
                    episode_i = min(episode_i + 1, len(episodes) - 1)
                    rows = read_episode_rows(episodes[episode_i])
                    frame_i = 0
                else:
                    frame_i = len(rows) - 1
                    paused = True
                    awaiting_decision = True

    write_marks(good, reject, good_path, reject_path)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
