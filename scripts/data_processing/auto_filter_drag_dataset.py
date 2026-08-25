#!/usr/bin/env python

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


IMAGE_COLUMNS = ("front_rgb", "wrist_rgb")
DEFAULT_TASK_ROOTS = ("red", "green", "yellow", "full")


@dataclass
class EpisodeCheck:
    episode: Path
    frames: int
    close_events: int
    open_events: int
    reasons: list[str]

    @property
    def ok(self) -> bool:
        return not self.reasons


def parse_args():
    parser = argparse.ArgumentParser(description="Automatically filter raw CR3 drag episodes.")
    parser.add_argument("--input-root", default="data/cr3_real_drag_raw")
    parser.add_argument("--output-dir", default="data/episode_lists")
    parser.add_argument(
        "--episode-pattern",
        default="drag_episode_*",
        help="Episode directory glob pattern under each task directory, e.g. drag_episode_20260721_*.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASK_ROOTS),
        help="Subdirectories under --input-root to filter, e.g. red green yellow full.",
    )
    parser.add_argument("--single-min-frames", type=int, default=30)
    parser.add_argument("--full-min-frames", type=int, default=120)
    parser.add_argument("--single-min-close-events", type=int, default=1)
    parser.add_argument("--single-min-open-events", type=int, default=1)
    parser.add_argument("--full-min-close-events", type=int, default=3)
    parser.add_argument("--full-min-open-events", type=int, default=3)
    parser.add_argument(
        "--allow-gripper-error",
        action="store_true",
        help="Keep episodes with GRIPPER_ERROR in data.csv. Not recommended for training.",
    )
    parser.add_argument(
        "--write-combined",
        action="store_true",
        help="Also write all_auto_good.txt and all_auto_reject.txt.",
    )
    return parser.parse_args()


def read_rows(episode: Path) -> list[dict[str, str]]:
    csv_path = episode / "data.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_action_events(actions: list[str | None], token: str) -> int:
    count = 0
    was_active = False
    for action in actions:
        action = action or ""
        active = token in action
        if active and not was_active:
            count += 1
        was_active = active
    return count


def has_missing_images(episode: Path, rows: list[dict[str, str]]) -> bool:
    if not rows:
        return True
    sample_rows = rows[:5] + rows[-5:]
    for row in sample_rows:
        for column in IMAGE_COLUMNS:
            rel_path = row.get(column, "")
            if not rel_path or not (episode / rel_path).is_file():
                return True
    return False


def check_episode(episode: Path, task: str, args) -> EpisodeCheck:
    reasons = []
    rows = read_rows(episode)
    frames = len(rows)

    if not rows:
        reasons.append("missing_or_empty_csv")

    min_frames = args.full_min_frames if task == "full" else args.single_min_frames
    min_close_events = args.full_min_close_events if task == "full" else args.single_min_close_events
    min_open_events = args.full_min_open_events if task == "full" else args.single_min_open_events

    if frames < min_frames:
        reasons.append("short")

    actions = [row.get("action") or "" for row in rows]
    close_events = count_action_events(actions, "GRIPPER_CLOSE")
    open_events = count_action_events(actions, "GRIPPER_OPEN")

    if not args.allow_gripper_error and any("GRIPPER_ERROR" in action for action in actions):
        reasons.append("gripper_error")
    if close_events < min_close_events:
        reasons.append("few_close")
    if open_events < min_open_events:
        reasons.append("few_open")
    if has_missing_images(episode, rows):
        reasons.append("missing_image")

    return EpisodeCheck(episode, frames, close_events, open_events, reasons)


def write_list(path: Path, episodes: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(episode) for episode in episodes) + ("\n" if episodes else ""), encoding="utf-8")


def write_report(path: Path, checks: list[EpisodeCheck]) -> None:
    lines = ["episode,frames,close_events,open_events,status,reasons"]
    for check in checks:
        status = "good" if check.ok else "reject"
        reasons = "|".join(check.reasons)
        lines.append(
            f"{check.episode},{check.frames},{check.close_events},{check.open_events},{status},{reasons}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    all_good = []
    all_reject = []

    for task in args.tasks:
        task_root = input_root / task
        episodes = sorted(path for path in task_root.glob(args.episode_pattern) if path.is_dir())
        checks = [check_episode(episode, task, args) for episode in episodes]
        good = [check.episode for check in checks if check.ok]
        reject = [check.episode for check in checks if not check.ok]
        all_good.extend(good)
        all_reject.extend(reject)

        write_list(output_dir / f"{task}_auto_good.txt", good)
        write_list(output_dir / f"{task}_auto_reject.txt", reject)
        write_report(output_dir / f"{task}_auto_report.csv", checks)

        reason_counts: dict[str, int] = {}
        for check in checks:
            for reason in check.reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        print(f"{task}: total={len(episodes)} good={len(good)} reject={len(reject)}")
        if reason_counts:
            print("  reject reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items())))

    if args.write_combined:
        write_list(output_dir / "all_auto_good.txt", all_good)
        write_list(output_dir / "all_auto_reject.txt", all_reject)
        print(f"all: good={len(all_good)} reject={len(all_reject)}")


if __name__ == "__main__":
    main()
