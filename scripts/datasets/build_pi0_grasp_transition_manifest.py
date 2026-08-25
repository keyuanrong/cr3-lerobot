#!/usr/bin/env python
"""Build total-goal grasp clips centered on valid gripper state transitions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if not __package__:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets.build_pi0_key_event_manifests import (
    CORE_PHASES,
    TOTAL_GOALS,
    group_core_phases,
    make_row,
    normalize_path,
    read_episode_rows,
    read_phase_rows,
    read_source_list,
    stable_events,
    write_jsonl,
)


SAMPLE_TYPE = "grasp_transition_event"


def event_sequence(
    events: list[tuple[int, bool]], *, start: int, end: int
) -> list[tuple[int, bool]]:
    return [(frame, is_open) for frame, is_open in events if start <= frame < end]


def build_transition_manifest(
    source_episodes: list[Path],
    phase_rows: list[dict[str, Any]],
    frame_counts: dict[Path, int],
    episode_rows: dict[Path, list[dict[str, str]]],
    *,
    context_pre_frames: int,
    open_pre_frames: int,
    close_pre_frames: int,
    lift_post_frames: int,
    threshold: float,
    debounce: int,
    min_event_frames: int = 90,
) -> tuple[list[dict[str, object]], Counter[str]]:
    """Select only successful open/close/lift transitions from grasp contexts."""
    sources = {normalize_path(source) for source in source_episodes}
    rows_out: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    grouped = group_core_phases(sources, phase_rows)

    for (source, color), phases in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        if set(phases) != set(CORE_PHASES):
            rejected["missing_core_phase"] += 1
            continue

        rows = episode_rows[source]
        frame_count = frame_counts[source]
        if frame_count != len(rows):
            raise ValueError(f"Frame count mismatch for {source}: {frame_count} != {len(rows)}")

        grasp, carry = phases["grasp"], phases["carry"]
        grasp_start = int(grasp["start"])
        carry_end = int(carry["end"])
        lift_frame = int(grasp["refined_boundaries"]["grasp_end"])
        context_start = max(0, grasp_start - context_pre_frames)

        if not grasp_start <= lift_frame <= carry_end:
            rejected["invalid_visual_lift_order"] += 1
            continue

        initial_open = float(rows[context_start]["gripper"]) >= threshold
        events = event_sequence(
            stable_events(rows, threshold=threshold, debounce=debounce),
            start=context_start,
            end=lift_frame + 1,
        )

        if initial_open:
            expected_values = [False]
            transition_type = "open_close_lift"
        else:
            expected_values = [True, False]
            transition_type = "closed_open_close_lift"

        values = [is_open for _, is_open in events]
        if values != expected_values:
            if not initial_open and True not in values:
                rejected["missing_stable_open_after_closed"] += 1
            elif False not in values:
                rejected["missing_stable_close"] += 1
            else:
                rejected["unexpected_gripper_event_sequence"] += 1
            continue

        if initial_open:
            close_frame = events[0][0]
            start = max(context_start, close_frame - close_pre_frames)
            open_frame = None
        else:
            open_frame, close_frame = events[0][0], events[1][0]
            start = max(context_start, open_frame - open_pre_frames)
        end = min(frame_count, carry_end, lift_frame + lift_post_frames)

        if end - start < min_event_frames:
            rejected["transition_too_short"] += 1
            continue

        anchors: dict[str, int | str] = {
            "transition_type": transition_type,
            "stable_close_frame": close_frame,
            "refined_grasp_end": lift_frame,
            "context_start": context_start,
        }
        if open_frame is not None:
            anchors["stable_open_frame"] = open_frame

        rows_out.append(
            make_row(
                source,
                start,
                end,
                SAMPLE_TYPE,
                color,
                "grasp_transition",
                anchors,
            )
        )

    rows_out.sort(key=lambda row: (str(row["episode"]), int(row["start"]), int(row["end"])))
    return rows_out, rejected


def summarize(rows: list[dict[str, object]], rejected: Counter[str]) -> dict[str, object]:
    anchors = [dict(row["anchors"]) for row in rows]
    return {
        "episodes": len(rows),
        "frames": sum(int(row["end"]) - int(row["start"]) for row in rows),
        "by_goal_group": dict(sorted(Counter(str(row["goal_group"]) for row in rows).items())),
        "by_color": dict(sorted(Counter(str(row["color"]) for row in rows).items())),
        "by_transition_type": dict(sorted(Counter(str(anchor["transition_type"]) for anchor in anchors).items())),
        "rejected": dict(sorted(rejected.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified total-goal grasp transition manifest.")
    parser.add_argument("--source-train-list", type=Path, default=Path("data/episode_lists/pi0_unified_goal_v1/source_train.txt"))
    parser.add_argument("--phase-manifest", type=Path, default=Path("data/episode_lists/第二版好的分割_60+_60/refined_segments.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/episode_lists/pi0_unified_goal_v3_grasp_transition"))
    parser.add_argument("--context-pre-frames", type=int, default=60)
    parser.add_argument("--open-pre-frames", type=int, default=45)
    parser.add_argument("--close-pre-frames", type=int, default=60)
    parser.add_argument("--lift-post-frames", type=int, default=45)
    parser.add_argument("--gripper-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-debounce-frames", type=int, default=5)
    parser.add_argument("--min-event-frames", type=int, default=90)
    parser.add_argument("--limit", type=int, help="Write at most this many valid clips after scanning all sources.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = read_source_list(args.source_train_list)
    phase_rows = read_phase_rows(args.phase_manifest)
    episode_rows = {source: read_episode_rows(source) for source in sources}
    frame_counts = {source: len(rows) for source, rows in episode_rows.items()}
    rows, rejected = build_transition_manifest(
        sources,
        phase_rows,
        frame_counts,
        episode_rows,
        context_pre_frames=args.context_pre_frames,
        open_pre_frames=args.open_pre_frames,
        close_pre_frames=args.close_pre_frames,
        lift_post_frames=args.lift_post_frames,
        threshold=args.gripper_threshold,
        debounce=args.gripper_debounce_frames,
        min_event_frames=args.min_event_frames,
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "grasp_transition_event_train.jsonl", rows)
    summary = {
        "source_train_list": str(args.source_train_list),
        "phase_manifest": str(args.phase_manifest),
        "clip_frames": {
            "context_pre": args.context_pre_frames,
            "open_pre": args.open_pre_frames,
            "close_pre": args.close_pre_frames,
            "lift_post": args.lift_post_frames,
        },
        "event_rules": {
            "gripper_threshold": args.gripper_threshold,
            "gripper_debounce_frames": args.gripper_debounce_frames,
            "min_event_frames": args.min_event_frames,
        },
        "manifest": summarize(rows, rejected),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{SAMPLE_TYPE}: episodes={len(rows)}, frames={sum(int(row['end']) - int(row['start']) for row in rows)}")
    print(f"rejected color tasks: {sum(rejected.values())}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
