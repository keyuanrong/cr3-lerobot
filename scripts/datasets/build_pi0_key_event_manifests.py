#!/usr/bin/env python
"""Build narrow grasp-lift and release manifests from reviewed Pi0 phase boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if not __package__:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_processing.filter_drag_phase_segments_with_qwen import stable_events


CORE_PHASES = ("approach", "grasp", "carry", "place")
EVENT_TYPES = ("grasp_lift_event", "release_event")
TOTAL_GOALS = {
    "red": "put the red block into the black frame",
    "green": "put the green block into the black frame",
    "yellow": "put the yellow block into the black frame",
    "full": (
        "put the red block into the black frame, then put the green block into the black frame, "
        "then put the yellow block into the black frame"
    ),
}


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def read_source_list(path: Path) -> list[Path]:
    sources = [normalize_path(Path(line.strip())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(sources) != len(set(sources)):
        raise ValueError(f"Duplicate source episode in {path}")
    invalid = [source for source in sources if source.parent.name not in TOTAL_GOALS]
    if invalid:
        raise ValueError(f"Unsupported source task group: {invalid[0]}")
    return sorted(sources)


def read_phase_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"episode", "start", "end", "phase", "color", "refined_boundaries"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
        row["episode"] = str(normalize_path(Path(row["episode"])))
        rows.append(row)
    return rows


def read_episode_rows(source: Path) -> list[dict[str, str]]:
    with (source / "data.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_core_phases(
    sources: set[Path], phase_rows: list[dict[str, Any]]
) -> dict[tuple[Path, str], dict[str, dict[str, Any]]]:
    grouped: dict[tuple[Path, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in phase_rows:
        source = normalize_path(Path(row["episode"]))
        if source not in sources or row["phase"] not in CORE_PHASES:
            continue
        color = str(row["color"])
        if color not in {"red", "green", "yellow"}:
            raise ValueError(f"Unsupported phase color {color!r} for {source}")
        phase = str(row["phase"])
        if phase in grouped[(source, color)]:
            raise ValueError(f"Duplicate {phase} phase for {source} / {color}")
        grouped[(source, color)][phase] = row
    return grouped


def make_row(
    source: Path,
    start: int,
    end: int,
    sample_type: str,
    color: str,
    phase: str,
    anchors: dict[str, int | str],
) -> dict[str, object]:
    group = source.parent.name
    return {
        "episode": str(source),
        "start": start,
        "end": end,
        "task": TOTAL_GOALS[group],
        "sample_type": sample_type,
        "goal_group": group,
        "source_kind": group,
        "color": color,
        "phase": phase,
        "anchors": anchors,
    }


def local_event_frame(
    events: list[tuple[int, bool]], *, start: int, end: int, is_open: bool
) -> int | None:
    return next((frame for frame, value in events if start <= frame < end and value is is_open), None)


def build_event_manifests(
    source_episodes: list[Path],
    phase_rows: list[dict[str, Any]],
    frame_counts: dict[Path, int],
    episode_rows: dict[Path, list[dict[str, str]]],
    *,
    close_pre_frames: int,
    lift_post_frames: int,
    open_pre_frames: int,
    release_post_frames: int,
    threshold: float,
    debounce: int,
    min_event_frames: int = 30,
) -> tuple[dict[str, list[dict[str, object]]], Counter[str]]:
    """Create only event clips whose gripper and refined visual anchors agree."""
    sources = {normalize_path(source) for source in source_episodes}
    manifests: dict[str, list[dict[str, object]]] = {sample_type: [] for sample_type in EVENT_TYPES}
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

        grasp, carry, place = phases["grasp"], phases["carry"], phases["place"]
        boundaries = grasp["refined_boundaries"]
        lift_frame = int(boundaries["grasp_end"])
        release_frame = int(place["refined_boundaries"]["place_end"])
        events = stable_events(rows, threshold=threshold, debounce=debounce)

        close_frame = local_event_frame(
            events, start=int(grasp["start"]), end=int(grasp["end"]), is_open=False
        )
        if close_frame is None:
            rejected["missing_stable_close"] += 1
        elif not int(grasp["start"]) <= close_frame <= lift_frame <= int(carry["end"]):
            rejected["invalid_grasp_order"] += 1
        else:
            start = max(int(grasp["start"]), close_frame - close_pre_frames)
            end = min(frame_count, int(carry["end"]), lift_frame + lift_post_frames)
            if end - start < min_event_frames:
                rejected["grasp_lift_too_short"] += 1
            else:
                manifests["grasp_lift_event"].append(
                    make_row(
                        source,
                        start,
                        end,
                        "grasp_lift_event",
                        color,
                        "grasp_lift",
                        {
                            "stable_close_frame": close_frame,
                            "refined_grasp_end": lift_frame,
                            "pre_close_frames": close_frame - start,
                            "post_lift_frames": end - lift_frame,
                        },
                    )
                )

        open_frame = local_event_frame(
            events, start=int(place["start"]), end=int(place["end"]), is_open=True
        )
        if open_frame is None:
            rejected["missing_stable_open"] += 1
        elif not int(place["start"]) <= open_frame <= release_frame <= int(place["end"]):
            rejected["invalid_release_order"] += 1
        else:
            start = max(int(place["start"]), open_frame - open_pre_frames)
            post_release = 0 if source.parent.name == "full" and color in {"red", "green"} else release_post_frames
            end = min(frame_count, int(place["end"]), release_frame + post_release)
            if end - start < min_event_frames:
                rejected["release_too_short"] += 1
            else:
                manifests["release_event"].append(
                    make_row(
                        source,
                        start,
                        end,
                        "release_event",
                        color,
                        "release",
                        {
                            "stable_open_frame": open_frame,
                            "refined_place_end": release_frame,
                            "pre_open_frames": open_frame - start,
                            "post_release_frames": end - release_frame,
                        },
                    )
                )

    for rows in manifests.values():
        rows.sort(key=lambda row: (str(row["episode"]), int(row["start"]), int(row["end"])))
    return manifests, rejected


def summarize(manifests: dict[str, list[dict[str, object]]], rejected: Counter[str]) -> dict[str, object]:
    return {
        sample_type: {
            "episodes": len(rows),
            "frames": sum(int(row["end"]) - int(row["start"]) for row in rows),
            "by_goal_group": dict(sorted(Counter(str(row["goal_group"]) for row in rows).items())),
            "by_color": dict(sorted(Counter(str(row["color"]) for row in rows).items())),
        }
        for sample_type, rows in manifests.items()
    } | {"rejected": dict(sorted(rejected.items()))}


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build narrow Pi0 grasp-lift and release event manifests.")
    parser.add_argument("--source-train-list", type=Path, default=Path("data/episode_lists/pi0_unified_goal_v1/source_train.txt"))
    parser.add_argument("--phase-manifest", type=Path, default=Path("data/episode_lists/第二版好的分割_60+_60/refined_segments.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/episode_lists/pi0_unified_goal_v2_key_events"))
    parser.add_argument("--close-pre-frames", type=int, default=45)
    parser.add_argument("--lift-post-frames", type=int, default=45)
    parser.add_argument("--open-pre-frames", type=int, default=45)
    parser.add_argument("--release-post-frames", type=int, default=30)
    parser.add_argument("--gripper-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-debounce-frames", type=int, default=5)
    parser.add_argument("--min-event-frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = read_source_list(args.source_train_list)
    phase_rows = read_phase_rows(args.phase_manifest)
    episode_rows = {source: read_episode_rows(source) for source in sources}
    frame_counts = {source: len(rows) for source, rows in episode_rows.items()}
    manifests, rejected = build_event_manifests(
        sources,
        phase_rows,
        frame_counts,
        episode_rows,
        close_pre_frames=args.close_pre_frames,
        lift_post_frames=args.lift_post_frames,
        open_pre_frames=args.open_pre_frames,
        release_post_frames=args.release_post_frames,
        threshold=args.gripper_threshold,
        debounce=args.gripper_debounce_frames,
        min_event_frames=args.min_event_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "grasp_lift_event_train.jsonl", manifests["grasp_lift_event"])
    write_jsonl(args.output_dir / "release_event_train.jsonl", manifests["release_event"])
    summary = {
        "source_train_list": str(args.source_train_list),
        "phase_manifest": str(args.phase_manifest),
        "clip_frames": {
            "close_pre": args.close_pre_frames,
            "lift_post": args.lift_post_frames,
            "open_pre": args.open_pre_frames,
            "release_post": args.release_post_frames,
        },
        "event_rules": {
            "gripper_threshold": args.gripper_threshold,
            "gripper_debounce_frames": args.gripper_debounce_frames,
            "min_event_frames": args.min_event_frames,
            "full_red_green_release_post_frames": 0,
        },
        "manifests": summarize(manifests, rejected),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for sample_type, rows in manifests.items():
        print(f"{sample_type}: episodes={len(rows)}, frames={sum(int(row['end']) - int(row['start']) for row in rows)}")
    print(f"rejected color tasks: {sum(rejected.values())}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
