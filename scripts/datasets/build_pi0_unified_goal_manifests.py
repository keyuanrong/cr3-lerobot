#!/usr/bin/env python
"""Build train-only conversion manifests for unified Pi0 total-goal training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORE_PHASES = ("approach", "grasp", "carry", "place")
SAMPLE_TYPES = ("complete_goal", "goal_grasp_event", "goal_place_event", "atomic_assist")
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
        raise ValueError(f"Unsupported task root for source: {invalid[0]}")
    return sorted(sources)


def read_phase_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"episode", "start", "end", "task", "phase"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
        row["episode"] = str(normalize_path(Path(row["episode"])))
        rows.append(row)
    return rows


def source_frame_counts(sources: list[Path]) -> dict[Path, int]:
    counts: dict[Path, int] = {}
    for source in sources:
        with (source / "data.csv").open(newline="", encoding="utf-8") as csv_file:
            frames = max(0, sum(1 for _ in csv.DictReader(csv_file)) - 1)
        if frames < 2:
            raise ValueError(f"{source} has fewer than two usable frames")
        counts[source] = frames
    return counts


def goal_group(source: Path) -> str:
    return source.parent.name


def make_row(
    source: Path,
    start: int,
    end: int,
    task: str,
    sample_type: str,
    *,
    color: str | None = None,
    phase: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode": str(source),
        "start": start,
        "end": end,
        "task": task,
        "sample_type": sample_type,
        "goal_group": goal_group(source),
        "source_kind": goal_group(source),
    }
    if color is not None:
        row["color"] = color
    if phase is not None:
        row["phase"] = phase
    return row


def validate_row(row: dict[str, Any], frame_counts: dict[Path, int]) -> None:
    source = normalize_path(Path(row["episode"]))
    frame_count = frame_counts[source]
    start, end = int(row["start"]), int(row["end"])
    if not 0 <= start < end <= frame_count:
        raise ValueError(f"Invalid [{start}, {end}) for {source} with {frame_count} frames")


def build_manifests(
    source_episodes: list[Path],
    phase_rows: list[dict[str, Any]],
    frame_counts: dict[Path, int],
    grasp_pre_frames: int,
    grasp_post_frames: int,
    place_pre_frames: int,
    place_post_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    sources = {normalize_path(source) for source in source_episodes}
    normalized_counts = {normalize_path(source): count for source, count in frame_counts.items()}
    if sources != set(normalized_counts):
        raise ValueError("frame_counts must contain exactly the selected source episodes")

    manifests = {sample_type: [] for sample_type in SAMPLE_TYPES}
    for source in sorted(sources):
        manifests["complete_goal"].append(
            make_row(source, 0, normalized_counts[source], TOTAL_GOALS[goal_group(source)], "complete_goal")
        )

    grouped: dict[tuple[Path, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw_row in phase_rows:
        source = normalize_path(Path(raw_row["episode"]))
        if source not in sources:
            continue
        phase = raw_row["phase"]
        if phase not in CORE_PHASES:
            continue
        color = raw_row.get("color") or goal_group(source)
        if color not in {"red", "green", "yellow"}:
            raise ValueError(f"Unsupported phase color {color!r} for {source}")
        if phase in grouped[(source, color)]:
            raise ValueError(f"Duplicate {phase} phase for {source} / {color}")
        grouped[(source, color)][phase] = raw_row

    for (source, color), phases in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        frame_count = normalized_counts[source]
        if set(phases) != set(CORE_PHASES):
            continue
        for phase in CORE_PHASES:
            row = phases[phase]
            start = max(0, int(row["start"]))
            end = min(frame_count, int(row["end"]))
            if start >= end:
                raise ValueError(f"Empty {phase} segment for {source} / {color}")
            atomic = make_row(
                source,
                start,
                end,
                str(row["task"]),
                "atomic_assist",
                color=color,
                phase=phase,
            )
            if goal_group(source) != "full":
                manifests["atomic_assist"].append(atomic)

        grasp, carry, place = phases["grasp"], phases["carry"], phases["place"]
        grasp_start = max(0, int(grasp["start"]) - grasp_pre_frames)
        grasp_end = min(frame_count, int(carry["start"]) + grasp_post_frames, int(carry["end"]))
        place_start = max(0, int(place["start"]) - place_pre_frames)
        place_end = min(frame_count, int(place["end"]) + place_post_frames)
        total_goal = TOTAL_GOALS[goal_group(source)]
        manifests["goal_grasp_event"].append(
            make_row(source, grasp_start, grasp_end, total_goal, "goal_grasp_event", color=color, phase="grasp")
        )
        manifests["goal_place_event"].append(
            make_row(source, place_start, place_end, total_goal, "goal_place_event", color=color, phase="place")
        )

    for rows in manifests.values():
        for row in rows:
            validate_row(row, normalized_counts)
        rows.sort(key=lambda row: (row["episode"], row["start"], row["end"], row["task"]))
    return manifests


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def summarize(manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sample_type, rows in manifests.items():
        result[sample_type] = {
            "episodes": len(rows),
            "frames": sum(int(row["end"]) - int(row["start"]) for row in rows),
            "by_goal_group": dict(sorted(Counter(row["goal_group"] for row in rows).items())),
            "by_source_kind": dict(sorted(Counter(row["source_kind"] for row in rows).items())),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only unified Pi0 goal manifests.")
    parser.add_argument(
        "--source-train-list",
        type=Path,
        default=Path("data/episode_lists/pi0_unified_goal_v1/source_train.txt"),
    )
    parser.add_argument(
        "--phase-manifest",
        type=Path,
        default=Path("data/episode_lists/第二版好的分割_60+_60/refined_segments.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/episode_lists/pi0_unified_goal_v1"))
    parser.add_argument("--grasp-pre-frames", type=int, default=60)
    parser.add_argument("--grasp-post-frames", type=int, default=45)
    parser.add_argument("--place-pre-frames", type=int, default=60)
    parser.add_argument("--place-post-frames", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = read_source_list(args.source_train_list)
    phase_rows = read_phase_rows(args.phase_manifest)
    frame_counts = source_frame_counts(sources)
    manifests = build_manifests(
        sources,
        phase_rows,
        frame_counts,
        args.grasp_pre_frames,
        args.grasp_post_frames,
        args.place_pre_frames,
        args.place_post_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "complete_goal": "complete_goal_train.jsonl",
        "goal_grasp_event": "goal_grasp_event_train.jsonl",
        "goal_place_event": "goal_place_event_train.jsonl",
        "atomic_assist": "atomic_assist_train.jsonl",
    }
    for sample_type, filename in filenames.items():
        write_jsonl(args.output_dir / filename, manifests[sample_type])
    combined = [row for sample_type in SAMPLE_TYPES for row in manifests[sample_type]]
    write_jsonl(args.output_dir / "mixed_train_manifest.jsonl", combined)
    summary = {
        "source_train_list": str(args.source_train_list),
        "phase_manifest": str(args.phase_manifest),
        "clip_frames": {
            "grasp_pre": args.grasp_pre_frames,
            "grasp_post": args.grasp_post_frames,
            "place_pre": args.place_pre_frames,
            "place_post": args.place_post_frames,
        },
        "sample_types": summarize(manifests),
        "combined": {
            "episodes": len(combined),
            "frames": sum(int(row["end"]) - int(row["start"]) for row in combined),
        },
        "target_sampling_ratio": {
            "complete_goal": 0.40,
            "goal_grasp_event": 0.25,
            "goal_place_event": 0.25,
            "atomic_assist": 0.10,
        },
    }
    (args.output_dir / "mixed_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for sample_type, values in summary["sample_types"].items():
        print(f"{sample_type}: episodes={values['episodes']} frames={values['frames']} {values['by_goal_group']}")
    print(f"combined: episodes={summary['combined']['episodes']} frames={summary['combined']['frames']}")


if __name__ == "__main__":
    main()
