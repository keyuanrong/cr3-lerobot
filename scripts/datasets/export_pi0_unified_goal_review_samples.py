#!/usr/bin/env python
"""Create a small, autoplay-compatible review manifest for all unified-goal sampling buckets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if not __package__:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTED_MIXTURE_PATH = REPO_ROOT / "src" / "lerobot" / "datasets" / "weighted_mixture.py"
WEIGHTED_MIXTURE_SPEC = importlib.util.spec_from_file_location("lerobot_datasets_weighted_mixture", WEIGHTED_MIXTURE_PATH)
assert WEIGHTED_MIXTURE_SPEC and WEIGHTED_MIXTURE_SPEC.loader
WEIGHTED_MIXTURE = importlib.util.module_from_spec(WEIGHTED_MIXTURE_SPEC)
sys.modules[WEIGHTED_MIXTURE_SPEC.name] = WEIGHTED_MIXTURE
WEIGHTED_MIXTURE_SPEC.loader.exec_module(WEIGHTED_MIXTURE)
UNIFIED_GOAL_BUCKET_QUOTAS = WEIGHTED_MIXTURE.UNIFIED_GOAL_BUCKET_QUOTAS
classify_unified_goal_task = WEIGHTED_MIXTURE.classify_unified_goal_task


MANIFEST_FILENAMES = {
    "complete_goal": "complete_goal_train.jsonl",
    "goal_grasp_event": "goal_grasp_event_train.jsonl",
    "goal_place_event": "goal_place_event_train.jsonl",
    "atomic_assist": "atomic_assist_train.jsonl",
}
SAMPLE_TYPE_PHASES = {
    "complete_goal": "complete",
    "goal_grasp_event": "grasp",
    "goal_place_event": "place",
    "atomic_assist": "atomic",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def review_row(row: dict[str, Any], sample_type: str) -> tuple[str, dict[str, Any]]:
    phase = SAMPLE_TYPE_PHASES[sample_type]
    bucket = classify_unified_goal_task(str(row["task"]), phase)
    target = bucket.rsplit("/", 1)[1]
    result = dict(row)
    result["review_bucket"] = bucket
    result["color"] = target
    result["phase"] = str(row.get("phase", phase)) if sample_type == "atomic_assist" else phase
    return bucket, result


def select_review_rows(
    rows_by_sample_type: dict[str, list[dict[str, Any]]],
    samples_per_bucket: int,
    seed: int,
    required_buckets: tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if samples_per_bucket <= 0:
        raise ValueError("samples_per_bucket must be positive.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_type, rows in rows_by_sample_type.items():
        if sample_type not in SAMPLE_TYPE_PHASES:
            raise ValueError(f"Unknown sample type: {sample_type}")
        for row in rows:
            bucket, normalized = review_row(row, sample_type)
            grouped[bucket].append(normalized)

    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    required_buckets = required_buckets or tuple(grouped)
    for bucket in required_buckets:
        candidates = list(grouped.get(bucket, []))
        rng.shuffle(candidates)
        chosen: list[dict[str, Any]] = []
        seen_episodes: set[str] = set()
        for row in candidates:
            episode = str(row["episode"])
            if episode in seen_episodes:
                continue
            chosen.append(row)
            seen_episodes.add(episode)
            if len(chosen) == samples_per_bucket:
                break
        if len(chosen) != samples_per_bucket:
            raise ValueError(
                f"Bucket {bucket!r} has only {len(chosen)} distinct episodes; need {samples_per_bucket}."
            )
        selected[bucket] = chosen
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/episode_lists/pi0_unified_goal_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/episode_lists/pi0_unified_goal_v1/review_samples_5"),
    )
    parser.add_argument("--samples-per-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_by_sample_type = {
        sample_type: read_jsonl(args.manifest_dir / filename)
        for sample_type, filename in MANIFEST_FILENAMES.items()
    }
    selected = select_review_rows(
        rows_by_sample_type,
        args.samples_per_bucket,
        args.seed,
        required_buckets=tuple(UNIFIED_GOAL_BUCKET_QUOTAS),
    )
    review_rows = [row for bucket in UNIFIED_GOAL_BUCKET_QUOTAS for row in selected[bucket]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "review_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    summary = {
        "samples_per_bucket": args.samples_per_bucket,
        "seed": args.seed,
        "total_segments": len(review_rows),
        "buckets": {bucket: len(rows) for bucket, rows in selected.items()},
    }
    (args.output_dir / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Review segments: {len(review_rows)}")
    print(f"Manifest: {manifest_path}")
    for bucket, rows in selected.items():
        print(f"  {bucket:16} {len(rows)}")


if __name__ == "__main__":
    main()
