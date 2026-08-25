#!/usr/bin/env python
"""Verify exact phase-and-task sampling quotas before a Pi0 unified-goal run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if not __package__:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPO_IDS = [
    "local/cr3_pi0_unified_goal_v1_complete_goal",
    "local/cr3_pi0_unified_goal_v3_grasp_transition_event",
    "local/cr3_pi0_unified_goal_v1_goal_grasp_event",
    "local/cr3_pi0_unified_goal_v2_release_event",
    "local/cr3_pi0_unified_goal_v1_goal_place_event",
    "local/cr3_pi0_unified_goal_v1_atomic_assist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lerobot_data"))
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        help=(
            "Dataset repo ID. Provide exactly six values in "
            "complete/grasp_transition/grasp/release/place/atomic order."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[1_000, 5_000],
        help="Audit lengths. Defaults to one and five complete 1000-sample blocks.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lerobot.datasets.multi_dataset import MultiLeRobotDataset
    from lerobot.datasets.weighted_mixture import (
        FixedQuotaBucketSampler,
        UNIFIED_GOAL_BUCKET_QUOTAS,
        build_unified_goal_bucket_indices,
        scaled_bucket_quotas,
        unified_goal_bucket_counts,
    )

    repo_ids = args.repo_ids or DEFAULT_REPO_IDS
    if len(repo_ids) != 6:
        raise ValueError(
            "Provide exactly six --repo-id values in "
            "complete/grasp_transition/grasp/release/place/atomic order."
        )
    if any(count <= 0 for count in args.samples):
        raise ValueError("Every --samples value must be positive.")

    dataset = MultiLeRobotDataset(repo_ids=repo_ids, root=args.root, download_videos=False)
    bucket_indices = build_unified_goal_bucket_indices(dataset._datasets)
    available_frames = {bucket: len(indices) for bucket, indices in bucket_indices.items()}
    print("Available frames by bucket:")
    for bucket, count in available_frames.items():
        print(f"  {bucket:16} {count}")

    report: dict[str, object] = {
        "repo_ids": repo_ids,
        "root": str(args.root),
        "available_frames": available_frames,
        "audits": {},
    }
    for sample_count in args.samples:
        sampler = FixedQuotaBucketSampler(
            bucket_indices=bucket_indices,
            quotas=UNIFIED_GOAL_BUCKET_QUOTAS,
            num_samples=sample_count,
            block_size=1_000,
            seed=args.seed,
        )
        actual = unified_goal_bucket_counts(list(sampler), bucket_indices)
        expected = scaled_bucket_quotas(UNIFIED_GOAL_BUCKET_QUOTAS, sample_count)
        if actual != expected:
            raise RuntimeError(f"Sampling audit failed for {sample_count} samples: {actual} != {expected}")

        print(f"\nAudit passed: {sample_count} samples")
        for bucket, count in actual.items():
            print(f"  {bucket:16} {count}")
        report["audits"][str(sample_count)] = {"expected": expected, "actual": actual}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
