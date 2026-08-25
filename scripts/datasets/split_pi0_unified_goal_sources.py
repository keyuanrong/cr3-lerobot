#!/usr/bin/env python
"""Create a fixed source-episode train/validation split for unified Pi0 training."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TASK_KINDS = ("red", "green", "yellow", "full")


def read_episode_list(path: Path) -> list[Path]:
    episodes = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(episodes) != len(set(episodes)):
        raise ValueError(f"Duplicate source episodes in {path}")
    invalid = [episode for episode in episodes if episode.parent.name not in TASK_KINDS]
    if invalid:
        raise ValueError(f"Unsupported task directory: {invalid[0]}")
    return episodes


def split_sources(
    episodes: list[Path], *, validation_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    by_kind: dict[str, list[Path]] = defaultdict(list)
    for episode in episodes:
        by_kind[episode.parent.name].append(episode)

    train: list[Path] = []
    validation: list[Path] = []
    for kind in TASK_KINDS:
        group = sorted(by_kind[kind])
        if not group:
            raise ValueError(f"No source episodes for {kind}")
        random.Random(f"{seed}:{kind}").shuffle(group)
        validation_count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])

    return sorted(train), sorted(validation)


def write_list(path: Path, episodes: list[Path]) -> None:
    path.write_text("\n".join(str(episode) for episode in episodes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible source-level split for unified Pi0 mixed training."
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=Path("data/episode_lists/train_20260721_20260726_final_good.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/episode_lists/pi0_unified_goal_v1"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = read_episode_list(args.input_list)
    train, validation = split_sources(
        episodes, validation_fraction=args.validation_fraction, seed=args.seed
    )

    if set(train) & set(validation) or set(train) | set(validation) != set(episodes):
        raise RuntimeError("Source split is not a disjoint partition")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_list(args.output_dir / "source_train.txt", train)
    write_list(args.output_dir / "source_validation.txt", validation)
    summary = {
        "input_list": str(args.input_list),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "source_counts": {
            "all": dict(sorted(Counter(path.parent.name for path in episodes).items())),
            "train": dict(sorted(Counter(path.parent.name for path in train).items())),
            "validation": dict(sorted(Counter(path.parent.name for path in validation).items())),
        },
        "totals": {"all": len(episodes), "train": len(train), "validation": len(validation)},
    }
    (args.output_dir / "source_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"all={len(episodes)} train={len(train)} validation={len(validation)}")
    for kind in TASK_KINDS:
        print(
            f"{kind}: train={summary['source_counts']['train'].get(kind, 0)} "
            f"validation={summary['source_counts']['validation'].get(kind, 0)}"
        )


if __name__ == "__main__":
    main()
