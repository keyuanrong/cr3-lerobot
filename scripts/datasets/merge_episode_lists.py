#!/usr/bin/env python
"""Merge episode-list text files while preserving order and removing duplicates."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw episode lists into one de-duplicated list.")
    parser.add_argument("--input-list", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-paths", action="store_true", help="Fail if an episode directory is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged: list[str] = []
    seen: set[str] = set()
    source_counts: list[tuple[Path, int]] = []

    for list_path in args.input_list:
        count = 0
        for line in list_path.read_text(encoding="utf-8").splitlines():
            episode = line.strip()
            if not episode or episode.startswith("#"):
                continue
            if args.verify_paths and not Path(episode).is_dir():
                raise SystemExit(f"Missing episode directory: {episode}")
            if episode not in seen:
                seen.add(episode)
                merged.append(episode)
                count += 1
        source_counts.append((list_path, count))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
    print(f"wrote {len(merged)} unique episodes: {args.output}")
    for list_path, count in source_counts:
        print(f"  {list_path}: added={count}")


if __name__ == "__main__":
    main()
