#!/usr/bin/env python

"""Build a clean LeRobotDataset copy from readable CR3 VLA recordings.

This is intentionally non-destructive: the source dataset is never modified.
It reads each valid data parquet + matching front/wrist videos, remaps episode
indices to a continuous 0..N-1 range, and writes a fresh LeRobotDataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import av


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.datasets import LeRobotDataset
from sim.cr3_mujoco.collect_stack_blocks_vla_dataset import make_features


DEFAULT_SOURCE_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla_teleop"
DEFAULT_OUTPUT_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla_teleop_clean"
DEFAULT_OUTPUT_REPO_ID = "local/cr3_lmg90_stack_blocks_vla_teleop_clean"


def read_task(source_root: Path) -> str:
    tasks_path = source_root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks = pd.read_parquet(tasks_path)
        if len(tasks.index) > 0:
            if "task" in tasks.columns:
                return str(tasks.iloc[0]["task"])
            if tasks.index.name == "task":
                return str(tasks.index[0])
    return "put the red block into the black frame, then put the green block into the black frame, finally put the yellow block into the black frame"


def readable_episode_files(source_root: Path) -> list[tuple[Path, pd.DataFrame]]:
    episodes: list[tuple[Path, pd.DataFrame]] = []
    for parquet_path in sorted((source_root / "data").glob("chunk-*/*.parquet")):
        try:
            table = pq.read_table(parquet_path)
        except Exception as exc:
            print(f"skip broken parquet: {parquet_path} ({exc})")
            continue
        df = table.to_pandas()
        if len(df.index) == 0:
            continue
        if "episode_index" not in df or "frame_index" not in df:
            print(f"skip parquet without episode/frame columns: {parquet_path}")
            continue
        episodes.append((parquet_path, df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)))
    return episodes


def video_path_for_parquet(source_root: Path, video_key: str, parquet_path: Path) -> Path:
    file_index = parquet_path.stem.split("-")[-1]
    chunk_name = parquet_path.parent.name
    return source_root / "videos" / video_key / chunk_name / f"file-{file_index}.mp4"


def read_video_rgb(video_path: Path, expected_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= expected_frames:
                break
    if len(frames) < expected_frames:
        raise RuntimeError(f"Video too short: {video_path} has {len(frames)} frames, need {expected_frames}")
    return frames


def infer_image_size(source_root: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    info = json.loads((source_root / "meta" / "info.json").read_text())
    front_shape = info["features"]["observation.images.front_rgb"]["shape"]
    wrist_shape = info["features"]["observation.images.wrist_rgb"]["shape"]
    return (int(front_shape[0]), int(front_shape[1])), (int(wrist_shape[0]), int(wrist_shape[1]))


def flush_saved_episode(dataset: LeRobotDataset) -> None:
    flush_metadata = getattr(dataset.meta, "_flush_metadata_buffer", None)
    if callable(flush_metadata):
        flush_metadata()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.source_root}")
    if args.output_root.exists() and not args.dry_run:
        if not args.overwrite_output:
            raise FileExistsError(f"Output already exists: {args.output_root}. Use --overwrite-output.")
        shutil.rmtree(args.output_root)

    info = json.loads((args.source_root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    task = read_task(args.source_root)
    front_size, wrist_size = infer_image_size(args.source_root)
    episode_files = readable_episode_files(args.source_root)

    readable = []
    for parquet_path, df in episode_files:
        front_video = video_path_for_parquet(args.source_root, "observation.images.front_rgb", parquet_path)
        wrist_video = video_path_for_parquet(args.source_root, "observation.images.wrist_rgb", parquet_path)
        if not front_video.exists() or not wrist_video.exists():
            print(f"skip {parquet_path}: missing matching videos")
            continue
        episode_ids = sorted(int(x) for x in df["episode_index"].unique().tolist())
        for episode_id in episode_ids:
            ep_df = df[df["episode_index"] == episode_id].sort_values("frame_index").reset_index(drop=True)
            readable.append((episode_id, parquet_path, front_video, wrist_video, ep_df))

    print("Clean VLA dataset plan")
    print(f"source: {args.source_root}")
    print(f"output: {args.output_root}")
    print(f"source meta episodes: {info.get('total_episodes')}")
    print(f"readable episodes: {len(readable)}")
    for new_idx, (old_idx, parquet_path, _front_video, _wrist_video, ep_df) in enumerate(readable):
        print(f"  old episode {old_idx} -> new episode {new_idx}: {len(ep_df)} frames from {parquet_path.name}")
    if args.dry_run:
        return

    dataset = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=args.output_root,
        fps=fps,
        features=make_features(front_size, wrist_size),
        robot_type="cr3_lmg90_mujoco",
        use_videos=True,
        image_writer_threads=4,
        metadata_buffer_size=1,
    )
    try:
        for new_idx, (old_idx, _parquet_path, front_video, wrist_video, ep_df) in enumerate(readable):
            print(f"writing clean episode {new_idx} from old episode {old_idx} ({len(ep_df)} frames)")
            front_frames = read_video_rgb(front_video, len(ep_df))
            wrist_frames = read_video_rgb(wrist_video, len(ep_df))
            for frame_idx, row in ep_df.iterrows():
                dataset.add_frame(
                    {
                        "observation.images.front_rgb": front_frames[frame_idx],
                        "observation.images.wrist_rgb": wrist_frames[frame_idx],
                        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                        "action": np.asarray(row["action"], dtype=np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode()
            flush_saved_episode(dataset)
    finally:
        dataset.finalize()

    print(f"clean dataset written to: {args.output_root}")


if __name__ == "__main__":
    main()
