#!/usr/bin/env python

"""Split long CR3 VLA episodes into short language-conditioned subtasks.

The original teleop dataset uses one long instruction:

    put the red block into the black frame, then put the green block into the
    black frame, finally put the yellow block into the black frame

For SmolVLA fine-tuning, this script rewrites each long episode as three
clearer subtask episodes:

    put the red block into the black frame
    put the green block into the black frame
    put the yellow block into the black frame

The source dataset is never modified. A new LeRobotDataset is written together
with a split manifest that records the source episode and frame range for each
new subtask episode.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.datasets import LeRobotDataset
from lerobot.configs import VideoEncoderConfig
from sim.cr3_mujoco.collect_stack_blocks_vla_dataset import make_features


DEFAULT_SOURCE_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla_teleop_clean"
DEFAULT_OUTPUT_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla_teleop_subtasks"
DEFAULT_OUTPUT_REPO_ID = "local/cr3_lmg90_stack_blocks_vla_teleop_subtasks"

FRONT_KEY = "observation.images.front_rgb"
WRIST_KEY = "observation.images.wrist_rgb"

SUBTASKS = (
    ("red", "put the red block into the black frame"),
    ("green", "put the green block into the black frame"),
    ("yellow", "put the yellow block into the black frame"),
)


@dataclass(frozen=True)
class SourceEpisode:
    old_episode_index: int
    parquet_path: Path
    front_video: Path
    wrist_video: Path
    df: pd.DataFrame


@dataclass(frozen=True)
class Segment:
    old_episode_index: int
    subtask_index: int
    block: str
    task: str
    start: int
    end: int
    method: str

    @property
    def length(self) -> int:
        return self.end - self.start


def infer_image_size(source_root: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    info = json.loads((source_root / "meta" / "info.json").read_text())
    front_shape = info["features"][FRONT_KEY]["shape"]
    wrist_shape = info["features"][WRIST_KEY]["shape"]
    return (int(front_shape[0]), int(front_shape[1])), (int(wrist_shape[0]), int(wrist_shape[1]))


def make_subtask_features(
    front_size: tuple[int, int],
    wrist_size: tuple[int, int],
    *,
    use_videos: bool,
) -> dict:
    features = make_features(front_size, wrist_size)
    if not use_videos:
        features[FRONT_KEY]["dtype"] = "image"
        features[WRIST_KEY]["dtype"] = "image"
    return features


def video_path_for_parquet(source_root: Path, video_key: str, parquet_path: Path) -> Path:
    file_index = parquet_path.stem.split("-")[-1]
    chunk_name = parquet_path.parent.name
    return source_root / "videos" / video_key / chunk_name / f"file-{file_index}.mp4"


def readable_source_episodes(source_root: Path) -> list[SourceEpisode]:
    episodes: list[SourceEpisode] = []
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

        front_video = video_path_for_parquet(source_root, FRONT_KEY, parquet_path)
        wrist_video = video_path_for_parquet(source_root, WRIST_KEY, parquet_path)
        if not front_video.exists() or not wrist_video.exists():
            print(f"skip {parquet_path}: missing matching videos")
            continue

        for episode_id in sorted(int(x) for x in df["episode_index"].unique().tolist()):
            ep_df = df[df["episode_index"] == episode_id].sort_values("frame_index").reset_index(drop=True)
            episodes.append(
                SourceEpisode(
                    old_episode_index=episode_id,
                    parquet_path=parquet_path,
                    front_video=front_video,
                    wrist_video=wrist_video,
                    df=ep_df,
                )
            )
    return episodes


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


def iter_video_rgb(video_path: Path):
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def action_gripper_values(df: pd.DataFrame) -> np.ndarray:
    return np.asarray([np.asarray(action, dtype=np.float32)[-1] for action in df["action"]], dtype=np.float32)


def split_by_thirds(num_frames: int) -> tuple[int, int]:
    return max(1, num_frames // 3), max(2, (2 * num_frames) // 3)


def split_by_gripper_release(
    df: pd.DataFrame,
    *,
    min_segment_frames: int,
    release_pad_frames: int,
) -> tuple[int, int] | None:
    """Infer phase boundaries from gripper close->open transitions.

    The CR3 MuJoCo recordings store gripper command as the last action value.
    Open is usually lower and grasp/close is higher. For a block-in-frame task,
    the end of each subtask often appears as a falling edge after the block is
    released. We use the first two well-separated falling edges as boundaries.
    """

    gripper = action_gripper_values(df)
    if len(gripper) < min_segment_frames * 3:
        return None

    low = float(np.nanpercentile(gripper, 20))
    high = float(np.nanpercentile(gripper, 80))
    if high - low < 1e-4:
        return None
    threshold = low + 0.45 * (high - low)
    closed = gripper > threshold

    release_edges: list[int] = []
    for idx in range(1, len(closed)):
        if closed[idx - 1] and not closed[idx]:
            boundary = min(idx + release_pad_frames, len(closed) - 1)
            if boundary < min_segment_frames:
                continue
            if release_edges and boundary - release_edges[-1] < min_segment_frames:
                continue
            release_edges.append(boundary)
            if len(release_edges) >= 2:
                break

    if len(release_edges) < 2:
        return None
    first, second = release_edges[:2]
    if first < min_segment_frames or second - first < min_segment_frames or len(df) - second < min_segment_frames:
        return None
    return first, second


def make_segments(
    episode: SourceEpisode,
    *,
    split_mode: str,
    min_segment_frames: int,
    release_pad_frames: int,
) -> list[Segment]:
    num_frames = len(episode.df)
    if num_frames < min_segment_frames * 3:
        return []

    method = split_mode
    if split_mode == "thirds":
        cut1, cut2 = split_by_thirds(num_frames)
    elif split_mode == "gripper":
        cuts = split_by_gripper_release(
            episode.df,
            min_segment_frames=min_segment_frames,
            release_pad_frames=release_pad_frames,
        )
        if cuts is None:
            return []
        cut1, cut2 = cuts
    elif split_mode == "auto":
        cuts = split_by_gripper_release(
            episode.df,
            min_segment_frames=min_segment_frames,
            release_pad_frames=release_pad_frames,
        )
        if cuts is None:
            cut1, cut2 = split_by_thirds(num_frames)
            method = "thirds-fallback"
        else:
            cut1, cut2 = cuts
            method = "gripper"
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    ranges = ((0, cut1), (cut1, cut2), (cut2, num_frames))
    segments: list[Segment] = []
    for subtask_idx, ((block, task), (start, end)) in enumerate(zip(SUBTASKS, ranges, strict=True)):
        if end - start < min_segment_frames:
            return []
        segments.append(
            Segment(
                old_episode_index=episode.old_episode_index,
                subtask_index=subtask_idx,
                block=block,
                task=task,
                start=int(start),
                end=int(end),
                method=method,
            )
        )
    return segments


def flush_saved_episode(dataset: LeRobotDataset) -> None:
    flush_metadata = getattr(dataset.meta, "_flush_metadata_buffer", None)
    if callable(flush_metadata):
        flush_metadata()


def write_manifest(output_root: Path, segments: list[Segment]) -> None:
    manifest_path = output_root / "subtask_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_episode_index",
                "source_episode_index",
                "subtask_index",
                "block",
                "task",
                "start_frame",
                "end_frame_exclusive",
                "num_frames",
                "split_method",
            ],
        )
        writer.writeheader()
        for new_idx, segment in enumerate(segments):
            writer.writerow(
                {
                    "new_episode_index": new_idx,
                    "source_episode_index": segment.old_episode_index,
                    "subtask_index": segment.subtask_index,
                    "block": segment.block,
                    "task": segment.task,
                    "start_frame": segment.start,
                    "end_frame_exclusive": segment.end,
                    "num_frames": segment.length,
                    "split_method": segment.method,
                }
            )
    print(f"manifest written to: {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--split-mode",
        choices=("auto", "gripper", "thirds"),
        default="auto",
        help="auto tries gripper release boundaries first and falls back to thirds.",
    )
    parser.add_argument("--min-segment-frames", type=int, default=30)
    parser.add_argument(
        "--release-pad-frames",
        type=int,
        default=15,
        help="Extra frames kept after a gripper release before ending the subtask.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Debug helper: only process the first N source episodes.",
    )
    parser.add_argument(
        "--parallel-encoding",
        action="store_true",
        help="Use LeRobot's parallel video encoding. Faster, but less stable for many short AV1 episodes.",
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        help="Store camera observations as videos. Defaults to image files for faster dataset rewriting.",
    )
    parser.add_argument(
        "--video-codec",
        default="h264",
        help="Video codec used with --use-videos. h264 is much faster than the default libsvtav1.",
    )
    parser.add_argument("--video-crf", type=float, default=28)
    parser.add_argument("--video-preset", default="ultrafast")
    parser.add_argument("--encoder-threads", type=int, default=2)
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
    front_size, wrist_size = infer_image_size(args.source_root)
    source_episodes = readable_source_episodes(args.source_root)
    if args.max_episodes is not None:
        source_episodes = source_episodes[: args.max_episodes]

    segments_by_episode: dict[int, list[Segment]] = {}
    all_segments: list[Segment] = []
    for episode in source_episodes:
        segments = make_segments(
            episode,
            split_mode=args.split_mode,
            min_segment_frames=max(args.min_segment_frames, 1),
            release_pad_frames=max(args.release_pad_frames, 0),
        )
        if not segments:
            print(f"skip episode {episode.old_episode_index}: could not create valid subtask splits")
            continue
        segments_by_episode[episode.old_episode_index] = segments
        all_segments.extend(segments)

    print("VLA subtask split plan")
    print(f"source: {args.source_root}")
    print(f"output: {args.output_root}")
    print(f"source episodes considered: {len(source_episodes)}")
    print(f"subtask episodes to write: {len(all_segments)}")
    for new_idx, segment in enumerate(all_segments):
        print(
            f"  new {new_idx:04d}: old {segment.old_episode_index:04d} "
            f"{segment.block:>6} frames [{segment.start}, {segment.end}) "
            f"len={segment.length} method={segment.method}"
        )
    if args.dry_run:
        return

    camera_encoder = None
    if args.use_videos:
        camera_encoder = VideoEncoderConfig(
            vcodec=args.video_codec,
            crf=args.video_crf,
            preset=args.video_preset,
            fast_decode=1,
        )

    dataset = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=args.output_root,
        fps=fps,
        features=make_subtask_features(front_size, wrist_size, use_videos=args.use_videos),
        robot_type="cr3_lmg90_mujoco",
        use_videos=args.use_videos,
        image_writer_threads=4,
        metadata_buffer_size=1,
        camera_encoder=camera_encoder,
        encoder_threads=args.encoder_threads if args.use_videos else None,
    )

    written_segments: list[Segment] = []
    try:
        for episode in source_episodes:
            segments = segments_by_episode.get(episode.old_episode_index)
            if not segments:
                continue

            print(f"streaming source episode {episode.old_episode_index} ({len(episode.df)} frames)")
            segment_idx = 0
            segment = segments[segment_idx]
            print(
                f"writing {segment.block} subtask from old episode {episode.old_episode_index}: "
                f"frames [{segment.start}, {segment.end})"
            )

            frame_count = 0
            frame_iter = zip(iter_video_rgb(episode.front_video), iter_video_rgb(episode.wrist_video), strict=False)
            for frame_idx, (front_frame, wrist_frame) in enumerate(frame_iter):
                if frame_idx >= len(episode.df):
                    break

                while frame_idx >= segment.end:
                    dataset.save_episode(parallel_encoding=args.parallel_encoding)
                    flush_saved_episode(dataset)
                    written_segments.append(segment)
                    segment_idx += 1
                    if segment_idx >= len(segments):
                        segment = None
                        break
                    segment = segments[segment_idx]
                    print(
                        f"writing {segment.block} subtask from old episode {episode.old_episode_index}: "
                        f"frames [{segment.start}, {segment.end})"
                    )
                if segment is None:
                    break

                if frame_idx < segment.start:
                    continue
                row = episode.df.iloc[frame_idx]
                dataset.add_frame(
                    {
                        FRONT_KEY: front_frame,
                        WRIST_KEY: wrist_frame,
                        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                        "action": np.asarray(row["action"], dtype=np.float32),
                        "task": segment.task,
                    }
                )
                frame_count += 1

            if segment_idx < len(segments):
                dataset.save_episode(parallel_encoding=args.parallel_encoding)
                flush_saved_episode(dataset)
                written_segments.append(segments[segment_idx])
                segment_idx += 1

            if frame_count < len(episode.df):
                print(
                    f"warning: source episode {episode.old_episode_index} wrote {frame_count} "
                    f"frames from {len(episode.df)} expected frames"
                )

            for missing_segment in segments[segment_idx:]:
                print(
                    f"warning: did not write {missing_segment.block} subtask from old episode "
                    f"{episode.old_episode_index}"
                )
    finally:
        dataset.finalize()

    write_manifest(args.output_root, written_segments)
    print(f"subtask dataset written to: {args.output_root}")


if __name__ == "__main__":
    main()
