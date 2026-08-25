import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if not __package__:
    sys.path.insert(0, str(REPO_ROOT))


STATE_FIELDS = ["q1", "q2", "q3", "q4", "q5", "q6", "gripper"]
ACTION_FIELDS = ["q1", "q2", "q3", "q4", "q5", "q6", "gripper"]
DEFAULT_FPS = 30
JOINT_FIELDS = ["q1", "q2", "q3", "q4", "q5", "q6"]
TARGET_JOINT_FIELDS = ["target_q1", "target_q2", "target_q3", "target_q4", "target_q5", "target_q6"]
DEFAULT_IMAGE_COLUMNS = ["front_rgb", "wrist_rgb"]
DEFAULT_IMAGE_KEYS = {
    "front_rgb": "observation.images.front_rgb",
    "wrist_rgb": "observation.images.wrist_rgb",
}


def import_lerobot_dataset():
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def import_video_encoder_config():
    from lerobot.configs.video import VideoEncoderConfig

    return VideoEncoderConfig


def list_episodes(input_root: Path) -> list[Path]:
    episodes = [
        path
        for path in sorted(input_root.glob("drag_episode_*"))
        if (path / "data.csv").is_file()
    ]
    if not episodes and (input_root / "data.csv").is_file():
        episodes = [input_root]
    return episodes


def read_episode_list(path: Path) -> list[Path]:
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        episode_dir = Path(line).expanduser()
        if not episode_dir.is_absolute():
            episode_dir = (path.parent.parent.parent / episode_dir).resolve()
        episodes.append(episode_dir)
    return episodes


def read_segment_manifest(path: Path) -> list[dict]:
    segments = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        segment = json.loads(line)
        required = {"episode", "start", "end", "task"}
        missing = required - segment.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing keys: {sorted(missing)}")
        segment["episode"] = Path(segment["episode"]).expanduser()
        segments.append(segment)
    return segments


def read_rows(episode_dir: Path, image_columns: list[str]) -> list[dict]:
    with (episode_dir / "data.csv").open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) < 2:
        print(f"Skipping {episode_dir}: fewer than 2 frames")
        return []
    missing = [field for field in JOINT_FIELDS + image_columns if field not in rows[0]]
    if missing:
        raise ValueError(
            f"{episode_dir}/data.csv does not contain {missing}. "
            "Record new episodes with the updated RGB-D record_drag_dataset.py before converting."
        )
    return rows


def row_joints(row: dict) -> np.ndarray:
    return np.asarray([float(row[field]) for field in JOINT_FIELDS], dtype=np.float32)


def row_gripper_action(row: dict, threshold: float, semantics: str) -> float:
    value = row.get("gripper", "")
    if value == "":
        return 0.0 if semantics == "close_high" else 1.0

    is_open = float(value) >= threshold
    if semantics == "close_high":
        return 0.0 if is_open else 1.0
    if semantics == "open_high":
        return 1.0 if is_open else 0.0
    raise ValueError(f"Unsupported gripper action semantics: {semantics}")


def row_gripper_value(row: dict) -> float:
    value = row.get("gripper", "")
    if value == "":
        return 100.0
    return float(value)


def row_state(row: dict, gripper_threshold: float) -> np.ndarray:
    return np.concatenate(
        [row_joints(row), np.asarray([row_gripper_value(row)], dtype=np.float32)]
    )


def row_target_joints(row: dict, next_row: dict) -> np.ndarray:
    if all(field in row and row[field] != "" for field in TARGET_JOINT_FIELDS):
        return np.asarray([float(row[field]) for field in TARGET_JOINT_FIELDS], dtype=np.float32)
    return row_joints(next_row)


def row_action(row: dict, next_row: dict, gripper_threshold: float, gripper_action_semantics: str) -> np.ndarray:
    target = row_target_joints(row, next_row)
    gripper_action = np.asarray(
        [row_gripper_action(next_row, gripper_threshold, gripper_action_semantics)],
        dtype=np.float32,
    )
    return np.concatenate([target, gripper_action])


def clean_task_text(task: str) -> str:
    return task.strip().rstrip("、，,;；").strip()


def read_rgb_image(cv2, episode_dir: Path, row: dict, column: str) -> np.ndarray:
    image_path = episode_dir / row[column]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def infer_image_shapes(cv2, episodes: list[Path], image_columns: list[str]) -> dict[str, tuple[int, int, int]]:
    for episode_dir in episodes:
        rows = read_rows(episode_dir, image_columns)
        if not rows:
            continue
        return {
            column: read_rgb_image(cv2, episode_dir, rows[0], column).shape
            for column in image_columns
        }
    raise ValueError("No valid episode images found.")


def make_features(image_keys: dict[str, str], image_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features = {}
    for column, key in image_keys.items():
        features[key] = {
            "dtype": "video",
            "shape": image_shapes[column],
            "names": ["height", "width", "channel"],
        }
    features.update({
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_FIELDS),),
            "names": STATE_FIELDS,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_FIELDS),),
            "names": ACTION_FIELDS,
        },
    })
    return features


def convert_episode(
    cv2,
    dataset,
    episode_dir: Path,
    image_keys: dict[str, str],
    task: str,
    task_source: str,
    drop_last_frame: bool,
    gripper_threshold: float,
    gripper_action_semantics: str,
    image_columns: list[str],
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    rows = read_rows(episode_dir, image_columns)
    if not rows:
        return 0

    frame_count = len(rows) - 1 if drop_last_frame else len(rows)
    end_frame = frame_count if end_frame is None else min(end_frame, frame_count)
    if not 0 <= start_frame < end_frame:
        raise ValueError(f"Invalid segment [{start_frame}, {end_frame}) for {episode_dir} with {frame_count} frames")

    for index in range(start_frame, end_frame):
        row = rows[index]
        # Do not let the final action of one labelled phase point into the next phase.
        next_row = rows[min(index + 1, end_frame - 1)]
        frame = {
            "observation.state": row_state(row, gripper_threshold),
            "action": row_action(row, next_row, gripper_threshold, gripper_action_semantics),
            "task": clean_task_text(row.get("task", "")) if task_source == "row" else clean_task_text(task),
        }
        for column, key in image_keys.items():
            frame[key] = read_rgb_image(cv2, episode_dir, row, column)
        dataset.add_frame(frame)

    dataset.save_episode()
    return end_frame - start_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data", help="Folder containing drag_episode_* directories, or one episode directory.")
    parser.add_argument("--episode-list", help="Text file with one episode directory per line. Overrides --input-root.")
    parser.add_argument("--segment-manifest", help="JSONL segments from the visual phase splitter. Overrides --episode-list and --input-root.")
    parser.add_argument("--output-root", default="lerobot_data")
    parser.add_argument("--repo-id", default="local/dobot_cr3_drag")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--task", default="drag the leader robot and copy with the follower")
    parser.add_argument(
        "--task-source",
        choices=["argument", "row"],
        default="argument",
        help="Use --task for all frames, or read and clean the task column from each raw CSV row.",
    )
    parser.add_argument("--front-rgb-key", default=DEFAULT_IMAGE_KEYS["front_rgb"])
    parser.add_argument("--wrist-rgb-key", default=DEFAULT_IMAGE_KEYS["wrist_rgb"])
    parser.add_argument("--use-wrist-depth", action="store_true", help="Include wrist_depth if it exists in the raw episodes.")
    parser.add_argument("--wrist-depth-key", default="observation.images.wrist_depth")
    parser.add_argument("--robot-type", default="dobot_cr3")
    parser.add_argument("--gripper-threshold", type=float, default=50.0)
    parser.add_argument(
        "--gripper-action-semantics",
        choices=["close_high", "open_high"],
        default="close_high",
        help=(
            "How to encode the action gripper dimension. close_high matches PI0/OpenPI "
            "(0=open, 1=close); open_high keeps the previous local convention (1=open, 0=close)."
        ),
    )
    parser.add_argument("--no-videos", action="store_true", help="Store images without encoding videos if supported by your LeRobot version.")
    parser.add_argument(
        "--video-codec",
        default="h264",
        choices=["h264", "hevc", "libsvtav1", "auto"],
        help="Video codec for LeRobot videos. h264 is faster; libsvtav1 is smaller but much slower.",
    )
    parser.add_argument("--keep-last-frame", action="store_true", help="Keep the last frame by reusing its own joint target.")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed. Run: pip install opencv-python") from exc

    LeRobotDataset = import_lerobot_dataset()
    VideoEncoderConfig = import_video_encoder_config()

    input_root = Path(args.input_root)
    segments = read_segment_manifest(Path(args.segment_manifest)) if args.segment_manifest else None
    episodes = [segment["episode"] for segment in segments] if segments is not None else (read_episode_list(Path(args.episode_list)) if args.episode_list else list_episodes(input_root))
    if not episodes:
        raise SystemExit(f"No drag episodes found under {input_root}")

    missing = [episode for episode in episodes if not (episode / "data.csv").is_file()]
    if missing:
        raise SystemExit("Episode list contains missing data.csv files:\n" + "\n".join(str(path) for path in missing))

    image_columns = DEFAULT_IMAGE_COLUMNS.copy()
    image_keys = {
        "front_rgb": args.front_rgb_key,
        "wrist_rgb": args.wrist_rgb_key,
    }
    if args.use_wrist_depth:
        image_columns.append("wrist_depth")
        image_keys["wrist_depth"] = args.wrist_depth_key

    image_shapes = infer_image_shapes(cv2, episodes, image_columns)
    dataset_root = Path(args.output_root) / args.repo_id
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        fps=args.fps,
        features=make_features(image_keys, image_shapes),
        robot_type=args.robot_type,
        use_videos=not args.no_videos,
        camera_encoder=VideoEncoderConfig(vcodec=args.video_codec),
    )

    total_frames = 0
    iterable = segments if segments is not None else [{"episode": episode, "start": 0, "end": None, "task": args.task} for episode in episodes]
    for segment in iterable:
        episode_dir = segment["episode"]
        frames = convert_episode(
            cv2,
            dataset,
            episode_dir,
            image_keys,
            segment["task"],
            "argument" if segments is not None else args.task_source,
            drop_last_frame=not args.keep_last_frame,
            gripper_threshold=args.gripper_threshold,
            gripper_action_semantics=args.gripper_action_semantics,
            image_columns=image_columns,
            start_frame=int(segment["start"]),
            end_frame=None if segment["end"] is None else int(segment["end"]),
        )
        total_frames += frames
        suffix = f" [{segment.get('phase')}]" if segments is not None else ""
        print(f"Converted {episode_dir}{suffix}: {frames} frames")

    dataset.finalize()
    print(f"Converted {len(iterable)} episodes, {total_frames} frames -> {dataset_root}")


if __name__ == "__main__":
    main()
