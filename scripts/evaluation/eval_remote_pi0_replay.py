#!/usr/bin/env python
"""Replay held-out LeRobot trajectories against a remote Pi0 policy server.

This is an offline, teacher-forced evaluation: recorded observations are replayed
at the same request/control cadence as the real robot client, but no robot command
is ever sent. The remote server protocol matches ``run_remote_pi0_policy.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle  # nosec B403 - trusted LAN policy server.
import socket
import struct
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if not __package__:
    sys.path.insert(0, str(REPO_ROOT))


def request_frame_indices(total_frames: int, fps: float, request_hz: float) -> list[int]:
    """Return frame indices at the same cadence as the live request loop."""
    if total_frames <= 0:
        return []
    if fps <= 0 or request_hz <= 0:
        raise ValueError("fps and request_hz must be positive.")
    stride = max(1, round(fps / request_hz))
    return list(range(0, total_frames, stride))


def apply_latency_compensation(
    actions: np.ndarray,
    roundtrip_ms: float,
    control_hz: float,
    min_remaining_actions: int,
) -> tuple[np.ndarray | None, int]:
    """Discard actions that would already be stale when a response arrives."""
    skipped = max(0, int(round(roundtrip_ms / 1000.0 * control_hz)))
    remaining = np.asarray(actions, dtype=np.float32)[skipped:]
    if len(remaining) < min_remaining_actions:
        return None, skipped
    return remaining, skipped


def _is_close(value: float, semantics: str) -> bool:
    high = float(value) >= 0.5
    if semantics == "close_high":
        return high
    if semantics == "open_high":
        return not high
    raise ValueError(f"Unsupported gripper semantics: {semantics}")


class GripperMetrics:
    def __init__(self, semantics: str) -> None:
        self.semantics = semantics
        self.total = 0
        self.correct = 0
        self.target_open = 0
        self.target_close = 0
        self.correct_open = 0
        self.correct_close = 0

    def update(self, target: float, prediction: float) -> None:
        target_close = _is_close(target, self.semantics)
        prediction_close = _is_close(prediction, self.semantics)
        self.total += 1
        self.correct += int(target_close == prediction_close)
        if target_close:
            self.target_close += 1
            self.correct_close += int(prediction_close)
        else:
            self.target_open += 1
            self.correct_open += int(not prediction_close)

    def summary(self) -> dict[str, float | int | None]:
        return {
            "frames": self.total,
            "accuracy": self.correct / self.total if self.total else None,
            "open_recall": self.correct_open / self.target_open if self.target_open else None,
            "close_recall": self.correct_close / self.target_close if self.target_close else None,
            "target_open_frames": self.target_open,
            "target_close_frames": self.target_close,
        }


class ActionMetrics:
    def __init__(self, semantics: str) -> None:
        self.gripper = GripperMetrics(semantics)
        self.joint_errors: list[np.ndarray] = []

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if target.shape[0] < 7 or prediction.shape[0] < 7:
            raise ValueError("Expected seven-dimensional CR3 actions.")
        self.joint_errors.append(np.abs(target[:6] - prediction[:6]))
        self.gripper.update(float(target[6]), float(prediction[6]))

    def summary(self) -> dict[str, Any]:
        gripper = self.gripper.summary()
        if not self.joint_errors:
            return {"joint_mae": None, "joint_p95_abs_error": None, "gripper": gripper}
        errors = np.stack(self.joint_errors)
        return {
            "joint_mae": float(errors.mean()),
            "joint_p95_abs_error": float(np.percentile(errors, 95)),
            "gripper": gripper,
        }


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, value: Any) -> None:
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock: socket.socket) -> Any:
    (size,) = struct.unpack("!I", recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, size))  # nosec B301 - trusted LAN policy server.


def parse_server(value: str) -> tuple[str, int]:
    try:
        host, port = value.rsplit(":", 1)
        return host, int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--server must be host:port") from exc


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def as_vector(value: Any) -> np.ndarray:
    array = as_numpy(value).astype(np.float32)
    while array.ndim > 1:
        array = array[0]
    return array.reshape(-1)


def image_to_rgb_u8(value: Any) -> np.ndarray:
    image = as_numpy(value)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image tensor, got {image.shape}.")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def encode_payload_frame(cv2: Any, frame: np.ndarray, width: int, height: int, jpeg_quality: int) -> dict[str, Any] | np.ndarray:
    if frame.shape[1] != width or frame.shape[0] != height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    if jpeg_quality <= 0:
        return frame
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG encode validation frame.")
    return {"encoding": "jpg", "data": encoded.tobytes()}


class RemotePolicyClient:
    def __init__(self, server: str, timeout_s: float) -> None:
        self.host, self.port = parse_server(server)
        self.timeout_s = timeout_s
        self.sock: socket.socket | None = None

    def __enter__(self) -> "RemotePolicyClient":
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self.sock.settimeout(self.timeout_s)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def request(self, payload: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        if self.sock is None:
            raise RuntimeError("RemotePolicyClient is not connected.")
        start = time.perf_counter()
        send_msg(self.sock, payload)
        response = recv_msg(self.sock)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        action = np.asarray(response.get("action"), dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(f"Remote server returned invalid action shape {action.shape}.")
        return action, roundtrip_ms, float(response.get("latency_ms", -1.0))


def parse_lock_joints(value: str) -> list[int]:
    if not value.strip():
        return []
    result = []
    for part in value.split(","):
        joint = int(part.strip())
        if joint < 1 or joint > 6:
            raise ValueError(f"--lock-joints only supports 1 through 6, got {joint}.")
        result.append(joint - 1)
    return sorted(set(result))


def clamp_absolute_action(current: np.ndarray, action: np.ndarray, max_joint_delta: float) -> np.ndarray:
    limited = np.asarray(action, dtype=np.float32).copy()
    limited[:6] = np.clip(limited[:6], current[:6] - max_joint_delta, current[:6] + max_joint_delta)
    limited[6] = 1.0 if limited[6] >= 0.5 else 0.0
    return limited


def apply_locked_joints(action: np.ndarray, reference: np.ndarray, locked: list[int]) -> np.ndarray:
    output = np.asarray(action, dtype=np.float32).copy()
    for index in locked:
        output[index] = reference[index]
    return output


def limited_action_chunk(base_state: np.ndarray, actions: np.ndarray, max_delta: float, locked: list[int]) -> list[np.ndarray]:
    commands: list[np.ndarray] = []
    rolling = base_state.copy()
    for action in actions:
        command = apply_locked_joints(action, rolling, locked)
        command = clamp_absolute_action(rolling, command, max_delta)
        command = apply_locked_joints(command, rolling, locked)
        commands.append(command)
        rolling = command
    return commands


def ensemble_commands(old: list[np.ndarray], new: list[np.ndarray], anchor: np.ndarray, steps: int, max_delta: float, locked: list[int]) -> list[np.ndarray]:
    overlap = min(len(old), len(new), steps)
    fused: list[np.ndarray] = []
    for index in range(overlap):
        new_weight = (index + 1) / (overlap + 1)
        command = (1.0 - new_weight) * old[index] + new_weight * new[index]
        command[6] = new[index][6]
        fused.append(command.astype(np.float32))
    fused.extend(new[overlap:])
    return limited_action_chunk(anchor, np.asarray(fused), max_delta, locked)


def semantic_gripper_action(width: float, semantics: str) -> float:
    is_open = width >= 50.0
    if semantics == "close_high":
        return 0.0 if is_open else 1.0
    return 1.0 if is_open else 0.0


class GripperCommandFilter:
    def __init__(self, debounce_steps: int, min_hold_steps: int, initial_width: int) -> None:
        self.debounce_steps = max(1, debounce_steps)
        self.min_hold_steps = max(0, min_hold_steps)
        self.width = initial_width
        self.pending_width: int | None = None
        self.pending_count = 0
        self.last_change_step = -self.min_hold_steps

    def update(self, width: int, step: int) -> int:
        if width == self.width:
            self.pending_width = None
            self.pending_count = 0
            return self.width
        if width != self.pending_width:
            self.pending_width = width
            self.pending_count = 1
        else:
            self.pending_count += 1
        if self.pending_count >= self.debounce_steps and step - self.last_change_step >= self.min_hold_steps:
            self.width = width
            self.pending_width = None
            self.pending_count = 0
            self.last_change_step = step
        return self.width


class ActionQueue:
    def __init__(self) -> None:
        self.values: deque[np.ndarray] = deque()
        self.last_action: np.ndarray | None = None

    def replace(self, commands: list[np.ndarray]) -> None:
        self.values = deque(commands)

    def pop(self) -> np.ndarray | None:
        if not self.values:
            return None
        self.last_action = self.values.popleft()
        return self.last_action.copy()

    def snapshot(self) -> list[np.ndarray]:
        return [value.copy() for value in self.values]


def parse_dataset_spec(value: str) -> tuple[str, Path]:
    try:
        repo_id, root = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--dataset must be repo_id=/absolute/or/relative/root") from exc
    return repo_id, Path(root)


def make_payload(cv2: Any, item: dict[str, Any], state: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "state": state,
        "front_rgb": encode_payload_frame(
            cv2, image_to_rgb_u8(item["observation.images.front_rgb"]), args.send_width, args.send_height, args.jpeg_quality
        ),
        "wrist_rgb": encode_payload_frame(
            cv2, image_to_rgb_u8(item["observation.images.wrist_rgb"]), args.send_width, args.send_height, args.jpeg_quality
        ),
        "task": str(item["task"]),
    }


def update_summary(bucket: dict[str, ActionMetrics], key: str, target: np.ndarray, prediction: np.ndarray, semantics: str) -> None:
    if key not in bucket:
        bucket[key] = ActionMetrics(semantics)
    bucket[key].update(target, prediction)


def replay_dataset(dataset_spec: tuple[str, Path], client: RemotePolicyClient, args: argparse.Namespace, per_frame: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, ActionMetrics]:
    import cv2
    from lerobot.datasets import LeRobotDataset

    repo_id, root = dataset_spec
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    metrics: dict[str, ActionMetrics] = {}
    locked = parse_lock_joints(args.lock_joints)
    request_stride = max(1, round(args.fps / args.request_hz))
    current_episode = None
    local_frame = 0
    queue = ActionQueue()
    pending: list[tuple[int, np.ndarray, float, float]] = []
    gripper_filter: GripperCommandFilter | None = None
    last_target_gripper: float | None = None
    last_prediction_gripper: float | None = None
    current_task = ""
    request_count = 0

    for index in range(dataset.num_frames):
        raw = dataset.get_raw_item(index)
        episode = int(as_numpy(raw["episode_index"]).reshape(-1)[0])
        if episode != current_episode:
            current_episode = episode
            local_frame = 0
            queue = ActionQueue()
            pending = []
            initial_width = float(as_vector(raw["observation.state"])[6])
            gripper_filter = GripperCommandFilter(args.gripper_debounce_steps, args.gripper_min_hold_steps, int(initial_width >= 50) * 100)
            last_target_gripper = None
            last_prediction_gripper = None
            current_task = ""

        state = as_vector(raw["observation.state"])
        target = as_vector(raw["action"])
        still_pending: list[tuple[int, np.ndarray, float, float]] = []
        for arrival_frame, actions, roundtrip_ms, server_ms in pending:
            if arrival_frame > local_frame:
                still_pending.append((arrival_frame, actions, roundtrip_ms, server_ms))
                continue
            anchor = queue.last_action if queue.last_action is not None else state
            commands = limited_action_chunk(state, actions, args.max_joint_delta, locked)
            if args.queue_mode == "ensemble":
                commands = ensemble_commands(queue.snapshot(), commands, anchor, args.ensemble_steps, args.max_joint_delta, locked)
            queue.replace(commands[: args.queue_max_actions])
        pending = still_pending

        request_info: dict[str, float | int] = {}
        if local_frame % request_stride == 0:
            item = dataset[index]
            current_task = str(item["task"])
            actions, roundtrip_ms, server_ms = client.request(make_payload(cv2, item, state, args))
            request_count += 1
            aligned, skipped = apply_latency_compensation(
                actions, roundtrip_ms, args.latency_compensation_hz, args.min_actions_after_latency_compensation
            )
            request_info = {"roundtrip_ms": roundtrip_ms, "server_ms": server_ms, "skipped_actions": skipped}
            if aligned is not None:
                pending.append((local_frame + skipped, aligned, roundtrip_ms, server_ms))
            if request_count % args.progress_every == 0:
                print(
                    f"{repo_id}: request={request_count} frame={index + 1}/{dataset.num_frames} "
                    f"roundtrip_ms={roundtrip_ms:.1f} server_ms={server_ms:.1f}",
                    flush=True,
                )

        command = queue.pop() if len(queue.values) >= args.min_queue_before_start else None
        if command is not None and gripper_filter is not None:
            width = 0 if _is_close(float(command[6]), args.gripper_action_semantics) else 100
            effective_width = gripper_filter.update(width, local_frame)
            prediction = command.copy()
            prediction[6] = semantic_gripper_action(effective_width, args.gripper_action_semantics)
            update_summary(metrics, "all", target, prediction, args.gripper_action_semantics)
            update_summary(metrics, repo_id, target, prediction, args.gripper_action_semantics)
            update_summary(metrics, current_task, target, prediction, args.gripper_action_semantics)
            target_close = _is_close(float(target[6]), args.gripper_action_semantics)
            prediction_close = _is_close(float(prediction[6]), args.gripper_action_semantics)
            if last_target_gripper is not None and target_close != last_target_gripper:
                events.append({"repo_id": repo_id, "episode_index": episode, "frame": local_frame, "kind": "target", "close": target_close})
            if last_prediction_gripper is not None and prediction_close != last_prediction_gripper:
                events.append({"repo_id": repo_id, "episode_index": episode, "frame": local_frame, "kind": "prediction", "close": prediction_close})
            last_target_gripper = target_close
            last_prediction_gripper = prediction_close
            per_frame.append(
                {
                    "repo_id": repo_id,
                    "episode_index": episode,
                    "frame": local_frame,
                    "queue_after_pop": len(queue.values),
                    "target_action": json.dumps(target.tolist()),
                    "prediction_action": json.dumps(prediction.tolist()),
                    **request_info,
                }
            )
        local_frame += 1
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a remote Pi0 server by replaying held-out LeRobot trajectories.")
    parser.add_argument("--server", default="192.168.1.125:8765")
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True, metavar="REPO_ID=ROOT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--request-hz", type=float, default=1.5)
    parser.add_argument("--latency-compensation-hz", type=float, default=30.0)
    parser.add_argument("--min-actions-after-latency-compensation", type=int, default=15)
    parser.add_argument("--queue-mode", choices=["ensemble"], default="ensemble")
    parser.add_argument("--queue-max-actions", type=int, default=50)
    parser.add_argument("--min-queue-before-start", type=int, default=20)
    parser.add_argument("--ensemble-steps", type=int, default=6)
    parser.add_argument("--max-joint-delta", type=float, default=0.12)
    parser.add_argument("--lock-joints", default="6")
    parser.add_argument("--gripper-action-semantics", choices=["close_high", "open_high"], default="close_high")
    parser.add_argument("--gripper-debounce-steps", type=int, default=3)
    parser.add_argument("--gripper-min-hold-steps", type=int, default=15)
    parser.add_argument("--send-width", type=int, default=224)
    parser.add_argument("--send-height", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_frame: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    all_metrics: dict[str, ActionMetrics] = {}
    with RemotePolicyClient(args.server, args.timeout_s) as client:
        for dataset_spec in args.dataset:
            print(f"Replaying {dataset_spec[0]} from {dataset_spec[1]}", flush=True)
            for key, metric in replay_dataset(dataset_spec, client, args, per_frame, events).items():
                if key not in all_metrics:
                    all_metrics[key] = metric
                else:
                    # Re-add frame-level values through the existing aggregates only when keys overlap.
                    all_metrics[key].joint_errors.extend(metric.joint_errors)
                    target = metric.gripper
                    merged = all_metrics[key].gripper
                    merged.total += target.total
                    merged.correct += target.correct
                    merged.target_open += target.target_open
                    merged.target_close += target.target_close
                    merged.correct_open += target.correct_open
                    merged.correct_close += target.correct_close
    summary = {key: metric.summary() for key, metric in sorted(all_metrics.items())}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "per_frame.csv", per_frame)
    write_csv(args.output_dir / "gripper_events.csv", events)
    print(json.dumps(summary.get("all", {}), ensure_ascii=False, indent=2), flush=True)
    print(f"Wrote evaluation report to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
