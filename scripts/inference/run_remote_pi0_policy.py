import argparse
from collections import deque
import pickle  # nosec B403 - trusted server reached through SSH tunnel.
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

LEROBOT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.robots.dobot_cr3 import DobotCR3, DobotCR3Config
from lerobot.policies.rtc.action_queue import ActionQueue as RTCActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from scripts.collection.record_drag_dataset import (
    CameraStream,
    RealSenseRGBDStream,
    default_camera_backend,
    default_gripper_port,
    frame_for_cv2,
    normalize_camera_frame,
)
from scripts.inference.run_act_policy import (
    clamp_absolute_action,
    parse_six_floats,
    send_action_to_robot,
    show_preview,
)


class LatestFrameReader:
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._error = None

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._timestamp = time.monotonic()

    def fail(self, exc: Exception) -> None:
        with self._lock:
            self._error = exc

    def latest(self, timeout_s: float = 2.0) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError(f"{self.name} reader failed: {self._error}") from self._error
                if self._frame is not None:
                    return self._frame.copy(), self._timestamp
            time.sleep(0.005)
        raise TimeoutError(f"Timed out waiting for {self.name} frame.")


class ActionQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque()
        self.last_action = None
        self.last_update_step = -1

    def replace(self, actions: list[np.ndarray], update_step: int) -> None:
        with self._lock:
            self._queue = deque(actions)
            self.last_update_step = update_step

    def append(self, actions: list[np.ndarray], update_step: int, maxlen: int) -> None:
        with self._lock:
            for action in actions:
                self._queue.append(action)
            while len(self._queue) > maxlen:
                self._queue.popleft()
            self.last_update_step = update_step

    def snapshot(self) -> list[np.ndarray]:
        with self._lock:
            return [action.copy() for action in self._queue]

    def last(self) -> np.ndarray | None:
        with self._lock:
            if self.last_action is None:
                return None
            return self.last_action.copy()

    def pop(self) -> tuple[np.ndarray | None, int, int]:
        with self._lock:
            if self._queue:
                action = self._queue.popleft()
                self.last_action = action
                return action.copy(), len(self._queue), self.last_update_step
            return None, 0, self.last_update_step

    def size(self) -> int:
        with self._lock:
            return len(self._queue)


def make_rtc_action_queue() -> RTCActionQueue:
    """Create the official LeRobot queue used only by the optional RTC path."""
    return RTCActionQueue(RTCConfig(enabled=True))


def snapshot_rtc_prefix(queue: RTCActionQueue) -> tuple[torch.Tensor | None, int]:
    """Atomically read the unexecuted model actions and their consumption index."""
    with queue.lock:
        action_index = queue.last_index
        prefix = None if queue.original_queue is None else queue.original_queue[action_index:].clone()
    return prefix, action_index


def merge_rtc_response(
    queue: RTCActionQueue,
    original_actions: torch.Tensor,
    processed_actions: np.ndarray,
    *,
    request_action_index: int,
) -> int:
    """Atomically merge both RTC queues using the real consumed action count."""
    processed_tensor = torch.from_numpy(np.ascontiguousarray(processed_actions, dtype=np.float32))
    return queue.merge_with_consumed_delay(
        original_actions.detach().cpu(),
        processed_tensor,
        request_action_index,
    )


class GripperCommandFilter:
    """Debounce discrete policy outputs before they are sent to the physical gripper."""

    def __init__(self, debounce_steps: int, min_hold_steps: int, initial_width: int | None = None):
        self.debounce_steps = max(1, debounce_steps)
        self.min_hold_steps = max(0, min_hold_steps)
        self.width = initial_width
        self.pending_width: int | None = None
        self.pending_count = 0
        self.last_change_step = -self.min_hold_steps

    def update(self, candidate_width: int, step: int) -> int | None:
        if self.width is None:
            self.width = candidate_width
            self.last_change_step = step
            return candidate_width
        if candidate_width == self.width:
            self.pending_width = None
            self.pending_count = 0
            return None
        if candidate_width != self.pending_width:
            self.pending_width = candidate_width
            self.pending_count = 1
        else:
            self.pending_count += 1
        if self.pending_count < self.debounce_steps:
            return None
        if step - self.last_change_step < self.min_hold_steps:
            return None
        self.width = candidate_width
        self.pending_width = None
        self.pending_count = 0
        self.last_change_step = step
        return candidate_width


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket):
    header = recv_exact(sock, 4)
    (size,) = struct.unpack("!I", header)
    return pickle.loads(recv_exact(sock, size))  # nosec B301 - trusted tunnel.


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)) + data)


def maybe_resize_frame(cv2, frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def encode_payload_frame(cv2, frame: np.ndarray, jpeg_quality: int):
    if jpeg_quality <= 0:
        return frame
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG encode camera frame.")
    return {"encoding": "jpg", "data": encoded.tobytes()}


def robot_state(robot: DobotCR3) -> np.ndarray:
    obs = robot.get_observation()
    gripper = obs["gripper.pos"] if obs["gripper.pos"] >= 0 else 100.0
    return np.asarray(
        [
            obs["q1.pos"],
            obs["q2.pos"],
            obs["q3.pos"],
            obs["q4.pos"],
            obs["q5.pos"],
            obs["q6.pos"],
            gripper,
        ],
        dtype=np.float32,
    )


def make_preview(cv2, frames: dict[str, np.ndarray], current: np.ndarray, action: np.ndarray, cmd: np.ndarray, execute: bool, step: int, args, mpl_state: dict) -> bool:
    return show_preview(
        cv2,
        frames,
        current,
        action,
        cmd,
        execute,
        step,
        args.preview_dir,
        args.preview_wait_ms,
        args.preview_backend,
        mpl_state,
    )


def parse_lock_joints(value: str) -> list[int]:
    if not value.strip():
        return []
    locked = []
    for part in value.split(","):
        index = int(part.strip())
        if index < 1 or index > 6:
            raise ValueError(f"--lock-joints only supports joints 1-6, got {index}.")
        locked.append(index - 1)
    return sorted(set(locked))


def apply_locked_joints(action: np.ndarray, reference: np.ndarray, locked_joints: list[int]) -> np.ndarray:
    if not locked_joints:
        return action
    output = action.copy()
    for joint_index in locked_joints:
        output[joint_index] = reference[joint_index]
    return output


def gripper_action_to_width(value: float, semantics: str) -> int:
    if semantics == "close_high":
        return 0 if value >= 0.5 else 100
    if semantics == "open_high":
        return 100 if value >= 0.5 else 0
    raise ValueError(f"Unsupported gripper action semantics: {semantics}")


def limited_action_chunk(
    base_state: np.ndarray,
    actions: np.ndarray,
    max_joint_delta: float,
    locked_joints: list[int],
) -> list[np.ndarray]:
    commands = []
    rolling = base_state.copy()
    for action in actions:
        action = apply_locked_joints(np.asarray(action, dtype=np.float32), rolling, locked_joints)
        limited = clamp_absolute_action(rolling, action, max_joint_delta)
        limited = apply_locked_joints(limited, rolling, locked_joints)
        commands.append(limited)
        rolling = limited
    return commands


def smooth_bridge(start: np.ndarray, target: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        return []
    bridge = []
    for index in range(1, steps + 1):
        alpha = index / (steps + 1)
        cmd = (1.0 - alpha) * start + alpha * target
        if cmd.shape[0] > 6:
            cmd[6] = target[6]
        bridge.append(cmd.astype(np.float32))
    return bridge


def clamp_consecutive_commands(
    commands: list[np.ndarray],
    anchor: np.ndarray,
    max_joint_delta: float,
    locked_joints: list[int],
) -> list[np.ndarray]:
    limited_commands = []
    rolling = anchor.copy()
    for command in commands:
        command = apply_locked_joints(command, rolling, locked_joints)
        limited = clamp_absolute_action(rolling, command, max_joint_delta)
        limited = apply_locked_joints(limited, rolling, locked_joints)
        limited_commands.append(limited)
        rolling = limited
    return limited_commands


def temporal_ensemble_chunk(
    old_remaining: list[np.ndarray],
    new_commands: list[np.ndarray],
    anchor: np.ndarray,
    ensemble_steps: int,
    max_joint_delta: float,
    locked_joints: list[int],
) -> list[np.ndarray]:
    if not old_remaining:
        return clamp_consecutive_commands(new_commands, anchor, max_joint_delta, locked_joints)

    overlap = min(len(old_remaining), len(new_commands), ensemble_steps)
    fused = []
    for index in range(overlap):
        new_weight = (index + 1) / (overlap + 1)
        old_weight = 1.0 - new_weight
        command = old_weight * old_remaining[index] + new_weight * new_commands[index]
        if command.shape[0] > 6:
            command[6] = new_commands[index][6]
        command = apply_locked_joints(command, anchor if index == 0 else fused[-1], locked_joints)
        fused.append(command.astype(np.float32))

    fused.extend(new_commands[overlap:])
    return clamp_consecutive_commands(fused, anchor, max_joint_delta, locked_joints)


def run_async_rollout(
    *,
    args,
    cv2,
    robot: DobotCR3,
    front_camera: CameraStream,
    realsense: RealSenseRGBDStream,
    sock: socket.socket,
    joint_action_sign: np.ndarray,
) -> None:
    front_reader = LatestFrameReader("front_rgb")
    wrist_reader = LatestFrameReader("wrist_rgb")
    action_queue = ActionQueue()
    use_rtc_chunking = args.rtc_enabled or args.rtc_trained_prefix
    rtc_action_queue = make_rtc_action_queue() if use_rtc_chunking else None
    stop = threading.Event()
    robot_lock = threading.Lock()
    stats_lock = threading.Lock()
    stats = {
        "requests": 0,
        "last_latency_ms": -1.0,
        "last_roundtrip_ms": -1.0,
        "last_request_step": -1,
    }
    rtc_lock = threading.Lock()
    rtc_state = {"estimated_delay_steps": max(0, args.rtc_delay_steps)}
    mpl_state = {}

    def camera_loop() -> None:
        while not stop.is_set():
            try:
                front_reader.update(normalize_camera_frame(cv2, front_camera.read(), "rgb"))
                wrist_color_bgr, _ = realsense.read()
                wrist_reader.update(normalize_camera_frame(cv2, wrist_color_bgr, "rgb"))
            except Exception as exc:
                front_reader.fail(exc)
                wrist_reader.fail(exc)
                stop.set()

    def request_loop() -> None:
        period = 1.0 / args.request_hz
        request_step = 0
        while not stop.is_set():
            start = time.monotonic()
            try:
                front_rgb, _ = front_reader.latest()
                wrist_rgb, _ = wrist_reader.latest()
                front_rgb = maybe_resize_frame(cv2, front_rgb, args.send_width, args.send_height)
                wrist_rgb = maybe_resize_frame(cv2, wrist_rgb, args.send_width, args.send_height)
                with robot_lock:
                    current = robot_state(robot)
                payload = {
                    "state": current,
                    "front_rgb": encode_payload_frame(cv2, front_rgb, args.jpeg_quality),
                    "wrist_rgb": encode_payload_frame(cv2, wrist_rgb, args.jpeg_quality),
                    "task": args.task,
                }
                if use_rtc_chunking:
                    assert rtc_action_queue is not None
                    prefix, request_action_index = snapshot_rtc_prefix(rtc_action_queue)
                    with rtc_lock:
                        estimated_delay_steps = rtc_state["estimated_delay_steps"]
                    payload["rtc_prev_chunk"] = None if prefix is None else prefix.numpy()
                    payload["rtc_estimated_delay_steps"] = estimated_delay_steps
                roundtrip_start = time.perf_counter()
                send_msg(sock, payload)
                response = recv_msg(sock)
                roundtrip_ms = (time.perf_counter() - roundtrip_start) * 1000.0
                actions = np.asarray(response["action"], dtype=np.float32)
                if actions.ndim == 1:
                    actions = actions.reshape(1, -1)
                skipped_actions = 0
                if use_rtc_chunking:
                    assert rtc_action_queue is not None
                    raw_original = response.get("rtc_original_action")
                    if raw_original is None:
                        raise RuntimeError("RTC server response is missing rtc_original_action.")
                    original_actions = torch.as_tensor(raw_original, dtype=torch.float32)
                    if original_actions.shape != actions.shape:
                        raise RuntimeError(
                            "RTC original/processed action shapes differ: "
                            f"{tuple(original_actions.shape)} vs {tuple(actions.shape)}."
                        )
                elif args.latency_compensation_hz > 0:
                    skipped_actions = int(roundtrip_ms / 1000.0 * args.latency_compensation_hz)
                    remaining_actions = len(actions) - skipped_actions
                    if remaining_actions < args.min_actions_after_latency_compensation:
                        print(
                            f"request {request_step:04d} dropped_stale_response "
                            f"roundtrip_ms={roundtrip_ms:.1f} skip={skipped_actions} "
                            f"chunk={len(actions)}",
                            flush=True,
                        )
                        request_step += 1
                        elapsed = time.monotonic() - start
                        if elapsed < period:
                            time.sleep(period - elapsed)
                        continue
                    actions = actions[skipped_actions:]
                actions[:, :6] *= joint_action_sign
                # The policy saw `current` at request start. Re-anchor the remaining, latency-aligned
                # actions to the robot's state when the response actually arrives.
                with robot_lock:
                    execution_state = robot_state(robot)
                commands = limited_action_chunk(
                    execution_state,
                    actions,
                    args.max_joint_delta,
                    args.locked_joint_indices,
                )
                if use_rtc_chunking:
                    assert rtc_action_queue is not None
                    skipped_actions = merge_rtc_response(
                        rtc_action_queue,
                        original_actions,
                        np.asarray(commands, dtype=np.float32),
                        request_action_index=request_action_index,
                    )
                    with rtc_lock:
                        rtc_state["estimated_delay_steps"] = skipped_actions
                elif args.queue_mode == "append":
                    action_queue.append(commands, request_step, args.queue_max_actions)
                elif args.queue_mode == "ensemble":
                    anchor = action_queue.last()
                    if anchor is None:
                        anchor = current
                    commands = temporal_ensemble_chunk(
                        action_queue.snapshot(),
                        commands,
                        anchor,
                        args.ensemble_steps,
                        args.max_joint_delta,
                        args.locked_joint_indices,
                    )
                    action_queue.replace(commands, request_step)
                elif args.queue_mode == "smooth_replace":
                    anchor = action_queue.last()
                    if anchor is None:
                        anchor = current
                    bridge = smooth_bridge(anchor, commands[0], args.blend_steps)
                    action_queue.replace(bridge + commands, request_step)
                else:
                    action_queue.replace(commands, request_step)
                with stats_lock:
                    stats["requests"] += 1
                    stats["last_latency_ms"] = float(response.get("latency_ms", -1.0))
                    stats["last_roundtrip_ms"] = roundtrip_ms
                    stats["last_request_step"] = request_step
                print(
                    f"request {request_step:04d} server_ms={response.get('latency_ms', -1):.1f} "
                    f"roundtrip_ms={roundtrip_ms:.1f} chunk={len(commands)} "
                    f"skip={skipped_actions} queue="
                    f"{rtc_action_queue.qsize() if use_rtc_chunking else action_queue.size()} "
                    f"rtc_prefix={0 if not use_rtc_chunking or prefix is None else len(prefix)} "
                    f"rtc_estimate={estimated_delay_steps if use_rtc_chunking else 0} "
                    f"rtc_actual={skipped_actions if use_rtc_chunking else 0} "
                    f"first_cmd={np.round(commands[0], 2).tolist()}",
                    flush=True,
                )
                request_step += 1
            except Exception as exc:
                print(f"WARNING: async request failed: {exc}", flush=True)
                time.sleep(min(period, 0.5))
            elapsed = time.monotonic() - start
            if elapsed < period:
                time.sleep(period - elapsed)

    camera_thread = threading.Thread(target=camera_loop, name="remote-pi0-camera", daemon=True)
    request_thread = threading.Thread(target=request_loop, name="remote-pi0-request", daemon=True)
    camera_thread.start()
    # Warm up frames before the first request.
    front_reader.latest()
    wrist_reader.latest()
    request_thread.start()

    period = 1.0 / args.hz
    initial_width = getattr(args, "initial_gripper_width", None)
    if initial_width is None and args.use_gripper:
        initial_width = 100 if robot_state(robot)[6] >= 50 else 0
    gripper_filter = GripperCommandFilter(
        args.gripper_debounce_steps,
        args.gripper_min_hold_steps,
        initial_width=initial_width,
    )
    try:
        for step in range(args.steps):
            start = time.monotonic()
            queue_size_now = rtc_action_queue.qsize() if use_rtc_chunking else action_queue.size()
            if args.execute and queue_size_now < args.min_queue_before_start and step == 0:
                print(f"waiting for initial action queue >= {args.min_queue_before_start}", flush=True)
                while not stop.is_set() and (
                    rtc_action_queue.qsize() if use_rtc_chunking else action_queue.size()
                ) < args.min_queue_before_start:
                    time.sleep(0.02)
            if use_rtc_chunking:
                assert rtc_action_queue is not None
                rtc_cmd = rtc_action_queue.get()
                cmd = None if rtc_cmd is None else rtc_cmd.numpy()
                queue_size = rtc_action_queue.qsize()
                update_step = -1
            else:
                cmd, queue_size, update_step = action_queue.pop()
            if cmd is None:
                print(f"{step:04d} waiting_for_action_queue", flush=True)
            else:
                gripper_width = None
                if args.use_gripper and cmd.shape[0] > 6:
                    candidate_width = gripper_action_to_width(float(cmd[6]), args.gripper_action_semantics)
                    gripper_width = gripper_filter.update(candidate_width, step)
                    if (
                        gripper_width is None
                        and args.gripper_every > 0
                        and step % args.gripper_every == 0
                    ):
                        gripper_width = gripper_filter.width
                if args.execute:
                    with robot_lock:
                        send_action_to_robot(
                            robot,
                            cmd,
                            action_mode="absolute",
                            command_mode=args.command_mode,
                            servo_t=args.servo_t if args.servo_t is not None else period,
                            servo_lookahead_time=args.servo_lookahead_time,
                            servo_gain=args.servo_gain,
                            gripper_width=gripper_width,
                        )
                with stats_lock:
                    server_ms = stats["last_latency_ms"]
                    roundtrip_ms = stats["last_roundtrip_ms"]
                if step % args.print_every == 0 or queue_size == 0:
                    print(
                        f"{step:04d} queue={queue_size} from_request={update_step} "
                        f"server_ms={server_ms:.1f} roundtrip_ms={roundtrip_ms:.1f} "
                        f"cmd={np.round(cmd, 2).tolist()}",
                        flush=True,
                    )
                if not args.no_preview:
                    try:
                        front_rgb, _ = front_reader.latest(timeout_s=0.05)
                        wrist_rgb, _ = wrist_reader.latest(timeout_s=0.05)
                        frames = {"front_rgb": front_rgb, "wrist_rgb": wrist_rgb}
                        with robot_lock:
                            current = robot_state(robot)
                        if not make_preview(
                            cv2,
                            frames,
                            current,
                            cmd,
                            cmd,
                            args.execute,
                            step,
                            args,
                            mpl_state,
                        ):
                            stop.set()
                            break
                    except Exception as exc:
                        print(f"WARNING: preview update failed: {exc}", flush=True)
            elapsed = time.monotonic() - start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        stop.set()
        camera_thread.join(timeout=1.0)
        request_thread.join(timeout=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="127.0.0.1:8765")
    parser.add_argument("--task", default="put the red block into the black frame, then put the green block into the black frame, finally put the yellow block into the black frame")
    parser.add_argument("--robot-ip", default="192.168.6.1")
    parser.add_argument("--front-rgb-index", default="0")
    parser.add_argument("--use-realsense-wrist", action="store_true")
    parser.add_argument("--realsense-serial", default=None)
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=15)
    parser.add_argument(
        "--send-width",
        type=int,
        default=0,
        help="Resize images to this width before sending to the remote server. 0 keeps captured size.",
    )
    parser.add_argument(
        "--send-height",
        type=int,
        default=0,
        help="Resize images to this height before sending to the remote server. 0 keeps captured size.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=70,
        help="JPEG quality for images sent to the remote server. 0 sends raw numpy arrays.",
    )
    parser.add_argument("--backend", default=default_camera_backend(), choices=["any", "dshow", "msmf", "v4l2"])
    parser.add_argument("--front-width", type=int, default=1280)
    parser.add_argument("--front-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--gripper-port", default=default_gripper_port())
    parser.add_argument("--no-gripper", dest="use_gripper", action="store_false")
    parser.set_defaults(use_gripper=True)
    parser.add_argument(
        "--initial-gripper",
        choices=["keep", "open", "close"],
        default="keep",
        help="Set the physical gripper once before rollout. Use 'open' when testing a pick task from an empty gripper.",
    )
    parser.add_argument(
        "--gripper-debounce-steps",
        type=int,
        default=5,
        help="Require this many consecutive policy commands before changing the gripper state.",
    )
    parser.add_argument(
        "--gripper-min-hold-steps",
        type=int,
        default=15,
        help="Keep a changed gripper state for at least this many control steps before another change.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument(
        "--cloud-pi0-preset",
        action="store_true",
        help="Use recommended async settings for cloud-server PI0 rollout over SSH.",
    )
    parser.add_argument("--async-rollout", action="store_true")
    parser.add_argument("--request-hz", type=float, default=1.0)
    parser.add_argument(
        "--rtc-enabled",
        action="store_true",
        help="Use server-side RTC-V1 chunk guidance. Requires async rollout and replace mode.",
    )
    parser.add_argument(
        "--rtc-trained-prefix",
        action="store_true",
        help="Use fixed-prefix inference for a checkpoint trained with Training-Time RTC.",
    )
    parser.add_argument(
        "--rtc-delay-steps",
        type=int,
        default=10,
        help="Expected actions consumed during one remote inference request for RTC-V1.",
    )
    parser.add_argument(
        "--latency-compensation-hz",
        type=float,
        default=0.0,
        help=(
            "Discard the leading part of a returned action chunk according to round-trip latency. "
            "Set this to the dataset action rate (30 for this CR3 dataset) when using remote inference."
        ),
    )
    parser.add_argument(
        "--min-actions-after-latency-compensation",
        type=int,
        default=10,
        help="Drop a stale response when fewer than this many actions remain after latency compensation.",
    )
    parser.add_argument("--queue-mode", choices=["replace", "append", "smooth_replace", "ensemble"], default="replace")
    parser.add_argument("--queue-max-actions", type=int, default=64)
    parser.add_argument("--min-queue-before-start", type=int, default=1)
    parser.add_argument("--blend-steps", type=int, default=8)
    parser.add_argument("--ensemble-steps", type=int, default=15)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--command-mode", choices=["jointmovj", "servoj"], default="servoj")
    parser.add_argument("--servo-t", type=float, default=None)
    parser.add_argument("--servo-lookahead-time", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)
    parser.add_argument("--joint-action-sign", default="1,1,1,1,1,1")
    parser.add_argument(
        "--lock-joints",
        default="",
        help="Comma-separated 1-based joint indices to keep at their current value, e.g. '6' or '5,6'.",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.25)
    parser.add_argument("--speed-factor", type=int, default=10)
    parser.add_argument(
        "--gripper-action-semantics",
        choices=["close_high", "open_high"],
        default="close_high",
        help=(
            "How to interpret the policy gripper action. close_high matches PI0/OpenPI "
            "(0=open, 1=close); open_high keeps the previous local convention (1=open, 0=close)."
        ),
    )
    parser.add_argument("--gripper-every", type=int, default=10)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-backend", choices=["cv2", "matplotlib"], default="matplotlib")
    parser.add_argument("--preview-wait-ms", type=int, default=80)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--verbose-api", action="store_true")
    args = parser.parse_args()

    if args.cloud_pi0_preset:
        args.async_rollout = True
        args.hz = 10.0
        args.request_hz = 0.5
        args.queue_mode = "replace" if (args.rtc_enabled or args.rtc_trained_prefix) else "ensemble"
        args.queue_max_actions = 80
        args.min_queue_before_start = 20
        args.blend_steps = 8
        args.ensemble_steps = 15
        args.max_joint_delta = 2.0
        args.command_mode = "servoj"
        args.servo_t = 0.1
        args.speed_factor = 60
        args.no_preview = True
        if args.send_width <= 0:
            args.send_width = 320
        if args.send_height <= 0:
            args.send_height = 240
        if args.jpeg_quality <= 0:
            args.jpeg_quality = 70

    if args.rtc_enabled and args.rtc_trained_prefix:
        parser.error("--rtc-enabled and --rtc-trained-prefix cannot be combined.")

    if args.rtc_enabled or args.rtc_trained_prefix:
        if not args.async_rollout:
            parser.error("RTC chunk modes require --async-rollout.")
        if args.queue_mode != "replace":
            parser.error("RTC chunk modes require --queue-mode replace; do not combine RTC with local ensembling.")
        if args.latency_compensation_hz <= 0:
            parser.error("RTC chunk modes require --latency-compensation-hz > 0.")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed.") from exc

    host, port_s = args.server.rsplit(":", 1)
    joint_action_sign = parse_six_floats(args.joint_action_sign)
    args.locked_joint_indices = parse_lock_joints(args.lock_joints)
    period = 1.0 / args.hz
    mpl_state = {}

    config = DobotCR3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        use_gripper=args.use_gripper,
        speed_factor=args.speed_factor,
        enable_robot_on_connect=args.execute,
        use_opencv_camera=False,
    )
    robot = DobotCR3(config)
    front_camera = None
    realsense = None
    sock = None
    args.initial_gripper_width = None
    try:
        front_camera = CameraStream(
            cv2,
            "front_rgb",
            args.front_rgb_index,
            args.backend,
            args.front_width,
            args.front_height,
            args.camera_fps,
        ).open()
        if not args.use_realsense_wrist:
            raise ValueError("This remote pi0 runner currently expects --use-realsense-wrist.")
        realsense = RealSenseRGBDStream(
            args.realsense_width,
            args.realsense_height,
            args.realsense_fps,
            serial=args.realsense_serial,
        ).open()
        robot.connect()
        robot.set_api_verbose(args.verbose_api)
        if args.execute and args.use_gripper and args.initial_gripper != "keep":
            if robot.gripper is None:
                raise RuntimeError("--initial-gripper was requested but the gripper is not connected.")
            initial_width = 100 if args.initial_gripper == "open" else 0
            robot.gripper.set_width(initial_width, wait=False)
            args.initial_gripper_width = initial_width
            print(f"initial gripper: {args.initial_gripper} (width={initial_width})", flush=True)
            time.sleep(0.5)
        sock = socket.create_connection((host, int(port_s)), timeout=20)
        sock.settimeout(None)
        print("remote pi0 rollout ready")
        print("mode:", "EXECUTE" if args.execute else "DRY-RUN")
        if args.async_rollout:
            print(
                f"async rollout: control_hz={args.hz} request_hz={args.request_hz} "
                f"queue_mode={args.queue_mode}",
                flush=True,
            )
            run_async_rollout(
                args=args,
                cv2=cv2,
                robot=robot,
                front_camera=front_camera,
                realsense=realsense,
                sock=sock,
                joint_action_sign=joint_action_sign,
            )
            return

        initial_width = args.initial_gripper_width
        if initial_width is None and args.use_gripper:
            initial_width = 100 if robot_state(robot)[6] >= 50 else 0
        gripper_filter = GripperCommandFilter(
            args.gripper_debounce_steps,
            args.gripper_min_hold_steps,
            initial_width=initial_width,
        )
        for step in range(args.steps):
            start = time.time()
            front_rgb = normalize_camera_frame(cv2, front_camera.read(), "rgb")
            wrist_color_bgr, _ = realsense.read()
            wrist_rgb = normalize_camera_frame(cv2, wrist_color_bgr, "rgb")
            front_rgb = maybe_resize_frame(cv2, front_rgb, args.send_width, args.send_height)
            wrist_rgb = maybe_resize_frame(cv2, wrist_rgb, args.send_width, args.send_height)
            frames = {"front_rgb": front_rgb, "wrist_rgb": wrist_rgb}
            current = robot_state(robot)
            payload = {
                "state": current,
                "front_rgb": encode_payload_frame(cv2, front_rgb, args.jpeg_quality),
                "wrist_rgb": encode_payload_frame(cv2, wrist_rgb, args.jpeg_quality),
                "task": args.task,
            }
            send_msg(sock, payload)
            response = recv_msg(sock)
            action = np.asarray(response["action"], dtype=np.float32).reshape(-1)
            signed_action = action.copy()
            signed_action[:6] *= joint_action_sign
            signed_action = apply_locked_joints(signed_action, current, args.locked_joint_indices)
            limited = clamp_absolute_action(current, signed_action, args.max_joint_delta)
            limited = apply_locked_joints(limited, current, args.locked_joint_indices)
            print(
                f"{step:04d} latency_ms={response.get('latency_ms', -1):.1f} "
                f"current={np.round(current, 2).tolist()} pred={np.round(action, 2).tolist()} "
                f"cmd={np.round(limited, 2).tolist()}"
            )
            if not args.no_preview and not make_preview(
                cv2, frames, current, action, limited, args.execute, step, args, mpl_state
            ):
                break
            if args.execute:
                gripper_width = None
                if args.use_gripper and limited.shape[0] > 6:
                    candidate_width = gripper_action_to_width(float(limited[6]), args.gripper_action_semantics)
                    gripper_width = gripper_filter.update(candidate_width, step)
                    if (
                        gripper_width is None
                        and args.gripper_every > 0
                        and step % args.gripper_every == 0
                    ):
                        gripper_width = gripper_filter.width
                send_action_to_robot(
                    robot,
                    limited,
                    action_mode="absolute",
                    command_mode=args.command_mode,
                    servo_t=args.servo_t if args.servo_t is not None else period,
                    servo_lookahead_time=args.servo_lookahead_time,
                    servo_gain=args.servo_gain,
                    gripper_width=gripper_width,
                )
            elapsed = time.time() - start
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        robot.disconnect()
        if sock is not None:
            sock.close()
        if front_camera is not None:
            front_camera.release()
        if realsense is not None:
            realsense.release()
        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
