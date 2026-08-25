import argparse
import contextlib
import csv
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LEROBOT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.robots.dobot_cr3 import DobotCR3, DobotCR3Config, LeaderFollowerCopyController

DEFAULT_USE_GRIPPER = True
DEFAULT_CAMERA_INDEX = None
DEFAULT_FRONT_RGB_INDEX = None
DEFAULT_WRIST_RGB_INDEX = -1
DEFAULT_WRIST_DEPTH_INDEX = -1
DEFAULT_USE_REALSENSE_WRIST = False
DEFAULT_USE_WRIST_DEPTH = False
DEFAULT_REALSENSE_WIDTH = 640
DEFAULT_REALSENSE_HEIGHT = 480
DEFAULT_REALSENSE_FPS = 15
DEFAULT_MANUAL_DRAG = False
DEFAULT_CAMERA_COLOR = "rgb"
DEFAULT_GRIPPER_OPEN_WIDTH = 100
DEFAULT_GRIPPER_CLOSE_WIDTH = 0
DEFAULT_GRIPPER_TIMEOUT = 3.0
DEFAULT_GRIPPER_TOLERANCE = 2.0
DEFAULT_INITIALIZE_GRIPPER = False
DEFAULT_SPEED_FACTOR = 20
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_DISPLAY_WIDTH = 1400
DEFAULT_DISPLAY_HEIGHT = 700
DEFAULT_CONTROL_HZ = 50.0
DEFAULT_CAMERA_READ_FPS = 30.0
DEFAULT_RECORD_FPS = 30.0
DEFAULT_CONTROL_STATS_INTERVAL = 10.0

CSV_FIELDS = [
    "frame",
    "timestamp",
    "front_rgb",
    "wrist_rgb",
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "leader_q1",
    "leader_q2",
    "leader_q3",
    "leader_q4",
    "leader_q5",
    "leader_q6",
    "target_q1",
    "target_q2",
    "target_q3",
    "target_q4",
    "target_q5",
    "target_q6",
    "gripper",
    "task",
    "action",
]


def csv_fields(use_wrist_depth: bool) -> list[str]:
    fields = CSV_FIELDS.copy()
    if use_wrist_depth:
        fields.insert(fields.index("x"), "wrist_depth")
    return fields


def default_camera_backend() -> str:
    return "msmf" if sys.platform.startswith("win") else "any"


def default_gripper_port() -> str:
    if sys.platform.startswith("win"):
        return "COM5"

    candidates = sorted(glob.glob("/dev/serial/by-id/*"))
    candidates.extend(sorted(glob.glob("/dev/ttyACM*")))
    candidates.extend(sorted(glob.glob("/dev/ttyUSB*")))
    return candidates[0] if candidates else "/dev/ttyACM0"


def video_index_from_path(path: str | Path) -> int | None:
    name = Path(path).name
    if not name.startswith("video"):
        return None
    suffix = name.removeprefix("video")
    return int(suffix) if suffix.isdigit() else None


def resolve_video_index(path_or_index: str | int | None) -> int | None:
    if path_or_index is None:
        return None
    if isinstance(path_or_index, int):
        return path_or_index
    value = str(path_or_index)
    if value.lstrip("-").isdigit():
        return int(value)
    return video_index_from_path(os.path.realpath(value))


def video_device_labels() -> dict[int, list[str]]:
    labels: dict[int, list[str]] = {}
    for symlink in glob.glob("/dev/v4l/by-id/*"):
        index = video_index_from_path(os.path.realpath(symlink))
        if index is None:
            continue
        labels.setdefault(index, []).append(Path(symlink).name.lower())
    for video_class in glob.glob("/sys/class/video4linux/video*"):
        index = video_index_from_path(video_class)
        if index is None:
            continue

        index_labels = labels.setdefault(index, [])
        name_path = Path(video_class) / "name"
        if name_path.exists():
            index_labels.append(name_path.read_text(encoding="utf-8", errors="ignore").strip().lower())

        device_path = Path(video_class).resolve() / "device"
        resolved_device = Path(os.path.realpath(device_path))
        index_labels.append(str(resolved_device).lower())
        for parent in (resolved_device, resolved_device.parent):
            for field in ("idVendor", "idProduct"):
                field_path = parent / field
                if field_path.exists():
                    index_labels.append(field_path.read_text(encoding="utf-8", errors="ignore").strip().lower())
    return labels


def is_realsense_video_index(index: int, labels: dict[int, list[str]]) -> bool:
    return any(
        "realsense" in label or "intel" in label or label == "8086" for label in labels.get(index, [])
    )


def is_builtin_video_index(index: int, labels: dict[int, list[str]]) -> bool:
    builtin_tokens = ("bisoncam", "nb pro", "integrated", "builtin", "built-in", "internal")
    return any(token in label for label in labels.get(index, []) for token in builtin_tokens)


def auto_front_rgb_index(avoid_realsense: bool) -> int:
    labels = video_device_labels()
    realsense_indices = {
        index for index in labels if avoid_realsense and is_realsense_video_index(index, labels)
    }
    builtin_indices = {index for index in labels if is_builtin_video_index(index, labels)}

    preferred_tokens = ("orbbec", "astra", "gemini", "dabai")
    for index, index_labels in sorted(labels.items()):
        if index in realsense_indices or index in builtin_indices:
            continue
        if any(token in label for label in index_labels for token in preferred_tokens):
            return index

    candidates = []
    video_indices = set(labels)
    for device in glob.glob("/dev/video*"):
        index = video_index_from_path(device)
        if index is not None:
            video_indices.add(index)

    for index in sorted(video_indices):
        if index in realsense_indices or index in builtin_indices:
            continue
        candidates.append(index)

    if candidates:
        return candidates[0]

    details = ""
    if labels:
        details = f" Detected video labels: {labels}"
    raise RuntimeError(
        "Could not auto-detect a front RGB camera that does not conflict with RealSense."
        " Pass --front-rgb-index explicitly after checking /sys/class/video4linux/ or /dev/v4l/by-id/."
        + details
    )


def read_cv_key(cv2) -> str | None:
    key = cv2.waitKey(1)
    if key < 0:
        return None

    key &= 0xFF
    if key == 27:
        return "ESC"

    try:
        return chr(key).upper()
    except ValueError:
        return None


def toggle_recording(recording, csv_file, writer, data_root):
    if recording:
        close_episode(csv_file)
        print("recording stopped")
        return False, None, None, None, None, 0, 0.0

    episode_dir, images_dir, csv_file, writer = create_episode(data_root, DEFAULT_USE_WRIST_DEPTH)
    print(f"recording started: {episode_dir}")
    return True, episode_dir, images_dir, csv_file, writer, 0, 0.0


def create_episode(root: Path, use_wrist_depth: bool = DEFAULT_USE_WRIST_DEPTH):
    episode_dir = root / f"drag_episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    images_dir = episode_dir / "images"
    image_names = ["front_rgb", "wrist_rgb"]
    if use_wrist_depth:
        image_names.append("wrist_depth")
    for name in image_names:
        (images_dir / name).mkdir(parents=True, exist_ok=False)
    csv_file = (episode_dir / "data.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields(use_wrist_depth))
    writer.writeheader()
    return episode_dir, images_dir, csv_file, writer


def close_episode(csv_file):
    if csv_file is not None:
        csv_file.flush()
        csv_file.close()


def stop_recording_episode(save_queue, csv_file):
    save_queue.join()
    close_episode(csv_file)
    print("recording stopped")
    return False, None, None, None, None, 0, 0.0


@contextlib.contextmanager
def suppress_stderr(enabled: bool):
    if not enabled:
        yield
        return

    original_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(original_stderr, 2)
        os.close(original_stderr)


def keep_display_awake(enabled: bool) -> None:
    """Best-effort X11 display sleep prevention during recording."""
    if not enabled or not os.environ.get("DISPLAY"):
        return
    commands = [
        ["xset", "s", "off"],
        ["xset", "s", "noblank"],
        ["xset", "-dpms"],
    ]
    for command in commands:
        try:
            subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            return


def normalize_camera_frame(cv2, frame, color_mode: str):
    if color_mode == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def frame_for_cv2(cv2, frame, color_mode: str):
    if color_mode == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def transform_frame(cv2, frame, rotate: int = 0, flip: str = "none"):
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate != 0:
        raise ValueError(f"Unsupported rotation: {rotate}. Use 0, 90, 180, or 270.")

    if flip == "horizontal":
        frame = cv2.flip(frame, 1)
    elif flip == "vertical":
        frame = cv2.flip(frame, 0)
    elif flip == "both":
        frame = cv2.flip(frame, -1)
    elif flip != "none":
        raise ValueError(f"Unsupported flip: {flip}. Use none, horizontal, vertical, or both.")
    return frame


class CameraStream:
    def __init__(self, cv2, name: str, index: int | str, backend: str, width: int, height: int, fps: int):
        self.cv2 = cv2
        self.name = name
        self.index = int(index) if isinstance(index, str) and index.lstrip("-").isdigit() else index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def _disabled(self) -> bool:
        return self.index == -1 or self.index == "-1"

    def open(self):
        if self._disabled():
            self.cap = None
            return self
        self.cap = self.cv2.VideoCapture(self.index, opencv_backend(self.cv2, self.backend))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {self.name} camera {self.index}.")
        self.cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(self.cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)

        ok = False
        for _ in range(60):
            ok, _frame = self.cap.read()
            if ok:
                break
            time.sleep(0.05)

        if not ok:
            self.release()
            raise RuntimeError(f"{self.name} camera {self.index} opened but failed to read.")
        return self

    def read(self):
        if self._disabled():
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if self.cap is None:
            raise RuntimeError(f"{self.name} camera is not open.")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"{self.name} camera failed to read.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class RealSenseRGBDStream:
    def __init__(self, width: int, height: int, fps: int, serial: str | None = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.rs = None
        self.pipeline = None
        self.align = None

    def open(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is not installed. Run: python -m pip install pyrealsense2") from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        for _ in range(60):
            frames = self.pipeline.wait_for_frames(3000)
            if frames and frames.get_color_frame() and frames.get_depth_frame():
                return self
        self.release()
        raise RuntimeError("RealSense opened but failed to read RGB-D frames.")

    def read(self):
        if self.pipeline is None or self.align is None:
            raise RuntimeError("RealSense stream is not open.")
        frames = self.pipeline.wait_for_frames(1000)
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense failed to read RGB-D frames.")
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())
        return color_bgr, depth_u16

    def release(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class LatestValue:
    def __init__(self, initial=None):
        self._lock = threading.Lock()
        self._value = initial

    def set(self, value):
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value


class ControlLoop(threading.Thread):
    def __init__(
        self,
        controller: LeaderFollowerCopyController,
        latest_state: LatestValue,
        stop_event: threading.Event,
        hz: float,
        stats_interval: float,
        verbose: bool,
    ):
        super().__init__(daemon=True)
        self.controller = controller
        self.latest_state = latest_state
        self.stop_event = stop_event
        self.period = 1.0 / hz
        self.target_hz = hz
        self.stats_interval = stats_interval
        self.verbose = verbose
        self.error = None

    def run(self):
        count = 0
        overruns = 0
        total_step_s = 0.0
        max_step_s = 0.0
        stats_start = time.perf_counter()
        next_tick = time.perf_counter()

        while not self.stop_event.is_set():
            started = time.perf_counter()
            try:
                state = self.controller.step()
                state["step_ms"] = (time.perf_counter() - started) * 1000.0
                self.latest_state.set(state)
            except Exception as exc:
                self.error = exc
                self.stop_event.set()
                break

            elapsed = time.perf_counter() - started
            total_step_s += elapsed
            max_step_s = max(max_step_s, elapsed)
            count += 1
            if elapsed > self.period:
                overruns += 1

            now = time.perf_counter()
            if now - stats_start >= self.stats_interval:
                actual_hz = count / (now - stats_start)
                avg_step_ms = (total_step_s / max(1, count)) * 1000.0
                should_warn = overruns > 0 or actual_hz < self.target_hz * 0.9
                if self.verbose or should_warn:
                    status = "warn" if should_warn else "ok"
                    print(
                        f"control {status}: {actual_hz:.1f}Hz "
                        f"avg {avg_step_ms:.1f}ms max {max_step_s * 1000.0:.1f}ms "
                        f"over {overruns}/{count}"
                    )
                count = 0
                overruns = 0
                total_step_s = 0.0
                max_step_s = 0.0
                stats_start = now

            next_tick += self.period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.perf_counter()


class CameraReader(threading.Thread):
    def __init__(
        self,
        cv2,
        camera: CameraStream,
        key: str,
        latest_frames: LatestValue,
        stop_event: threading.Event,
        color_mode: str,
        fps: float,
        rotate: int = 0,
        flip: str = "none",
    ):
        super().__init__(daemon=True)
        self.cv2 = cv2
        self.camera = camera
        self.key = key
        self.latest_frames = latest_frames
        self.stop_event = stop_event
        self.color_mode = color_mode
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.rotate = rotate
        self.flip = flip
        self.error = None

    def run(self):
        next_tick = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                if self.key.endswith("_rgb"):
                    frame = read_rgb_camera(self.cv2, self.camera, self.color_mode)
                else:
                    frame = depth_to_uint8_rgb(self.cv2, self.camera.read())
                frame = transform_frame(self.cv2, frame, self.rotate, self.flip)

                frames = self.latest_frames.get() or {}
                frames = dict(frames)
                frames[self.key] = frame
                self.latest_frames.set(frames)
            except Exception as exc:
                self.error = exc
                self.stop_event.set()
                break

            if self.period > 0:
                next_tick += self.period
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()


class RealSenseReader(threading.Thread):
    def __init__(
        self,
        cv2,
        stream: RealSenseRGBDStream,
        latest_frames: LatestValue,
        stop_event: threading.Event,
        color_mode: str,
        fps: float,
        save_depth: bool,
        rotate: int = 0,
        flip: str = "none",
    ):
        super().__init__(daemon=True)
        self.cv2 = cv2
        self.stream = stream
        self.latest_frames = latest_frames
        self.stop_event = stop_event
        self.color_mode = color_mode
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.save_depth = save_depth
        self.rotate = rotate
        self.flip = flip
        self.error = None

    def run(self):
        next_tick = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                color_bgr, depth_u16 = self.stream.read()
                frames = self.latest_frames.get() or {}
                frames = dict(frames)
                frames["wrist_rgb"] = transform_frame(
                    self.cv2,
                    normalize_camera_frame(self.cv2, color_bgr, self.color_mode),
                    self.rotate,
                    self.flip,
                )
                if self.save_depth:
                    frames["wrist_depth"] = transform_frame(self.cv2, depth_u16, self.rotate, self.flip)
                self.latest_frames.set(frames)
            except Exception as exc:
                self.error = exc
                self.stop_event.set()
                break

            if self.period > 0:
                next_tick += self.period
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()


class SaveWorker(threading.Thread):
    def __init__(self, cv2, save_queue: queue.Queue, stop_event: threading.Event, color_mode: str):
        super().__init__(daemon=True)
        self.cv2 = cv2
        self.save_queue = save_queue
        self.stop_event = stop_event
        self.color_mode = color_mode
        self.error = None

    def run(self):
        while not self.stop_event.is_set() or not self.save_queue.empty():
            try:
                sample = self.save_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                save_sample(self.cv2, color_mode=self.color_mode, **sample)
            except Exception as exc:
                self.error = exc
                self.stop_event.set()
            finally:
                self.save_queue.task_done()


class GripperWorker(threading.Thread):
    def __init__(
        self,
        robot: DobotCR3,
        stop_event: threading.Event,
        initial_pos: float | None,
        *,
        readback: bool = False,
        readback_delay_s: float = 0.0,
    ):
        super().__init__(daemon=True)
        self.robot = robot
        self.stop_event = stop_event
        self.commands: queue.Queue[tuple[int, str]] = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.position = initial_pos
        self.action = None
        self.error = None
        self.busy = False
        self.readback = readback
        self.readback_delay_s = readback_delay_s

    def command(self, width: int, action: str) -> None:
        while True:
            try:
                self.commands.get_nowait()
                self.commands.task_done()
            except queue.Empty:
                break
        with self.lock:
            self.position = float(width)
            self.action = f"{action}_PENDING"
            self.error = None
            self.busy = True
        self.commands.put((width, action))

    def snapshot(self) -> tuple[float | None, str | None]:
        with self.lock:
            if self.error:
                return self.position, f"GRIPPER_ERROR {self.error}"
            if self.busy:
                return self.position, self.action
            return self.position, self.action

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                width, action = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if self.robot.gripper is None:
                    raise RuntimeError("gripper is not configured")
                if not self.robot.gripper.is_connected():
                    self.robot.gripper.connect()

                self.robot.gripper.set_width(width, wait=False)
                actual_pos = float(width)
                if self.readback:
                    if self.readback_delay_s > 0:
                        time.sleep(self.readback_delay_s)
                    try:
                        actual_pos = float(self.robot.gripper.get_position())
                    except Exception as exc:
                        print(f"Gripper readback failed, using commanded width {width}: {exc}")

                with self.lock:
                    self.position = actual_pos
                    self.action = action
                    self.error = None
                    self.busy = False
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.action = "GRIPPER_ERROR"
                    self.busy = False
                print(f"Gripper command failed: {exc}")
                try:
                    if self.robot.gripper is not None:
                        self.robot.gripper.disconnect()
                except Exception:
                    pass
            finally:
                self.commands.task_done()


def opencv_backend(cv2, name: str) -> int:
    backends = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }
    return backends[name]


def read_rgb_camera(cv2, camera: CameraStream, color_mode: str):
    return normalize_camera_frame(cv2, camera.read(), color_mode)


def depth_to_uint8_rgb(cv2, frame):
    if frame.ndim == 3:
        if frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame[:, :, 0]
    else:
        gray = frame

    if gray.dtype == "uint8":
        depth_u8 = gray
    else:
        valid = gray[gray > 0]
        if valid.size == 0:
            depth_u8 = np.zeros(gray.shape, dtype=np.uint8)
        else:
            low, high = np.percentile(valid, [1, 99])
            if high <= low:
                high = low + 1
            depth_u8 = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
    return cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2RGB)


def set_gripper_width_safe(robot: DobotCR3, width: int, timeout_s: float, tolerance: float) -> tuple[bool, float | None]:
    if robot.gripper is None:
        return False, None

    try:
        if not robot.gripper.is_connected():
            robot.gripper.connect()
        deadline = time.time() + timeout_s
        last_pos = None
        while True:
            robot.gripper.set_width(width, wait=False)
            try:
                last_pos = float(robot.gripper.get_position())
                if abs(last_pos - width) <= tolerance:
                    return True, last_pos
            except Exception:
                last_pos = None

            if time.time() >= deadline:
                return True, last_pos
            time.sleep(0.1)
    except Exception as exc:
        print(f"Gripper command failed, ignored: {exc}")
        try:
            robot.gripper.disconnect()
        except Exception:
            pass
        return False, None


def update_gripper(
    robot: DobotCR3,
    gripper_pos: float,
    key: str | None,
    open_width: int,
    close_width: int,
    timeout_s: float,
    tolerance: float,
):
    if key == "O" and robot.gripper is not None:
        ok, actual_pos = set_gripper_width_safe(robot, open_width, timeout_s, tolerance)
        if ok:
            return actual_pos if actual_pos is not None else float(open_width), "GRIPPER_OPEN"
        return gripper_pos, "GRIPPER_ERROR"

    if key == "P" and robot.gripper is not None:
        ok, actual_pos = set_gripper_width_safe(robot, close_width, timeout_s, tolerance)
        if ok:
            return actual_pos if actual_pos is not None else float(close_width), "GRIPPER_CLOSE"
        return gripper_pos, "GRIPPER_ERROR"

    return gripper_pos, None


def draw_overlay(cv2, frame, pose, recording, episode_dir, frame_count, gripper_pos, gripper_action):
    lines = [
        "Drag robot by hand   R record   O/P gripper   ESC quit",
        f"recording: {'ON' if recording else 'OFF'}",
    ]
    if recording:
        lines.append(f"frames: {frame_count}")
        lines.append(str(episode_dir))
    if pose is not None:
        lines.extend(
            [
                f"x {pose[0]:.2f}  y {pose[1]:.2f}  z {pose[2]:.2f}",
                f"rx {pose[3]:.2f}  ry {pose[4]:.2f}  rz {pose[5]:.2f}",
            ]
        )
    if gripper_pos is not None:
        lines.append(f"gripper {gripper_pos:.0f}  {gripper_action or ''}")

    x, y = 12, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def resize_for_display(cv2, frame, max_width: int, max_height: int):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale <= 0:
        return frame

    display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if display_size == (width, height):
        return frame
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_LINEAR)


def label_frame(cv2, frame, label: str):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (220, 34), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def make_preview_frame(cv2, frames: dict[str, np.ndarray], color_mode: str):
    front = frame_for_cv2(cv2, frames["front_rgb"], color_mode)
    wrist = frame_for_cv2(cv2, frames["wrist_rgb"], color_mode)
    preview_frames = [("front_rgb", front), ("wrist_rgb", wrist)]
    if "wrist_depth" in frames:
        depth = frame_for_cv2(cv2, depth_to_uint8_rgb(cv2, frames["wrist_depth"]), "rgb")
        preview_frames.append(("wrist_depth", depth))

    height = min(frame.shape[0] for _, frame in preview_frames)
    panels = []
    for label, frame in preview_frames:
        width = max(1, int(frame.shape[1] * height / frame.shape[0]))
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        panels.append(label_frame(cv2, resized, label))
    return np.hstack(panels)


def save_image_set(cv2, images_dir: Path, frame_count: int, frames: dict[str, np.ndarray], color_mode: str):
    names = {}
    for key, frame in frames.items():
        suffix = ".png" if key.endswith("_depth") else ".jpg"
        image_name = f"{key}/{frame_count:06d}{suffix}"
        if key.endswith("_rgb"):
            output = frame_for_cv2(cv2, frame, color_mode)
        elif frame.ndim == 2:
            output = frame
        else:
            output = frame_for_cv2(cv2, frame, "rgb")
        cv2.imwrite(str(images_dir / image_name), output)
        names[key] = f"images/{image_name}"
    return names


def save_sample(
    cv2,
    frames,
    writer,
    images_dir,
    frame_count,
    timestamp,
    pose,
    joints,
    leader_joints,
    action_target,
    gripper_pos,
    task,
    action,
    color_mode,
):
    image_names = save_image_set(cv2, images_dir, frame_count, frames, color_mode)
    row = {
        "frame": frame_count,
        "timestamp": f"{timestamp:.6f}",
        "front_rgb": image_names["front_rgb"],
        "wrist_rgb": image_names["wrist_rgb"],
        "x": f"{pose[0]:.6f}",
        "y": f"{pose[1]:.6f}",
        "z": f"{pose[2]:.6f}",
        "rx": f"{pose[3]:.6f}",
        "ry": f"{pose[4]:.6f}",
        "rz": f"{pose[5]:.6f}",
        "q1": f"{joints[0]:.6f}",
        "q2": f"{joints[1]:.6f}",
        "q3": f"{joints[2]:.6f}",
        "q4": f"{joints[3]:.6f}",
        "q5": f"{joints[4]:.6f}",
        "q6": f"{joints[5]:.6f}",
        "leader_q1": f"{leader_joints[0]:.6f}",
        "leader_q2": f"{leader_joints[1]:.6f}",
        "leader_q3": f"{leader_joints[2]:.6f}",
        "leader_q4": f"{leader_joints[3]:.6f}",
        "leader_q5": f"{leader_joints[4]:.6f}",
        "leader_q6": f"{leader_joints[5]:.6f}",
        "target_q1": f"{action_target[0]:.6f}",
        "target_q2": f"{action_target[1]:.6f}",
        "target_q3": f"{action_target[2]:.6f}",
        "target_q4": f"{action_target[3]:.6f}",
        "target_q5": f"{action_target[4]:.6f}",
        "target_q6": f"{action_target[5]:.6f}",
        "gripper": "" if gripper_pos is None else f"{gripper_pos:.3f}",
        "task": task,
        "action": action or "HAND_GUIDE",
    }
    if "wrist_depth" in image_names and "wrist_depth" in writer.fieldnames:
        row["wrist_depth"] = image_names["wrist_depth"]
    writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--follower-ip", default="192.168.6.1")
    parser.add_argument("--no-align-follower", action="store_true")
    drag_group = parser.add_mutually_exclusive_group()
    drag_group.add_argument(
        "--manual-drag",
        dest="manual_drag",
        action="store_true",
        help="Do not call StartDrag automatically. Enable drag mode manually.",
    )
    drag_group.add_argument(
        "--auto-drag",
        dest="manual_drag",
        action="store_false",
        help="Call StartDrag automatically after connecting.",
    )
    parser.set_defaults(manual_drag=DEFAULT_MANUAL_DRAG)
    gripper_group = parser.add_mutually_exclusive_group()
    gripper_group.add_argument("--with-gripper", dest="use_gripper", action="store_true")
    gripper_group.add_argument("--no-gripper", dest="use_gripper", action="store_false")
    parser.set_defaults(use_gripper=DEFAULT_USE_GRIPPER)
    parser.add_argument("--allow-missing-gripper", action="store_true", help="Continue recording without gripper data if the gripper cannot connect.")
    parser.add_argument("--gripper-port", default=default_gripper_port())
    parser.add_argument("--gripper-baudrate", type=int, default=115200)
    parser.add_argument("--gripper-slave-address", type=int, default=1)
    parser.add_argument("--gripper-modbus-timeout", type=float, default=1.0)
    parser.add_argument("--gripper-modbus-retries", type=int, default=3)
    parser.add_argument("--camera-index", default=DEFAULT_CAMERA_INDEX, help="Legacy alias for --front-rgb-index.")
    parser.add_argument(
        "--front-rgb-index",
        default=DEFAULT_FRONT_RGB_INDEX,
        help="OpenCV index or stable device path for the front RGB camera.",
    )
    parser.add_argument("--wrist-rgb-index", default=DEFAULT_WRIST_RGB_INDEX)
    parser.add_argument("--wrist-depth-index", default=DEFAULT_WRIST_DEPTH_INDEX)
    wrist_group = parser.add_mutually_exclusive_group()
    wrist_group.add_argument("--realsense-wrist", dest="use_realsense_wrist", action="store_true", help="Use RealSense for wrist RGB-D streams.")
    wrist_group.add_argument("--opencv-wrist", dest="use_realsense_wrist", action="store_false", help="Use OpenCV indices for wrist streams instead of RealSense.")
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--wrist-depth", dest="use_wrist_depth", action="store_true", help="Record wrist depth images.")
    depth_group.add_argument("--no-wrist-depth", dest="use_wrist_depth", action="store_false", help="Record only front RGB and wrist RGB images.")
    parser.add_argument("--realsense-serial")
    parser.add_argument("--realsense-width", type=int, default=DEFAULT_REALSENSE_WIDTH)
    parser.add_argument("--realsense-height", type=int, default=DEFAULT_REALSENSE_HEIGHT)
    parser.add_argument("--realsense-fps", type=int, default=DEFAULT_REALSENSE_FPS)
    parser.set_defaults(use_realsense_wrist=DEFAULT_USE_REALSENSE_WRIST)
    parser.set_defaults(use_wrist_depth=DEFAULT_USE_WRIST_DEPTH)
    parser.add_argument("--backend", choices=["any", "dshow", "msmf", "v4l2"], default=default_camera_backend())
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--camera-color", choices=["rgb", "bgr"], default=DEFAULT_CAMERA_COLOR)
    parser.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_WIDTH)
    parser.add_argument("--display-height", type=int, default=DEFAULT_DISPLAY_HEIGHT)
    parser.add_argument("--front-rotate", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--wrist-rotate", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--front-flip", choices=["none", "horizontal", "vertical", "both"], default="none")
    parser.add_argument("--wrist-flip", choices=["none", "horizontal", "vertical", "both"], default="none")
    parser.add_argument(
        "--allow-builtin-front-camera",
        action="store_true",
        help="Allow using the laptop built-in camera as front_rgb.",
    )
    parser.add_argument("--speed-factor", type=int, default=DEFAULT_SPEED_FACTOR)
    parser.add_argument("--record-fps", type=float, default=DEFAULT_RECORD_FPS)
    parser.add_argument("--control-only", action="store_true", help="Only run 50Hz leader-follower control; do not open cameras, display, or save.")
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ)
    parser.add_argument("--control-stats-interval", type=float, default=DEFAULT_CONTROL_STATS_INTERVAL)
    parser.add_argument("--verbose-control", action="store_true", help="Print regular control-loop timing stats.")
    parser.add_argument("--camera-read-fps", type=float, default=DEFAULT_CAMERA_READ_FPS)
    parser.add_argument("--gripper-open-width", type=int, default=DEFAULT_GRIPPER_OPEN_WIDTH)
    parser.add_argument("--gripper-close-width", type=int, default=DEFAULT_GRIPPER_CLOSE_WIDTH)
    parser.add_argument("--gripper-timeout", type=float, default=DEFAULT_GRIPPER_TIMEOUT)
    parser.add_argument("--gripper-tolerance", type=float, default=DEFAULT_GRIPPER_TOLERANCE)
    parser.add_argument(
        "--gripper-readback",
        action="store_true",
        help="Read gripper position after each command. Disabled by default because some Modbus grippers time out while moving.",
    )
    parser.add_argument("--gripper-readback-delay-s", type=float, default=0.0)
    init_gripper_group = parser.add_mutually_exclusive_group()
    init_gripper_group.add_argument("--initialize-gripper", dest="initialize_gripper", action="store_true")
    init_gripper_group.add_argument("--no-initialize-gripper", dest="initialize_gripper", action="store_false")
    parser.set_defaults(initialize_gripper=DEFAULT_INITIALIZE_GRIPPER)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--task",
        required=True,
        help="Language instruction saved with each recorded frame and episode.",
    )
    parser.add_argument(
        "--allow-display-sleep",
        action="store_true",
        help="Do not disable screen saver / DPMS during recording.",
    )
    parser.add_argument("--verbose-api", action="store_true")
    parser.add_argument("--show-opencv-warnings", action="store_true")
    args = parser.parse_args()

    if not args.show_opencv_warnings:
        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

    keep_display_awake(not args.allow_display_sleep)

    print(f"wrist camera mode: {'RealSense' if args.use_realsense_wrist else 'OpenCV/placeholder'}")
    print(f"wrist depth recording: {'on' if args.use_wrist_depth else 'off'}")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed. Run: pip install opencv-python") from exc

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    if args.camera_index is not None:
        front_rgb_index = args.camera_index
    elif args.front_rgb_index is not None:
        front_rgb_index = args.front_rgb_index
    else:
        front_rgb_index = auto_front_rgb_index(avoid_realsense=args.use_realsense_wrist)

    front_video_index = resolve_video_index(front_rgb_index)
    if (
        front_video_index is not None
        and is_builtin_video_index(front_video_index, video_device_labels())
        and not args.allow_builtin_front_camera
    ):
        raise RuntimeError(
            f"Selected front_rgb camera {front_rgb_index} is detected as the laptop built-in camera. "
            "Use the external camera, for example --front-rgb-index 0 or its /dev/v4l/by-id path. "
            "Pass --allow-builtin-front-camera only if this is intentional."
        )

    print(
        "camera layout: front_rgb=Orbbec/OpenCV "
        f"index {front_rgb_index}, "
        f"wrist_rgb={'RealSense' if args.use_realsense_wrist else 'OpenCV index'}, "
        f"wrist_depth={'enabled' if args.use_wrist_depth else 'disabled'}"
    )

    config = DobotCR3Config(
        robot_ip=args.robot_ip,
        gripper_port=args.gripper_port,
        gripper_baudrate=args.gripper_baudrate,
        gripper_slave_address=args.gripper_slave_address,
        gripper_modbus_timeout=args.gripper_modbus_timeout,
        gripper_modbus_retries=args.gripper_modbus_retries,
        use_gripper=args.use_gripper,
        speed_factor=args.speed_factor,
        use_opencv_camera=False,
        opencv_camera_index=front_rgb_index,
        opencv_camera_backend=args.backend,
        opencv_camera_width=args.width,
        opencv_camera_height=args.height,
        opencv_camera_fps=args.fps,
    )

    robot = DobotCR3(config)
    recording = False
    r_was_pressed = False
    episode_dir = None
    images_dir = None
    csv_file = None
    writer = None
    frame_count = 0
    last_record_time = 0.0
    record_period = 1.0 / args.record_fps
    o_was_pressed = False
    p_was_pressed = False
    pose = None
    joints = None
    gripper_pos = None
    gripper_action = None
    follower_controller = None
    control_stop = threading.Event()
    camera_stop = threading.Event()
    save_stop = threading.Event()
    latest_control_state = LatestValue()
    latest_frames = LatestValue({})
    save_queue = queue.Queue(maxsize=128)
    control_thread = None
    camera_threads: list[threading.Thread] = []
    save_worker = None
    gripper_stop = threading.Event()
    gripper_worker = None
    cameras: list[CameraStream] = []
    realsense_stream = None
    try:
        try:
            robot.connect()
        except ConnectionError as exc:
            if not args.use_gripper or not args.allow_missing_gripper:
                raise

            print(f"Gripper unavailable, continuing without it: {exc}")
            robot.disconnect()
            config.use_gripper = False
            robot = DobotCR3(config)
            robot.connect()

        robot.set_api_verbose(args.verbose_api)
        if robot.gripper is not None and args.initialize_gripper:
            try:
                robot.gripper.initialize(wait=True)
            except Exception as exc:
                print(f"Gripper initialize failed, continuing without initialization: {exc}")
        obs = robot.get_observation()
        pose = [obs[k] for k in ["x.pos", "y.pos", "z.pos", "rx.pos", "ry.pos", "rz.pos"]]
        joints = [obs[k] for k in ["q1.pos", "q2.pos", "q3.pos", "q4.pos", "q5.pos", "q6.pos"]]
        gripper_pos = obs["gripper.pos"] if obs["gripper.pos"] >= 0 else 100.0
        if robot.gripper is not None:
            gripper_worker = GripperWorker(
                robot,
                gripper_stop,
                gripper_pos,
                readback=args.gripper_readback,
                readback_delay_s=args.gripper_readback_delay_s,
            )
            gripper_worker.start()

        follower_controller = LeaderFollowerCopyController(
            auto_enable=True,
            start_drag=not args.manual_drag,
            leader_ip=args.robot_ip,
            follower_ip=args.follower_ip,
            align_on_start=not args.no_align_follower,
            leader_dashboard=robot.dashboard,
            leader_move=robot.move,
        )
        follower_controller.connect()
        control_thread = ControlLoop(
            follower_controller,
            latest_control_state,
            control_stop,
            args.control_hz,
            args.control_stats_interval,
            args.verbose_control,
        )
        control_thread.start()

        if args.control_only:
            print("control-only mode ready. Press Ctrl+C to stop.")
            while not control_stop.is_set():
                if control_thread.error is not None:
                    raise control_thread.error
                time.sleep(0.2)
            return

        with suppress_stderr(not args.show_opencv_warnings):
            front_camera = CameraStream(cv2, "front_rgb", front_rgb_index, args.backend, args.width, args.height, args.fps).open()
        cameras = [front_camera]
        front_reader = CameraReader(
            cv2,
            front_camera,
            "front_rgb",
            latest_frames,
            camera_stop,
            args.camera_color,
            args.camera_read_fps,
            args.front_rotate,
            args.front_flip,
        )
        front_reader.start()
        camera_threads.append(front_reader)

        if args.use_realsense_wrist:
            realsense_stream = RealSenseRGBDStream(
                args.realsense_width,
                args.realsense_height,
                args.realsense_fps,
                args.realsense_serial,
            ).open()
            realsense_reader = RealSenseReader(
                cv2,
                realsense_stream,
                latest_frames,
                camera_stop,
                args.camera_color,
                args.camera_read_fps,
                args.use_wrist_depth,
                args.wrist_rotate,
                args.wrist_flip,
            )
            realsense_reader.start()
            camera_threads.append(realsense_reader)
            print("wrist RGB: RealSense color stream")
        else:
            camera_map = {}
            with suppress_stderr(not args.show_opencv_warnings):
                camera_map["wrist_rgb"] = CameraStream(cv2, "wrist_rgb", args.wrist_rgb_index, args.backend, args.width, args.height, args.fps).open()
                if args.use_wrist_depth:
                    camera_map["wrist_depth"] = CameraStream(cv2, "wrist_depth", args.wrist_depth_index, args.backend, args.width, args.height, args.fps).open()
            cameras.extend(camera_map.values())
            for key, camera in camera_map.items():
                reader = CameraReader(
                    cv2,
                    camera,
                    key,
                    latest_frames,
                    camera_stop,
                    args.camera_color,
                    args.camera_read_fps,
                    args.wrist_rotate,
                    args.wrist_flip,
                )
                reader.start()
                camera_threads.append(reader)
            print("wrist RGB: OpenCV camera index")

        save_worker = SaveWorker(cv2, save_queue, save_stop, args.camera_color)
        save_worker.start()
        print("recorder ready. Window keys: R record, O open, P close, ESC quit.")
        with suppress_stderr(not args.show_opencv_warnings):
            cv2.namedWindow("Dobot CR3 drag recorder", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Dobot CR3 drag recorder", args.display_width, args.display_height)

        while True:
            now = time.time()
            if control_thread.error is not None:
                raise control_thread.error
            for reader in camera_threads:
                if reader.error is not None:
                    raise reader.error
            if save_worker is not None and save_worker.error is not None:
                raise save_worker.error

            control_state = latest_control_state.get()
            if control_state is not None:
                joints = control_state["follower_actual"]
                leader_joints = control_state["leader_joints"]
                action_target = control_state["action_target"]
            else:
                leader_joints = joints
                action_target = joints

            if gripper_worker is not None:
                gripper_pos, gripper_action = gripper_worker.snapshot()

            frames = latest_frames.get()
            required_frame_keys = ["front_rgb", "wrist_rgb"]
            if args.use_wrist_depth:
                required_frame_keys.append("wrist_depth")
            if not all(key in frames for key in required_frame_keys):
                time.sleep(0.01)
                continue

            display_frame = resize_for_display(
                cv2,
                make_preview_frame(cv2, frames, args.camera_color),
                args.display_width,
                args.display_height,
            )
            display_frame = display_frame.copy()
            draw_overlay(cv2, display_frame, pose, recording, episode_dir, frame_count, gripper_pos if robot.gripper is not None else None, gripper_action)
            with suppress_stderr(not args.show_opencv_warnings):
                cv2.imshow("Dobot CR3 drag recorder", display_frame)
                key = read_cv_key(cv2)
            if key == "ESC":
                break

            if key == "R" and not r_was_pressed:
                if recording:
                    (
                        recording,
                        episode_dir,
                        images_dir,
                        csv_file,
                        writer,
                        frame_count,
                        last_record_time,
                    ) = stop_recording_episode(save_queue, csv_file)
                else:
                    episode_dir, images_dir, csv_file, writer = create_episode(data_root, args.use_wrist_depth)
                    (episode_dir / "task.txt").write_text(args.task + "\n", encoding="utf-8")
                    print(f"recording started: {episode_dir}")
                    recording = True
                    frame_count = 0
                    last_record_time = 0.0
            r_was_pressed = key == "R"

            if gripper_worker is not None:
                if key == "O" and not o_was_pressed:
                    gripper_worker.command(args.gripper_open_width, "GRIPPER_OPEN")
                elif key == "P" and not p_was_pressed:
                    gripper_worker.command(args.gripper_close_width, "GRIPPER_CLOSE")
            elif robot.gripper is None:
                gripper_action = None
            o_was_pressed = key == "O"
            p_was_pressed = key == "P"

            if recording and writer is not None and now - last_record_time >= record_period:
                save_queue.put(
                    {
                        "frames": {key: frame.copy() for key, frame in frames.items()},
                        "writer": writer,
                        "images_dir": images_dir,
                        "frame_count": frame_count,
                        "timestamp": now,
                        "pose": pose[:],
                        "joints": joints[:],
                        "leader_joints": leader_joints[:],
                        "action_target": action_target[:],
                        "gripper_pos": gripper_pos if robot.gripper is not None else None,
                        "task": args.task,
                        "action": gripper_action or "LEADER_FOLLOWER_COPY",
                    }
                )
                frame_count += 1
                last_record_time = now

    except KeyboardInterrupt:
        print("\ndrag recorder stopped")
    finally:
        control_stop.set()
        camera_stop.set()
        gripper_stop.set()
        save_stop.set()
        if gripper_worker is not None:
            gripper_worker.join(timeout=1.0)
        if control_thread is not None:
            control_thread.join(timeout=2.0)
        for reader in camera_threads:
            reader.join(timeout=1.0)
        if save_worker is not None:
            save_queue.join()
            save_worker.join(timeout=2.0)
        close_episode(csv_file)
        if follower_controller is not None:
            follower_controller.close()
        for camera in cameras:
            camera.release()
        if realsense_stream is not None:
            realsense_stream.release()
        robot.disconnect()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
