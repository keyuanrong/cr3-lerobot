import argparse
import os
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

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots.dobot_cr3 import DobotCR3, DobotCR3Config

from scripts.collection.record_drag_dataset import (
    CameraStream,
    RealSenseRGBDStream,
    default_camera_backend,
    default_gripper_port,
    depth_to_uint8_rgb,
    frame_for_cv2,
    normalize_camera_frame,
)

DEFAULT_POLICY_PATH = (
    LEROBOT_ROOT
    / "outputs/train/cr3_act_review_good_from_scratch_100k_v2/checkpoints/070000/pretrained_model"
)


class LatestCameraReader:
    def __init__(self, cv2, camera: CameraStream, color_mode: str = "rgb"):
        self.cv2 = cv2
        self.camera = camera
        self.color_mode = color_mode
        self._lock = threading.Lock()
        self._frame = None
        self._timestamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.error = None

    def start(self):
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = normalize_camera_frame(self.cv2, self.camera.read(), self.color_mode)
                timestamp = time.monotonic()
                with self._lock:
                    self._frame = frame
                    self._timestamp = timestamp
            except Exception as exc:
                self.error = exc
                self._stop.set()
                break

    def latest(self, timeout_s: float = 2.0) -> tuple[np.ndarray, float]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError(f"{self.camera.name} reader failed: {self.error}") from self.error
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy(), self._timestamp
            time.sleep(0.005)
        raise TimeoutError(f"Timed out waiting for {self.camera.name} frame.")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def image_to_tensor(cv2, frame: np.ndarray, image_width: int, image_height: int) -> torch.Tensor:
    if frame.shape[1] != image_width or frame.shape[0] != image_height:
        frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0


def feature_hw(policy: PreTrainedPolicy, key: str) -> tuple[int, int]:
    shape = policy.config.input_features[key].shape
    return int(shape[1]), int(shape[2])


def make_observation(
    cv2,
    robot: DobotCR3,
    frames: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
) -> dict:
    obs = robot.get_observation()
    gripper = obs["gripper.pos"] if obs["gripper.pos"] >= 0 else 100.0
    state = torch.tensor(
        [
            obs["q1.pos"],
            obs["q2.pos"],
            obs["q3.pos"],
            obs["q4.pos"],
            obs["q5.pos"],
            obs["q6.pos"],
            gripper,
        ],
        dtype=torch.float32,
    )
    observation = {"observation.state": state}
    for key, frame_name in [
        ("observation.images.front_rgb", "front_rgb"),
        ("observation.images.wrist_rgb", "wrist_rgb"),
        ("observation.images.wrist_depth", "wrist_depth"),
    ]:
        if key not in policy.config.input_features:
            continue
        if frame_name not in frames:
            raise KeyError(f"Policy expects {key}, but frame '{frame_name}' was not captured.")
        image_h, image_w = feature_hw(policy, key)
        observation[key] = image_to_tensor(cv2, frames[frame_name], image_w, image_h)
    return observation


def load_policy(policy_path: Path, device: str, base_policy_path: str | None = None):
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    config.device = device
    policy_cls = get_policy_class(config.type)

    if getattr(config, "use_peft", False):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(policy_path))
        if base_policy_path is not None:
            peft_config.base_model_name_or_path = str(base_policy_path)
        base_path = peft_config.base_model_name_or_path
        if not base_path:
            raise ValueError("LoRA adapter has no base_model_name_or_path; pass --base-policy-path.")
        print(f"Loading base policy from: {base_path}")
        policy = policy_cls.from_pretrained(
            base_path,
            config=config,
            local_files_only=True,
        )
        print(f"Loading LoRA adapter from: {policy_path}")
        policy = PeftModel.from_pretrained(policy, str(policy_path), config=peft_config, is_trainable=False)
        policy.eval()
        policy.to(device)
        return policy

    policy = policy_cls.from_pretrained(policy_path, config=config, local_files_only=True)
    policy.eval()
    policy.to(device)
    return policy


def clamp_action(action: np.ndarray, max_joint_delta: float) -> np.ndarray:
    limited = action.copy()
    limited[:6] = np.clip(limited[:6], -max_joint_delta, max_joint_delta)
    if limited.shape[0] > 6:
        limited[6] = 1.0 if limited[6] >= 0.5 else 0.0
    return limited


def clamp_absolute_action(current: np.ndarray, action: np.ndarray, max_joint_delta: float) -> np.ndarray:
    limited = action.copy()
    lower = current[:6] - max_joint_delta
    upper = current[:6] + max_joint_delta
    limited[:6] = np.clip(limited[:6], lower, upper)
    if limited.shape[0] > 6:
        limited[6] = 1.0 if limited[6] >= 0.5 else 0.0
    return limited


def parse_six_floats(value: str) -> np.ndarray:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(values)}: {value}")
    return np.asarray(values, dtype=np.float32)


def delta_action_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    return {
        "dq1": float(action[0]),
        "dq2": float(action[1]),
        "dq3": float(action[2]),
        "dq4": float(action[3]),
        "dq5": float(action[4]),
        "dq6": float(action[5]),
        "gripper.open": float(action[6]),
    }


def send_action_to_robot(
    robot: DobotCR3,
    action: np.ndarray,
    *,
    action_mode: str,
    command_mode: str,
    servo_t: float,
    servo_lookahead_time: float,
    servo_gain: float,
    gripper_width: int | None,
) -> None:
    if action_mode == "delta":
        target = np.asarray(robot.get_joints(), dtype=np.float32) + action[:6]
        if command_mode == "jointmovj":
            robot.send_action(delta_action_to_robot_dict(action))
            return
    elif action_mode == "absolute":
        target = action[:6]
    else:
        raise ValueError(f"Unsupported action mode: {action_mode}")

    if command_mode == "jointmovj":
        if robot.move is None:
            raise ConnectionError("Dobot move interface is not connected.")
        robot.move.JointMovJ(*[float(v) for v in target])
        return

    if command_mode == "servoj":
        if robot.move is None:
            raise ConnectionError("Dobot move interface is not connected.")
        robot.move.ServoJ(
            *[float(v) for v in target],
            t=float(servo_t),
            lookahead_time=float(servo_lookahead_time),
            gain=float(servo_gain),
        )
        if gripper_width is not None and robot.gripper is not None:
            try:
                robot.gripper.set_width(gripper_width, wait=False)
            except Exception as exc:
                print(f"WARNING: gripper command failed and was skipped: {exc}", flush=True)
        return

    raise ValueError(f"Unsupported command mode: {command_mode}")


def resize_for_panel(cv2, frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def put_lines(cv2, frame: np.ndarray, lines: list[str]) -> np.ndarray:
    output = frame.copy()
    x, y = 12, 28
    for line in lines:
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(output, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return output


def show_preview(
    cv2,
    frames: dict[str, np.ndarray],
    current: np.ndarray,
    pred: np.ndarray,
    cmd: np.ndarray,
    execute: bool,
    step: int,
    preview_dir: Path | None,
    wait_ms: int,
    preview_backend: str,
    mpl_state: dict,
) -> bool:
    panel_w, panel_h = 426, 240
    front = resize_for_panel(cv2, frame_for_cv2(cv2, frames["front_rgb"], "rgb"), panel_w, panel_h)
    wrist = resize_for_panel(cv2, frame_for_cv2(cv2, frames["wrist_rgb"], "rgb"), panel_w, panel_h)
    panels = [front, wrist]
    if "wrist_depth" in frames:
        depth = resize_for_panel(cv2, frame_for_cv2(cv2, frames["wrist_depth"], "rgb"), panel_w, panel_h)
        panels.append(depth)

    top = np.hstack(panels)
    lines = [
        f"step {step:04d}  mode={'EXECUTE' if execute else 'DRY-RUN'}  q/ESC quit",
        "current " + np.array2string(current, precision=2, suppress_small=True),
        "pred    " + np.array2string(pred, precision=2, suppress_small=True),
        "cmd     " + np.array2string(cmd, precision=2, suppress_small=True),
    ]
    bottom = np.zeros((150, top.shape[1], 3), dtype=np.uint8)
    bottom = put_lines(cv2, bottom, lines)
    preview = np.vstack([top, bottom])
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_dir / f"step_{step:06d}.jpg"), preview)
    if preview_backend == "matplotlib":
        import matplotlib.pyplot as plt

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        if "fig" not in mpl_state:
            plt.ion()
            fig, ax = plt.subplots(num="CR3 policy real cameras", figsize=(13, 5))
            image = ax.imshow(preview_rgb)
            ax.axis("off")
            mpl_state.update({"fig": fig, "ax": ax, "image": image})
            plt.show(block=False)
        else:
            mpl_state["image"].set_data(preview_rgb)
        mpl_state["fig"].canvas.draw_idle()
        plt.pause(max(wait_ms, 1) / 1000.0)
        return plt.fignum_exists(mpl_state["fig"].number)

    try:
        if step == 0:
            cv2.namedWindow("CR3 policy real cameras", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CR3 policy real cameras", preview.shape[1], preview.shape[0])
        cv2.imshow("CR3 policy real cameras", preview)
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        return key not in {27, ord("q"), ord("Q")}
    except cv2.error as exc:
        print(f"OpenCV preview window is unavailable; continuing without imshow: {exc}", flush=True)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--base-policy-path", default=None, help="Local base policy path for PEFT/LoRA checkpoints.")
    parser.add_argument("--robot-ip", default="192.168.6.1")
    parser.add_argument("--front-rgb-index", default="0", help="OpenCV camera index or stable device path.")
    parser.add_argument("--wrist-rgb-index", default="-1", help="OpenCV camera index or stable device path.")
    parser.add_argument("--wrist-depth-index", default="-1", help="OpenCV camera index or stable device path.")
    parser.add_argument("--use-realsense-wrist", action="store_true")
    parser.add_argument("--realsense-serial", default=None)
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=15)
    parser.add_argument("--backend", default=default_camera_backend(), choices=["any", "dshow", "msmf", "v4l2"])
    parser.add_argument("--front-backend", choices=["any", "dshow", "msmf", "v4l2"], default=None)
    parser.add_argument("--wrist-backend", choices=["any", "dshow", "msmf", "v4l2"], default=None)
    parser.add_argument("--front-width", type=int, default=1280)
    parser.add_argument("--front-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--gripper-port", default=default_gripper_port())
    parser.add_argument("--no-gripper", dest="use_gripper", action="store_false")
    parser.set_defaults(use_gripper=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--replan-every",
        type=int,
        default=0,
        help="Reset ACT action queue every N steps so the policy replans from the latest observation. 0 keeps the checkpoint default chunking.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--command-mode", choices=["jointmovj", "servoj"], default="servoj")
    parser.add_argument("--servo-t", type=float, default=None)
    parser.add_argument("--servo-lookahead-time", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)
    parser.add_argument(
        "--joint-action-sign",
        default="1,1,1,1,1,1",
        help="Per-joint multiplier applied to predicted dq before limiting/sending.",
    )
    parser.add_argument("--no-action-limit", action="store_true")
    parser.add_argument("--max-joint-delta", type=float, default=0.5, help="deg per control step")
    parser.add_argument("--speed-factor", type=int, default=10)
    parser.add_argument(
        "--gripper-every",
        type=int,
        default=10,
        help="Send gripper command every N policy steps or when the open/close target changes. 0 sends only on target changes.",
    )
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-backend", choices=["cv2", "matplotlib"], default="matplotlib")
    parser.add_argument("--preview-wait-ms", type=int, default=80)
    parser.add_argument("--preview-dir", type=Path, default=None, help="Save preview frames as jpg files.")
    parser.add_argument("--max-camera-skew-ms", type=float, default=120.0, help="Warn when front/wrist frame timestamps differ by more than this; does not skip actions.")
    parser.add_argument("--print-camera-skew", action="store_true")
    parser.add_argument("--verbose-api", action="store_true")
    args = parser.parse_args()
    front_backend = args.front_backend or args.backend
    wrist_backend = args.wrist_backend or args.backend

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled; CUDA may still be used, but convolution will use non-cuDNN kernels.")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed.") from exc
    print(f"DISPLAY={os.environ.get('DISPLAY')!r} WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}", flush=True)
    mpl_state = {}

    policy_path = Path(args.policy_path)
    joint_action_sign = parse_six_floats(args.joint_action_sign)
    policy = load_policy(policy_path, args.device, args.base_policy_path)
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    expected_image_keys = set(policy.config.input_features) & {
        "observation.images.front_rgb",
        "observation.images.wrist_rgb",
        "observation.images.wrist_depth",
    }
    needs_wrist_depth = "observation.images.wrist_depth" in expected_image_keys

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
    wrist_rgb_camera = None
    wrist_depth_camera = None
    front_reader = None
    wrist_reader = None
    realsense = None
    period = 1.0 / args.hz
    try:
        front_camera = CameraStream(
            cv2,
            "front_rgb",
            args.front_rgb_index,
            front_backend,
            args.front_width,
            args.front_height,
            args.camera_fps,
        ).open()
        if args.use_realsense_wrist:
            realsense = RealSenseRGBDStream(
                args.realsense_width,
                args.realsense_height,
                args.realsense_fps,
                serial=args.realsense_serial,
            ).open()
        else:
            wrist_rgb_camera = CameraStream(
                cv2,
                "wrist_rgb",
                args.wrist_rgb_index,
                wrist_backend,
                args.realsense_width,
                args.realsense_height,
                args.camera_fps,
            ).open()
            if needs_wrist_depth:
                wrist_depth_camera = CameraStream(
                    cv2,
                    "wrist_depth",
                    args.wrist_depth_index,
                    wrist_backend,
                    args.realsense_width,
                    args.realsense_height,
                    args.camera_fps,
                ).open()
            front_reader = LatestCameraReader(cv2, front_camera).start()
            wrist_reader = LatestCameraReader(cv2, wrist_rgb_camera).start()

        robot.connect()
        robot.set_api_verbose(args.verbose_api)
        print("policy rollout ready")
        print("mode:", "EXECUTE" if args.execute else "DRY-RUN")
        print("Ctrl-C to stop.")
        last_gripper_width = None

        for step in range(args.steps):
            start = time.time()
            if realsense is not None:
                frames = {"front_rgb": normalize_camera_frame(cv2, front_camera.read(), "rgb")}
                wrist_color_bgr, wrist_depth_u16 = realsense.read()
                frames["wrist_rgb"] = normalize_camera_frame(cv2, wrist_color_bgr, "rgb")
                if needs_wrist_depth:
                    frames["wrist_depth"] = depth_to_uint8_rgb(cv2, wrist_depth_u16)
            else:
                front_frame, front_ts = front_reader.latest()
                wrist_frame, wrist_ts = wrist_reader.latest()
                skew_ms = abs(front_ts - wrist_ts) * 1000.0
                if args.print_camera_skew or skew_ms > args.max_camera_skew_ms:
                    print(f"{step:04d} camera_skew_ms={skew_ms:.1f}", flush=True)
                frames = {"front_rgb": front_frame, "wrist_rgb": wrist_frame}
                if needs_wrist_depth:
                    frames["wrist_depth"] = depth_to_uint8_rgb(cv2, wrist_depth_camera.read())

            raw_obs = make_observation(cv2, robot, frames, policy)
            batch = preprocessor(raw_obs)
            if args.replan_every > 0 and step % args.replan_every == 0:
                policy.reset()
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action).detach().cpu().numpy().reshape(-1)
            signed_action = action.copy()
            signed_action[:6] *= joint_action_sign

            current = raw_obs["observation.state"].numpy()
            if args.no_action_limit:
                limited = signed_action.copy()
                if limited.shape[0] > 6:
                    limited[6] = 1.0 if limited[6] >= 0.5 else 0.0
            elif args.action_mode == "absolute":
                limited = clamp_absolute_action(current, signed_action, args.max_joint_delta)
            else:
                limited = clamp_action(signed_action, args.max_joint_delta)
            print(
                f"{step:04d} current={np.round(current, 2).tolist()} "
                f"pred={np.round(action, 2).tolist()} "
                f"signed={np.round(signed_action, 2).tolist()} "
                f"cmd={np.round(limited, 2).tolist()}"
            )

            if not args.no_preview and not show_preview(
                cv2,
                frames,
                current,
                action,
                limited,
                args.execute,
                step,
                args.preview_dir,
                args.preview_wait_ms,
                args.preview_backend,
                mpl_state,
            ):
                print("preview requested stop")
                break

            if args.execute:
                gripper_width = None
                if args.use_gripper and limited.shape[0] > 6:
                    candidate_width = 100 if float(limited[6]) >= 0.5 else 0
                    should_send_gripper = candidate_width != last_gripper_width
                    if args.gripper_every > 0:
                        should_send_gripper = should_send_gripper or step % args.gripper_every == 0
                    if should_send_gripper:
                        gripper_width = candidate_width
                        last_gripper_width = candidate_width
                print(f"{step:04d} sending cmd to robot...")
                send_action_to_robot(
                    robot,
                    limited,
                    action_mode=args.action_mode,
                    command_mode=args.command_mode,
                    servo_t=args.servo_t if args.servo_t is not None else period,
                    servo_lookahead_time=args.servo_lookahead_time,
                    servo_gain=args.servo_gain,
                    gripper_width=gripper_width,
                )
                print(f"{step:04d} robot command returned")

            elapsed = time.time() - start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        robot.disconnect()
        if front_reader is not None:
            front_reader.stop()
        if wrist_reader is not None:
            wrist_reader.stop()
        if front_camera is not None:
            front_camera.release()
        if wrist_rgb_camera is not None:
            wrist_rgb_camera.release()
        if wrist_depth_camera is not None:
            wrist_depth_camera.release()
        if realsense is not None:
            realsense.release()
        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            if mpl_state:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(mpl_state["fig"])
                except Exception:
                    pass


if __name__ == "__main__":
    main()
