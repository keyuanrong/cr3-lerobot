#!/usr/bin/env python

"""Roll out a fine-tuned SmolVLA checkpoint in the CR3 + LMG-90 MuJoCo scene."""

from __future__ import annotations

import argparse
import math
from queue import Empty, Queue
from pathlib import Path
import sys
import threading
import time

import glfw
import mujoco
import numpy as np
import torch


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.configs import RTCAttentionSchedule
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot.policies.smolvla import SmolVLAPolicy
from sim.cr3_mujoco.collect_stack_blocks_vla_dataset import CameraRenderer
from sim.cr3_mujoco.record_vla_teleop_dataset import TASK_DESCRIPTION
from sim.cr3_mujoco.teleop_cr3_eef import (
    ARM_ACTUATORS,
    DEFAULT_INITIAL_DISPLAY_STATE,
    GRIPPER_GRASP_CMD,
    GRIPPER_OPEN_CMD,
    GraspAssist,
    MinimalGLFWViewer,
    SCENE_XML,
    arm_actuator_ids,
    arm_dof_addrs,
    arm_qpos_addrs,
    gripper_actuator_ids,
    parse_seven,
    set_display_state,
    set_gripper,
)


DEFAULT_POLICY_PATH = (
    LEROBOT_ROOT
    / "outputs/train/cr3_smolvla_100k_v6_bs8/checkpoints/070000/pretrained_model"
)
DEFAULT_LOCAL_VLM_PATH = Path("/home/kyr/hf_models/SmolVLM2-500M-Video-Instruct")


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def move_to_device(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def policy_image_shape(policy: SmolVLAPolicy, key: str, fallback: tuple[int, int] = (480, 640)) -> tuple[int, int]:
    feature = policy.config.input_features.get(key)
    if feature is None:
        return fallback
    # Policy shape is channel-first: C,H,W.
    return int(feature.shape[1]), int(feature.shape[2])


def state_vector(model: mujoco.MjModel, data: mujoco.MjData, gripper_cmd: float) -> np.ndarray:
    state = np.empty(7, dtype=np.float32)
    state[:6] = data.qpos[arm_qpos_addrs(model)].astype(np.float32)
    state[6] = np.float32(gripper_cmd)
    return state


def make_observation(
    *,
    renderer: CameraRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    policy: SmolVLAPolicy,
    gripper_cmd: float,
    task: str,
    front_camera: str,
    wrist_camera: str,
) -> dict:
    front_h, front_w = policy_image_shape(policy, "observation.images.camera1")
    wrist_h, wrist_w = policy_image_shape(policy, "observation.images.camera2")
    return {
        # The saved policy preprocessor renames these to camera1/camera2.
        "observation.images.front_rgb": image_to_tensor(renderer.render(data, front_camera, front_h, front_w)),
        "observation.images.wrist_rgb": image_to_tensor(renderer.render(data, wrist_camera, wrist_h, wrist_w)),
        "observation.state": torch.tensor(state_vector(model, data, gripper_cmd), dtype=torch.float32),
        "task": task,
    }


def restore_viewer_context(viewer: MinimalGLFWViewer | None) -> None:
    if viewer is not None and viewer.window is not None:
        glfw.make_context_current(viewer.window)


def normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected RTC prefix shape [T, A], got {tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


def clip_action(
    action: np.ndarray,
    model: mujoco.MjModel,
    *,
    previous_qpos_target: np.ndarray,
    previous_gripper_target: float,
    smoothing: float,
) -> tuple[np.ndarray, float]:
    raw_qpos = np.asarray(action[:6], dtype=np.float64)
    limits = np.asarray([model.jnt_range[int(model.actuator_trnid[i, 0])] for i in arm_actuator_ids(model)])
    raw_qpos = np.clip(raw_qpos, limits[:, 0], limits[:, 1])
    alpha = float(np.clip(smoothing, 0.0, 1.0))
    target_qpos = previous_qpos_target * alpha + raw_qpos * (1.0 - alpha)

    raw_gripper = float(np.clip(action[6], GRIPPER_OPEN_CMD, GRIPPER_GRASP_CMD))
    gripper_target = float(previous_gripper_target * alpha + raw_gripper * (1.0 - alpha))
    return target_qpos.astype(np.float64), gripper_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--vlm-path",
        type=Path,
        default=DEFAULT_LOCAL_VLM_PATH,
        help="Local SmolVLM2 backbone path. Used when it exists to avoid downloading from Hugging Face.",
    )
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--hz", type=float, default=30.0, help="Target policy inference request frequency.")
    parser.add_argument("--physics-steps-per-policy-step", type=int, default=33, help=argparse.SUPPRESS)
    parser.add_argument("--render-every-physics-steps", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--render-hz", type=float, default=30.0)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--front-camera", default="front")
    parser.add_argument("--wrist-camera", default="wrist")
    parser.add_argument("--task", default=TASK_DESCRIPTION)
    parser.add_argument("--initial-display-state", default=DEFAULT_INITIAL_DISPLAY_STATE)
    parser.add_argument("--viewer-camera-overlays", default="front,wrist")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--disable-grasp-assist", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true", help="Use CUDA without cuDNN convolutions.")
    parser.add_argument("--disable-rtc", action="store_true", help="Fall back to single-action async inference.")
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=[item.value for item in RTCAttentionSchedule],
        default=RTCAttentionSchedule.LINEAR.value,
    )
    parser.add_argument("--rtc-queue-threshold", type=int, default=8)
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=0.05,
        help="EMA smoothing for absolute policy actions. 0 tracks actions directly, 1 freezes the target.",
    )
    parser.add_argument("--settle-steps", type=int, default=80)
    return parser


def policy_worker(
    *,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    request_queue: Queue,
    result_queue: Queue,
    action_queue: ActionQueue | None,
    action_period: float,
    rtc_enabled: bool,
    rtc_execution_horizon: int,
    device: str,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            item = request_queue.get(timeout=0.05)
        except Empty:
            continue
        if item is None:
            break
        seq, obs = item
        start = time.perf_counter()
        try:
            batch = preprocessor(obs)
            batch = move_to_device(batch, device)
            if rtc_enabled:
                if action_queue is None:
                    raise RuntimeError("RTC worker requires an ActionQueue")
                idx_before = action_queue.get_action_index()
                prev_actions = action_queue.get_left_over()
                if prev_actions is not None:
                    prev_actions = normalize_prev_actions_length(prev_actions.to(device), rtc_execution_horizon)
                with torch.no_grad():
                    actions = policy.predict_action_chunk(
                        batch,
                        inference_delay=0,
                        prev_chunk_left_over=prev_actions,
                        execution_horizon=rtc_execution_horizon,
                    )
                original = actions.squeeze(0).detach().clone()
                processed = postprocessor(actions).squeeze(0).detach().cpu()
                infer_time = time.perf_counter() - start
                real_delay = int(math.ceil(infer_time / max(action_period, 1e-6)))
                action_queue.merge(original.cpu(), processed, real_delay, idx_before)
                result_queue.put((seq, None, infer_time, None))
            else:
                with torch.no_grad():
                    action = policy.select_action(batch)
                action = postprocessor(action).detach().cpu().numpy().reshape(-1)
                result_queue.put((seq, action, time.perf_counter() - start, None))
        except Exception as exc:  # noqa: BLE001 - surfaced in the main thread with context.
            result_queue.put((seq, None, time.perf_counter() - start, exc))


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled; CUDA kernels will be used where available.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested {args.device}, but CUDA is unavailable. Falling back to cpu.")
        args.device = "cpu"
    if not args.policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {args.policy_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"MuJoCo scene does not exist: {args.model_path}")

    print(f"Loading policy: {args.policy_path}")
    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.device = args.device
    rtc_enabled = not args.disable_rtc
    if rtc_enabled:
        policy_config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.rtc_execution_horizon,
            max_guidance_weight=args.rtc_max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
        )
        print(
            "RTC enabled: "
            f"horizon={args.rtc_execution_horizon}, "
            f"guidance={args.rtc_max_guidance_weight}, "
            f"schedule={args.rtc_prefix_attention_schedule}"
        )
    if args.vlm_path and args.vlm_path.exists():
        policy_config.vlm_model_name = str(args.vlm_path)
        print(f"Using local VLM backbone: {args.vlm_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=policy_config)
    policy.to(torch.device(args.device))
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": policy_config.vlm_model_name},
            "device_processor": {"device": args.device},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    set_display_state(model, data, parse_seven(args.initial_display_state))
    arm_ids = arm_actuator_ids(model)
    qpos_addrs = arm_qpos_addrs(model)
    dof_addrs = arm_dof_addrs(model)
    left_gripper_id, right_gripper_id = gripper_actuator_ids(model)

    qpos_target = np.asarray(data.qpos[qpos_addrs], dtype=np.float64).copy()
    gripper_cmd = GRIPPER_OPEN_CMD
    data.ctrl[arm_ids] = qpos_target
    set_gripper(data, left_gripper_id, right_gripper_id, gripper_cmd)
    for _ in range(args.settle_steps):
        data.ctrl[arm_ids] = qpos_target
        set_gripper(data, left_gripper_id, right_gripper_id, gripper_cmd)
        mujoco.mj_step(model, data)

    renderer = CameraRenderer(model)
    grasp_assist = GraspAssist()
    viewer = None
    camera_overlays = tuple(item.strip() for item in args.viewer_camera_overlays.split(",") if item.strip())
    if not args.no_viewer:
        viewer = MinimalGLFWViewer(model, data, title="CR3 SmolVLA Rollout")
        viewer.cam.lookat[:] = (-0.16, 0.0, 0.42)
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = -80
        viewer.cam.elevation = -35
        for _ in range(3):
            viewer.poll()
            viewer.render("loading SmolVLA...", camera_overlays)
            time.sleep(0.03)

    request_queue: Queue = Queue(maxsize=1)
    result_queue: Queue = Queue()
    action_queue = ActionQueue(policy_config.rtc_config) if rtc_enabled else None
    stop_event = threading.Event()
    worker = threading.Thread(
        target=policy_worker,
        kwargs={
            "policy": policy,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "request_queue": request_queue,
            "result_queue": result_queue,
            "action_queue": action_queue,
            "action_period": 1.0 / float(args.hz),
            "rtc_enabled": rtc_enabled,
            "rtc_execution_horizon": args.rtc_execution_horizon,
            "device": args.device,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    worker.start()

    print("Running async SmolVLA RTC in MuJoCo. Close the viewer or press Ctrl-C to stop.")
    print(f"task: {args.task}")
    policy_period = 1.0 / float(args.hz)
    render_period = 1.0 / float(args.render_hz)
    physics_dt = float(model.opt.timestep) / max(args.realtime_factor, 1e-6)
    next_policy_time = 0.0
    next_action_time = 0.0
    next_render_time = 0.0
    policy_seq = 0
    completed_policy_steps = 0
    in_flight_seq: int | None = None
    latest_action = np.full(7, np.nan, dtype=np.float64)
    latest_infer_ms = 0.0
    last_policy_status = "warming up"
    try:
        while completed_policy_steps < args.steps:
            if viewer is not None and not viewer.is_running():
                break
            loop_start = time.perf_counter()

            while True:
                try:
                    seq, action, infer_time, error = result_queue.get_nowait()
                except Empty:
                    break
                if error is not None:
                    raise RuntimeError(f"Policy worker failed on request {seq}") from error
                if seq == in_flight_seq:
                    in_flight_seq = None
                latest_infer_ms = infer_time * 1000.0
                if rtc_enabled:
                    queue_size = action_queue.qsize() if action_queue is not None else 0
                    last_policy_status = f"merged RTC chunk {seq} in {latest_infer_ms:.0f}ms q={queue_size}"
                    print(
                        f"chunk seq={seq} infer={latest_infer_ms:.0f}ms queue={queue_size}",
                        flush=True,
                    )
                else:
                    latest_action = np.asarray(action, dtype=np.float64)
                    qpos_target, gripper_cmd = clip_action(
                        latest_action,
                        model,
                        previous_qpos_target=qpos_target,
                        previous_gripper_target=gripper_cmd,
                        smoothing=args.action_smoothing,
                    )
                    completed_policy_steps += 1
                    last_policy_status = f"updated action {seq} in {latest_infer_ms:.0f}ms"
                    print(
                        f"{completed_policy_steps:04d} seq={seq} infer={latest_infer_ms:.0f}ms "
                        f"action={np.round(latest_action, 3).tolist()} "
                        f"target_qpos={np.round(qpos_target, 3).tolist()} gripper={gripper_cmd:.3f}",
                        flush=True,
                    )

            now = time.perf_counter()
            queue_size = action_queue.qsize() if action_queue is not None else 0
            should_request_chunk = (
                now >= next_policy_time
                and in_flight_seq is None
                and request_queue.empty()
                and (not rtc_enabled or queue_size <= args.rtc_queue_threshold)
            )
            if should_request_chunk:
                obs = make_observation(
                    renderer=renderer,
                    model=model,
                    data=data,
                    policy=policy,
                    gripper_cmd=gripper_cmd,
                    task=args.task,
                    front_camera=args.front_camera,
                    wrist_camera=args.wrist_camera,
                )
                restore_viewer_context(viewer)
                policy_seq += 1
                request_queue.put((policy_seq, obs))
                in_flight_seq = policy_seq
                last_policy_status = f"thinking request {policy_seq}"
                next_policy_time = now + policy_period

            now = time.perf_counter()
            if rtc_enabled and action_queue is not None and now >= next_action_time:
                queued_action = action_queue.get()
                if queued_action is not None:
                    latest_action = queued_action.detach().cpu().numpy().reshape(-1)
                    qpos_target, gripper_cmd = clip_action(
                        latest_action,
                        model,
                        previous_qpos_target=qpos_target,
                        previous_gripper_target=gripper_cmd,
                        smoothing=args.action_smoothing,
                    )
                    completed_policy_steps += 1
                    last_policy_status = f"executing RTC action q={action_queue.qsize()}"
                    print(
                        f"{completed_policy_steps:04d} rtc_action "
                        f"action={np.round(latest_action, 3).tolist()} "
                        f"target_qpos={np.round(qpos_target, 3).tolist()} gripper={gripper_cmd:.3f}",
                        flush=True,
                    )
                next_action_time = now + policy_period

            data.ctrl[arm_ids] = qpos_target
            set_gripper(data, left_gripper_id, right_gripper_id, gripper_cmd)
            mujoco.mj_step(model, data)
            if not args.disable_grasp_assist:
                grasp_assist.update(
                    model,
                    data,
                    gripper_cmd=gripper_cmd,
                    gripper_target=gripper_cmd,
                    enabled=True,
                )

            now = time.perf_counter()
            if viewer is not None and now >= next_render_time:
                viewer.poll()
                action_text = "pending" if np.isnan(latest_action).any() else np.round(latest_action, 3).tolist()
                current_qpos = np.asarray(data.qpos[qpos_addrs], dtype=np.float64)
                arm_err = float(np.max(np.abs(qpos_target - current_qpos)))
                overlay = (
                    f"policy: {args.policy_path.name}\n"
                    f"RTC:{'ON' if rtc_enabled else 'OFF'} q:{queue_size} actions:{completed_policy_steps}/{args.steps}\n"
                    f"gripper:{gripper_cmd:.3f} arm_err:{arm_err:.3f}rad\n"
                    f"{last_policy_status}\n"
                    f"target:{np.round(qpos_target, 2).tolist()}\n"
                    f"action:{action_text}"
                )
                viewer.render(overlay, camera_overlays)
                next_render_time = now + render_period

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(physics_dt - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        stop_event.set()
        try:
            request_queue.put_nowait(None)
        except Exception:
            pass
        worker.join(timeout=1.0)
        renderer.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
