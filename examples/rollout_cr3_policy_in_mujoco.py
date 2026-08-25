#!/usr/bin/env python

"""Run a trained CR3 ACT policy in the MuJoCo CR3 scene.

The trained policy outputs 7 values: joint deltas ``dq1..dq6`` in the same
display/real-robot coordinate system used by the Dobot CR3 scripts, plus a
binary gripper-open scalar. The MuJoCo scene has 8 actuators, so the CR3
mapping converts this to ``J1..J6`` plus two main gripper controls.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch

LEROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.envs.my_mujoco import MyMujocoEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from sim.cr3_mujoco.joint_mapping import policy_action_to_mujoco_ctrl, sim_to_display_deg


DEFAULT_POLICY_PATH = (
    LEROBOT_ROOT
    / "outputs/train/cr3_act_review_good_from_scratch_100k_v2/checkpoints/050000/pretrained_model"
)
DEFAULT_MODEL_PATH = LEROBOT_ROOT / "sim/cr3_mujoco/cr3_scene.xml"
DEFAULT_INITIAL_DISPLAY_STATE = "0,-5,-97,-8,91,180,0"


class CameraRenderer:
    def __init__(self, env: MyMujocoEnv):
        self.env = env
        self.renderers: dict[tuple[int, int], object] = {}

    def render(self, camera_name: str, height: int, width: int) -> np.ndarray:
        key = (height, width)
        if key not in self.renderers:
            self.renderers[key] = self.env._mujoco.Renderer(self.env._model, height=height, width=width)
        renderer = self.renderers[key]
        renderer.update_scene(self.env._data, camera=camera_name)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def move_to_device(value, device: torch.device | str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def sim_display_state(env: MyMujocoEnv) -> np.ndarray:
    joints_rad = []
    for idx in range(6):
        joint_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_JOINT, f"J{idx + 1}")
        qpos_addr = int(env._model.jnt_qposadr[joint_id])
        joints_rad.append(float(env._data.qpos[qpos_addr]))
    return sim_to_display_deg(np.rad2deg(np.asarray(joints_rad, dtype=np.float32)))


def parse_seven(value: str) -> np.ndarray:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 7:
        raise ValueError(f"Expected 7 comma-separated values, got {len(values)}: {value}")
    return np.asarray(values, dtype=np.float32)


def set_display_state(env: MyMujocoEnv, display_state: np.ndarray) -> None:
    ctrl = policy_action_to_mujoco_ctrl(
        display_state,
        env._model.actuator_ctrlrange,
        current_ctrl=env._data.ctrl,
    )
    env._data.ctrl[:] = ctrl

    for actuator_id in range(env._model.nu):
        joint_id = int(env._model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_type = int(env._model.jnt_type[joint_id])
        if joint_type != env._mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qpos_addr = int(env._model.jnt_qposadr[joint_id])
        env._data.qpos[qpos_addr] = ctrl[actuator_id]

    env._mujoco.mj_forward(env._model, env._data)


def lock_ctrl_to_qpos(env: MyMujocoEnv, ctrl: np.ndarray) -> None:
    env._data.ctrl[:] = ctrl
    for actuator_id in range(env._model.nu):
        joint_id = int(env._model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_type = int(env._model.jnt_type[joint_id])
        if joint_type != env._mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qpos_addr = int(env._model.jnt_qposadr[joint_id])
        qvel_addr = int(env._model.jnt_dofadr[joint_id])
        env._data.qpos[qpos_addr] = ctrl[actuator_id]
        env._data.qvel[qvel_addr] = 0.0
    env._mujoco.mj_forward(env._model, env._data)


def gripper_position_state(env: MyMujocoEnv) -> float:
    values = []
    for joint_name, actuator_name in [
        ("Left_joint1", "Left_joint1_pos"),
        ("Right_joint", "Right_joint_pos"),
    ]:
        joint_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if joint_id < 0 or actuator_id < 0 or actuator_id >= env._model.nu:
            continue
        qpos_addr = int(env._model.jnt_qposadr[joint_id])
        low, high = env._model.actuator_ctrlrange[actuator_id]
        qpos = float(env._data.qpos[qpos_addr])
        if actuator_name == "Right_joint_pos":
            values.append((high - qpos) / max(high - low, 1e-6))
        else:
            values.append((qpos - low) / max(high - low, 1e-6))
    if not values:
        return 100.0
    return float(np.clip(np.mean(values) * 100.0, 0.0, 100.0))


def dataset_initial_state(dataset_root: Path, episode_index: int, frame_index: int | None = None) -> np.ndarray:
    import pyarrow.parquet as pq

    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset data directory does not exist: {data_dir}")
    parquet_paths = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    columns = ["episode_index", "frame_index", "observation.state"]
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=columns)
        episodes = np.asarray(table["episode_index"])
        matches = np.where(episodes == episode_index)[0]
        if matches.size == 0:
            continue
        frame_indices = np.asarray(table["frame_index"])[matches]
        if frame_index is None:
            first_row = int(matches[np.argmin(frame_indices)])
        else:
            frame_matches = matches[np.where(frame_indices == frame_index)[0]]
            if frame_matches.size == 0:
                continue
            first_row = int(frame_matches[0])
        state = np.asarray(table["observation.state"][first_row].as_py(), dtype=np.float32)
        if state.shape[0] != 7:
            raise ValueError(f"Expected dataset observation.state shape (7,), got {state.shape}.")
        # Dataset gripper state is a 0..100 width; action mapping expects 0..1.
        action_state = state.copy()
        action_state[6] = 1.0 if state[6] >= 50.0 else 0.0
        return action_state

    if frame_index is None:
        raise ValueError(f"Episode {episode_index} was not found in {data_dir}.")
    raise ValueError(f"Episode {episode_index} frame {frame_index} was not found in {data_dir}.")


def print_episode_motion(dataset_root: Path, episode_index: int, top_k: int) -> None:
    import pyarrow.parquet as pq

    rows: list[tuple[int, float, list[float]]] = []
    for parquet_path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "action"])
        episodes = np.asarray(table["episode_index"])
        matches = np.where(episodes == episode_index)[0]
        if matches.size == 0:
            continue
        frame_indices = np.asarray(table["frame_index"])[matches]
        actions = table["action"]
        for row_idx, frame_idx in zip(matches, frame_indices, strict=True):
            action = np.asarray(actions[int(row_idx)].as_py(), dtype=np.float32)
            motion = float(np.linalg.norm(action[:6]))
            rows.append((int(frame_idx), motion, action.tolist()))

    if not rows:
        raise ValueError(f"Episode {episode_index} was not found in {dataset_root / 'data'}.")
    rows.sort(key=lambda item: item[1], reverse=True)
    print(f"Top {min(top_k, len(rows))} action-motion frames for episode {episode_index}:")
    for frame_idx, motion, action in rows[:top_k]:
        print(f"frame={frame_idx:04d} motion={motion:.4f} action={np.round(action, 4).tolist()}")


def feature_shape(policy: ACTPolicy, key: str) -> tuple[int, int]:
    shape = policy.config.input_features[key].shape
    return int(shape[1]), int(shape[2])


def make_observation(
    env: MyMujocoEnv,
    renderer: CameraRenderer,
    policy: ACTPolicy,
    *,
    front_camera: str,
    wrist_camera: str,
    depth_mode: Literal["zeros", "wrist_rgb"],
) -> dict[str, torch.Tensor]:
    front_h, front_w = feature_shape(policy, "observation.images.front_rgb")
    wrist_h, wrist_w = feature_shape(policy, "observation.images.wrist_rgb")
    needs_wrist_depth = "observation.images.wrist_depth" in policy.config.input_features

    front = renderer.render(front_camera, front_h, front_w)
    wrist = renderer.render(wrist_camera, wrist_h, wrist_w)

    state = np.concatenate([sim_display_state(env), [gripper_position_state(env)]]).astype(np.float32)
    observation = {
        "observation.images.front_rgb": image_to_tensor(front),
        "observation.images.wrist_rgb": image_to_tensor(wrist),
        "observation.state": torch.tensor(state, dtype=torch.float32),
    }
    if needs_wrist_depth:
        depth_h, depth_w = feature_shape(policy, "observation.images.wrist_depth")
        if depth_mode == "wrist_rgb":
            wrist_depth = renderer.render(wrist_camera, depth_h, depth_w)
        else:
            wrist_depth = np.zeros((depth_h, depth_w, 3), dtype=np.uint8)
        observation["observation.images.wrist_depth"] = image_to_tensor(wrist_depth)
    return observation


def clamp_policy_action(action: np.ndarray, max_joint_delta_deg: float) -> np.ndarray:
    limited = action.copy()
    limited[:6] = np.clip(limited[:6], -max_joint_delta_deg, max_joint_delta_deg)
    limited[6] = 1.0 if limited[6] >= 0.5 else 0.0
    return limited


def action_to_absolute_target(current_state: np.ndarray, action: np.ndarray, mode: str) -> np.ndarray:
    target = np.zeros(7, dtype=np.float32)
    if mode == "delta":
        target[:6] = current_state[:6] + action[:6]
    elif mode == "absolute":
        target[:6] = action[:6]
    else:
        raise ValueError(f"Unsupported action mode: {mode}")
    target[6] = float(np.clip(action[6], 0.0, 1.0))
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Disable cuDNN. Useful when CUDA works but conv2d fails with CUDNN_STATUS_NOT_INITIALIZED.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--action-mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--max-joint-delta-deg", type=float, default=3.0)
    parser.add_argument("--no-action-limit", action="store_true")
    parser.add_argument(
        "--lock-sim-to-target",
        action="store_true",
        help="Kinematically lock MuJoCo qpos to policy targets; useful for visual policy checks without gravity sag.",
    )
    parser.add_argument("--front-camera", default="front")
    parser.add_argument("--wrist-camera", default="wrist")
    parser.add_argument("--depth-mode", choices=["zeros", "wrist_rgb"], default="wrist_rgb")
    parser.add_argument("--viewer-camera", default=None, help="Optional fixed viewer camera, e.g. front or wrist.")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--hold-initial-s", type=float, default=2.0, help="Seconds to show the initial pose before rollout.")
    parser.add_argument(
        "--initial-display-state",
        default=DEFAULT_INITIAL_DISPLAY_STATE,
        help="Initial [J1..J6, gripper] in policy/display coordinates.",
    )
    parser.add_argument(
        "--initial-from-dataset-episode",
        type=int,
        default=None,
        help="Initialize from the first observation.state of a local training dataset episode.",
    )
    parser.add_argument(
        "--initial-from-dataset-frame",
        type=int,
        default=None,
        help="Frame index to use with --initial-from-dataset-episode. Defaults to the first frame.",
    )
    parser.add_argument(
        "--print-episode-motion",
        action="store_true",
        help="Print frames with largest dataset action magnitude, then exit.",
    )
    parser.add_argument("--top-k-motion-frames", type=int, default=10)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=LEROBOT_ROOT / "lerobot_data/local/dobot_cr3_drag_review_good",
    )
    parser.add_argument(
        "--use-xml-initial-state",
        action="store_true",
        help="Do not remap the XML reset pose into policy/display coordinates before rollout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled; CUDA may still be used, but convolution will use non-cuDNN kernels.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{args.device}' is not available; falling back to cpu.")
        args.device = "cpu"
    if args.print_episode_motion:
        if args.initial_from_dataset_episode is None:
            raise ValueError("--initial-from-dataset-episode is required with --print-episode-motion")
        print_episode_motion(args.dataset_root, args.initial_from_dataset_episode, args.top_k_motion_frames)
        return
    policy = ACTPolicy.from_pretrained(args.policy_path, device=args.device)
    policy.config.device = args.device
    policy.to(torch.device(args.device))
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    env = MyMujocoEnv(
        model_path=str(args.model_path),
        obs_type="state",
        action_dim=8,
        state_dim=8,
        frame_skip=args.frame_skip,
    )
    env.reset()
    if not args.use_xml_initial_state:
        if args.initial_from_dataset_episode is None:
            initial_display_state = parse_seven(args.initial_display_state)
        else:
            initial_display_state = dataset_initial_state(
                args.dataset_root,
                args.initial_from_dataset_episode,
                args.initial_from_dataset_frame,
            )
            print(
                f"Using dataset episode {args.initial_from_dataset_episode}"
                f"{'' if args.initial_from_dataset_frame is None else f' frame {args.initial_from_dataset_frame}'}"
                " initial state: "
                f"{np.round(initial_display_state, 3).tolist()}"
            )
        set_display_state(env, initial_display_state)
    renderer = CameraRenderer(env)
    viewer_context = None
    viewer = None

    try:
        if not args.no_viewer:
            import mujoco.viewer

            viewer_context = mujoco.viewer.launch_passive(env._model, env._data)
            viewer = viewer_context.__enter__()
            if args.viewer_camera is not None:
                camera_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_CAMERA, args.viewer_camera)
                if camera_id < 0:
                    raise ValueError(f"Camera not found: {args.viewer_camera}")
                viewer.cam.type = env._mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            else:
                viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
                viewer.cam.distance = 1.9
                viewer.cam.azimuth = -90
                viewer.cam.elevation = -55

        initial_state = np.concatenate([sim_display_state(env), [gripper_position_state(env)]]).astype(np.float32)
        print(f"Initial display state: {np.round(initial_state, 3).tolist()}")
        if viewer is not None and args.hold_initial_s > 0:
            hold_until = time.perf_counter() + args.hold_initial_s
            while time.perf_counter() < hold_until and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

        period = 1.0 / args.hz
        print(f"Loaded policy: {args.policy_path}")
        print("Running CR3 policy in MuJoCo. Ctrl-C to stop.")
        for step in range(args.steps):
            start = time.perf_counter()
            raw_obs = make_observation(
                env,
                renderer,
                policy,
                front_camera=args.front_camera,
                wrist_camera=args.wrist_camera,
                depth_mode=args.depth_mode,
            )
            batch = preprocessor(raw_obs)
            batch = move_to_device(batch, args.device)
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action).detach().cpu().numpy().reshape(-1)
            limited_action = action.copy() if args.no_action_limit else clamp_policy_action(
                action, args.max_joint_delta_deg
            )

            current = raw_obs["observation.state"].numpy()
            target = action_to_absolute_target(current, limited_action, args.action_mode)
            ctrl = policy_action_to_mujoco_ctrl(target, env._model.actuator_ctrlrange, current_ctrl=env._data.ctrl)
            if args.lock_sim_to_target:
                lock_ctrl_to_qpos(env, ctrl)
            else:
                env._data.ctrl[:] = ctrl
                for _ in range(env.frame_skip):
                    env._mujoco.mj_step(env._model, env._data)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            print(
                f"{step:04d} state={np.round(current, 2).tolist()} "
                f"action={np.round(action, 2).tolist()} target={np.round(target, 2).tolist()}"
            )
            elapsed = time.perf_counter() - start
            time.sleep(max(period - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        renderer.close()
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)
        env.close()


if __name__ == "__main__":
    main()
