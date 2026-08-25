#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open a MuJoCo viewer for the demo environment and optionally replay dataset actions.

Examples:

    # View your generated CR3 scene when sim/cr3_mujoco/cr3_scene.xml exists.
    uv run python examples/view_my_mujoco.py

    # View the built-in placeholder demo scene instead.
    uv run python examples/view_my_mujoco.py --demo-scene

    # Watch the arm move with a simple scripted signal.
    uv run python examples/view_my_mujoco.py --mode sine

    # Replay one LeRobot dataset episode into the MuJoCo actuators.
    uv run python examples/view_my_mujoco.py \
        --mode dataset \
        --dataset.repo-id <user>/<dataset_name> \
        --dataset.root /path/to/local/dataset \
        --dataset.episode 0

    # Replay actions that are already actuator control targets.
    uv run python examples/view_my_mujoco.py \
        --mode dataset \
        --dataset.repo-id <user>/<dataset_name> \
        --dataset.root /path/to/local/dataset \
        --dataset.action-mode absolute_ctrl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

LEROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.envs.my_mujoco import MyMujocoEnv
from lerobot.utils.constants import ACTION
from sim.cr3_mujoco.joint_mapping import gripper_to_finger_ctrl, policy_action_to_mujoco_ctrl

DEFAULT_CR3_URDF = Path("sim/cr3_mujoco/urdf/cr3_mujoco.urdf")
DEFAULT_CR3_SCENE = Path("sim/cr3_mujoco/cr3_scene.xml")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _load_dataset_actions(args: argparse.Namespace) -> tuple[list[np.ndarray], int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=[args.dataset_episode],
        download_videos=False,
    )
    actions_ds = dataset.select_columns(ACTION)
    actions = [_as_numpy(actions_ds[idx][ACTION]).reshape(-1) for idx in range(dataset.num_frames)]
    fps = args.fps if args.fps is not None else dataset.fps
    print(
        f"Loaded {len(actions)} action frames from episode {args.dataset_episode} "
        f"of {args.dataset_repo_id} at {fps} fps."
    )
    return actions, fps


def _clip_ctrl(env: MyMujocoEnv, ctrl: np.ndarray) -> np.ndarray:
    limited = np.asarray(env._model.actuator_ctrllimited, dtype=bool)
    if limited.any():
        low = env._model.actuator_ctrlrange[:, 0]
        high = env._model.actuator_ctrlrange[:, 1]
        ctrl = ctrl.copy()
        ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])
    return ctrl


def _apply_ctrl(env: MyMujocoEnv, ctrl: np.ndarray) -> None:
    if env.action_dim == 0:
        for _ in range(env.frame_skip):
            env._mujoco.mj_step(env._model, env._data)
        return
    if ctrl.shape != (env.action_dim,):
        raise ValueError(f"Expected action shape {(env.action_dim,)}, got {ctrl.shape}.")
    env._data.ctrl[:] = _clip_ctrl(env, ctrl)
    for _ in range(env.frame_skip):
        env._mujoco.mj_step(env._model, env._data)


def _action_to_ctrl(env: MyMujocoEnv, action: np.ndarray, action_mode: str) -> np.ndarray:
    if action_mode == "cr3_policy":
        return policy_action_to_mujoco_ctrl(
            action,
            env._model.actuator_ctrlrange,
            current_ctrl=np.asarray(env._data.ctrl, dtype=np.float32),
        )

    if action_mode == "cr3_recorded":
        if action.shape[0] != 7 or env.action_dim < 8:
            raise ValueError(
                "cr3_recorded expects dataset action shape (7,) and a sim with at least 8 actuators "
                "(J1-J6, Left_joint1, Right_joint)."
            )
        ctrl = np.asarray(env._data.ctrl, dtype=np.float32).copy()
        ctrl[:6] = ctrl[:6] + action[:6] * env.action_scale
        gripper_binary = float(np.clip(action[6], 0.0, 1.0))
        ctrl[6:8] = gripper_to_finger_ctrl(gripper_binary, env._model.actuator_ctrlrange)
        return ctrl

    if action.shape[0] < env.action_dim:
        padded = np.zeros(env.action_dim, dtype=np.float32)
        padded[: action.shape[0]] = action
        action = padded
    elif action.shape[0] > env.action_dim:
        action = action[: env.action_dim]

    if action_mode == "normalized":
        return env._home_ctrl + np.clip(action, -1.0, 1.0) * env.action_scale
    if action_mode == "absolute_ctrl":
        return action
    if action_mode == "delta_ctrl":
        return np.asarray(env._data.ctrl, dtype=np.float32) + action * env.action_scale
    raise ValueError(f"Unsupported action mode: {action_mode}")


def _scripted_action(step: int, action_dim: int, mode: str) -> np.ndarray:
    if action_dim == 0:
        return np.zeros(0, dtype=np.float32)
    if mode == "idle":
        return np.zeros(action_dim, dtype=np.float32)
    if mode == "random":
        return np.random.uniform(-0.5, 0.5, size=action_dim).astype(np.float32)
    if mode == "sine":
        phase = step * 0.035
        offsets = np.arange(action_dim, dtype=np.float32) * 0.7
        return (0.45 * np.sin(phase + offsets)).astype(np.float32)
    raise ValueError(f"Unsupported scripted mode: {mode}")


def _print_body_pose(env: MyMujocoEnv, body_name: str) -> None:
    body_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        print(f"Body not found: {body_name}")
        return
    pos = np.asarray(env._data.xpos[body_id], dtype=np.float64)
    quat = np.asarray(env._data.xquat[body_id], dtype=np.float64)
    print(
        f"{body_name} world pos={np.array2string(pos, precision=6, suppress_small=True)} "
        f"quat={np.array2string(quat, precision=6, suppress_small=True)}"
    )
    print(f'XML body pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"')


def _infer_action_dim(model_path: str | None, demo_scene: bool) -> int:
    if model_path is None:
        return 7

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as exc:
        if demo_scene:
            return 7
        raise RuntimeError(f"Could not infer action_dim from {model_path}. Pass --action-dim explicitly.") from exc
    return int(model.nu)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["manual", "idle", "sine", "random", "dataset"],
        default="manual",
        help=(
            "manual leaves MuJoCo controls untouched so you can move joints from the viewer UI; "
            "idle/sine/random/dataset drive actuators from this script."
        ),
    )
    parser.add_argument("--model-path", type=str, default=None, help="Optional MuJoCo XML/MJCF/URDF path.")
    parser.add_argument(
        "--demo-scene",
        action="store_true",
        help="Use the built-in placeholder scene instead of auto-loading sim/cr3_mujoco/urdf/cr3_mujoco.urdf.",
    )
    parser.add_argument("--camera-name", type=str, default="front", help="Camera used by the env render path.")
    parser.add_argument(
        "--viewer-camera",
        type=str,
        default=None,
        help="Open the MuJoCo viewer directly from a fixed camera, e.g. front or wrist.",
    )
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=7)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=None, help="Playback fps. Defaults to 30 or dataset fps.")
    parser.add_argument("--loop", action="store_true", help="Loop scripted motion or dataset replay.")
    parser.add_argument("--print-every", type=int, default=30, help="Print a small status line every N frames.")
    parser.add_argument(
        "--print-cube-every",
        type=int,
        default=0,
        help="Print the red cube world position every N frames. Disabled by default.",
    )
    parser.add_argument(
        "--view-lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.35),
        metavar=("X", "Y", "Z"),
        help="Free-view camera look-at point. Defaults to the table center.",
    )
    parser.add_argument("--view-distance", type=float, default=1.9, help="Free-view camera distance.")
    parser.add_argument("--view-azimuth", type=float, default=-90.0, help="Free-view camera azimuth.")
    parser.add_argument("--view-elevation", type=float, default=-55.0, help="Free-view camera elevation.")

    parser.add_argument("--dataset.repo-id", dest="dataset_repo_id", type=str, default=None)
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, default=None)
    parser.add_argument("--dataset.episode", dest="dataset_episode", type=int, default=0)
    parser.add_argument(
        "--dataset.action-mode",
        dest="dataset_action_mode",
        choices=["normalized", "absolute_ctrl", "delta_ctrl", "cr3_recorded", "cr3_policy"],
        default="cr3_recorded",
        help=(
            "How to interpret dataset actions: normalized means LeRobot-style [-1, 1] actions; "
            "absolute_ctrl means direct MuJoCo actuator controls; delta_ctrl means increments from current ctrl; "
            "cr3_recorded maps [dq1..dq6, gripper_binary] onto J1-J6 plus closed-loop gripper open/close; "
            "cr3_policy maps [J1..J6 in display degrees, gripper_binary] onto calibrated CR3 MuJoCo ctrl."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "dataset" and args.dataset_repo_id is None:
        raise ValueError("--dataset.repo-id is required when --mode=dataset")

    import mujoco.viewer

    model_path = args.model_path
    if model_path is None and not args.demo_scene and DEFAULT_CR3_SCENE.exists():
        model_path = str(DEFAULT_CR3_SCENE)
        print(f"Auto-loading CR3 MuJoCo scene: {model_path}")
    elif model_path is None and not args.demo_scene and DEFAULT_CR3_URDF.exists():
        model_path = str(DEFAULT_CR3_URDF)
        print(f"Auto-loading CR3 URDF view-only model: {model_path}")
    elif model_path is None:
        print("Using built-in placeholder demo scene. Pass --model-path to load your robot.")

    action_dim = args.action_dim
    if action_dim is None:
        action_dim = _infer_action_dim(model_path, args.demo_scene)
        if action_dim == 0:
            print("No --action-dim provided for an external model; using action_dim=0 for view-only mode.")

    env = MyMujocoEnv(
        model_path=model_path,
        obs_type="state",
        camera_name=args.camera_name,
        action_dim=action_dim,
        state_dim=args.state_dim,
        frame_skip=args.frame_skip,
        action_scale=args.action_scale,
    )
    env.reset()

    dataset_actions: list[np.ndarray] = []
    fps = args.fps if args.fps is not None else 30
    if args.mode == "dataset":
        dataset_actions, fps = _load_dataset_actions(args)

    frame_dt = 1.0 / fps
    frame_idx = 0
    print("MuJoCo viewer is running. Close the viewer window to stop.")
    print("Press p in the viewer to print the red cube world position.")
    print(f"mode={args.mode}, fps={fps}, action_dim={action_dim}, model_path={model_path}")

    def key_callback(key: int) -> None:
        if key == ord("p"):
            _print_body_pose(env, "cube")

    try:
        with mujoco.viewer.launch_passive(env._model, env._data, key_callback=key_callback) as viewer:
            if args.viewer_camera is not None:
                camera_id = env._mujoco.mj_name2id(
                    env._model,
                    env._mujoco.mjtObj.mjOBJ_CAMERA,
                    args.viewer_camera,
                )
                if camera_id < 0:
                    camera_names = [
                        env._mujoco.mj_id2name(env._model, env._mujoco.mjtObj.mjOBJ_CAMERA, i)
                        for i in range(env._model.ncam)
                    ]
                    raise ValueError(f"Camera '{args.viewer_camera}' not found. Available cameras: {camera_names}")
                viewer.cam.type = env._mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            else:
                viewer.cam.lookat[:] = args.view_lookat
                viewer.cam.distance = args.view_distance
                viewer.cam.azimuth = args.view_azimuth
                viewer.cam.elevation = args.view_elevation
            while viewer.is_running():
                start_t = time.perf_counter()

                if args.mode == "manual":
                    viewer.sync()
                    for _ in range(env.frame_skip):
                        env._mujoco.mj_step(env._model, env._data)
                    viewer.sync()
                    if args.print_cube_every > 0 and frame_idx % args.print_cube_every == 0:
                        _print_body_pose(env, "cube")
                    frame_idx += 1
                    elapsed = time.perf_counter() - start_t
                    time.sleep(max(frame_dt - elapsed, 0.0))
                    continue

                if args.mode == "dataset":
                    if frame_idx >= len(dataset_actions):
                        if not args.loop:
                            break
                        frame_idx = 0
                        env.reset()
                    action = dataset_actions[frame_idx]
                    ctrl = _action_to_ctrl(env, action, args.dataset_action_mode)
                else:
                    action = _scripted_action(frame_idx, action_dim, args.mode)
                    ctrl = _action_to_ctrl(env, action, "normalized")

                _apply_ctrl(env, ctrl)
                viewer.sync()

                if args.print_every > 0 and frame_idx % args.print_every == 0:
                    print(f"frame={frame_idx} ctrl={np.array2string(ctrl, precision=3, suppress_small=True)}")
                if args.print_cube_every > 0 and frame_idx % args.print_cube_every == 0:
                    _print_body_pose(env, "cube")

                frame_idx += 1
                elapsed = time.perf_counter() - start_t
                time.sleep(max(frame_dt - elapsed, 0.0))
    finally:
        env.close()


if __name__ == "__main__":
    main()
