#!/usr/bin/env python

"""Replay a recorded CR3 dataset episode in the MuJoCo scene."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

LEROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.envs.my_mujoco import MyMujocoEnv
from sim.cr3_mujoco.joint_mapping import gripper_to_finger_ctrl, policy_action_to_mujoco_ctrl


DEFAULT_DATASET_ROOT = LEROBOT_ROOT / "lerobot_data/local/dobot_cr3_drag_review_good"
DEFAULT_MODEL_PATH = LEROBOT_ROOT / "sim/cr3_mujoco/cr3_scene.xml"


def episode_states(dataset_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray | None]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    rows: list[tuple[int, float | None, list[float]]] = []
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        episodes = np.asarray(table["episode_index"])
        matches = np.where(episodes == episode_index)[0]
        if matches.size == 0:
            continue

        has_timestamp = "timestamp" in table.column_names
        frame_indices = np.asarray(table["frame_index"])[matches]
        timestamps = np.asarray(table["timestamp"])[matches] if has_timestamp else [None] * len(matches)
        states = table["observation.state"]
        for row_idx, frame_idx, timestamp in zip(matches, frame_indices, timestamps, strict=True):
            rows.append((int(frame_idx), None if timestamp is None else float(timestamp), states[int(row_idx)].as_py()))

    if not rows:
        raise ValueError(f"Episode {episode_index} not found in {dataset_root}")

    rows.sort(key=lambda item: item[0])
    states_np = np.asarray([row[2] for row in rows], dtype=np.float32)
    timestamps_list = [row[1] for row in rows]
    timestamps_np = None if any(ts is None for ts in timestamps_list) else np.asarray(timestamps_list, dtype=np.float32)
    return states_np, timestamps_np


def parse_three(value: str) -> np.ndarray:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got {len(values)}: {value}")
    return np.asarray(values, dtype=np.float64)


def set_cube_pos(env: MyMujocoEnv, pos: np.ndarray) -> None:
    joint_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    if joint_id < 0:
        raise ValueError("Joint not found: cube_free")
    qpos_addr = int(env._model.jnt_qposadr[joint_id])
    qvel_addr = int(env._model.jnt_dofadr[joint_id])
    env._data.qpos[qpos_addr : qpos_addr + 3] = pos
    env._data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    env._data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    env._mujoco.mj_forward(env._model, env._data)


def set_display_state(env: MyMujocoEnv, state: np.ndarray, *, use_mapping: bool) -> np.ndarray:
    action_state = state.copy()
    action_state[6] = 1.0 if state[6] >= 50.0 else 0.0
    if use_mapping:
        ctrl = policy_action_to_mujoco_ctrl(action_state, env._model.actuator_ctrlrange, current_ctrl=env._data.ctrl)
    else:
        ctrl = env._data.ctrl.copy()
        ctrl[:6] = np.deg2rad(state[:6])
        ctrl[6:8] = gripper_to_finger_ctrl(float(action_state[6]), env._model.actuator_ctrlrange)
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
    return ctrl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--viewer-camera", default=None, help="Optional fixed camera, e.g. front or wrist.")
    parser.add_argument("--no-mapping", action="store_true", help="Write dataset J1-J6 directly into MuJoCo.")
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS. Defaults to dataset timestamps or 30.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop the episode until the viewer closes.")
    parser.add_argument(
        "--cube-pos",
        default=None,
        help="Optional cube world position x,y,z in meters, e.g. --cube-pos=-0.35,-0.13,0.3225.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    states, timestamps = episode_states(args.dataset_root, args.episode)
    print(f"Loaded episode {args.episode}: {len(states)} frames")
    print("First observation.state:")
    print(np.round(states[0], 3).tolist())
    print("Mapping:", "off" if args.no_mapping else "on")

    env = MyMujocoEnv(model_path=str(args.model_path), obs_type="state", action_dim=8, state_dim=8)
    env.reset()
    cube_pos = parse_three(args.cube_pos) if args.cube_pos is not None else None
    if cube_pos is not None:
        set_cube_pos(env, cube_pos)

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
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

        frame = 0
        while viewer.is_running():
            set_display_state(env, states[frame], use_mapping=not args.no_mapping)
            if cube_pos is not None:
                set_cube_pos(env, cube_pos)
            viewer.sync()

            if frame + 1 < len(states):
                if args.fps is not None:
                    dt = 1.0 / args.fps
                elif timestamps is not None:
                    dt = max(float(timestamps[frame + 1] - timestamps[frame]), 0.0)
                else:
                    dt = 1.0 / 30.0
                frame += 1
            elif args.loop:
                dt = 1.0 / (args.fps or 30.0)
                frame = 0
            else:
                print("Replay finished. Close the viewer to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                break

            time.sleep(dt / max(args.speed, 1e-6))

    env.close()


if __name__ == "__main__":
    main()
