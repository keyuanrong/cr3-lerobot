#!/usr/bin/env python

"""Keyboard test for the CR3 MuJoCo gripper.

Keys in the MuJoCo viewer:
  o: open gripper
  p: close gripper
  k: print current gripper joint positions and controls
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

LEROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEROBOT_ROOT / "src"))
sys.path.insert(0, str(LEROBOT_ROOT))

from lerobot.envs.my_mujoco import MyMujocoEnv
from sim.cr3_mujoco.joint_mapping import GRIPPER_CLOSED, GRIPPER_OPEN, gripper_to_finger_ctrl


DEFAULT_MODEL_PATH = LEROBOT_ROOT / "sim/cr3_mujoco/cr3_scene.xml"
GRIPPER_JOINTS = (
    "Left_joint1",
    "Left_joint2",
    "Left_static1_joint",
    "Right_joint",
    "Right_joint2",
    "Right_static1_joint",
)
GRIPPER_ACTUATORS = ("Left_joint1_pos", "Right_joint_pos")


def actuator_id(env: MyMujocoEnv, name: str) -> int:
    actuator = env._mujoco.mjtObj.mjOBJ_ACTUATOR
    index = env._mujoco.mj_name2id(env._model, actuator, name)
    if index < 0:
        raise ValueError(f"Actuator not found: {name}")
    return int(index)


def joint_qpos_addr(env: MyMujocoEnv, name: str) -> int:
    joint = env._mujoco.mjtObj.mjOBJ_JOINT
    index = env._mujoco.mj_name2id(env._model, joint, name)
    if index < 0:
        raise ValueError(f"Joint not found: {name}")
    return int(env._model.jnt_qposadr[index])


def set_home_arm_ctrl(env: MyMujocoEnv) -> None:
    for i in range(min(6, env._model.nu)):
        joint_id = int(env._model.actuator_trnid[i, 0])
        if joint_id < 0:
            continue
        qpos_addr = int(env._model.jnt_qposadr[joint_id])
        env._data.ctrl[i] = env._data.qpos[qpos_addr]


def set_gripper(env: MyMujocoEnv, actuator_ids: list[int], open_fraction: float) -> None:
    targets = gripper_to_finger_ctrl(float(open_fraction), env._model.actuator_ctrlrange)
    for actuator_id, target in zip(actuator_ids, targets, strict=True):
        low, high = env._model.actuator_ctrlrange[actuator_id]
        env._data.ctrl[actuator_id] = float(np.clip(target, low, high))


def move_toward(current: float, target: float, max_delta: float) -> float:
    if abs(target - current) <= max_delta:
        return target
    return current + max_delta if target > current else current - max_delta


def print_gripper_state(env: MyMujocoEnv, actuator_ids: list[int], qpos_addrs: list[int]) -> None:
    qpos = ", ".join(f"{env._data.qpos[addr]:.3f}" for addr in qpos_addrs)
    ctrl = ", ".join(f"{env._data.ctrl[actuator_id]:.3f}" for actuator_id in actuator_ids)
    print(f"gripper qpos=({qpos}) ctrl=({ctrl})", flush=True)


def key_matches(key: int, letter: str) -> bool:
    return key in {ord(letter.lower()), ord(letter.upper())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--open", type=float, default=1.0, help="Open fraction; 1.0 maps to max gripper opening.")
    parser.add_argument("--close", type=float, default=0.0, help="Closed fraction; 0.0 fully closes the gripper.")
    parser.add_argument("--speed", type=float, default=1.0, help="Fallback gripper open-fraction speed per second.")
    parser.add_argument("--open-speed", type=float, default=0.5, help="Opening speed in open-fraction per second.")
    parser.add_argument("--close-speed", type=float, default=1.0, help="Closing speed in open-fraction per second.")
    parser.add_argument("--start-open", action="store_true", help="Start with the gripper open.")
    parser.add_argument("--auto", action="store_true", help="Automatically toggle open/close once per second.")
    parser.add_argument("--viewer-camera", default=None, help="Optional fixed camera name, e.g. wrist or front.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import mujoco.viewer

    env = MyMujocoEnv(
        model_path=str(args.model_path),
        obs_type="state",
        action_dim=8,
        state_dim=10,
        render_mode="rgb_array",
    )
    env.reset()

    actuator_ids = [actuator_id(env, name) for name in GRIPPER_ACTUATORS]
    qpos_addrs = [joint_qpos_addr(env, name) for name in GRIPPER_JOINTS]

    set_home_arm_ctrl(env)
    gripper_value = float(np.clip(args.open if args.start_open else args.close, 0.0, 1.0))
    gripper_target = gripper_value
    set_gripper(env, actuator_ids, gripper_value)

    print("CR3 gripper keyboard test")
    print("Keys: o=open, p=close, k=print state. Close the viewer window to stop.")
    print(f"Gripper actuators: {', '.join(GRIPPER_ACTUATORS)}")
    print(
        f"open=1.00 -> +/-{GRIPPER_OPEN:.2f} rad actuator targets; "
        f"close=0.00 -> +/-{GRIPPER_CLOSED:.2f} rad actuator targets"
    )
    print_gripper_state(env, actuator_ids, qpos_addrs)

    auto_open = args.start_open
    last_auto_toggle = time.perf_counter()

    def key_callback(key: int) -> None:
        nonlocal gripper_target
        if key_matches(key, "o"):
            gripper_target = args.open
            print("open")
            print_gripper_state(env, actuator_ids, qpos_addrs)
        elif key_matches(key, "p"):
            gripper_target = args.close
            print("close")
            print_gripper_state(env, actuator_ids, qpos_addrs)
        elif key_matches(key, "k"):
            print_gripper_state(env, actuator_ids, qpos_addrs)
        else:
            print(f"unhandled key code: {key}", flush=True)

    frame_dt = 1.0 / max(args.fps, 1e-6)
    try:
        with mujoco.viewer.launch_passive(env._model, env._data, key_callback=key_callback) as viewer:
            if args.viewer_camera is not None:
                camera_id = env._mujoco.mj_name2id(
                    env._model,
                    env._mujoco.mjtObj.mjOBJ_CAMERA,
                    args.viewer_camera,
                )
                if camera_id < 0:
                    raise ValueError(f"Camera not found: {args.viewer_camera}")
                viewer.cam.type = env._mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            else:
                viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = -90
                viewer.cam.elevation = -45

            while viewer.is_running():
                start = time.perf_counter()
                if args.auto and start - last_auto_toggle >= 1.0:
                    auto_open = not auto_open
                    gripper_target = args.open if auto_open else args.close
                    print("auto open" if auto_open else "auto close")
                    print_gripper_state(env, actuator_ids, qpos_addrs)
                    last_auto_toggle = start
                if gripper_target > gripper_value:
                    step_speed = args.open_speed
                elif gripper_target < gripper_value:
                    step_speed = args.close_speed
                else:
                    step_speed = args.speed
                gripper_value = move_toward(gripper_value, gripper_target, step_speed * frame_dt)
                set_gripper(env, actuator_ids, gripper_value)
                env._mujoco.mj_step(env._model, env._data)
                viewer.sync()
                time.sleep(max(frame_dt - (time.perf_counter() - start), 0.0))
    finally:
        env.close()


if __name__ == "__main__":
    main()
