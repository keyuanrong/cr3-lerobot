#!/usr/bin/env python

"""Send the same CR3 joint-angle command to MuJoCo and, optionally, the real robot.

By default this script controls only the MuJoCo simulation. It will not move the
real robot unless --enable-real is passed.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

LEROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.envs.my_mujoco import MyMujocoEnv
from sim.cr3_mujoco.joint_mapping import (
    DISPLAY_TO_SIM_JOINT_OFFSET_DEG,
    DISPLAY_TO_SIM_JOINT_SIGN,
    display_to_sim_deg,
    sim_to_display_deg,
)


DEFAULT_MODEL_PATH = Path("sim/cr3_mujoco/cr3_scene.xml")


def parse_six_floats(value: str, name: str) -> np.ndarray:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError(f"{name} must contain exactly 6 comma-separated values, got {len(values)}: {value}")
    return np.asarray(values, dtype=np.float32)


def sim_joint_qpos(env: MyMujocoEnv) -> np.ndarray:
    qpos = np.zeros(6, dtype=np.float32)
    for idx in range(6):
        joint_id = env._mujoco.mj_name2id(env._model, env._mujoco.mjtObj.mjOBJ_JOINT, f"J{idx + 1}")
        if joint_id < 0:
            raise ValueError(f"Joint J{idx + 1} not found in sim model")
        qpos_addr = int(env._model.jnt_qposadr[joint_id])
        qpos[idx] = env._data.qpos[qpos_addr]
    return qpos


def set_sim_joint_targets(env: MyMujocoEnv, target_rad: np.ndarray) -> None:
    if env._model.nu < 6:
        raise ValueError(f"Expected at least 6 sim actuators, got {env._model.nu}")
    ctrl = env._data.ctrl.copy()
    ctrl[:6] = target_rad
    limited = np.asarray(env._model.actuator_ctrllimited, dtype=bool)
    if limited.any():
        low = env._model.actuator_ctrlrange[:, 0]
        high = env._model.actuator_ctrlrange[:, 1]
        ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])
    env._data.ctrl[:] = ctrl


def interpolate(current: np.ndarray, target: np.ndarray, max_delta: float) -> np.ndarray:
    delta = np.clip(target - current, -max_delta, max_delta)
    return current + delta


def run_sim(
    target_deg: np.ndarray,
    *,
    model_path: Path,
    joint_sign: np.ndarray,
    joint_offset_deg: np.ndarray,
    duration_s: float,
    fps: int,
    max_step_deg: float,
    viewer_camera: str | None,
) -> np.ndarray:
    import mujoco.viewer

    env = MyMujocoEnv(model_path=str(model_path), obs_type="state", action_dim=8, state_dim=8)
    env.reset()

    internal_target_deg = display_to_sim_deg(target_deg, joint_sign, joint_offset_deg)
    target_rad = np.deg2rad(internal_target_deg)
    max_step_rad = math.radians(max_step_deg)
    dt = 1.0 / fps
    steps = max(1, int(duration_s * fps))

    print(f"Sim target deg: {np.array2string(target_deg, precision=3, suppress_small=True)}")
    print(f"Sim internal target deg after mapping: {np.array2string(internal_target_deg, precision=3, suppress_small=True)}")
    print(f"Sim internal target rad after mapping: {np.array2string(target_rad, precision=3, suppress_small=True)}")

    try:
        with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
            if viewer_camera is not None:
                camera_id = env._mujoco.mj_name2id(
                    env._model,
                    env._mujoco.mjtObj.mjOBJ_CAMERA,
                    viewer_camera,
                )
                if camera_id < 0:
                    raise ValueError(f"Camera not found: {viewer_camera}")
                viewer.cam.type = env._mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            else:
                viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
                viewer.cam.distance = 1.9
                viewer.cam.azimuth = -90
                viewer.cam.elevation = -55

            ctrl = env._data.ctrl.copy()
            ctrl[:6] = sim_joint_qpos(env)
            env._data.ctrl[:] = ctrl

            for step in range(steps):
                start = time.perf_counter()
                next_target = interpolate(env._data.ctrl[:6], target_rad, max_step_rad)
                set_sim_joint_targets(env, next_target)
                for _ in range(env.frame_skip):
                    env._mujoco.mj_step(env._model, env._data)
                viewer.sync()
                time.sleep(max(dt - (time.perf_counter() - start), 0.0))
                if not viewer.is_running():
                    break

            print("Holding final target. Close the MuJoCo viewer to finish.")
            while viewer.is_running():
                start = time.perf_counter()
                set_sim_joint_targets(env, target_rad)
                for _ in range(env.frame_skip):
                    env._mujoco.mj_step(env._model, env._data)
                viewer.sync()
                time.sleep(max(dt - (time.perf_counter() - start), 0.0))

        final_internal_deg = np.rad2deg(sim_joint_qpos(env))
        return sim_to_display_deg(final_internal_deg, joint_sign, joint_offset_deg)
    finally:
        env.close()


def run_real_robot(target_deg: np.ndarray, *, robot_ip: str, speed_factor: int, use_gripper: bool) -> np.ndarray:
    from lerobot.robots.dobot_cr3 import DobotCR3, DobotCR3Config

    cfg = DobotCR3Config(robot_ip=robot_ip, speed_factor=speed_factor, use_gripper=use_gripper)
    robot = DobotCR3(cfg)
    robot.connect()
    try:
        before = np.asarray(robot.get_joints(), dtype=np.float32)
        print(f"Real before deg: {np.array2string(before, precision=3, suppress_small=True)}")
        print(f"Sending real JointMovJ target deg: {np.array2string(target_deg, precision=3, suppress_small=True)}")
        reply = robot.move.JointMovJ(*[float(v) for v in target_deg])
        print(f"JointMovJ reply: {reply}")
        time.sleep(0.5)
        after = np.asarray(robot.get_joints(), dtype=np.float32)
        print(f"Real after deg: {np.array2string(after, precision=3, suppress_small=True)}")
        return after
    finally:
        robot.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joints-deg",
        required=True,
        help="Six absolute joint targets in degrees, e.g. '0,-20,30,0,45,0'.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mode", choices=["sim", "real", "both"], default="sim")
    parser.add_argument(
        "--enable-real",
        action="store_true",
        help="Required for mode=real/both. Without this flag the real robot will not move.",
    )
    parser.add_argument("--robot-ip", default="192.168.5.1")
    parser.add_argument("--speed-factor", type=int, default=20)
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument("--duration-s", type=float, default=4.0, help="Sim interpolation duration.")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-step-deg", type=float, default=1.0, help="Max sim target change per frame.")
    parser.add_argument("--joint-sign", default=DISPLAY_TO_SIM_JOINT_SIGN)
    parser.add_argument("--joint-offset-deg", default=DISPLAY_TO_SIM_JOINT_OFFSET_DEG)
    parser.add_argument("--viewer-camera", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_deg = parse_six_floats(args.joints_deg, "--joints-deg")
    joint_sign = parse_six_floats(args.joint_sign, "--joint-sign")
    joint_offset_deg = parse_six_floats(args.joint_offset_deg, "--joint-offset-deg")

    if args.mode in ("real", "both") and not args.enable_real:
        raise RuntimeError("Refusing to move the real robot without --enable-real.")

    real_after = None
    if args.mode in ("real", "both"):
        real_after = run_real_robot(
            target_deg,
            robot_ip=args.robot_ip,
            speed_factor=args.speed_factor,
            use_gripper=args.use_gripper,
        )

    sim_after = None
    if args.mode in ("sim", "both"):
        sim_after = run_sim(
            target_deg,
            model_path=args.model_path,
            joint_sign=joint_sign,
            joint_offset_deg=joint_offset_deg,
            duration_s=args.duration_s,
            fps=args.fps,
            max_step_deg=args.max_step_deg,
            viewer_camera=args.viewer_camera,
        )
        print(f"Sim after deg: {np.array2string(sim_after, precision=3, suppress_small=True)}")

    if real_after is not None and sim_after is not None:
        print(f"Real-sim joint difference deg: {np.array2string(real_after - sim_after, precision=3, suppress_small=True)}")


if __name__ == "__main__":
    main()
