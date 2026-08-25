#!/usr/bin/env python

"""Stable first-pass cube contact/grasp test for the CR3 + Lebai LMG-90 gripper."""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))


ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "cr3_scene.xml"

LEFT_ACTUATOR = "Left_joint1_pos"
RIGHT_ACTUATOR = "Right_joint_pos"
LEFT_FINGER_GEOM = "left_static1_finger_collision"
RIGHT_FINGER_GEOM = "right_static1_finger_collision"
CUBE_JOINT = "cube_free"
CUBE_BODY = "cube"
CUBE_GEOM = "cube_collision"

GRIPPER_OPEN_CMD = 0.12
GRIPPER_GRASP_CMD = 0.35
GRIPPER_GRASP_CMD_MAX = 0.35
OPEN_SETTLE_S = 1.0
CLOSE_DURATION_S = 3.0
HOLD_DURATION_S = 1.0
POST_RELEASE_HOLD_S = 0.0
LIFT_DURATION_S = 0.05
LIFT_HEIGHT_M = 0.001


def obj_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    index = mujoco.mj_name2id(model, objtype, name)
    if index < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(index)


def set_gripper(data: mujoco.MjData, left_id: int, right_id: int, cmd: float) -> float:
    cmd = float(np.clip(cmd, 0.0, GRIPPER_GRASP_CMD_MAX))
    data.ctrl[left_id] = cmd
    data.ctrl[right_id] = -cmd
    return cmd


def hold_arm_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for actuator_id in range(min(6, model.nu)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = data.qpos[qpos_addr]


def reset_cube_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def place_cube_between_fingers(model: mujoco.MjModel, data: mujoco.MjData, *, z_lift: float = 0.0) -> np.ndarray:
    left_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qpos_addr = int(model.jnt_qposadr[cube_joint_id])
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])

    left_pos = np.asarray(data.geom_xpos[left_geom_id], dtype=np.float64)
    right_pos = np.asarray(data.geom_xpos[right_geom_id], dtype=np.float64)
    cube_pos = 0.5 * (left_pos + right_pos)
    cube_pos[2] += z_lift

    data.qpos[qpos_addr : qpos_addr + 3] = cube_pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return cube_pos


def cube_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def cube_finger_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    cube_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
    finger_ids = {
        obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM),
        obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM),
    }
    count = 0
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (geom1 == cube_geom_id and geom2 in finger_ids) or (geom2 == cube_geom_id and geom1 in finger_ids):
            count += 1
    return count


def step_for(model: mujoco.MjModel, data: mujoco.MjData, seconds: float, viewer=None) -> None:
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def print_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_id: int,
    right_id: int,
    cmd: float,
    label: str,
    start_time: float,
    cube_start_z: float | None = None,
) -> None:
    pos = cube_pos(model, data)
    drop_note = ""
    if cube_start_z is not None and pos[2] < cube_start_z - 0.04:
        drop_note = " DROPPED: likely friction/force/collision alignment is insufficient"
    print(
        f"t={time.perf_counter() - start_time:.2f}s {label}: cmd={cmd:.3f}, "
        f"left_ctrl={data.ctrl[left_id]:.3f}, right_ctrl={data.ctrl[right_id]:.3f}, "
        f"cube_z={pos[2]:.5f}, cube={np.round(pos, 5).tolist()}, contacts={cube_finger_contacts(model, data)}"
        f"{drop_note}",
        flush=True,
    )


def lift_robot_mount(model: mujoco.MjModel, data: mujoco.MjData, *, dz: float, alpha: float) -> None:
    body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "cr3_base_mount")
    if not hasattr(lift_robot_mount, "_start_z"):
        lift_robot_mount._start_z = float(model.body_pos[body_id, 2])
    model.body_pos[body_id, 2] = lift_robot_mount._start_z + dz * alpha


def run_sequence(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    viewer=None,
    cube_z_lift: float = 0.0,
    assist_placement: bool = True,
    lift_test: bool = True,
) -> None:
    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACTUATOR)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACTUATOR)

    start_time = time.perf_counter()
    hold_arm_pose(model, data)
    cmd = set_gripper(data, left_id, right_id, GRIPPER_OPEN_CMD)
    mujoco.mj_forward(model, data)
    print_status(model, data, left_id, right_id, cmd, "open", start_time)
    step_for(model, data, OPEN_SETTLE_S, viewer)

    cube_start = place_cube_between_fingers(model, data, z_lift=cube_z_lift)
    print(f"placed cube at {np.round(cube_start, 5).tolist()}", flush=True)
    step_for(model, data, 0.2, viewer)

    close_start = time.perf_counter()
    next_print = close_start
    while True:
        elapsed = time.perf_counter() - close_start
        alpha = min(elapsed / CLOSE_DURATION_S, 1.0)
        # Smoothstep keeps acceleration gentle at the beginning and end.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        cmd = GRIPPER_OPEN_CMD + (GRIPPER_GRASP_CMD - GRIPPER_OPEN_CMD) * alpha
        cmd = set_gripper(data, left_id, right_id, cmd)
        if assist_placement:
            place_cube_between_fingers(model, data, z_lift=cube_z_lift)
            reset_cube_velocity(model, data)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        now = time.perf_counter()
        if now >= next_print:
            print_status(model, data, left_id, right_id, cmd, "closing", start_time, cube_start[2])
            next_print = now + 0.25
        if elapsed >= CLOSE_DURATION_S:
            break

    print("release placement assist; testing gravity hold", flush=True)
    hold_until = time.perf_counter() + HOLD_DURATION_S
    next_print = time.perf_counter()
    while time.perf_counter() < hold_until:
        cmd = set_gripper(data, left_id, right_id, GRIPPER_GRASP_CMD)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        now = time.perf_counter()
        if now >= next_print:
            print_status(model, data, left_id, right_id, cmd, "hold", start_time, cube_start[2])
            next_print = now + 0.5

    release_start_z = float(cube_pos(model, data)[2])
    release_until = time.perf_counter() + POST_RELEASE_HOLD_S
    next_print = time.perf_counter()
    while time.perf_counter() < release_until:
        cmd = set_gripper(data, left_id, right_id, GRIPPER_GRASP_CMD)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        now = time.perf_counter()
        if now >= next_print:
            print_status(model, data, left_id, right_id, cmd, "gravity-hold", start_time, release_start_z)
            next_print = now + 0.5

    if not lift_test:
        return

    lift_start = time.perf_counter()
    lift_start_cube_z = float(cube_pos(model, data)[2])
    next_print = lift_start
    while True:
        elapsed = time.perf_counter() - lift_start
        alpha = min(elapsed / LIFT_DURATION_S, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        lift_robot_mount(model, data, dz=LIFT_HEIGHT_M, alpha=alpha)
        hold_arm_pose(model, data)
        cmd = set_gripper(data, left_id, right_id, GRIPPER_GRASP_CMD)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        now = time.perf_counter()
        if now >= next_print:
            print_status(model, data, left_id, right_id, cmd, "lift", start_time, lift_start_cube_z)
            next_print = now + 0.5
        if elapsed >= LIFT_DURATION_S:
            break

    final_lift = float(cube_pos(model, data)[2] - lift_start_cube_z)
    print(f"lift result: cube dz={final_lift:.5f}m target dz={LIFT_HEIGHT_M:.5f}m", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--no-gravity", action="store_true", help="Disable gravity for contact-only debugging.")
    parser.add_argument("--no-placement-assist", action="store_true", help="Do not hold cube between fingers during closing.")
    parser.add_argument("--lift-test", action="store_true", help="Run the optional slow base lift after grasp.")
    parser.add_argument("--no-lift-test", action="store_true", help="Deprecated: lift test is skipped by default.")
    parser.add_argument("--cube-z-lift", type=float, default=0.0, help="Small z offset when placing cube between fingers.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    if args.no_gravity:
        model.opt.gravity[:] = np.asarray([0.0, 0.0, 0.0])
    else:
        model.opt.gravity[:] = np.asarray([0.0, 0.0, -9.81])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.headless:
        run_sequence(
            model,
            data,
            cube_z_lift=args.cube_z_lift,
            assist_placement=not args.no_placement_assist,
            lift_test=args.lift_test and not args.no_lift_test,
        )
        return

    viewer_module = importlib.import_module("mujoco.viewer")

    with viewer_module.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -35
        run_sequence(
            model,
            data,
            viewer=viewer,
            cube_z_lift=args.cube_z_lift,
            assist_placement=not args.no_placement_assist,
            lift_test=args.lift_test and not args.no_lift_test,
        )
        print("grasp test finished; close the viewer window to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
