#!/usr/bin/env python

"""Scripted left-table pick demo for CR3 + Lebai LMG-90 in MuJoCo."""

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

ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
ARM_ACTUATORS = tuple(f"{name}_pos" for name in ARM_JOINTS)
LEFT_ACTUATOR = "Left_joint1_pos"
RIGHT_ACTUATOR = "Right_joint_pos"
LEFT_FINGER_GEOM = "left_static1_finger_collision"
RIGHT_FINGER_GEOM = "right_static1_finger_collision"
CUBE_BODY = "cube"
CUBE_GEOM = "cube"
CUBE_JOINT = "cube_free"

GRIPPER_OPEN_CMD = 0.12
GRIPPER_GRASP_CMD = 0.45
LEFT_TABLE_CUBE_XY = (-0.35, -0.12)
APPROACH_HEIGHT = 0.11
LIFT_HEIGHT = 0.08


def obj_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    index = mujoco.mj_name2id(model, objtype, name)
    if index < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(index)


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def arm_qpos_addrs(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_qposadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)] for joint in ARM_JOINTS],
        dtype=np.int32,
    )


def arm_qvel_addrs(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_dofadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)] for joint in ARM_JOINTS],
        dtype=np.int32,
    )


def arm_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) for actuator in ARM_ACTUATORS],
        dtype=np.int32,
    )


def set_gripper(data: mujoco.MjData, left_id: int, right_id: int, cmd: float) -> None:
    data.ctrl[left_id] = cmd
    data.ctrl[right_id] = -cmd


def finger_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    return 0.5 * (data.geom_xpos[left_id] + data.geom_xpos[right_id])


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


def table_top_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    return float(data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2])


def cube_half_size(model: mujoco.MjModel) -> float:
    geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
    return float(max(model.geom_size[geom_id]))


def place_cube_on_left_table(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qpos_addr = int(model.jnt_qposadr[cube_joint_id])
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])
    pos = np.asarray(
        [LEFT_TABLE_CUBE_XY[0], LEFT_TABLE_CUBE_XY[1], table_top_z(model, data) + cube_half_size(model) + 0.001],
        dtype=np.float64,
    )
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return pos


def set_cube_pose(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray) -> None:
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qpos_addr = int(model.jnt_qposadr[cube_joint_id])
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def align_cube_between_fingers_on_table(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    pos = finger_center(model, data).copy()
    pos[2] = table_top_z(model, data) + cube_half_size(model) + 0.001
    set_cube_pose(model, data, pos)
    mujoco.mj_forward(model, data)
    return pos


def carry_cube_with_fingers(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    pos = finger_center(model, data).copy()
    set_cube_pose(model, data, pos)
    mujoco.mj_forward(model, data)
    return pos


def set_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    data.qpos[arm_qpos_addrs(model)] = qpos
    data.qvel[arm_qvel_addrs(model)] = 0.0
    mujoco.mj_forward(model, data)


def solve_finger_center_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    seed: np.ndarray,
    *,
    tries: int = 2800,
) -> np.ndarray:
    addrs = arm_qpos_addrs(model)
    joint_ids = [obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in ARM_JOINTS]
    limits = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids], dtype=np.float64)
    original_qpos = np.asarray(data.qpos[addrs], dtype=np.float64).copy()

    def score(qpos: np.ndarray) -> float:
        data.qpos[addrs] = qpos
        mujoco.mj_forward(model, data)
        delta = finger_center(model, data) - target
        # Prefer poses that keep close to the previous segment, so the demo moves calmly.
        return float(delta @ delta + 0.002 * np.mean((qpos - seed) ** 2))

    best = np.clip(seed.copy(), limits[:, 0], limits[:, 1])
    best_score = score(best)
    scales = np.asarray([0.30, 0.22, 0.28, 0.35, 0.35, 0.50], dtype=np.float64)
    rng = np.random.default_rng(7)
    for idx in range(tries):
        temperature = 1.0 - 0.85 * (idx / max(tries - 1, 1))
        candidate = best + rng.normal(0.0, scales * temperature)
        candidate = np.clip(candidate, limits[:, 0], limits[:, 1])
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score

    data.qpos[addrs] = original_qpos
    mujoco.mj_forward(model, data)
    return best


def move_arm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_ids: np.ndarray,
    start_qpos: np.ndarray,
    goal_qpos: np.ndarray,
    seconds: float,
    *,
    left_id: int,
    right_id: int,
    gripper_cmd: float,
    carry_assist: bool = False,
    viewer=None,
    label: str,
) -> None:
    start_time = time.perf_counter()
    next_print = start_time
    addrs = arm_qpos_addrs(model)
    dofs = arm_qvel_addrs(model)
    while True:
        elapsed = time.perf_counter() - start_time
        alpha = smoothstep(elapsed / seconds)
        qpos = start_qpos + (goal_qpos - start_qpos) * alpha
        data.qpos[addrs] = qpos
        data.qvel[dofs] = 0.0
        data.ctrl[arm_ids] = qpos
        set_gripper(data, left_id, right_id, gripper_cmd)
        if carry_assist:
            carry_cube_with_fingers(model, data)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        now = time.perf_counter()
        if now >= next_print:
            print_status(model, data, label, gripper_cmd)
            next_print = now + 0.5
        if elapsed >= seconds:
            break


def close_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_ids: np.ndarray,
    hold_qpos: np.ndarray,
    left_id: int,
    right_id: int,
    *,
    carry_assist: bool,
    viewer=None,
) -> None:
    start_time = time.perf_counter()
    next_print = start_time
    addrs = arm_qpos_addrs(model)
    dofs = arm_qvel_addrs(model)
    while True:
        elapsed = time.perf_counter() - start_time
        alpha = smoothstep(elapsed / 3.0)
        cmd = GRIPPER_OPEN_CMD + (GRIPPER_GRASP_CMD - GRIPPER_OPEN_CMD) * alpha
        data.qpos[addrs] = hold_qpos
        data.qvel[dofs] = 0.0
        data.ctrl[arm_ids] = hold_qpos
        set_gripper(data, left_id, right_id, cmd)
        if carry_assist:
            align_cube_between_fingers_on_table(model, data)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        now = time.perf_counter()
        if now >= next_print:
            print_status(model, data, "closing", cmd)
            next_print = now + 0.4
        if elapsed >= 3.0:
            break


def print_status(model: mujoco.MjModel, data: mujoco.MjData, label: str, cmd: float) -> None:
    cpos = cube_pos(model, data)
    fpos = finger_center(model, data)
    print(
        f"{label}: cmd={cmd:.3f}, cube={np.round(cpos, 4).tolist()}, "
        f"finger={np.round(fpos, 4).tolist()}, contacts={cube_finger_contacts(model, data)}",
        flush=True,
    )


def run_demo(model: mujoco.MjModel, data: mujoco.MjData, *, viewer=None, carry_assist: bool = True) -> None:
    model.opt.gravity[:] = np.asarray([0.0, 0.0, -9.81])
    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACTUATOR)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACTUATOR)
    arm_ids = arm_actuator_ids(model)
    addrs = arm_qpos_addrs(model)

    set_gripper(data, left_id, right_id, GRIPPER_OPEN_CMD)
    cube_start = place_cube_on_left_table(model, data)
    print(f"red cube placed on left table at {np.round(cube_start, 4).tolist()}", flush=True)

    current = np.asarray(data.qpos[addrs], dtype=np.float64).copy()
    grasp_target = cube_start.copy()
    grasp_target[2] += 0.004
    approach_target = grasp_target.copy()
    approach_target[2] += APPROACH_HEIGHT
    lift_target = grasp_target.copy()
    lift_target[2] += LIFT_HEIGHT

    approach_qpos = solve_finger_center_ik(model, data, approach_target, current)
    grasp_qpos = solve_finger_center_ik(model, data, grasp_target, approach_qpos)
    lift_qpos = solve_finger_center_ik(model, data, lift_target, grasp_qpos)
    print("planned arm poses in degrees:", flush=True)
    for name, pose in (("approach", approach_qpos), ("grasp", grasp_qpos), ("lift", lift_qpos)):
        print(f"  {name}: {np.round(np.rad2deg(pose), 2).tolist()}", flush=True)

    set_arm_qpos(model, data, current)
    data.ctrl[arm_ids] = current
    set_gripper(data, left_id, right_id, GRIPPER_OPEN_CMD)
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()

    move_arm(
        model,
        data,
        arm_ids,
        current,
        approach_qpos,
        3.0,
        left_id=left_id,
        right_id=right_id,
        gripper_cmd=GRIPPER_OPEN_CMD,
        viewer=viewer,
        label="approach",
    )
    move_arm(
        model,
        data,
        arm_ids,
        approach_qpos,
        grasp_qpos,
        2.0,
        left_id=left_id,
        right_id=right_id,
        gripper_cmd=GRIPPER_OPEN_CMD,
        viewer=viewer,
        label="descend",
    )
    if carry_assist:
        aligned = align_cube_between_fingers_on_table(model, data)
        print(f"demo assist aligned cube between fingers at {np.round(aligned, 4).tolist()}", flush=True)
    close_gripper(model, data, arm_ids, grasp_qpos, left_id, right_id, carry_assist=carry_assist, viewer=viewer)

    if carry_assist:
        carried = carry_cube_with_fingers(model, data)
        print(f"demo assist attached cube to gripper at {np.round(carried, 4).tolist()}", flush=True)

    move_arm(
        model,
        data,
        arm_ids,
        grasp_qpos,
        lift_qpos,
        2.5,
        left_id=left_id,
        right_id=right_id,
        gripper_cmd=GRIPPER_GRASP_CMD,
        carry_assist=carry_assist,
        viewer=viewer,
        label="lift",
    )

    if carry_assist:
        start_time = time.perf_counter()
        next_print = start_time
        addrs = arm_qpos_addrs(model)
        dofs = arm_qvel_addrs(model)
        while time.perf_counter() - start_time < 0.8:
            data.qpos[addrs] = lift_qpos
            data.qvel[dofs] = 0.0
            data.ctrl[arm_ids] = lift_qpos
            set_gripper(data, left_id, right_id, GRIPPER_GRASP_CMD)
            carry_cube_with_fingers(model, data)
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            now = time.perf_counter()
            if now >= next_print:
                print_status(model, data, "carry-assist hold", GRIPPER_GRASP_CMD)
                next_print = now + 0.4

    final = cube_pos(model, data)
    print(f"demo finished: cube final={np.round(final, 4).tolist()}, dz={final[2] - cube_start[2]:.4f}m", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument(
        "--no-carry-assist",
        action="store_true",
        help="Disable the visual cube alignment/carry assist and run pure contact physics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.headless:
        run_demo(model, data, carry_assist=not args.no_carry_assist)
        return

    viewer_module = importlib.import_module("mujoco.viewer")
    with viewer_module.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (-0.35, -0.08, 0.45)
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = -65
        viewer.cam.elevation = -30
        run_demo(model, data, viewer=viewer, carry_assist=not args.no_carry_assist)
        print("left-table pick demo finished; close the viewer window to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
