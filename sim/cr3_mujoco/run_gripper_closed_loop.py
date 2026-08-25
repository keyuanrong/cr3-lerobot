#!/usr/bin/env python

"""Open the CR3 MuJoCo scene and cycle the Lebai LMG-90 closed-loop gripper."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import mujoco

LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))

from sim.cr3_mujoco.joint_mapping import GRIPPER_CLOSED, GRIPPER_OPEN


ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "cr3_scene.xml"
LEFT_ACTUATOR = "Left_joint1_pos"
RIGHT_ACTUATOR = "Right_joint_pos"
ARM_ACTUATORS = ("J1_pos", "J2_pos", "J3_pos", "J4_pos", "J5_pos", "J6_pos")
LOOP_SITE_PAIRS = (
    ("left_link2_pin_site", "left_static_pin_site"),
    ("left_link2_axis_site", "left_static_axis_site"),
    ("right_link2_pin_site", "right_static_pin_site"),
    ("right_link2_axis_site", "right_static_axis_site"),
)


def name_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return obj_id


def actuator_joint_qpos(model: mujoco.MjModel, actuator_id: int) -> float | None:
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
        return None
    qpos_addr = int(model.jnt_qposadr[joint_id])
    return qpos_addr


def hold_current_arm_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for actuator_name in ARM_ACTUATORS:
        actuator_id = name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        qpos_addr = actuator_joint_qpos(model, actuator_id)
        if qpos_addr is not None:
            data.ctrl[actuator_id] = data.qpos[qpos_addr]


def site_distance(model: mujoco.MjModel, data: mujoco.MjData, site1: str, site2: str) -> float:
    site1_id = name_id(model, mujoco.mjtObj.mjOBJ_SITE, site1)
    site2_id = name_id(model, mujoco.mjtObj.mjOBJ_SITE, site2)
    delta = data.site_xpos[site1_id] - data.site_xpos[site2_id]
    return float(math.sqrt(float(delta @ delta)))


def step_gripper(model: mujoco.MjModel, data: mujoco.MjData, left_id: int, right_id: int, t: float) -> float:
    midpoint = 0.5 * (GRIPPER_OPEN + GRIPPER_CLOSED)
    amplitude = 0.5 * (GRIPPER_OPEN - GRIPPER_CLOSED)
    cmd = midpoint + amplitude * math.sin(2.0 * math.pi * 0.15 * t)
    data.ctrl[left_id] = cmd
    data.ctrl[right_id] = -cmd
    return cmd


def print_status(model: mujoco.MjModel, data: mujoco.MjData, cmd: float) -> None:
    distances = ", ".join(f"{a}/{b}={site_distance(model, data, a, b):.5f}m" for a, b in LOOP_SITE_PAIRS)
    print(f"cmd left={cmd:.3f}, right={-cmd:.3f}, site distance: {distances}", flush=True)


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, duration: float) -> None:
    left_id = name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACTUATOR)
    right_id = name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACTUATOR)
    hold_current_arm_pose(model, data)

    start = time.time()
    next_print = start
    while time.time() - start < duration:
        now = time.time()
        cmd = step_gripper(model, data, left_id, right_id, now - start)
        mujoco.mj_step(model, data)
        if now >= next_print:
            print_status(model, data, cmd)
            next_print = now + 0.5


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    import mujoco.viewer

    left_id = name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACTUATOR)
    right_id = name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACTUATOR)
    hold_current_arm_pose(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        next_print = start
        while viewer.is_running():
            now = time.time()
            cmd = step_gripper(model, data, left_id, right_id, now - start)
            mujoco.mj_step(model, data)
            viewer.sync()
            if now >= next_print:
                print_status(model, data, cmd)
                next_print = now + 0.5
            time.sleep(model.opt.timestep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--duration", type=float, default=8.0, help="Headless run duration in seconds.")
    args = parser.parse_args()

    if not SCENE_XML.exists():
        raise FileNotFoundError(f"Scene XML not found: {SCENE_XML}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.headless:
        run_headless(model, data, args.duration)
    else:
        run_viewer(model, data)


if __name__ == "__main__":
    main()
