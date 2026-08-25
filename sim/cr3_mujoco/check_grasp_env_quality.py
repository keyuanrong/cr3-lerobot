#!/usr/bin/env python

"""Quality gate for the CR3 + Lebai LMG-90 MuJoCo grasp environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

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
CUBE_BODY = "cube"
CUBE_JOINT = "cube_free"
CUBE_COLLISION_GEOM = "cube_collision"
CUBE_VISUAL_GEOM = "cube"
TABLE_COLLISION_GEOMS = (
    "table_top_collision",
    "table_edge_collision_xmin",
    "table_edge_collision_xmax",
    "table_edge_collision_ymin",
    "table_edge_collision_ymax",
)
LOOP_SITE_PAIRS = (
    ("left_link2_pin_site", "left_static_pin_site"),
    ("left_link2_axis_site", "left_static_axis_site"),
    ("right_link2_pin_site", "right_static_pin_site"),
    ("right_link2_axis_site", "right_static_axis_site"),
)
CONNECT_NAMES = (
    "left_link2_static_connect",
    "left_link2_static_axis_connect",
    "right_link2_static_connect",
    "right_link2_static_axis_connect",
)

GRIPPER_OPEN_CMD = 0.12
GRIPPER_GRASP_CMD = 0.35
OPEN_SETTLE_S = 1.0
CLOSE_DURATION_S = 3.0
HOLD_DURATION_S = 3.0
EMPTY_SIM_S = 4.0
SITE_WARN_M = 0.005
SITE_PASS_M = 0.008
DROP_WARN_M = 0.03
FINGER_TABLE_PENETRATION_WARN_M = 0.004


@dataclass
class CheckReport:
    scene_load: bool = False
    no_nan_no_explosion: bool = False
    gripper_connect_stability: bool = False
    finger_cube_contact: bool = False
    cube_table_contact: bool = False
    finger_table_obvious_penetration: bool = False
    cube_stable_during_hold: bool = False
    replay_consistency: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def ready_for_vla_data(self) -> bool:
        return all(
            (
                self.scene_load,
                self.no_nan_no_explosion,
                self.gripper_connect_stability,
                self.finger_cube_contact,
                self.cube_table_contact,
                self.finger_table_obvious_penetration,
                self.cube_stable_during_hold,
                self.replay_consistency,
            )
        )


def obj_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    index = mujoco.mj_name2id(model, objtype, name)
    if index < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(index)


def maybe_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    return int(mujoco.mj_name2id(model, objtype, name))


def finite_state(data: mujoco.MjData) -> bool:
    return bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.xpos).all())


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def set_gripper(data: mujoco.MjData, left_id: int, right_id: int, cmd: float) -> None:
    data.ctrl[left_id] = float(cmd)
    data.ctrl[right_id] = -float(cmd)


def hold_arm_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for actuator_id in range(min(6, model.nu)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        data.ctrl[actuator_id] = data.qpos[qpos_addr]


def step_for(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> bool:
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not finite_state(data):
            return False
    return True


def site_distance(model: mujoco.MjModel, data: mujoco.MjData, site1: str, site2: str) -> float:
    site1_id = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, site1)
    site2_id = obj_id(model, mujoco.mjtObj.mjOBJ_SITE, site2)
    delta = data.site_xpos[site1_id] - data.site_xpos[site2_id]
    return float(np.linalg.norm(delta))


def all_site_distances(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    return {f"{a}<->{b}": site_distance(model, data, a, b) for a, b in LOOP_SITE_PAIRS}


def reset_cube_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def place_cube_between_fingers(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qpos_addr = int(model.jnt_qposadr[cube_joint_id])
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])

    pos = 0.5 * (data.geom_xpos[left_geom_id] + data.geom_xpos[right_geom_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return np.asarray(pos, dtype=np.float64)


def contact_pair_names(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or f"geom{contact.geom1}"
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or f"geom{contact.geom2}"
        pairs.append((geom1, geom2, float(contact.dist)))
    return pairs


def count_contacts_with(model: mujoco.MjModel, data: mujoco.MjData, geom_a: str, geom_b: str) -> int:
    a_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a)
    b_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b)
    count = 0
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (geom1 == a_id and geom2 == b_id) or (geom1 == b_id and geom2 == a_id):
            count += 1
    return count


def count_contacts_in_set(model: mujoco.MjModel, data: mujoco.MjData, geom: str, others: tuple[str, ...]) -> int:
    geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    other_ids = {obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, other) for other in others if maybe_id(model, mujoco.mjtObj.mjOBJ_GEOM, other) >= 0}
    count = 0
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (geom1 == geom_id and geom2 in other_ids) or (geom2 == geom_id and geom1 in other_ids):
            count += 1
    return count


def run_empty_stability(model: mujoco.MjModel) -> tuple[bool, int]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ok = finite_state(data) and step_for(model, data, EMPTY_SIM_S)
    return ok, int(data.ncon)


def run_replay(model: mujoco.MjModel) -> dict[str, object]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACTUATOR)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACTUATOR)
    hold_arm_pose(model, data)
    set_gripper(data, left_id, right_id, GRIPPER_OPEN_CMD)
    mujoco.mj_forward(model, data)
    open_distances = all_site_distances(model, data)
    open_ok = finite_state(data) and step_for(model, data, OPEN_SETTLE_S)

    placed_cube = place_cube_between_fingers(model, data)
    close_steps = max(1, int(CLOSE_DURATION_S / model.opt.timestep))
    max_site_distance = max(open_distances.values())
    max_cube_speed = 0.0
    for step in range(close_steps):
        alpha = step / max(close_steps - 1, 1)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        cmd = GRIPPER_OPEN_CMD + (GRIPPER_GRASP_CMD - GRIPPER_OPEN_CMD) * alpha
        set_gripper(data, left_id, right_id, cmd)
        place_cube_between_fingers(model, data)
        reset_cube_velocity(model, data)
        mujoco.mj_step(model, data)
        if not finite_state(data):
            break
        max_site_distance = max(max_site_distance, max(all_site_distances(model, data).values()))
        cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
        qvel_addr = int(model.jnt_dofadr[cube_joint_id])
        max_cube_speed = max(max_cube_speed, float(np.linalg.norm(data.qvel[qvel_addr : qvel_addr + 3])))

    closed_distances = all_site_distances(model, data)
    left_contacts_at_close = count_contacts_with(model, data, CUBE_COLLISION_GEOM, LEFT_FINGER_GEOM)
    right_contacts_at_close = count_contacts_with(model, data, CUBE_COLLISION_GEOM, RIGHT_FINGER_GEOM)
    contact_pairs_at_close = contact_pair_names(model, data)

    hold_start = body_pos(model, data, CUBE_BODY).copy()
    hold_steps = max(1, int(HOLD_DURATION_S / model.opt.timestep))
    min_hold_z = float(hold_start[2])
    max_hold_disp = 0.0
    hold_contacts_left = 0
    hold_contacts_right = 0
    for _ in range(hold_steps):
        set_gripper(data, left_id, right_id, GRIPPER_GRASP_CMD)
        mujoco.mj_step(model, data)
        if not finite_state(data):
            break
        pos = body_pos(model, data, CUBE_BODY)
        min_hold_z = min(min_hold_z, float(pos[2]))
        max_hold_disp = max(max_hold_disp, float(np.linalg.norm(pos - hold_start)))
        hold_contacts_left = max(hold_contacts_left, count_contacts_with(model, data, CUBE_COLLISION_GEOM, LEFT_FINGER_GEOM))
        hold_contacts_right = max(hold_contacts_right, count_contacts_with(model, data, CUBE_COLLISION_GEOM, RIGHT_FINGER_GEOM))
        max_site_distance = max(max_site_distance, max(all_site_distances(model, data).values()))

    final_cube = body_pos(model, data, CUBE_BODY).copy()
    return {
        "open_ok": open_ok,
        "placed_cube": placed_cube,
        "hold_start": hold_start,
        "final_cube": final_cube,
        "min_hold_z": min_hold_z,
        "max_hold_disp": max_hold_disp,
        "max_cube_speed": max_cube_speed,
        "open_site_distances": open_distances,
        "closed_site_distances": closed_distances,
        "max_site_distance": max_site_distance,
        "left_contacts_at_close": left_contacts_at_close,
        "right_contacts_at_close": right_contacts_at_close,
        "hold_contacts_left": hold_contacts_left,
        "hold_contacts_right": hold_contacts_right,
        "contact_pairs_at_close": contact_pairs_at_close,
        "finite": finite_state(data),
    }


def initial_penetration_checks(model: mujoco.MjModel) -> dict[str, object]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    table_contacts = count_contacts_in_set(model, data, CUBE_COLLISION_GEOM, TABLE_COLLISION_GEOMS)
    cube_finger_initial = (
        count_contacts_with(model, data, CUBE_COLLISION_GEOM, LEFT_FINGER_GEOM)
        + count_contacts_with(model, data, CUBE_COLLISION_GEOM, RIGHT_FINGER_GEOM)
    )

    finger_table_negative: list[tuple[str, str, float]] = []
    table_ids = {obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in TABLE_COLLISION_GEOMS if maybe_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0}
    finger_ids = {
        obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM),
        obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM),
    }
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if (geom1 in finger_ids and geom2 in table_ids) or (geom2 in finger_ids and geom1 in table_ids):
            if contact.dist < -FINGER_TABLE_PENETRATION_WARN_M:
                name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or str(geom1)
                name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or str(geom2)
                finger_table_negative.append((name1, name2, float(contact.dist)))

    cube_pos = body_pos(model, data, CUBE_BODY)
    table_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    cube_geom_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_COLLISION_GEOM)
    table_top_z = float(data.geom_xpos[table_id, 2] + model.geom_size[table_id, 2])
    cube_bottom_z = float(cube_pos[2] - max(model.geom_size[cube_geom_id]))
    return {
        "table_contacts": table_contacts,
        "cube_finger_initial": cube_finger_initial,
        "finger_table_negative": finger_table_negative,
        "cube_bottom_z": cube_bottom_z,
        "table_top_z": table_top_z,
        "contact_pairs": contact_pair_names(model, data),
    }


def required_objects_check(model: mujoco.MjModel, report: CheckReport) -> bool:
    required = [
        (mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY),
        (mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT),
        (mujoco.mjtObj.mjOBJ_GEOM, CUBE_COLLISION_GEOM),
        (mujoco.mjtObj.mjOBJ_GEOM, CUBE_VISUAL_GEOM),
        (mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM),
        (mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM),
        (mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_ACTUATOR),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_ACTUATOR),
    ]
    ok = True
    for objtype, name in required:
        if maybe_id(model, objtype, name) < 0:
            report.warnings.append(f"missing required object: {name}")
            ok = False
    for name in CONNECT_NAMES:
        if maybe_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name) < 0:
            report.warnings.append(f"missing equality/connect: {name}")
            ok = False
    for site1, site2 in LOOP_SITE_PAIRS:
        for site in (site1, site2):
            if maybe_id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
                report.warnings.append(f"missing loop site: {site}")
                ok = False
    return ok


def print_bool(label: str, value: bool) -> None:
    print(f"{label}: {'PASS' if value else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = CheckReport()

    print("==== CR3 + LMG-90 MuJoCo Grasp Env Check ====")
    print()

    try:
        model = mujoco.MjModel.from_xml_path(str(args.model_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        report.scene_load = finite_state(data) and required_objects_check(model, report)
    except Exception as exc:
        report.warnings.append(f"scene load failed: {exc}")
        print_bool("Scene load", False)
        print_bool("No NaN / no explosion", False)
        print_bool("Gripper connect stability", False)
        print_bool("Finger-cube contact", False)
        print_bool("Cube-table contact", False)
        print_bool("Finger-table obvious penetration", False)
        print_bool("Cube stable during hold", False)
        print_bool("Replay consistency", False)
        print()
        print("Final result:")
        print("READY_FOR_VLA_DATA = False")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        return

    empty_ok, empty_contacts = run_empty_stability(model)
    report.no_nan_no_explosion = empty_ok

    initial = initial_penetration_checks(model)
    report.cube_table_contact = bool(initial["table_contacts"] > 0 and initial["cube_bottom_z"] >= initial["table_top_z"] - 0.003)
    report.finger_table_obvious_penetration = len(initial["finger_table_negative"]) == 0
    if initial["cube_finger_initial"] > 0:
        report.warnings.append("cube initially contacts finger collision; check task reset pose if this is unexpected")
    if not report.cube_table_contact:
        report.warnings.append(
            f"cube/table check weak: contacts={initial['table_contacts']} cube_bottom_z={initial['cube_bottom_z']:.5f} table_top_z={initial['table_top_z']:.5f}"
        )
    for name1, name2, dist in initial["finger_table_negative"]:
        report.warnings.append(f"finger/table penetration: {name1} vs {name2}, dist={dist:.5f}m")

    replay = run_replay(model)
    max_site_distance = float(replay["max_site_distance"])
    report.gripper_connect_stability = bool(replay["finite"] and max_site_distance <= SITE_PASS_M)
    report.finger_cube_contact = bool(replay["left_contacts_at_close"] > 0 and replay["right_contacts_at_close"] > 0)
    cube_drop = float(replay["hold_start"][2] - replay["min_hold_z"])
    report.cube_stable_during_hold = bool(replay["finite"] and cube_drop <= DROP_WARN_M and replay["hold_contacts_left"] > 0 and replay["hold_contacts_right"] > 0)
    final_displacement = float(np.linalg.norm(replay["final_cube"] - replay["hold_start"]))
    report.replay_consistency = bool(
        replay["open_ok"]
        and replay["finite"]
        and report.finger_cube_contact
        and report.gripper_connect_stability
        and final_displacement < 0.20
    )

    if max_site_distance > SITE_WARN_M:
        report.warnings.append(f"max loop site distance {max_site_distance * 1000:.2f}mm exceeds {SITE_WARN_M * 1000:.1f}mm warning threshold")
    if not report.finger_cube_contact:
        report.warnings.append(
            f"finger/cube contact missing or one-sided: left={replay['left_contacts_at_close']} right={replay['right_contacts_at_close']}"
        )
    if cube_drop > DROP_WARN_M:
        report.warnings.append(f"cube z dropped by {cube_drop:.5f}m during hold")

    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    print(f"Model path: {args.model_path}")
    print(f"Gravity: {np.round(gravity, 5).tolist()}")
    print(f"Empty sim contacts after {EMPTY_SIM_S:.1f}s: {empty_contacts}")
    print()
    print("Loop site distances at open:")
    for name, dist in replay["open_site_distances"].items():
        print(f"  {name}: {dist * 1000:.3f} mm")
    print("Loop site distances after close:")
    for name, dist in replay["closed_site_distances"].items():
        print(f"  {name}: {dist * 1000:.3f} mm")
    print(f"  max during replay: {max_site_distance * 1000:.3f} mm")
    print()
    print("Initial task contacts:")
    print(f"  cube-table contacts: {initial['table_contacts']}")
    print(f"  cube-finger initial contacts: {initial['cube_finger_initial']}")
    for geom1, geom2, dist in initial["contact_pairs"]:
        print(f"  {geom1} <-> {geom2}, dist={dist:.6f}")
    print()
    print("Replay grasp contacts at close:")
    print(f"  left finger contacts: {replay['left_contacts_at_close']}")
    print(f"  right finger contacts: {replay['right_contacts_at_close']}")
    for geom1, geom2, dist in replay["contact_pairs_at_close"]:
        if CUBE_COLLISION_GEOM in (geom1, geom2) or LEFT_FINGER_GEOM in (geom1, geom2) or RIGHT_FINGER_GEOM in (geom1, geom2):
            print(f"  {geom1} <-> {geom2}, dist={dist:.6f}")
    print()
    print("Replay stability:")
    print(f"  placed cube: {np.round(replay['placed_cube'], 5).tolist()}")
    print(f"  hold start cube: {np.round(replay['hold_start'], 5).tolist()}")
    print(f"  final cube: {np.round(replay['final_cube'], 5).tolist()}")
    print(f"  hold z drop: {cube_drop:.5f} m")
    print(f"  max hold displacement: {final_displacement:.5f} m")
    print()

    print_bool("Scene load", report.scene_load)
    print_bool("No NaN / no explosion", report.no_nan_no_explosion)
    print_bool("Gripper connect stability", report.gripper_connect_stability)
    print_bool("Finger-cube contact", report.finger_cube_contact)
    print_bool("Cube-table contact", report.cube_table_contact)
    print_bool("Finger-table obvious penetration", report.finger_table_obvious_penetration)
    print_bool("Cube stable during hold", report.cube_stable_during_hold)
    print_bool("Replay consistency", report.replay_consistency)

    print()
    print("Final result:")
    print(f"READY_FOR_VLA_DATA = {report.ready_for_vla_data}")
    if report.warnings:
        print()
        for warning in report.warnings:
            print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
