#!/usr/bin/env python

"""Terminal joint controller for CR3 sim and optional real robot.

Controls:
  Up/Down or j/k     select joint
  Left/Right or -/+  decrement/increment selected joint
  1..6               select joint directly
  s                  send current target once
  h                  hold/send continuously toggle
  o/p or g           open/stable-grasp sim gripper
  c                  place cube between current sim fingers
  v                  print sim grasp/collision status
  z                  zero all targets
  r                  read real robot joints into targets (real mode only)
  q                  quit

Examples:
  # Sim only.
  python examples/terminal_cr3_joint_control.py --mode sim

  # Real robot only. This will move the robot.
  python examples/terminal_cr3_joint_control.py --mode real --enable-real --robot-ip 192.168.6.1
"""

from __future__ import annotations

import argparse
import csv
import curses
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

LEROBOT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.envs.my_mujoco import MyMujocoEnv
from sim.cr3_mujoco.joint_mapping import (
    DISPLAY_TO_SIM_JOINT_OFFSET_DEG,
    DISPLAY_TO_SIM_JOINT_SIGN,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    display_to_sim_deg,
    gripper_to_finger_ctrl,
    parse_six,
    sim_to_display_deg,
)


DEFAULT_MODEL_PATH = Path("sim/cr3_mujoco/cr3_scene.xml")
JOINT_NAMES = [f"J{i}" for i in range(1, 7)]
JOINT_LOW_DEG = np.asarray([-360, -180, -180, -180, -180, -360], dtype=np.float32)
JOINT_HIGH_DEG = np.asarray([360, 180, 180, 180, 180, 360], dtype=np.float32)
GRIPPER_ACTUATOR_NAMES = ("Left_joint1_pos", "Right_joint_pos")
GRIPPER_GRASP_CMD = 0.50
GRIPPER_GRASP_CMD_MAX = 0.65
LEFT_FINGER_GEOM = "left_static1_finger_collision"
RIGHT_FINGER_GEOM = "right_static1_finger_collision"
CUBE_BODY = "cube"
CUBE_JOINT = "cube_free"
CUBE_COLLISION_GEOM = "cube_collision"
TABLE_COLLISION_GEOM = "table_top_collision"
LOOP_SITE_PAIRS = (
    ("left_link2_pin_site", "left_static_pin_site"),
    ("left_link2_axis_site", "left_static_axis_site"),
    ("right_link2_pin_site", "right_static_pin_site"),
    ("right_link2_axis_site", "right_static_axis_site"),
)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def move_toward(current: float, target: float, max_delta: float) -> float:
    if abs(target - current) <= max_delta:
        return target
    return current + max_delta if target > current else current - max_delta


def gripper_open_fraction_to_cmd(open_fraction: float) -> float:
    open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
    return GRIPPER_CLOSED + open_fraction * (GRIPPER_OPEN - GRIPPER_CLOSED)


def gripper_cmd_to_open_fraction(cmd: float) -> float:
    cmd = float(np.clip(cmd, GRIPPER_OPEN, GRIPPER_CLOSED))
    return float(np.clip((cmd - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0, 1.0))


class SimController:
    def __init__(
        self,
        model_path: Path,
        joint_sign: np.ndarray,
        joint_offset_deg: np.ndarray,
        use_joint_mapping: bool,
        robot_z_offset: float,
        show_viewer: bool,
        viewer_camera: str | None,
    ):
        self.env = MyMujocoEnv(model_path=str(model_path), obs_type="state", action_dim=8, state_dim=8)
        self.env.reset()
        self.joint_sign = joint_sign
        self.joint_offset_deg = joint_offset_deg
        self.use_joint_mapping = use_joint_mapping
        self.viewer = None
        self.viewer_context = None
        self.viewer_key_queue: list[int] = []
        if robot_z_offset:
            body_id = self.env._mujoco.mj_name2id(
                self.env._model,
                self.env._mujoco.mjtObj.mjOBJ_BODY,
                "cr3_base_mount",
            )
            if body_id < 0:
                raise ValueError("Body not found: cr3_base_mount")
            self.env._model.body_pos[body_id, 2] += robot_z_offset
            self.env._mujoco.mj_forward(self.env._model, self.env._data)

        if show_viewer:
            import mujoco.viewer

            def key_callback(key: int) -> None:
                self.viewer_key_queue.append(int(key))

            self.viewer_context = mujoco.viewer.launch_passive(
                self.env._model,
                self.env._data,
                key_callback=key_callback,
            )
            self.viewer = self.viewer_context.__enter__()
            if viewer_camera is not None:
                camera_id = self.env._mujoco.mj_name2id(
                    self.env._model,
                    self.env._mujoco.mjtObj.mjOBJ_CAMERA,
                    viewer_camera,
                )
                if camera_id < 0:
                    raise ValueError(f"Camera not found: {viewer_camera}")
                self.viewer.cam.type = self.env._mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = camera_id
            else:
                self.viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
                self.viewer.cam.distance = 1.9
                self.viewer.cam.azimuth = -90
                self.viewer.cam.elevation = -55

    def pop_viewer_keys(self) -> list[int]:
        keys = self.viewer_key_queue
        self.viewer_key_queue = []
        return keys

    def actuator_id(self, name: str) -> int:
        actuator_id = self.env._mujoco.mj_name2id(
            self.env._model,
            self.env._mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if actuator_id < 0:
            raise ValueError(f"Actuator not found: {name}")
        return int(actuator_id)

    def object_id(self, objtype: int, name: str) -> int:
        obj_id = self.env._mujoco.mj_name2id(self.env._model, objtype, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(obj_id)

    def close(self) -> None:
        if self.viewer_context is not None:
            self.viewer_context.__exit__(None, None, None)
            self.viewer_context = None
            self.viewer = None
        self.env.close()

    def internal_current_deg(self) -> np.ndarray:
        joints = np.zeros(6, dtype=np.float32)
        for idx, name in enumerate(JOINT_NAMES):
            joint_id = self.env._mujoco.mj_name2id(self.env._model, self.env._mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = int(self.env._model.jnt_qposadr[joint_id])
            joints[idx] = math.degrees(float(self.env._data.qpos[qpos_addr]))
        return joints

    def current_deg(self) -> np.ndarray:
        joints = self.internal_current_deg()
        if not self.use_joint_mapping:
            return joints
        return sim_to_display_deg(joints, self.joint_sign, self.joint_offset_deg)

    def send(self, target_deg: np.ndarray) -> None:
        if self.use_joint_mapping:
            sim_deg = display_to_sim_deg(target_deg, self.joint_sign, self.joint_offset_deg)
        else:
            sim_deg = np.asarray(target_deg[:6], dtype=np.float32)
        ctrl = self.env._data.ctrl.copy()
        ctrl[:6] = np.deg2rad(sim_deg)
        limited = np.asarray(self.env._model.actuator_ctrllimited, dtype=bool)
        if limited.any():
            low = self.env._model.actuator_ctrlrange[:, 0]
            high = self.env._model.actuator_ctrlrange[:, 1]
            ctrl[limited] = np.clip(ctrl[limited], low[limited], high[limited])
        self.env._data.ctrl[:] = ctrl

    def set_gripper(self, open_fraction: float) -> None:
        targets = gripper_to_finger_ctrl(float(open_fraction), self.env._model.actuator_ctrlrange)
        ctrl = self.env._data.ctrl.copy()
        for actuator_name, target in zip(GRIPPER_ACTUATOR_NAMES, targets, strict=True):
            actuator_id = self.actuator_id(actuator_name)
            low, high = self.env._model.actuator_ctrlrange[actuator_id]
            ctrl[actuator_id] = float(np.clip(target, low, high))
        self.env._data.ctrl[:] = ctrl

    def place_cube_between_fingers(self) -> np.ndarray:
        left_geom_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
        right_geom_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
        cube_joint_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
        qpos_addr = int(self.env._model.jnt_qposadr[cube_joint_id])
        qvel_addr = int(self.env._model.jnt_dofadr[cube_joint_id])
        pos = 0.5 * (self.env._data.geom_xpos[left_geom_id] + self.env._data.geom_xpos[right_geom_id])
        self.env._data.qpos[qpos_addr : qpos_addr + 3] = pos
        self.env._data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.env._data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        self.env._mujoco.mj_forward(self.env._model, self.env._data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
        return np.asarray(pos, dtype=np.float64)

    def _contact_count(self, geom_a: str, geom_b: str) -> int:
        geom_a_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_GEOM, geom_a)
        geom_b_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_GEOM, geom_b)
        count = 0
        for idx in range(self.env._data.ncon):
            contact = self.env._data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (geom1 == geom_a_id and geom2 == geom_b_id) or (geom1 == geom_b_id and geom2 == geom_a_id):
                count += 1
        return count

    def _site_distance(self, site_a: str, site_b: str) -> float:
        site_a_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_SITE, site_a)
        site_b_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_SITE, site_b)
        delta = self.env._data.site_xpos[site_a_id] - self.env._data.site_xpos[site_b_id]
        return float(np.linalg.norm(delta))

    def grasp_status(self) -> str:
        cube_body_id = self.object_id(self.env._mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
        cube_pos = np.asarray(self.env._data.xpos[cube_body_id], dtype=np.float64)
        left_contacts = self._contact_count(CUBE_COLLISION_GEOM, LEFT_FINGER_GEOM)
        right_contacts = self._contact_count(CUBE_COLLISION_GEOM, RIGHT_FINGER_GEOM)
        table_contacts = self._contact_count(CUBE_COLLISION_GEOM, TABLE_COLLISION_GEOM)
        max_site_mm = max(self._site_distance(a, b) for a, b in LOOP_SITE_PAIRS) * 1000.0
        return (
            f"cube={np.round(cube_pos, 4).tolist()} "
            f"contacts L/R/table={left_contacts}/{right_contacts}/{table_contacts} "
            f"max_site={max_site_mm:.3f}mm ncon={self.env._data.ncon}"
        )

    def snap_to(self, target_deg: np.ndarray) -> None:
        self.send(target_deg)
        for idx, name in enumerate(JOINT_NAMES):
            joint_id = self.env._mujoco.mj_name2id(self.env._model, self.env._mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr = int(self.env._model.jnt_qposadr[joint_id])
            qvel_addr = int(self.env._model.jnt_dofadr[joint_id])
            self.env._data.qpos[qpos_addr] = self.env._data.ctrl[idx]
            self.env._data.qvel[qvel_addr] = 0.0
        self.env._mujoco.mj_forward(self.env._model, self.env._data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def step(self, n: int = 10) -> None:
        for _ in range(n):
            self.env._mujoco.mj_step(self.env._model, self.env._data)
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()


class RealController:
    def __init__(self, robot_ip: str, speed_factor: int, use_gripper: bool, read_only: bool):
        from lerobot.robots.dobot_cr3 import DobotCR3, DobotCR3Config
        from lerobot.robots.dobot_cr3.dobot_api import DobotApiDashboard

        cfg = DobotCR3Config(
            robot_ip=robot_ip,
            speed_factor=speed_factor,
            use_gripper=use_gripper,
            enable_robot_on_connect=not read_only,
        )
        self.robot = DobotCR3(cfg)
        self.read_only = read_only
        if read_only:
            self.robot.dashboard = DobotApiDashboard(cfg.robot_ip, cfg.dashboard_port)
            return
        try:
            self.robot.connect()
        except Exception:
            self.robot.disconnect()
            raise

    def close(self) -> None:
        self.robot.disconnect()

    def current_deg(self) -> np.ndarray:
        return np.asarray(self.robot.get_joints(), dtype=np.float32)

    def send(self, target_deg: np.ndarray) -> None:
        if self.read_only:
            raise RuntimeError("Real robot is connected in read-only mode.")
        self.robot.move.JointMovJ(*[float(v) for v in target_deg])


def draw(
    stdscr: Any,
    *,
    mode: str,
    selected: int,
    targets: np.ndarray,
    step_deg: float,
    hold: bool,
    gripper_open: float,
    gripper_target: float,
    gripper_grasp: float,
    sim_current: np.ndarray | None,
    real_current: np.ndarray | None,
    last_msg: str,
) -> None:
    height, width = stdscr.getmaxyx()

    def safe_addstr(row: int, col: int, text: str) -> None:
        if row < 0 or row >= height or col >= width:
            return
        try:
            stdscr.addstr(row, col, text[: max(width - col - 1, 0)])
        except curses.error:
            pass

    stdscr.erase()
    safe_addstr(0, 0, "CR3 Joint Terminal Controller")
    safe_addstr(
        1,
        0,
        f"mode={mode}  step={step_deg:.3f} deg  hold={'on' if hold else 'off'}  "
        f"gripper_cmd={gripper_open_fraction_to_cmd(gripper_open):.2f}->{gripper_open_fraction_to_cmd(gripper_target):.2f}",
    )
    safe_addstr(2, 0, "keys: up/down select | left/right +/- adjust | 1..6 select | s send | h hold | z zero | q quit")
    safe_addstr(
        3,
        0,
        f"      o open | p grasp({gripper_grasp:.2f}) | g toggle | c cube-to-fingers | v status | [/] step | r read | l log",
    )
    safe_addstr(5, 0, "Joint    Target(deg)      Sim(deg)       Real(deg)")
    for idx, name in enumerate(JOINT_NAMES):
        marker = ">" if idx == selected else " "
        sim_val = "   n/a" if sim_current is None else f"{sim_current[idx]:8.3f}"
        real_val = "   n/a" if real_current is None else f"{real_current[idx]:8.3f}"
        safe_addstr(6 + idx, 0, f"{marker} {name:<4} {targets[idx]:12.3f} {sim_val:>13} {real_val:>13}")
    safe_addstr(min(14, height - 1), 0, f"status: {last_msg[:100]}")
    stdscr.refresh()


def append_calibration_sample(path: Path, targets: np.ndarray, sim_current: np.ndarray, real_current: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = ["time_s"]
    fieldnames += [f"target_j{i}" for i in range(1, 7)]
    fieldnames += [f"sim_j{i}" for i in range(1, 7)]
    fieldnames += [f"real_j{i}" for i in range(1, 7)]

    row = {"time_s": f"{time.time():.3f}"}
    row.update({f"target_j{i + 1}": f"{float(v):.6f}" for i, v in enumerate(targets)})
    row.update({f"sim_j{i + 1}": f"{float(v):.6f}" for i, v in enumerate(sim_current)})
    row.update({f"real_j{i + 1}": f"{float(v):.6f}" for i, v in enumerate(real_current)})

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sim", "real", "both"], default="sim")
    parser.add_argument("--enable-real", action="store_true", help="Required for mode=real/both.")
    parser.add_argument("--robot-ip", default="192.168.6.1")
    parser.add_argument("--speed-factor", type=int, default=10)
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument(
        "--real-read-only",
        action="store_true",
        help="Read real robot joints without EnableRobot and never send real motion commands.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sim-robot-z-offset", type=float, default=0.0, help="Raise/lower the sim robot base in meters.")
    parser.add_argument(
        "--lock-sim-to-target",
        action="store_true",
        help="Kinematically snap sim qpos to the target each frame; useful for calibration without gravity sag.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Do not open the MuJoCo viewer in sim/both mode.")
    parser.add_argument("--viewer-camera", default=None, help="Optional fixed sim camera, e.g. front or wrist.")
    parser.add_argument("--initial-deg", default="0,0,0,0,0,0")
    parser.add_argument("--initial-gripper", type=float, default=1.0, help="Initial sim gripper open fraction, 0=closed, 1=open.")
    parser.add_argument(
        "--grasp-cmd",
        type=float,
        default=GRIPPER_GRASP_CMD,
        help="Manual sim grasp command in actuator radians; capped at 0.65.",
    )
    parser.add_argument("--open-speed", type=float, default=0.5, help="Opening speed in open-fraction per second.")
    parser.add_argument("--close-speed", type=float, default=1.0, help="Closing speed in open-fraction per second.")
    parser.add_argument("--step-deg", type=float, default=1.0)
    parser.add_argument("--joint-sign", default=DISPLAY_TO_SIM_JOINT_SIGN)
    parser.add_argument("--joint-offset-deg", default=DISPLAY_TO_SIM_JOINT_OFFSET_DEG)
    parser.add_argument(
        "--use-joint-mapping",
        action="store_true",
        help="Map Dobot/display joint angles into MuJoCo joint angles. Default is to write angles directly.",
    )
    parser.add_argument("--calib-log", type=Path, default=Path("cr3_calibration_samples.csv"))
    parser.add_argument("--loop-hz", type=float, default=20.0)
    return parser


def run_tui(stdscr: Any, args: argparse.Namespace) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    targets = parse_six(args.initial_deg)
    gripper_open = float(np.clip(args.initial_gripper, 0.0, 1.0))
    gripper_target = gripper_open
    gripper_grasp_cmd = float(np.clip(args.grasp_cmd, GRIPPER_OPEN, GRIPPER_GRASP_CMD_MAX))
    gripper_grasp = gripper_cmd_to_open_fraction(gripper_grasp_cmd)
    joint_sign = parse_six(args.joint_sign, default=1.0)
    joint_offset_deg = parse_six(args.joint_offset_deg)
    selected = 0
    step_deg = args.step_deg
    hold = args.mode == "sim"
    last_msg = "ready"

    sim: SimController | None = None
    real: RealController | None = None
    real_current: np.ndarray | None = None
    last_real_poll = 0.0

    if args.mode in ("sim", "both"):
        sim = SimController(
            args.model_path,
            joint_sign,
            joint_offset_deg,
            use_joint_mapping=args.use_joint_mapping,
            robot_z_offset=args.sim_robot_z_offset,
            show_viewer=not args.no_viewer,
            viewer_camera=args.viewer_camera,
        )
        if args.mode == "sim" and not args.use_joint_mapping and args.initial_deg == "0,0,0,0,0,0":
            targets[:] = sim.current_deg()
        sim.set_gripper(gripper_open)
        last_msg = "sim connected; viewer open" if not args.no_viewer else "sim connected; viewer disabled"

    if args.mode in ("real", "both"):
        if not args.enable_real:
            raise RuntimeError("Refusing to move the real robot without --enable-real.")
        real_read_only = args.real_read_only
        try:
            real = RealController(args.robot_ip, args.speed_factor, args.use_gripper, read_only=real_read_only)
        except RuntimeError as exc:
            if "EnableRobot failed" not in str(exc):
                raise
            real_read_only = True
            real = RealController(args.robot_ip, args.speed_factor, args.use_gripper, read_only=True)
        real_current = real.current_deg()
        last_real_poll = time.perf_counter()
        targets[:] = real_current
        if sim is not None:
            sim.snap_to(targets)
            sim.set_gripper(gripper_open)
        if real_read_only:
            last_msg = f"real read-only at {args.robot_ip}; targets initialized from real joints"
        else:
            last_msg = f"real connected at {args.robot_ip}; targets initialized from real joints"

    dt = 1.0 / args.loop_hz
    try:
        while True:
            start = time.perf_counter()
            terminal_key = stdscr.getch()
            keys = [] if terminal_key == -1 else [terminal_key]
            if sim is not None:
                keys.extend(sim.pop_viewer_keys())
            send_now = False
            log_now = False
            should_quit = False

            for key in keys:
                if key in (ord("q"), 27):
                    should_quit = True
                    break
                if key in (curses.KEY_UP, ord("k")):
                    selected = (selected - 1) % 6
                elif key in (curses.KEY_DOWN, ord("j")):
                    selected = (selected + 1) % 6
                elif key in (curses.KEY_LEFT, ord("-")):
                    targets[selected] -= step_deg
                    send_now = True
                elif key in (curses.KEY_RIGHT, ord("+"), ord("=")):
                    targets[selected] += step_deg
                    send_now = True
                elif ord("1") <= key <= ord("6"):
                    selected = key - ord("1")
                elif key == ord("s"):
                    send_now = True
                elif key == ord("h"):
                    hold = not hold
                    last_msg = f"hold {'on' if hold else 'off'}"
                elif key in (ord("o"), ord("O")):
                    gripper_target = 1.0
                    last_msg = "sim gripper opened"
                elif key in (ord("p"), ord("P")):
                    gripper_target = gripper_grasp
                    last_msg = f"sim gripper grasp cmd={gripper_grasp_cmd:.2f}"
                elif key in (ord("g"), ord("G")):
                    midpoint = 0.5 * (1.0 + gripper_grasp)
                    gripper_target = gripper_grasp if gripper_target >= midpoint else 1.0
                    last_msg = "sim gripper opened" if gripper_target >= midpoint else f"sim gripper grasp cmd={gripper_grasp_cmd:.2f}"
                elif key in (ord("c"), ord("C")):
                    if sim is None:
                        last_msg = "cube placement is sim-only"
                    else:
                        pos = sim.place_cube_between_fingers()
                        last_msg = f"cube placed between fingers at {np.round(pos, 4).tolist()}"
                elif key in (ord("v"), ord("V")):
                    if sim is None:
                        last_msg = "grasp status is sim-only"
                    else:
                        last_msg = sim.grasp_status()
                elif key == ord("z"):
                    targets[:] = 0.0
                    send_now = True
                elif key == ord("["):
                    step_deg = max(step_deg * 0.5, 0.001)
                elif key == ord("]"):
                    step_deg = step_deg * 2.0
                elif key == ord("r") and real is not None:
                    real_current = real.current_deg()
                    last_real_poll = time.perf_counter()
                    targets[:] = real_current
                    last_msg = "targets read from real robot"
                elif key == ord("l"):
                    log_now = True

            if should_quit:
                break

            targets[:] = np.clip(targets, JOINT_LOW_DEG, JOINT_HIGH_DEG)
            if gripper_target > gripper_open:
                gripper_speed = args.open_speed
            elif gripper_target < gripper_open:
                gripper_speed = args.close_speed
            else:
                gripper_speed = 0.0
            gripper_open = move_toward(gripper_open, gripper_target, gripper_speed * dt)
            if sim is not None:
                sim.set_gripper(gripper_open)

            if send_now or hold:
                if sim is not None:
                    if args.lock_sim_to_target:
                        sim.snap_to(targets)
                    else:
                        sim.send(targets)
                    sim.set_gripper(gripper_open)
                    last_msg = "sent target to sim"
                if real is not None and send_now:
                    if real.read_only:
                        last_msg = "real robot is read-only; not sent"
                    else:
                        real.send(targets)
                        last_msg = "sent target to real robot"

            if sim is not None and args.lock_sim_to_target:
                sim.snap_to(targets)
                sim.set_gripper(gripper_open)
            elif sim is not None:
                sim.step()

            sim_current = sim.current_deg() if sim is not None else None
            if real is not None and time.perf_counter() - last_real_poll >= 0.25:
                real_current = real.current_deg()
                last_real_poll = time.perf_counter()
            if log_now:
                if sim_current is None or real_current is None:
                    last_msg = "need both sim and real values before logging"
                else:
                    append_calibration_sample(args.calib_log, targets, sim_current, real_current)
                    last_msg = f"logged calibration sample to {args.calib_log}"

            draw(
                stdscr,
                mode=args.mode,
                selected=selected,
                targets=targets,
                step_deg=step_deg,
                hold=hold,
                gripper_open=gripper_open,
                gripper_target=gripper_target,
                gripper_grasp=gripper_grasp_cmd,
                sim_current=sim_current,
                real_current=real_current,
                last_msg=last_msg,
            )

            time.sleep(max(dt - (time.perf_counter() - start), 0.0))
    finally:
        if sim is not None:
            sim.close()
        if real is not None:
            real.close()


def main() -> None:
    args = build_parser().parse_args()
    curses.wrapper(run_tui, args)


if __name__ == "__main__":
    main()
