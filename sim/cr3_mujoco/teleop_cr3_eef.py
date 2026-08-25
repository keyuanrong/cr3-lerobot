#!/usr/bin/env python

"""Keyboard end-effector teleoperation for CR3 + Lebai LMG-90 in MuJoCo.

This follows the same high-level idea as lerobot-mujoco-tutorial:
keyboard changes an end-effector target, IK converts it to arm joints, and a
single gripper scalar opens/closes the gripper.

Default mode intentionally mimics lerobot-mujoco-tutorial while avoiding
MuJoCo viewer shortcut conflicts:
- position target is the midpoint of the two finger collision geoms;
- keyboard updates that target by small increments;
- IK uses the current joint target as its initial guess and applies the result;
- a minimal GLFW viewer captures key states directly, so commands do not need
  Enter and do not trigger MuJoCo official viewer shortcuts;
- extra rejection/clamping is only enabled with --safe-mode.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import queue
import sys
import threading
import time

import glfw
import mujoco
import numpy as np


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))

from sim.cr3_mujoco.joint_mapping import policy_action_to_mujoco_ctrl


ROOT = Path(__file__).resolve().parent
SCENE_XML = ROOT / "cr3_scene.xml"

ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
ARM_ACTUATORS = tuple(f"{joint}_pos" for joint in ARM_JOINTS)
LEFT_GRIPPER_ACTUATOR = "Left_joint1_pos"
RIGHT_GRIPPER_ACTUATOR = "Right_joint_pos"
LEFT_FINGER_GEOM = "left_static1_finger_collision"
RIGHT_FINGER_GEOM = "right_static1_finger_collision"
CUBE_JOINT = "cube_free"
CUBE_BODY = "cube"
CUBE_GEOM = "cube_collision"
BLOCKS = (
    ("red", "cube", "cube_free", "cube_collision"),
    ("yellow", "yellow_cube", "yellow_cube_free", "yellow_cube_collision"),
    ("green", "green_cube", "green_cube_free", "green_cube_collision"),
)
IK_ORIENTATION_BODY = "Link6"
TABLE_TOP_Z = 0.31
CUBE_SIZE = 0.025

GRIPPER_OPEN_CMD = 0.12
GRIPPER_GRASP_CMD = 0.65
GRIPPER_RATE_DEFAULT = 0.55
GRIPPER_FAST_OPEN = True
GRASP_ASSIST_MIN_CMD = 0.20
CONTACT_STOP_MIN_CMD = 0.42
GRASP_ASSIST_CENTER_THRESH_M = 0.055
GRASP_ASSIST_Z_THRESH_M = 0.050
CENTER_OK_XY_THRESH_M = 0.012
CENTER_OK_Z_THRESH_M = 0.020
GRASP_RELEASE_DROP_M = 0.080
INITIAL_SETTLE_STEPS = 80
# Use the previous ACT rollout arm posture, but keep the gripper open for EEF
# teleop startup. In this mapping gripper=0 means hard closed, which can fight
# the closed-loop gripper constraints and visually deform the linkage.
DEFAULT_INITIAL_DISPLAY_STATE = "0,-5,-97,-8,91,180,1"

KEY_ESC = 256


class MinimalGLFWViewer:
    """Small MuJoCo GLFW viewer with tutorial-style key-state polling."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, title: str = "CR3 EEF Teleop") -> None:
        self.model = model
        self.data = data
        self._keys_down: set[int] = set()
        self._keys_once: set[int] = set()
        self._button_left = False
        self._button_right = False
        self._last_x = 0.0
        self._last_y = 0.0

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor) if monitor is not None else None
        width = int(mode.size.width * 0.82) if mode is not None else 1280
        height = int(mode.size.height * 0.82) if mode is not None else 800
        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        fb_width, fb_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, fb_width, fb_height)

    def close(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def poll(self) -> None:
        glfw.poll_events()

    def is_key_down(self, key: int) -> bool:
        return key in self._keys_down

    def pop_key_once(self, key: int) -> bool:
        if key not in self._keys_once:
            return False
        self._keys_once.remove(key)
        return True

    def render(self, overlay_text: str = "", camera_overlays: tuple[str, ...] = ()) -> None:
        if self.window is None:
            return
        width, height = glfw.get_framebuffer_size(self.window)
        self.viewport.left = 0
        self.viewport.bottom = 0
        self.viewport.width = width
        self.viewport.height = height
        mujoco.mjr_rectangle(self.viewport, 0.0, 0.0, 0.0, 1.0)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        self._render_camera_overlays(width, height, camera_overlays)
        if overlay_text:
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                self.viewport,
                "CR3 teleop",
                overlay_text,
                self.context,
            )
        glfw.swap_buffers(self.window)

    def _render_camera_overlays(self, width: int, height: int, camera_names: tuple[str, ...]) -> None:
        overlay_width = max(220, width // 4)
        overlay_height = int(overlay_width * 9 / 16)
        margin = 12
        for idx, camera_name in enumerate(camera_names):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                continue
            small_cam = mujoco.MjvCamera()
            small_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            small_cam.fixedcamid = camera_id
            if idx == 0:
                bottom = height - overlay_height - margin
            elif idx == 1:
                bottom = margin
            else:
                bottom = margin + (idx - 1) * (overlay_height + margin)
            viewport = mujoco.MjrRect(
                width - overlay_width - margin,
                bottom,
                overlay_width,
                overlay_height,
            )
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.opt,
                None,
                small_cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                camera_name,
                "",
                self.context,
            )

    def _key_callback(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        del window, scancode, mods
        if action == glfw.PRESS:
            self._keys_down.add(key)
            self._keys_once.add(key)
        elif action == glfw.RELEASE:
            self._keys_down.discard(key)
        elif action == glfw.REPEAT:
            self._keys_down.add(key)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

    def _cursor_pos_callback(self, window, xpos: float, ypos: float) -> None:
        del window
        dx = xpos - self._last_x
        dy = ypos - self._last_y
        self._last_x = xpos
        self._last_y = ypos
        if not (self._button_left or self._button_right):
            return
        width, height = glfw.get_framebuffer_size(self.window)
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if self._button_left else mujoco.mjtMouse.mjMOUSE_MOVE_H
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(
            self.window, glfw.KEY_RIGHT_SHIFT
        ) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if self._button_left else mujoco.mjtMouse.mjMOUSE_MOVE_V
        mujoco.mjv_moveCamera(self.model, action, dx / max(height, 1), dy / max(height, 1), self.scene, self.cam)

    def _mouse_button_callback(self, window, button: int, action: int, mods: int) -> None:
        del mods
        self._button_left = button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
        self._button_right = button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS
        self._last_x, self._last_y = glfw.get_cursor_pos(window)

    def _scroll_callback(self, window, x_offset: float, y_offset: float) -> None:
        del window, x_offset
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * y_offset, self.scene, self.cam)


def unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return fallback.astype(np.float64).copy()
    return vec / norm


def obj_id(model: mujoco.MjModel, objtype: int, name: str) -> int:
    index = mujoco.mj_name2id(model, objtype, name)
    if index < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(index)


def arm_qpos_addrs(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_qposadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)] for joint in ARM_JOINTS],
        dtype=np.int32,
    )


def arm_dof_addrs(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_dofadr[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)] for joint in ARM_JOINTS],
        dtype=np.int32,
    )


def arm_joint_limits(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [model.jnt_range[obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)] for joint in ARM_JOINTS],
        dtype=np.float64,
    )


def arm_actuator_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) for actuator in ARM_ACTUATORS],
        dtype=np.int32,
    )


def gripper_actuator_ids(model: mujoco.MjModel) -> tuple[int, int]:
    return (
        obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_GRIPPER_ACTUATOR),
        obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_GRIPPER_ACTUATOR),
    )


def gripper_joint_values(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    left_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "Left_joint1")
    right_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "Right_joint")
    left_qpos = float(data.qpos[int(model.jnt_qposadr[left_joint])])
    right_qpos = float(data.qpos[int(model.jnt_qposadr[right_joint])])
    return left_qpos, right_qpos


def set_gripper(data: mujoco.MjData, left_id: int, right_id: int, cmd: float) -> None:
    data.ctrl[left_id] = float(cmd)
    data.ctrl[right_id] = -float(cmd)


def parse_seven(value: str) -> np.ndarray:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 7:
        raise ValueError(f"Expected 7 comma-separated values, got {len(values)}: {value}")
    return np.asarray(values, dtype=np.float32)


def dataset_initial_state(dataset_root: Path, episode_index: int, frame_index: int | None = None) -> np.ndarray:
    import pyarrow.parquet as pq

    data_dir = dataset_root / "data"
    parquet_paths = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "observation.state"])
        episodes = np.asarray(table["episode_index"])
        matches = np.where(episodes == episode_index)[0]
        if matches.size == 0:
            continue
        frame_indices = np.asarray(table["frame_index"])[matches]
        if frame_index is None:
            row = int(matches[np.argmin(frame_indices)])
        else:
            frame_matches = matches[np.where(frame_indices == frame_index)[0]]
            if frame_matches.size == 0:
                continue
            row = int(frame_matches[0])
        state = np.asarray(table["observation.state"][row].as_py(), dtype=np.float32)
        if state.shape[0] != 7:
            raise ValueError(f"Expected observation.state shape (7,), got {state.shape}.")
        initial = state.copy()
        # Older CR3 ACT datasets store gripper as 0..100 opening width; policy_action_to_mujoco_ctrl expects 0..1.
        if initial[6] > 1.5:
            initial[6] = 1.0 if initial[6] >= 50.0 else 0.0
        return initial

    raise ValueError(f"Episode {episode_index} frame {frame_index} not found in {data_dir}.")


def set_display_state(model: mujoco.MjModel, data: mujoco.MjData, display_state: np.ndarray) -> None:
    ctrl = policy_action_to_mujoco_ctrl(
        display_state,
        model.actuator_ctrlrange,
        current_ctrl=data.ctrl,
    )
    for actuator_id in range(min(6, model.nu)):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        low, high = model.jnt_range[joint_id]
        value = float(ctrl[actuator_id])
        while value < low:
            value += 2.0 * np.pi
        while value > high:
            value -= 2.0 * np.pi
        ctrl[actuator_id] = float(np.clip(value, low, high))
    data.ctrl[:] = ctrl

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        qvel_addr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_addr] = ctrl[actuator_id]
        data.qvel[qvel_addr] = 0.0
    mujoco.mj_forward(model, data)


def finger_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    return 0.5 * (np.asarray(data.geom_xpos[left_id]) + np.asarray(data.geom_xpos[right_id]))


def grasp_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    # The gripper grasp point is the midpoint of the two task finger collision
    # pads. Grasp assist snaps the cube here instead of preserving a bad offset.
    return finger_center(model, data)


def finger_center_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    jacp_left = np.zeros((3, model.nv), dtype=np.float64)
    jacr_left = np.zeros((3, model.nv), dtype=np.float64)
    jacp_right = np.zeros((3, model.nv), dtype=np.float64)
    jacr_right = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacGeom(model, data, jacp_left, jacr_left, left_id)
    mujoco.mj_jacGeom(model, data, jacp_right, jacr_right, right_id)
    return 0.5 * (jacp_left + jacp_right)


def body_rotation(model: mujoco.MjModel, data: mujoco.MjData, body_name: str = IK_ORIENTATION_BODY) -> np.ndarray:
    body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()


def orientation_error(current_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    # Small-angle orientation error in world coordinates. The sign convention is
    # chosen so the DLS step reduces current_rot -> target_rot for MuJoCo jacr.
    return 0.5 * (
        np.cross(target_rot[:, 0], current_rot[:, 0])
        + np.cross(target_rot[:, 1], current_rot[:, 1])
        + np.cross(target_rot[:, 2], current_rot[:, 2])
    )


def link6_rotation_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, IK_ORIENTATION_BODY)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jacr


def solve_6d_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    seed_qpos: np.ndarray,
    *,
    iterations: int,
    damping: float,
    max_step: float,
    pos_tolerance: float,
    rot_tolerance: float,
    orientation_weight: float,
    joint_regularization: float,
    joint_weights: np.ndarray,
    max_total_delta: float,
) -> tuple[np.ndarray, float, float, bool]:
    qpos_addrs = arm_qpos_addrs(model)
    dof_addrs = arm_dof_addrs(model)
    limits = arm_joint_limits(model)
    qpos = np.clip(seed_qpos.astype(np.float64).copy(), limits[:, 0], limits[:, 1])
    last_pos_err = float("inf")
    last_rot_err = float("inf")

    for _ in range(iterations):
        data.qpos[qpos_addrs] = qpos
        data.qvel[dof_addrs] = 0.0
        mujoco.mj_forward(model, data)
        pos_err = target_pos - finger_center(model, data)
        rot_err = orientation_error(body_rotation(model, data), target_rot) if orientation_weight > 0 else np.zeros(3)
        last_pos_err = float(np.linalg.norm(pos_err))
        last_rot_err = float(np.linalg.norm(rot_err))
        if last_pos_err < pos_tolerance and last_rot_err < rot_tolerance:
            return qpos, last_pos_err, last_rot_err, True

        jac_pos = finger_center_jacobian(model, data)[:, dof_addrs]
        jac_rot = link6_rotation_jacobian(model, data)[:, dof_addrs] if orientation_weight > 0 else np.zeros((3, 6))
        err = np.concatenate([pos_err, orientation_weight * rot_err])
        jac = np.vstack([jac_pos, orientation_weight * jac_rot])
        regularizer = (damping * damping) * np.eye(6)
        regularizer += joint_regularization * np.diag(joint_weights * joint_weights)
        lhs = jac.T @ jac + regularizer
        rhs = jac.T @ err
        dq = np.linalg.solve(lhs, rhs)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_step:
            dq *= max_step / dq_norm
        qpos = np.clip(qpos + dq, limits[:, 0], limits[:, 1])
        total_delta = qpos - seed_qpos
        total_delta_norm = float(np.linalg.norm(total_delta))
        if total_delta_norm > max_total_delta:
            qpos = seed_qpos + total_delta * (max_total_delta / total_delta_norm)
            qpos = np.clip(qpos, limits[:, 0], limits[:, 1])

    data.qpos[qpos_addrs] = qpos
    data.qvel[dof_addrs] = 0.0
    mujoco.mj_forward(model, data)
    last_pos_err = float(np.linalg.norm(target_pos - finger_center(model, data)))
    last_rot_err = (
        float(np.linalg.norm(orientation_error(body_rotation(model, data), target_rot))) if orientation_weight > 0 else 0.0
    )
    return qpos, last_pos_err, last_rot_err, last_pos_err < pos_tolerance * 2.5


def solve_random_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    seed_qpos: np.ndarray,
    *,
    tries: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    qpos_addrs = arm_qpos_addrs(model)
    dof_addrs = arm_dof_addrs(model)
    limits = arm_joint_limits(model)
    original_qpos = np.asarray(data.qpos[qpos_addrs], dtype=np.float64).copy()

    def score(qpos: np.ndarray) -> float:
        data.qpos[qpos_addrs] = qpos
        data.qvel[dof_addrs] = 0.0
        mujoco.mj_forward(model, data)
        delta = finger_center(model, data) - target
        # Keep solutions near the previous posture to avoid branch jumping.
        return float(delta @ delta + 0.0015 * np.mean((qpos - seed_qpos) ** 2))

    best = np.clip(seed_qpos.astype(np.float64).copy(), limits[:, 0], limits[:, 1])
    best_score = score(best)
    scales = np.asarray([0.25, 0.20, 0.25, 0.30, 0.30, 0.40], dtype=np.float64)
    for idx in range(tries):
        temperature = 1.0 - 0.85 * (idx / max(tries - 1, 1))
        candidate = best + rng.normal(0.0, scales * temperature)
        candidate = np.clip(candidate, limits[:, 0], limits[:, 1])
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score

    data.qpos[qpos_addrs] = best
    data.qvel[dof_addrs] = 0.0
    mujoco.mj_forward(model, data)
    err_norm = float(np.linalg.norm(target - finger_center(model, data)))
    data.qpos[qpos_addrs] = original_qpos
    data.qvel[dof_addrs] = 0.0
    mujoco.mj_forward(model, data)
    return best, err_norm


def place_cube_between_fingers(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    cube_joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    qpos_addr = int(model.jnt_qposadr[cube_joint_id])
    qvel_addr = int(model.jnt_dofadr[cube_joint_id])
    pos = grasp_center(model, data).copy()
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return pos


def cube_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    if body_id < 0:
        return None
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def block_center(model: mujoco.MjModel, data: mujoco.MjData, block: tuple[str, str, str, str]) -> np.ndarray | None:
    _label, body_name, _joint_name, _geom_name = block
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def cube_qpos_qvel_addrs(model: mujoco.MjModel) -> tuple[int, int]:
    joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def block_qpos_qvel_addrs(model: mujoco.MjModel, block: tuple[str, str, str, str]) -> tuple[int, int]:
    _label, _body_name, joint_name, _geom_name = block
    joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def nearest_grasp_block(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[tuple[str, str, str, str] | None, np.ndarray, float]:
    center = grasp_center(model, data)
    best_block = None
    best_delta = np.zeros(3, dtype=np.float64)
    best_score = float("inf")
    for block in BLOCKS:
        pos = block_center(model, data, block)
        if pos is None:
            continue
        delta = pos - center
        score = float(np.linalg.norm(delta))
        if score < best_score:
            best_block = block
            best_delta = delta
            best_score = score
    return best_block, best_delta, best_score


def cube_alignment_text(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    block, delta, _score = nearest_grasp_block(model, data)
    if block is None:
        return "block=missing"
    return f"{block[0]}_delta_mm={np.round(delta * 1000.0, 1).tolist()}"


def finger_cube_contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool, int]:
    block, _delta, _score = nearest_grasp_block(model, data)
    geom_name = CUBE_GEOM if block is None else block[3]
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    if min(cube_id, left_id, right_id) < 0:
        return False, False, 0

    left_contact = False
    right_contact = False
    contact_count = 0
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if cube_id not in geom_pair:
            continue
        if left_id in geom_pair:
            left_contact = True
            contact_count += 1
        if right_id in geom_pair:
            right_contact = True
            contact_count += 1
    return left_contact, right_contact, contact_count


class GraspAssist:
    def __init__(self) -> None:
        self.active = False
        self.block: tuple[str, str, str, str] | None = None
        self.state = "OPEN"

    def release(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.active:
            self.state = "RELEASED"
            return
        self.active = False
        block = self.block
        self.block = None
        self.state = "RELEASED"
        if block is None:
            return
        qpos_addr, qvel_addr = block_qpos_qvel_addrs(model, block)
        center = grasp_center(model, data)
        release_pos = center + np.asarray([0.0, 0.0, -GRASP_RELEASE_DROP_M])
        min_release_z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + 0.002
        release_pos[2] = max(float(release_pos[2]), min_release_z)
        data.qpos[qpos_addr : qpos_addr + 3] = release_pos
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        gripper_cmd: float,
        gripper_target: float,
        enabled: bool,
    ) -> bool:
        if not enabled:
            self.active = False
            self.block = None
            self.state = "OPEN"
            return False

        if self.active and self.block is not None:
            block = self.block
            delta = np.zeros(3, dtype=np.float64)
        else:
            block, delta, _score = nearest_grasp_block(model, data)

        if block is None:
            self.active = False
            self.block = None
            self.state = "OPEN"
            return False

        if gripper_target <= GRIPPER_OPEN_CMD + 0.04:
            self.release(model, data)
            return False

        if not self.active:
            self.state = "CLOSING"

        near_grasp = (
            float(np.linalg.norm(delta[:2])) <= GRASP_ASSIST_CENTER_THRESH_M
            and abs(float(delta[2])) <= GRASP_ASSIST_Z_THRESH_M
        )
        should_attach = gripper_cmd >= GRASP_ASSIST_MIN_CMD and near_grasp
        if not self.active and should_attach:
            self.active = True
            self.block = block
            self.state = f"HOLDING({block[0]})"

        if not self.active:
            return False

        self.state = f"HOLDING({block[0]})"

        center = grasp_center(model, data)
        qpos_addr, qvel_addr = block_qpos_qvel_addrs(model, block)
        data.qpos[qpos_addr : qpos_addr + 3] = center
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        mujoco.mj_forward(model, data)
        return True


def teleop_overlay_text(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    gripper_cmd: float,
    gripper_target: float,
    step_m: float,
    ik_pos_err: float,
    grasp_assist_active: bool,
    grasp_state: str,
) -> str:
    block, delta, _score = nearest_grasp_block(model, data)
    block_label = "none" if block is None else block[0]
    if block is None:
        cube_delta = np.zeros(3)
        cube_delta_text = "missing"
    else:
        cube_delta = delta * 1000.0
        cube_delta_text = f"{cube_delta[0]:+.1f}, {cube_delta[1]:+.1f}, {cube_delta[2]:+.1f} mm"
    left_contact, right_contact, contact_count = finger_cube_contact_flags(model, data)
    left_joint, right_joint = gripper_joint_values(model, data)
    if block is None:
        centered = "missing"
    elif float(np.linalg.norm(cube_delta[:2])) <= CENTER_OK_XY_THRESH_M * 1000.0 and abs(float(cube_delta[2])) <= (
        CENTER_OK_Z_THRESH_M * 1000.0
    ):
        centered = "OK"
    elif abs(float(cube_delta[2])) > CENTER_OK_Z_THRESH_M * 1000.0:
        centered = "LOW/HIGH"
    else:
        centered = "OFFSET"
    grasp = "BOTH" if left_contact and right_contact else "NO"
    if left_contact and right_contact:
        hint = "hold"
    elif left_contact or right_contact:
        hint = "one-sided contact: re-center or raise/lower"
    elif centered == "OK":
        hint = "centered but no contact: keep closing"
    elif centered == "LOW/HIGH":
        hint = "raise/lower cube into the grasp box"
    else:
        hint = "move cube/fingers to center first"
    return (
        f"WASD/RF move | O/P slow gripper | V print | Q quit\n"
        f"gripper cmd: {gripper_cmd:.3f} -> {gripper_target:.3f} "
        f"joint: {left_joint:.3f}/{right_joint:.3f}   step: {step_m * 1000:.1f} mm\n"
        f"nearest: {block_label}  block - grasp: {cube_delta_text}   center: {centered}\n"
        f"finger contact cube: L:{'Y' if left_contact else 'N'} R:{'Y' if right_contact else 'N'} "
        f"contacts:{contact_count} grasp:{grasp}\n"
        f"state:{grasp_state} | ik pos err:{ik_pos_err * 1000:.1f} mm | "
        f"assist:{'ON' if grasp_assist_active else 'off'} | {hint}"
    )


def move_towards(current: float, target: float, max_delta: float) -> float:
    if abs(target - current) <= max_delta:
        return float(target)
    return float(current + math.copysign(max_delta, target - current))


def print_help(control_frame: str, *, terminal_commands: bool = False) -> None:
    if terminal_commands:
        move_hint = f"terminal commands w/s, a/d, r/f + Enter ({control_frame} frame)"
    else:
        move_hint = f"hold W/S, A/D, R/F in the MuJoCo window ({control_frame} frame)"
    print(
        "\nCR3 EEF teleop controls:\n"
        f"  move: {move_hint}\n"
        "  o/p: smoothly open / close gripper\n"
        "  c: debug-place red cube between fingers\n"
        "  z: reset target to current finger center\n"
        "  [ / ]: halve / double translation step\n"
        "  v: print target/finger status\n"
        "  h: print this help\n"
        "  q: quit\n",
        flush=True,
    )


def line_command_collector(key_queue: queue.SimpleQueue[int], *, terminal_commands: bool) -> None:
    if not terminal_commands:
        return
    print(
        "Terminal command mode: type w/a/s/d/r/f/o/p/c/z/v/h/q then Enter. "
        "Repeats work, e.g. 'www' or 'ddp'.",
        flush=True,
    )
    mapping = {
        "w": "i",
        "s": "k",
        "a": "j",
        "d": "l",
        "r": "u",
        "f": "m",
    }
    while True:
        try:
            line = input("eef> ").strip()
        except EOFError:
            key_queue.put(ord("q"))
            return
        if not line:
            continue
        if line.lower() in {"help", "h"}:
            key_queue.put(ord("h"))
            continue
        for char in line:
            mapped = mapping.get(char.lower(), char)
            key_queue.put(ord(mapped))
            if mapped.lower() == "q":
                return


def glfw_control_keys(viewer: MinimalGLFWViewer) -> list[int]:
    keys: list[int] = []
    repeat_mapping = {
        glfw.KEY_W: "i",
        glfw.KEY_S: "k",
        glfw.KEY_A: "j",
        glfw.KEY_D: "l",
        glfw.KEY_R: "u",
        glfw.KEY_F: "m",
    }
    once_mapping = {
        glfw.KEY_O: "o",
        glfw.KEY_P: "p",
        glfw.KEY_SPACE: " ",
        glfw.KEY_C: "c",
        glfw.KEY_Z: "z",
        glfw.KEY_LEFT_BRACKET: "[",
        glfw.KEY_RIGHT_BRACKET: "]",
        glfw.KEY_H: "h",
        glfw.KEY_V: "v",
        glfw.KEY_Q: "q",
        glfw.KEY_ESCAPE: chr(KEY_ESC),
    }
    for glfw_key, mapped in repeat_mapping.items():
        if viewer.is_key_down(glfw_key):
            keys.append(ord(mapped))
    for glfw_key, mapped in once_mapping.items():
        if viewer.pop_key_once(glfw_key):
            keys.append(ord(mapped))
    return keys


def apply_key(
    key: int,
    target: np.ndarray,
    step_m: float,
    gripper_cmd: float,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_name: str | None,
    control_frame: str,
) -> tuple[bool, float, float]:
    running = True
    right = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    up = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if control_frame == "camera" and camera_name is not None:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id >= 0:
            camera_xmat = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
            right = unit(np.asarray(camera_xmat[0], dtype=np.float64) * np.asarray([1.0, 1.0, 0.0]), right)
            up = unit(np.asarray(camera_xmat[1], dtype=np.float64) * np.asarray([1.0, 1.0, 0.0]), up)
    elif control_frame == "world":
        right = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        up = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    if key in (KEY_ESC, ord("q"), ord("Q")):
        running = False
    elif key in (ord("i"), ord("I")):
        target[:] = target + up * step_m
    elif key in (ord("k"), ord("K")):
        target[:] = target - up * step_m
    elif key in (ord("j"), ord("J")):
        target[:] = target - right * step_m
    elif key in (ord("l"), ord("L")):
        target[:] = target + right * step_m
    elif key in (ord("u"), ord("U")):
        target[2] += step_m
    elif key in (ord("m"), ord("M")):
        target[2] -= step_m
    elif key in (ord("o"), ord("O")):
        gripper_cmd = GRIPPER_OPEN_CMD
    elif key in (ord("p"), ord("P")):
        gripper_cmd = GRIPPER_GRASP_CMD
    elif key == ord(" "):
        midpoint = 0.5 * (GRIPPER_OPEN_CMD + GRIPPER_GRASP_CMD)
        gripper_cmd = GRIPPER_OPEN_CMD if gripper_cmd > midpoint else GRIPPER_GRASP_CMD
    elif key in (ord("c"), ord("C")):
        pos = place_cube_between_fingers(model, data)
        print(f"placed red cube at {np.round(pos, 4).tolist()}", flush=True)
    elif key in (ord("z"), ord("Z")):
        target[:] = grasp_center(model, data)
        print(f"target reset to {np.round(target, 4).tolist()}", flush=True)
    elif key == ord("["):
        step_m = max(step_m * 0.5, 0.0005)
        print(f"step={step_m * 1000:.1f}mm", flush=True)
    elif key == ord("]"):
        step_m = min(step_m * 2.0, 0.05)
        print(f"step={step_m * 1000:.1f}mm", flush=True)
    elif key in (ord("h"), ord("H")):
        print_help(control_frame)
    elif key in (ord("v"), ord("V")):
        left_contact, right_contact, contact_count = finger_cube_contact_flags(model, data)
        print(
            f"target={np.round(target, 4).tolist()} "
            f"grasp={np.round(grasp_center(model, data), 4).tolist()} "
            f"{cube_alignment_text(model, data)} "
            f"finger_cube_contact=L:{left_contact} R:{right_contact} n:{contact_count}",
            flush=True,
        )
    return running, step_m, gripper_cmd


def clamp_workspace(target: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.clip(target, bounds[:, 0], bounds[:, 1])


def parse_workspace(value: str) -> np.ndarray:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("Expected xmin,xmax,ymin,ymax,zmin,zmax")
    return np.asarray([[parts[0], parts[1]], [parts[2], parts[3]], [parts[4], parts[5]]], dtype=np.float64)


def parse_joint_weights(value: str) -> np.ndarray:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("Expected 6 comma-separated joint weights")
    return np.asarray(parts, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    parser.add_argument(
        "--viewer-camera",
        default=None,
        help="Optional fixed camera, e.g. front or wrist. Omit for free viewer camera.",
    )
    parser.add_argument(
        "--camera-overlays",
        default="front,wrist",
        help="Comma-separated fixed camera mini views drawn in the top-right corner. Use '' to disable, e.g. front,wrist.",
    )
    parser.add_argument("--control-frame", choices=["camera", "world"], default="world")
    parser.add_argument("--step-m", type=float, default=0.003, help="Translation step per key press, in meters.")
    parser.add_argument("--ik-iterations", type=int, default=20)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-max-step", type=float, default=0.035, help="Max joint update per IK iteration, rad.")
    parser.add_argument("--ik-pos-tolerance", type=float, default=0.001, help="Position IK tolerance, meters.")
    parser.add_argument("--ik-rot-tolerance", type=float, default=10.0, help="Orientation IK tolerance, radians.")
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=0.0,
        help="0 means position-only IK. Increase gently, e.g. 0.02, after position control feels good.",
    )
    parser.add_argument("--joint-regularization", type=float, default=0.002)
    parser.add_argument("--max-total-joint-delta", type=float, default=10.0, help="Max total IK change per key update, rad.")
    parser.add_argument(
        "--joint-weights",
        type=parse_joint_weights,
        default=parse_joint_weights("1,1,1,1.5,2,10"),
        help="Weighted IK regularization for J1..J6. J6 is high to avoid wrist spin.",
    )
    parser.add_argument("--reject-pos-error", type=float, default=999.0, help="Only used with --safe-mode.")
    parser.add_argument("--max-target-from-finger", type=float, default=0.08, help="Max target lead distance, meters.")
    parser.add_argument(
        "--workspace",
        type=parse_workspace,
        default=parse_workspace("-1.15,0.20,-0.35,0.35,0.30,1.20"),
        help="xmin,xmax,ymin,ymax,zmin,zmax workspace clamp in world coordinates.",
    )
    parser.add_argument(
        "--ik-method",
        choices=["dls6d", "random"],
        default="dls6d",
        help="dls6d is the final/default teleop IK; random is kept only as a fallback.",
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Enable workspace clamp, target lead clamp, and IK rejection. Default mimics the tutorial: solve and apply.",
    )
    parser.add_argument("--ik-tries", type=int, default=600, help="Random IK samples per changed target.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--loop-hz", type=float, default=60.0)
    parser.add_argument(
        "--physics-steps-per-frame",
        type=int,
        default=16,
        help="MuJoCo mj_step calls per rendered frame. 16 at 60Hz is close to real time for timestep=0.001.",
    )
    parser.add_argument(
        "--gripper-rate",
        type=float,
        default=GRIPPER_RATE_DEFAULT,
        help="Max gripper command change per second. Lower values reduce pushing during grasp.",
    )
    parser.add_argument(
        "--no-contact-stop",
        action="store_true",
        help="Keep closing to the grasp target even after both fingers contact the cube.",
    )
    parser.add_argument(
        "--no-grasp-assist",
        action="store_true",
        help="Disable teleop grasp assist that keeps a centered cube attached while the gripper is closed.",
    )
    parser.add_argument("--status-period", type=float, default=0.0, help="Seconds between status prints. 0 disables periodic prints.")
    parser.add_argument(
        "--viewer-hotkeys",
        action="store_true",
        help="Deprecated; this script now uses its own GLFW viewer to avoid MuJoCo shortcut conflicts.",
    )
    parser.add_argument(
        "--terminal-commands",
        action="store_true",
        help="Use Enter-based terminal commands as a fallback if the separate control window cannot open.",
    )
    parser.add_argument(
        "--use-xml-initial-state",
        action="store_true",
        help="Keep the XML default posture instead of the CR3 dataset first-frame posture.",
    )
    parser.add_argument(
        "--initial-display-state",
        default=DEFAULT_INITIAL_DISPLAY_STATE,
        help="7D display state J1..J6 degrees plus gripper scalar. Default matches the previous ACT rollout start.",
    )
    parser.add_argument("--initial-from-dataset-root", type=Path, default=None)
    parser.add_argument("--initial-from-dataset-episode", type=int, default=0)
    parser.add_argument("--initial-from-dataset-frame", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing MuJoCo scene: {args.model_path}")

    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not args.use_xml_initial_state:
        if args.initial_from_dataset_root is not None:
            initial_display_state = dataset_initial_state(
                args.initial_from_dataset_root,
                args.initial_from_dataset_episode,
                args.initial_from_dataset_frame,
            )
            print(f"using dataset initial display state: {np.round(initial_display_state, 3).tolist()}", flush=True)
        else:
            initial_display_state = parse_seven(args.initial_display_state)
            print(f"using initial display state: {np.round(initial_display_state, 3).tolist()}", flush=True)
        set_display_state(model, data, initial_display_state)

    arm_ids = arm_actuator_ids(model)
    qpos_addrs = arm_qpos_addrs(model)
    dof_addrs = arm_dof_addrs(model)
    left_id, right_id = gripper_actuator_ids(model)

    qpos_target = np.asarray(data.qpos[qpos_addrs], dtype=np.float64).copy()
    data.ctrl[arm_ids] = qpos_target
    gripper_cmd = GRIPPER_OPEN_CMD
    gripper_target = GRIPPER_OPEN_CMD
    set_gripper(data, left_id, right_id, gripper_cmd)
    for _ in range(INITIAL_SETTLE_STEPS):
        data.ctrl[arm_ids] = qpos_target
        set_gripper(data, left_id, right_id, gripper_cmd)
        mujoco.mj_step(model, data)
    target = grasp_center(model, data).copy()
    target_rot = body_rotation(model, data)
    key_queue: queue.SimpleQueue[int] = queue.SimpleQueue()
    step_m = float(args.step_m)
    running = True
    target_dirty = True
    ik_pos_err = 0.0
    ik_rot_err = 0.0
    rng = np.random.default_rng(args.seed)
    next_status = time.perf_counter()
    grasp_assist = GraspAssist()
    grasp_assist_active = False

    print_help(args.control_frame, terminal_commands=args.terminal_commands)
    print(f"initial target={np.round(target, 4).tolist()} step={step_m * 1000:.1f}mm", flush=True)
    camera_overlays = tuple(item.strip() for item in args.camera_overlays.split(",") if item.strip())
    if camera_overlays:
        print(f"camera overlays: {camera_overlays}", flush=True)

    if args.terminal_commands:
        command_thread = threading.Thread(
            target=line_command_collector,
            args=(key_queue,),
            kwargs={"terminal_commands": True},
            daemon=True,
        )
        command_thread.start()

    viewer = MinimalGLFWViewer(model, data)
    try:
        if args.viewer_camera is not None:
            camera_id = obj_id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.viewer_camera)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
        else:
            viewer.cam.lookat[:] = (-0.15, 0.0, 0.45)
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = -80
            viewer.cam.elevation = -35

        dt = 1.0 / args.loop_hz
        while viewer.is_running() and running:
            start = time.perf_counter()
            viewer.poll()
            previous_qpos_target = qpos_target.copy()
            pending_keys: list[int] = []
            while True:
                try:
                    pending_keys.append(key_queue.get_nowait())
                except queue.Empty:
                    break
            if not args.terminal_commands:
                pending_keys.extend(glfw_control_keys(viewer))
            for key in pending_keys:
                old_target = target.copy()
                old_step = step_m
                old_gripper_target = gripper_target
                running, step_m, gripper_target = apply_key(
                    key,
                    target,
                    step_m,
                    gripper_target,
                    model,
                    data,
                    camera_name=args.viewer_camera,
                    control_frame=args.control_frame,
                )
                if gripper_target <= GRIPPER_OPEN_CMD + 0.04:
                    grasp_assist.release(model, data)
                    if GRIPPER_FAST_OPEN:
                        gripper_cmd = GRIPPER_OPEN_CMD
                        set_gripper(data, left_id, right_id, gripper_cmd)
                if args.safe_mode:
                    target[:] = clamp_workspace(target, args.workspace)
                    target_lead = target - grasp_center(model, data)
                    target_lead_norm = float(np.linalg.norm(target_lead))
                    if target_lead_norm > args.max_target_from_finger:
                        target[:] = grasp_center(model, data) + target_lead * (
                            args.max_target_from_finger / target_lead_norm
                        )
                if not np.allclose(old_target, target) or old_step != step_m or old_gripper_target != gripper_target:
                    target_dirty = True

            if target_dirty:
                if args.ik_method == "dls6d":
                    candidate_qpos, ik_pos_err, ik_rot_err, ik_ok = solve_6d_ik(
                        model,
                        data,
                        target,
                        target_rot,
                        previous_qpos_target,
                        iterations=args.ik_iterations,
                        damping=args.ik_damping,
                        max_step=args.ik_max_step,
                        pos_tolerance=args.ik_pos_tolerance,
                        rot_tolerance=args.ik_rot_tolerance,
                        orientation_weight=args.orientation_weight,
                        joint_regularization=args.joint_regularization,
                        joint_weights=args.joint_weights,
                        max_total_delta=args.max_total_joint_delta,
                    )
                else:
                    candidate_qpos, ik_pos_err = solve_random_position_ik(
                        model,
                        data,
                        target,
                        previous_qpos_target,
                        tries=args.ik_tries,
                        rng=rng,
                    )
                    ik_rot_err = 0.0
                    ik_ok = True
                if not args.safe_mode or (ik_ok and ik_pos_err <= args.reject_pos_error):
                    qpos_target = candidate_qpos
                else:
                    qpos_target = previous_qpos_target
                    if args.status_period > 0:
                        print(
                            f"\rIK rejected: pos_err={ik_pos_err * 1000:.1f}mm "
                            f"rot_err={ik_rot_err:.3f}rad target restored",
                            flush=True,
                        )
                target_dirty = False
            data.qpos[qpos_addrs] = qpos_target
            data.qvel[dof_addrs] = 0.0
            data.ctrl[arm_ids] = qpos_target
            gripper_cmd = move_towards(gripper_cmd, gripper_target, max(float(args.gripper_rate) * dt, 0.0))
            set_gripper(data, left_id, right_id, gripper_cmd)
            for _ in range(max(int(args.physics_steps_per_frame), 1)):
                data.qpos[qpos_addrs] = qpos_target
                data.qvel[dof_addrs] = 0.0
                data.ctrl[arm_ids] = qpos_target
                set_gripper(data, left_id, right_id, gripper_cmd)
                mujoco.mj_step(model, data)
                grasp_assist_active = grasp_assist.update(
                    model,
                    data,
                    gripper_cmd=gripper_cmd,
                    gripper_target=gripper_target,
                    enabled=not args.no_grasp_assist,
                )
            viewer.render(
                teleop_overlay_text(
                    model,
                    data,
                    gripper_cmd=gripper_cmd,
                    gripper_target=gripper_target,
                    step_m=step_m,
                    ik_pos_err=ik_pos_err,
                    grasp_assist_active=grasp_assist_active,
                    grasp_state=grasp_assist.state,
                ),
                camera_overlays,
            )

            now = time.perf_counter()
            if args.status_period > 0 and now >= next_status:
                print(
                    f"target={np.round(target, 4).tolist()} "
                    f"grasp={np.round(grasp_center(model, data), 4).tolist()} "
                    f"ik_pos={ik_pos_err * 1000:.1f}mm ik_rot={ik_rot_err:.3f}rad "
                    f"gripper={gripper_cmd:.2f}->{gripper_target:.2f} "
                    f"{cube_alignment_text(model, data)}",
                    flush=True,
                )
                next_status = now + args.status_period

            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
