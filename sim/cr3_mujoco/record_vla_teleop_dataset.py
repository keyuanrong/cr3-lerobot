#!/usr/bin/env python

"""Teleop-record a CR3 + LMG-90 language dataset for SmolVLA / pi0.

Controls:
  W/S, A/D, R/F: move the gripper target
  O/P or Space: open / close gripper with grasp assist
  1: start / pause recording frames
  2: save current recorded frames as one episode
  3: discard current recorded frames and reset scene
  0: reset scene for the next episode
  V: print status
  Q or Esc: quit

The saved dataset is model-agnostic LeRobot format:
  observation.images.front_rgb
  observation.images.wrist_rgb
  observation.state          7D joint+gripper state
  action                     7D joint+gripper command
  task                       language instruction
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

import glfw
import mujoco
import numpy as np


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.datasets import LeRobotDataset
from sim.cr3_mujoco.build_cr3_scene import (
    CUBE_START_POS,
    GREEN_CUBE_START_POS,
    TARGET_CENTER,
    TARGET_OUTER_SIZE,
    YELLOW_CUBE_START_POS,
)
from sim.cr3_mujoco.collect_stack_blocks_vla_dataset import CameraRenderer, make_features, tuple_hw
from sim.cr3_mujoco.teleop_cr3_eef import (
    ARM_JOINTS,
    DEFAULT_INITIAL_DISPLAY_STATE,
    GRIPPER_FAST_OPEN,
    GRIPPER_OPEN_CMD,
    GRIPPER_RATE_DEFAULT,
    KEY_ESC,
    SCENE_XML,
    GraspAssist,
    MinimalGLFWViewer,
    apply_key,
    arm_actuator_ids,
    arm_dof_addrs,
    arm_qpos_addrs,
    block_qpos_qvel_addrs,
    body_rotation,
    grasp_center,
    gripper_actuator_ids,
    move_towards,
    parse_seven,
    set_display_state,
    set_gripper,
    solve_6d_ik,
)


TASK_DESCRIPTION = (
    "put the red block into the black frame, then put the green block into the black frame, "
    "finally put the yellow block into the black frame"
)
DEFAULT_REPO_ID = "local/cr3_lmg90_stack_blocks_vla_teleop_clean"
DEFAULT_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla_teleop_clean"
DEFAULT_FPS = 30
DEFAULT_FRONT_SIZE = (480, 640)
DEFAULT_WRIST_SIZE = (480, 640)
INITIAL_RESET_SETTLE_STEPS = 30
RANDOMIZE_BLOCKS_ON_RESET = True
BLOCK_RANDOM_RADIUS_RANGE_M = (0.02, 0.05)
MIN_BLOCK_DISTANCE_M = 0.065
TARGET_CLEARANCE_M = 0.045
TABLE_X_RANGE_M = (-0.50, -0.08)
TABLE_Y_RANGE_M = (-0.18, 0.16)
BLOCK_RESET_SPECS = (
    ("red", "cube", "cube_free", "cube_collision", np.asarray(CUBE_START_POS, dtype=np.float64)),
    ("yellow", "yellow_cube", "yellow_cube_free", "yellow_cube_collision", np.asarray(YELLOW_CUBE_START_POS, dtype=np.float64)),
    ("green", "green_cube", "green_cube_free", "green_cube_collision", np.asarray(GREEN_CUBE_START_POS, dtype=np.float64)),
)


def state_vector(model: mujoco.MjModel, data: mujoco.MjData, gripper_cmd: float) -> np.ndarray:
    state = np.empty(7, dtype=np.float32)
    state[:6] = data.qpos[arm_qpos_addrs(model)].astype(np.float32)
    state[6] = np.float32(gripper_cmd)
    return state


def action_vector(qpos_target: np.ndarray, gripper_target: float) -> np.ndarray:
    action = np.empty(7, dtype=np.float32)
    action[:6] = qpos_target.astype(np.float32)
    action[6] = np.float32(gripper_target)
    return action


def add_dataset_frame(
    dataset: LeRobotDataset,
    renderer: CameraRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_target: np.ndarray,
    gripper_target: float,
    gripper_cmd: float,
    front_size: tuple[int, int],
    wrist_size: tuple[int, int],
    task: str,
) -> None:
    front_h, front_w = front_size
    wrist_h, wrist_w = wrist_size
    dataset.add_frame(
        {
            "observation.images.front_rgb": renderer.render(data, "front", front_h, front_w),
            "observation.images.wrist_rgb": renderer.render(data, "wrist", wrist_h, wrist_w),
            "observation.state": state_vector(model, data, gripper_cmd),
            "action": action_vector(qpos_target, gripper_target),
            "task": task,
        }
    )


def restore_viewer_context(viewer: MinimalGLFWViewer) -> None:
    if viewer.window is not None:
        glfw.make_context_current(viewer.window)


def flush_saved_episode(dataset: LeRobotDataset) -> None:
    flush_metadata = getattr(dataset.meta, "_flush_metadata_buffer", None)
    if callable(flush_metadata):
        flush_metadata()


def target_clearance_ok(pos_xy: np.ndarray) -> bool:
    half_x = TARGET_OUTER_SIZE[0] / 2.0 + TARGET_CLEARANCE_M
    half_y = TARGET_OUTER_SIZE[1] / 2.0 + TARGET_CLEARANCE_M
    return not (
        abs(float(pos_xy[0]) - TARGET_CENTER[0]) < half_x
        and abs(float(pos_xy[1]) - TARGET_CENTER[1]) < half_y
    )


def sample_block_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    for _ in range(300):
        positions: dict[str, np.ndarray] = {}
        ok = True
        for label, _body, _joint, _geom, base_pos in BLOCK_RESET_SPECS:
            for _attempt in range(80):
                radius = rng.uniform(*BLOCK_RANDOM_RADIUS_RANGE_M)
                angle = rng.uniform(0.0, 2.0 * np.pi)
                candidate = base_pos.copy()
                candidate[0] += radius * np.cos(angle)
                candidate[1] += radius * np.sin(angle)
                candidate[0] = np.clip(candidate[0], TABLE_X_RANGE_M[0], TABLE_X_RANGE_M[1])
                candidate[1] = np.clip(candidate[1], TABLE_Y_RANGE_M[0], TABLE_Y_RANGE_M[1])
                candidate[2] = base_pos[2]

                if not target_clearance_ok(candidate[:2]):
                    continue
                if any(np.linalg.norm(candidate[:2] - prev[:2]) < MIN_BLOCK_DISTANCE_M for prev in positions.values()):
                    continue
                positions[label] = candidate
                break
            else:
                ok = False
                break
        if ok and len(positions) == len(BLOCK_RESET_SPECS):
            return positions
    return {label: base_pos.copy() for label, _body, _joint, _geom, base_pos in BLOCK_RESET_SPECS}


def randomize_block_positions(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict[str, np.ndarray]:
    positions = sample_block_positions(rng)
    for block in BLOCK_RESET_SPECS:
        label = block[0]
        qpos_addr, qvel_addr = block_qpos_qvel_addrs(model, block[:4])
        data.qpos[qpos_addr : qpos_addr + 3] = positions[label]
        data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return positions


def restore_scene_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    arm_ids: np.ndarray,
    qpos_addrs: np.ndarray,
    left_gripper_id: int,
    right_gripper_id: int,
    rng: np.random.Generator,
    randomize_blocks: bool,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)

    if randomize_blocks:
        randomize_block_positions(model, data, rng)

    qpos_target = np.asarray(data.qpos[qpos_addrs], dtype=np.float64).copy()
    data.ctrl[arm_ids] = qpos_target
    gripper_cmd = GRIPPER_OPEN_CMD
    gripper_target = GRIPPER_OPEN_CMD
    set_gripper(data, left_gripper_id, right_gripper_id, gripper_cmd)

    for _ in range(INITIAL_RESET_SETTLE_STEPS):
        data.qpos[qpos_addrs] = qpos_target
        data.ctrl[arm_ids] = qpos_target
        set_gripper(data, left_gripper_id, right_gripper_id, gripper_cmd)
        mujoco.mj_step(model, data)

    target = grasp_center(model, data).copy()
    target_rot = body_rotation(model, data)
    return qpos_target, gripper_cmd, gripper_target, target, target_rot


def control_keys(viewer: MinimalGLFWViewer) -> list[int]:
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


def recorder_keys(viewer: MinimalGLFWViewer) -> list[str]:
    events: list[str] = []
    if viewer.pop_key_once(glfw.KEY_0):
        events.append("reset_scene")
    if viewer.pop_key_once(glfw.KEY_1):
        events.append("toggle_record")
    if viewer.pop_key_once(glfw.KEY_2):
        events.append("save")
    if viewer.pop_key_once(glfw.KEY_3):
        events.append("discard")
    return events


def overlay_text(
    *,
    recording: bool,
    frame_count: int,
    saved_episodes: int,
    fps: int,
    last_event: str,
) -> str:
    rec = "REC" if recording else "PAUSED"
    return (
        f"DATASET: {rec} | frames:{frame_count} | saved:{saved_episodes} | fps:{fps}\n"
        f"1 rec/pause | 2 save+reset | 3 discard+reset | 0 reset | Q quit\n"
        f"{last_event}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--front-size", type=tuple_hw, default=DEFAULT_FRONT_SIZE, help="HEIGHT,WIDTH")
    parser.add_argument("--wrist-size", type=tuple_hw, default=DEFAULT_WRIST_SIZE, help="HEIGHT,WIDTH")
    parser.add_argument("--task", default=TASK_DESCRIPTION)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated safety alias. Existing datasets are resumed, not deleted.",
    )
    parser.add_argument(
        "--reset-dataset",
        action="store_true",
        help="DANGER: delete the existing dataset directory before recording.",
    )
    parser.add_argument("--no-videos", action="store_true", help="Store images instead of videos.")
    parser.add_argument("--loop-hz", type=float, default=60.0)
    parser.add_argument("--physics-steps-per-frame", type=int, default=16)
    parser.add_argument("--gripper-rate", type=float, default=GRIPPER_RATE_DEFAULT)
    parser.add_argument("--step-m", type=float, default=0.003)
    parser.add_argument("--camera-overlays", default="front,wrist")
    parser.add_argument("--initial-display-state", default=DEFAULT_INITIAL_DISPLAY_STATE)
    parser.add_argument("--no-randomize-blocks", action="store_true", help="Keep block positions fixed after scene reset.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for randomized block reset positions.")
    parser.add_argument("--ik-iterations", type=int, default=20)
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument("--ik-max-step", type=float, default=0.035)
    parser.add_argument("--ik-pos-tolerance", type=float, default=0.001)
    parser.add_argument("--ik-rot-tolerance", type=float, default=10.0)
    parser.add_argument("--orientation-weight", type=float, default=0.0)
    parser.add_argument("--joint-regularization", type=float, default=0.002)
    parser.add_argument("--max-total-joint-delta", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing MuJoCo scene: {args.model_path}")

    if args.root.exists():
        if args.reset_dataset:
            print(f"RESET DATASET: deleting existing dataset at {args.root}", flush=True)
            shutil.rmtree(args.root)
        else:
            if args.overwrite:
                print(
                    "NOTE: --overwrite is now safe and will NOT delete existing saved episodes. "
                    "Appending with LeRobotDataset.resume(). Use --reset-dataset only if you truly want to delete data.",
                    flush=True,
                )

    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.root.exists():
        dataset = LeRobotDataset.resume(
            repo_id=args.repo_id,
            root=args.root,
            image_writer_threads=4,
        )
        dataset.meta._metadata_buffer_size = 1
        if dataset.fps != args.fps:
            print(f"NOTE: existing dataset fps={dataset.fps}; using existing fps instead of requested {args.fps}.")
            args.fps = dataset.fps
        print(
            f"Appending to existing dataset: episodes={dataset.num_episodes}, frames={dataset.num_frames}",
            flush=True,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.root,
            fps=args.fps,
            features=make_features(args.front_size, args.wrist_size),
            robot_type="cr3_lmg90_mujoco",
            use_videos=not args.no_videos,
            image_writer_threads=4,
            metadata_buffer_size=1,
        )
        print(f"Created new dataset: {args.root}", flush=True)
    renderer = CameraRenderer(model)
    rng = np.random.default_rng(args.random_seed)
    randomize_blocks = RANDOMIZE_BLOCKS_ON_RESET and not args.no_randomize_blocks

    initial_display_state = parse_seven(args.initial_display_state)
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

    for _ in range(80):
        data.ctrl[arm_ids] = qpos_target
        set_gripper(data, left_id, right_id, gripper_cmd)
        mujoco.mj_step(model, data)

    reset_qpos = data.qpos.copy()
    reset_qvel = data.qvel.copy()
    reset_ctrl = data.ctrl.copy()
    qpos_target, gripper_cmd, gripper_target, target, target_rot = restore_scene_snapshot(
        model,
        data,
        qpos=reset_qpos,
        qvel=reset_qvel,
        ctrl=reset_ctrl,
        arm_ids=arm_ids,
        qpos_addrs=qpos_addrs,
        left_gripper_id=left_id,
        right_gripper_id=right_id,
        rng=rng,
        randomize_blocks=randomize_blocks,
    )
    joint_weights = np.asarray([1, 1, 1, 1.5, 2, 10], dtype=np.float64)
    step_m = float(args.step_m)
    ik_pos_err = 0.0
    ik_rot_err = 0.0
    grasp_assist = GraspAssist()
    grasp_assist_active = False
    recording = False
    frame_count = 0
    saved_episodes = 0
    last_event = "ready: press 1 to start recording"
    next_record_time = time.perf_counter()
    record_period = 1.0 / float(args.fps)
    camera_overlays = tuple(item.strip() for item in args.camera_overlays.split(",") if item.strip())

    print("CR3 VLA teleop recorder")
    print(
        "Move: W/S A/D R/F | gripper: O/P/Space | "
        "record: 1 toggle, 2 save+reset, 3 discard+reset, 0 reset | Q quit"
    )
    print(f"dataset root: {args.root}")
    print(f"task: {args.task}")
    print(f"randomize blocks on reset: {randomize_blocks}")

    viewer = MinimalGLFWViewer(model, data, title="CR3 VLA Teleop Recorder")
    running = True
    try:
        viewer.cam.lookat[:] = (-0.15, 0.0, 0.45)
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = -80
        viewer.cam.elevation = -35

        dt = 1.0 / args.loop_hz
        while viewer.is_running() and running:
            start = time.perf_counter()
            restore_viewer_context(viewer)
            viewer.poll()
            previous_qpos_target = qpos_target.copy()

            for event in recorder_keys(viewer):
                if event == "reset_scene":
                    grasp_assist.release(model, data)
                    qpos_target, gripper_cmd, gripper_target, target, target_rot = restore_scene_snapshot(
                        model,
                        data,
                        qpos=reset_qpos,
                        qvel=reset_qvel,
                        ctrl=reset_ctrl,
                        arm_ids=arm_ids,
                        qpos_addrs=qpos_addrs,
                        left_gripper_id=left_id,
                        right_gripper_id=right_id,
                        rng=rng,
                        randomize_blocks=randomize_blocks,
                    )
                    recording = False
                    next_record_time = time.perf_counter()
                    last_event = "scene reset; recording paused"
                    if viewer.window is not None:
                        glfw.set_window_title(viewer.window, "CR3 VLA Teleop Recorder - RESET")
                    print(
                        "\n==== SCENE RESET ====\n"
                        "robot, gripper, and blocks restored for the next episode\n"
                        "press 1 to start recording\n",
                        flush=True,
                    )
                elif event == "toggle_record":
                    recording = not recording
                    next_record_time = time.perf_counter()
                    last_event = "recording started" if recording else "recording paused"
                    if viewer.window is not None:
                        title_state = "RECORDING" if recording else "PAUSED"
                        glfw.set_window_title(viewer.window, f"CR3 VLA Teleop Recorder - {title_state}")
                    print(
                        f"\n==== {'RECORDING STARTED' if recording else 'RECORDING PAUSED'} ====\n"
                        f"frames in current episode: {frame_count}\n"
                        f"press 2 to save+reset, 3 to discard+reset\n",
                        flush=True,
                    )
                elif event == "save":
                    if frame_count <= 0:
                        last_event = "save ignored: no frames"
                        print("\n==== SAVE IGNORED: no frames recorded ====\n", flush=True)
                    else:
                        dataset.save_episode()
                        flush_saved_episode(dataset)
                        saved_episodes += 1
                        frame_count = 0
                        recording = False
                        grasp_assist.release(model, data)
                        qpos_target, gripper_cmd, gripper_target, target, target_rot = restore_scene_snapshot(
                            model,
                            data,
                            qpos=reset_qpos,
                            qvel=reset_qvel,
                            ctrl=reset_ctrl,
                            arm_ids=arm_ids,
                            qpos_addrs=qpos_addrs,
                            left_gripper_id=left_id,
                            right_gripper_id=right_id,
                            rng=rng,
                            randomize_blocks=randomize_blocks,
                        )
                        last_event = f"saved episode {saved_episodes}; scene reset"
                        if viewer.window is not None:
                            glfw.set_window_title(viewer.window, "CR3 VLA Teleop Recorder - SAVED")
                        print(
                            f"\n==== SAVED EPISODE {saved_episodes} ====\n"
                            f"dataset root: {args.root}\n",
                            "scene reset for next episode\n",
                            flush=True,
                        )
                elif event == "discard":
                    dataset.clear_episode_buffer()
                    frame_count = 0
                    recording = False
                    grasp_assist.release(model, data)
                    qpos_target, gripper_cmd, gripper_target, target, target_rot = restore_scene_snapshot(
                        model,
                        data,
                        qpos=reset_qpos,
                        qvel=reset_qvel,
                        ctrl=reset_ctrl,
                        arm_ids=arm_ids,
                        qpos_addrs=qpos_addrs,
                        left_gripper_id=left_id,
                        right_gripper_id=right_id,
                        rng=rng,
                        randomize_blocks=randomize_blocks,
                    )
                    last_event = "discarded current episode; scene reset"
                    if viewer.window is not None:
                        glfw.set_window_title(viewer.window, "CR3 VLA Teleop Recorder - DISCARDED")
                    print(
                        "\n==== DISCARDED CURRENT EPISODE ====\n"
                        "scene reset for next episode\n",
                        flush=True,
                    )

            for key in control_keys(viewer):
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
                    camera_name=None,
                    control_frame="world",
                )
                if gripper_target <= GRIPPER_OPEN_CMD + 0.04:
                    grasp_assist.release(model, data)
                    if GRIPPER_FAST_OPEN:
                        gripper_cmd = GRIPPER_OPEN_CMD
                        set_gripper(data, left_id, right_id, gripper_cmd)
                target_dirty = (
                    not np.allclose(old_target, target)
                    or old_step != step_m
                    or old_gripper_target != gripper_target
                )
                if target_dirty:
                    qpos_target, ik_pos_err, ik_rot_err, _ik_ok = solve_6d_ik(
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
                        joint_weights=joint_weights,
                        max_total_delta=args.max_total_joint_delta,
                    )

            gripper_cmd = move_towards(gripper_cmd, gripper_target, max(float(args.gripper_rate) * dt, 0.0))
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
                    enabled=True,
                )

            now = time.perf_counter()
            if recording and now >= next_record_time:
                add_dataset_frame(
                    dataset,
                    renderer,
                    model,
                    data,
                    qpos_target=qpos_target,
                    gripper_target=gripper_target,
                    gripper_cmd=gripper_cmd,
                    front_size=args.front_size,
                    wrist_size=args.wrist_size,
                    task=args.task,
                )
                frame_count += 1
                next_record_time += record_period
                restore_viewer_context(viewer)

            viewer.render(
                overlay_text(
                    recording=recording,
                    frame_count=frame_count,
                    saved_episodes=saved_episodes,
                    fps=args.fps,
                    last_event=last_event,
                ),
                camera_overlays,
            )

            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        renderer.close()
        viewer.close()
        dataset.finalize()

    print(f"dataset written to: {args.root}")


if __name__ == "__main__":
    main()
