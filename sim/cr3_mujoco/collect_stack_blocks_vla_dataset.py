#!/usr/bin/env python

"""Collect a small language-conditioned stacking dataset in the CR3 MuJoCo scene.

The scripted task is:
    put the red cube in the black frame, then put the yellow cube on top,
    then put the green cube on top of the yellow cube.

This is intentionally a minimal VLA data-collection bridge: it records
front/wrist RGB, 7D state, 7D action, and one natural-language task string in
LeRobotDataset format. It does not modify the scene XML, URDF, meshes, gripper
sites, or closed-loop constraints.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


LEROBOT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

# Dataset collection renders cameras without opening a viewer. EGL avoids
# requiring an X11 window when the script is run from a terminal or server.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from lerobot.datasets import LeRobotDataset
from sim.cr3_mujoco.build_cr3_scene import CUBE_SIZE, SCENE_XML, TABLE_TOP_Z, TARGET_CENTER


ARM_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")
ARM_ACTUATORS = tuple(f"{joint}_pos" for joint in ARM_JOINTS)
LEFT_GRIPPER_ACTUATOR = "Left_joint1_pos"
RIGHT_GRIPPER_ACTUATOR = "Right_joint_pos"
LEFT_FINGER_GEOM = "left_static1_finger_collision"
RIGHT_FINGER_GEOM = "right_static1_finger_collision"

TASK_DESCRIPTION = (
    "put the red cube in the black frame, then put the yellow cube on top, "
    "then put the green cube on top of the yellow cube"
)

DEFAULT_REPO_ID = "local/cr3_lmg90_stack_blocks_vla"
DEFAULT_ROOT = LEROBOT_ROOT / "lerobot_data" / "local" / "cr3_lmg90_stack_blocks_vla"
DEFAULT_FPS = 15
DEFAULT_FRONT_SIZE = (480, 640)
DEFAULT_WRIST_SIZE = (480, 640)

GRIPPER_OPEN_CMD = 0.12
GRIPPER_GRASP_CMD = 0.45
APPROACH_HEIGHT = 0.13
LIFT_HEIGHT = 0.11
PLACE_CLEARANCE = 0.006


@dataclass(frozen=True)
class BlockSpec:
    label: str
    body: str
    joint: str
    visual_geom: str
    collision_geom: str


BLOCKS = (
    BlockSpec("red", "cube", "cube_free", "cube", "cube_collision"),
    BlockSpec("yellow", "yellow_cube", "yellow_cube_free", "yellow_cube", "yellow_cube_collision"),
    BlockSpec("green", "green_cube", "green_cube_free", "green_cube", "green_cube_collision"),
)


class CameraRenderer:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def render(self, data: mujoco.MjData, camera: str, height: int, width: int) -> np.ndarray:
        key = (height, width)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self.renderers[key]
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()


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


def gripper_actuator_ids(model: mujoco.MjModel) -> tuple[int, int]:
    return (
        obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_GRIPPER_ACTUATOR),
        obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_GRIPPER_ACTUATOR),
    )


def set_gripper(data: mujoco.MjData, left_id: int, right_id: int, cmd: float) -> None:
    data.ctrl[left_id] = float(cmd)
    data.ctrl[right_id] = -float(cmd)


def finger_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    left_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, LEFT_FINGER_GEOM)
    right_id = obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, RIGHT_FINGER_GEOM)
    return 0.5 * (np.asarray(data.geom_xpos[left_id]) + np.asarray(data.geom_xpos[right_id]))


def block_pos(model: mujoco.MjModel, data: mujoco.MjData, block: BlockSpec) -> np.ndarray:
    body_id = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, block.body)
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def set_block_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    block: BlockSpec,
    pos: np.ndarray,
    *,
    zero_velocity: bool = True,
) -> None:
    joint_id = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, block.joint)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    if zero_velocity:
        data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)


def attach_block_to_fingers(model: mujoco.MjModel, data: mujoco.MjData, block: BlockSpec) -> None:
    set_block_pose(model, data, block, finger_center(model, data), zero_velocity=True)


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
    tries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    addrs = arm_qpos_addrs(model)
    joint_ids = [obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in ARM_JOINTS]
    limits = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids], dtype=np.float64)
    original_qpos = np.asarray(data.qpos[addrs], dtype=np.float64).copy()

    def score(qpos: np.ndarray) -> float:
        data.qpos[addrs] = qpos
        mujoco.mj_forward(model, data)
        delta = finger_center(model, data) - target
        return float(delta @ delta + 0.0015 * np.mean((qpos - seed) ** 2))

    best = np.clip(seed.copy(), limits[:, 0], limits[:, 1])
    best_score = score(best)
    scales = np.asarray([0.30, 0.22, 0.28, 0.35, 0.35, 0.50], dtype=np.float64)
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


def state_vector(model: mujoco.MjModel, data: mujoco.MjData, gripper_cmd: float) -> np.ndarray:
    state = np.empty(7, dtype=np.float32)
    state[:6] = data.qpos[arm_qpos_addrs(model)].astype(np.float32)
    state[6] = np.float32(gripper_cmd)
    return state


def action_vector(qpos_target: np.ndarray, gripper_cmd: float) -> np.ndarray:
    action = np.empty(7, dtype=np.float32)
    action[:6] = qpos_target.astype(np.float32)
    action[6] = np.float32(gripper_cmd)
    return action


def make_features(front_size: tuple[int, int], wrist_size: tuple[int, int]) -> dict:
    front_h, front_w = front_size
    wrist_h, wrist_w = wrist_size
    return {
        "observation.images.front_rgb": {
            "dtype": "video",
            "shape": (front_h, front_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "video",
            "shape": (wrist_h, wrist_w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]},
        },
    }


def add_dataset_frame(
    dataset: LeRobotDataset,
    renderer: CameraRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos_target: np.ndarray,
    gripper_cmd: float,
    front_size: tuple[int, int],
    wrist_size: tuple[int, int],
) -> None:
    front_h, front_w = front_size
    wrist_h, wrist_w = wrist_size
    dataset.add_frame(
        {
            "observation.images.front_rgb": renderer.render(data, "front", front_h, front_w),
            "observation.images.wrist_rgb": renderer.render(data, "wrist", wrist_h, wrist_w),
            "observation.state": state_vector(model, data, gripper_cmd),
            "action": action_vector(qpos_target, gripper_cmd),
            "task": TASK_DESCRIPTION,
        }
    )


def run_segment(
    dataset: LeRobotDataset,
    renderer: CameraRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    arm_ids: np.ndarray,
    gripper_ids: tuple[int, int],
    start_qpos: np.ndarray,
    goal_qpos: np.ndarray,
    gripper_start: float,
    gripper_goal: float,
    seconds: float,
    fps: int,
    front_size: tuple[int, int],
    wrist_size: tuple[int, int],
    carried_block: BlockSpec | None = None,
    pinned_blocks: dict[str, tuple[BlockSpec, np.ndarray]] | None = None,
) -> tuple[np.ndarray, float]:
    addrs = arm_qpos_addrs(model)
    dofs = arm_qvel_addrs(model)
    left_id, right_id = gripper_ids
    sim_steps = max(1, int(round(seconds / model.opt.timestep)))
    record_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    current_cmd = gripper_start

    for step in range(sim_steps):
        alpha = smoothstep(step / max(sim_steps - 1, 1))
        qpos = start_qpos + (goal_qpos - start_qpos) * alpha
        current_cmd = gripper_start + (gripper_goal - gripper_start) * alpha
        data.qpos[addrs] = qpos
        data.qvel[dofs] = 0.0
        data.ctrl[arm_ids] = qpos
        set_gripper(data, left_id, right_id, current_cmd)
        if pinned_blocks is not None:
            for pinned_block, pinned_pos in pinned_blocks.values():
                set_block_pose(model, data, pinned_block, pinned_pos)
        if carried_block is not None:
            attach_block_to_fingers(model, data, carried_block)
        mujoco.mj_step(model, data)
        if step % record_every == 0:
            add_dataset_frame(
                dataset,
                renderer,
                model,
                data,
                qpos_target=qpos,
                gripper_cmd=current_cmd,
                front_size=front_size,
                wrist_size=wrist_size,
            )

    return goal_qpos.copy(), float(gripper_goal)


def settle_and_record(
    dataset: LeRobotDataset,
    renderer: CameraRenderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    qpos: np.ndarray,
    gripper_cmd: float,
    seconds: float,
    fps: int,
    front_size: tuple[int, int],
    wrist_size: tuple[int, int],
    arm_ids: np.ndarray,
    gripper_ids: tuple[int, int],
    pinned_blocks: dict[str, tuple[BlockSpec, np.ndarray]] | None = None,
) -> None:
    left_id, right_id = gripper_ids
    addrs = arm_qpos_addrs(model)
    dofs = arm_qvel_addrs(model)
    steps = max(1, int(round(seconds / model.opt.timestep)))
    record_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
    for step in range(steps):
        data.qpos[addrs] = qpos
        data.qvel[dofs] = 0.0
        data.ctrl[arm_ids] = qpos
        set_gripper(data, left_id, right_id, gripper_cmd)
        if pinned_blocks is not None:
            for pinned_block, pinned_pos in pinned_blocks.values():
                set_block_pose(model, data, pinned_block, pinned_pos)
        mujoco.mj_step(model, data)
        if step % record_every == 0:
            add_dataset_frame(
                dataset,
                renderer,
                model,
                data,
                qpos_target=qpos,
                gripper_cmd=gripper_cmd,
                front_size=front_size,
                wrist_size=wrist_size,
            )


def plan_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    seed: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> np.ndarray:
    return solve_finger_center_ik(model, data, target, seed, tries=args.ik_tries, rng=rng)


def collect_episode(
    dataset: LeRobotDataset,
    renderer: CameraRenderer,
    model: mujoco.MjModel,
    *,
    episode_idx: int,
    args: argparse.Namespace,
) -> None:
    data = mujoco.MjData(model)
    model.opt.gravity[:] = np.asarray([0.0, 0.0, -9.81])
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(args.seed + episode_idx)
    arm_ids = arm_actuator_ids(model)
    gripper_ids = gripper_actuator_ids(model)
    addrs = arm_qpos_addrs(model)

    # Slight XY jitter keeps the dataset from being one exact trajectory.
    jitter = lambda scale: rng.uniform(-scale, scale, size=2)
    start_offsets = {
        "red": jitter(args.object_jitter),
        "yellow": jitter(args.object_jitter),
        "green": jitter(args.object_jitter),
    }
    target_offset = jitter(args.target_jitter)

    table_cube_z = TABLE_TOP_Z + CUBE_SIZE / 2 + 0.001
    for block in BLOCKS:
        start = block_pos(model, data, block).copy()
        start[:2] += start_offsets[block.label]
        start[2] = table_cube_z
        set_block_pose(model, data, block, start)

    current_qpos = np.asarray(data.qpos[addrs], dtype=np.float64).copy()
    current_cmd = GRIPPER_OPEN_CMD
    data.ctrl[arm_ids] = current_qpos
    set_gripper(data, *gripper_ids, current_cmd)

    settle_and_record(
        dataset,
        renderer,
        model,
        data,
        qpos=current_qpos,
        gripper_cmd=current_cmd,
        seconds=0.5,
        fps=args.fps,
        front_size=args.front_size,
        wrist_size=args.wrist_size,
        arm_ids=arm_ids,
        gripper_ids=gripper_ids,
    )

    stack_xy = np.asarray([TARGET_CENTER[0], TARGET_CENTER[1]], dtype=np.float64) + target_offset
    pinned_blocks: dict[str, tuple[BlockSpec, np.ndarray]] = {}
    for stack_level, block in enumerate(BLOCKS):
        pick = block_pos(model, data, block).copy()
        pick[2] += 0.004
        pick_approach = pick.copy()
        pick_approach[2] += APPROACH_HEIGHT

        place = np.asarray(
            [
                stack_xy[0],
                stack_xy[1],
                TABLE_TOP_Z + CUBE_SIZE / 2 + stack_level * CUBE_SIZE + PLACE_CLEARANCE,
            ],
            dtype=np.float64,
        )
        place_approach = place.copy()
        place_approach[2] += APPROACH_HEIGHT
        lift = pick.copy()
        lift[2] += LIFT_HEIGHT

        approach_qpos = plan_pose(model, data, pick_approach, current_qpos, args, rng)
        grasp_qpos = plan_pose(model, data, pick, approach_qpos, args, rng)
        lift_qpos = plan_pose(model, data, lift, grasp_qpos, args, rng)
        place_approach_qpos = plan_pose(model, data, place_approach, lift_qpos, args, rng)
        place_qpos = plan_pose(model, data, place, place_approach_qpos, args, rng)

        print(
            f"episode {episode_idx}: {block.label} pick={np.round(pick, 4).tolist()} "
            f"place={np.round(place, 4).tolist()}",
            flush=True,
        )

        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=approach_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_OPEN_CMD,
            seconds=1.6,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            pinned_blocks=pinned_blocks,
        )
        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=grasp_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_OPEN_CMD,
            seconds=1.0,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            pinned_blocks=pinned_blocks,
        )

        # Align before closing so the recorded demo is successful and repeatable.
        set_block_pose(model, data, block, finger_center(model, data))
        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=current_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_GRASP_CMD,
            seconds=1.2,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            carried_block=block,
            pinned_blocks=pinned_blocks,
        )
        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=lift_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_GRASP_CMD,
            seconds=1.2,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            carried_block=block,
            pinned_blocks=pinned_blocks,
        )
        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=place_approach_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_GRASP_CMD,
            seconds=1.8,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            carried_block=block,
            pinned_blocks=pinned_blocks,
        )
        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=place_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_GRASP_CMD,
            seconds=1.0,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            carried_block=block,
            pinned_blocks=pinned_blocks,
        )

        # Snap to the intended stack pose before release. This keeps this first
        # dataset about VLA plumbing, not about solving block stacking physics.
        set_block_pose(model, data, block, place)
        pinned_blocks[block.label] = (block, place.copy())
        current_qpos, current_cmd = run_segment(
            dataset,
            renderer,
            model,
            data,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            start_qpos=current_qpos,
            goal_qpos=current_qpos,
            gripper_start=current_cmd,
            gripper_goal=GRIPPER_OPEN_CMD,
            seconds=0.9,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            pinned_blocks=pinned_blocks,
        )
        settle_and_record(
            dataset,
            renderer,
            model,
            data,
            qpos=current_qpos,
            gripper_cmd=current_cmd,
            seconds=0.4,
            fps=args.fps,
            front_size=args.front_size,
            wrist_size=args.wrist_size,
            arm_ids=arm_ids,
            gripper_ids=gripper_ids,
            pinned_blocks=pinned_blocks,
        )

    dataset.save_episode()
    final_positions = {block.label: np.round(block_pos(model, data, block), 4).tolist() for block in BLOCKS}
    print(f"episode {episode_idx}: saved. final block positions={final_positions}", flush=True)


def tuple_hw(value: str) -> tuple[int, int]:
    try:
        h_str, w_str = value.lower().replace("x", ",").split(",", maxsplit=1)
        return int(h_str), int(w_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected HEIGHT,WIDTH or HEIGHTxWIDTH, e.g. 480,640") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=SCENE_XML)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--front-size", type=tuple_hw, default=DEFAULT_FRONT_SIZE, help="HEIGHT,WIDTH")
    parser.add_argument("--wrist-size", type=tuple_hw, default=DEFAULT_WRIST_SIZE, help="HEIGHT,WIDTH")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--ik-tries", type=int, default=1800)
    parser.add_argument("--object-jitter", type=float, default=0.008)
    parser.add_argument("--target-jitter", type=float, default=0.004)
    parser.add_argument("--overwrite", action="store_true", help="Delete the output root before recording.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing MuJoCo scene: {args.model_path}")

    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset root already exists: {args.root}. Use --overwrite to replace it.")
        shutil.rmtree(args.root)

    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    renderer = CameraRenderer(model)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        features=make_features(args.front_size, args.wrist_size),
        robot_type="cr3_lmg90_mujoco",
        use_videos=True,
        image_writer_threads=4,
    )

    try:
        for episode_idx in range(args.episodes):
            collect_episode(dataset, renderer, model, episode_idx=episode_idx, args=args)
    finally:
        renderer.close()
        dataset.finalize()

    print(f"dataset written to: {args.root}", flush=True)
    print(f"repo_id: {args.repo_id}", flush=True)
    print(f"task: {TASK_DESCRIPTION}", flush=True)


if __name__ == "__main__":
    main()
