#!/usr/bin/env python3
"""Isolate CR3 ServoJ tracking from Pi0 inference and action-queue behavior."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LEROBOT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.robots.dobot_cr3 import DobotCR3, DobotCR3Config


def smoothstep_targets(
    start: np.ndarray,
    *,
    joint_index: int,
    delta_deg: float,
    duration_s: float,
    hz: float,
) -> np.ndarray:
    """Create a fixed-rate S-curve that moves only one joint."""
    if start.shape != (6,):
        raise ValueError(f"Expected six joint angles, got {start.shape}.")
    if not 0 <= joint_index < 6:
        raise ValueError("joint_index must be in [0, 5].")
    if duration_s <= 0 or hz <= 0:
        raise ValueError("duration_s and hz must be positive.")

    count = max(2, round(duration_s * hz) + 1)
    progress = np.linspace(0.0, 1.0, count, dtype=np.float32)
    curve = progress * progress * (3.0 - 2.0 * progress)
    targets = np.repeat(start[np.newaxis, :], count, axis=0)
    targets[:, joint_index] += float(delta_deg) * curve
    return targets


def parse_cmd_log(path: Path) -> np.ndarray:
    """Extract local execution commands, deliberately excluding server first_cmd values."""
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(?:^|\s)cmd=\[([^\]]+)\]", text, flags=re.MULTILINE)
    actions = []
    for item in matches:
        values = np.fromstring(item, sep=",", dtype=np.float32)
        if values.shape == (7,):
            actions.append(values)
    if not actions:
        raise ValueError(f"No seven-value local cmd=[...] records found in {path}.")
    return np.stack(actions)


def resample_actions(actions: np.ndarray, *, source_hz: float, target_hz: float) -> np.ndarray:
    """Linearly resample joint targets while preserving the prior discrete gripper state."""
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected [N, 7] actions, got {actions.shape}.")
    if len(actions) < 2:
        raise ValueError("At least two actions are required for replay.")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("source_hz and target_hz must be positive.")

    duration_s = (len(actions) - 1) / source_hz
    count = max(2, round(duration_s * target_hz) + 1)
    target_times = np.linspace(0.0, duration_s, count, dtype=np.float64)
    source_times = np.arange(len(actions), dtype=np.float64) / source_hz
    replay = np.empty((count, 7), dtype=np.float32)
    for joint in range(6):
        replay[:, joint] = np.interp(target_times, source_times, actions[:, joint])
    source_indices = np.minimum((target_times * source_hz).astype(int), len(actions) - 1)
    replay[:, 6] = actions[source_indices, 6]
    return replay


def validate_replay_start(start: np.ndarray, targets: np.ndarray, *, tolerance_deg: float) -> None:
    """Reject an absolute-action replay unless the robot starts near its recorded pose."""
    max_error = float(np.max(np.abs(start - targets[0])))
    if max_error > tolerance_deg:
        raise ValueError(
            f"replay start pose differs from the first recorded cmd by {max_error:.2f} degrees "
            f"(limit {tolerance_deg:.2f}). Move the robot to the recorded start pose first."
        )


def make_hold_targets(start: np.ndarray, *, duration_s: float, hz: float) -> np.ndarray:
    if duration_s <= 0 or hz <= 0:
        raise ValueError("duration_s and hz must be positive.")
    return np.repeat(start[np.newaxis, :], max(1, round(duration_s * hz)), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("hold", "smooth-j1", "replay"))
    parser.add_argument("--robot-ip", default="192.168.6.1")
    parser.add_argument("--speed-factor", type=int, default=20)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--servo-t", type=float, default=0.05)
    parser.add_argument("--servo-lookahead-time", type=float, default=100.0)
    parser.add_argument("--servo-gain", type=float, default=250.0)
    parser.add_argument("--joint-index", type=int, default=0)
    parser.add_argument("--delta-deg", type=float, default=3.0)
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument(
        "--replay-source-hz",
        type=float,
        default=6.0,
        help="Rate of cmd records in the input log; use 30 when the source used --print-every 1.",
    )
    parser.add_argument("--replay-start-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--execute", action="store_true", help="Connect to CR3 and send ServoJ commands.")
    args = parser.parse_args()

    if args.mode == "replay" and args.replay_log is None:
        parser.error("replay requires --replay-log PATH")
    if not 0.02 <= args.servo_t <= 3600.0:
        parser.error("--servo-t must be in [0.02, 3600.0]")
    if not 20.0 <= args.servo_lookahead_time <= 100.0:
        parser.error("--servo-lookahead-time must be in [20, 100]")
    if not 200.0 <= args.servo_gain <= 1000.0:
        parser.error("--servo-gain must be in [200, 1000]")
    if abs(args.delta_deg) > 5.0:
        parser.error("--delta-deg is safety-limited to +/-5 degrees")
    if args.replay_start_tolerance_deg <= 0:
        parser.error("--replay-start-tolerance-deg must be positive")
    return args


def targets_for_mode(args: argparse.Namespace, start: np.ndarray) -> np.ndarray:
    if args.mode == "hold":
        return make_hold_targets(start, duration_s=args.duration_s, hz=args.hz)
    if args.mode == "smooth-j1":
        return smoothstep_targets(
            start,
            joint_index=args.joint_index,
            delta_deg=args.delta_deg,
            duration_s=args.duration_s,
            hz=args.hz,
        )
    replay = parse_cmd_log(args.replay_log)
    return resample_actions(replay, source_hz=args.replay_source_hz, target_hz=args.hz)[:, :6]


def default_output_path(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "servoj_diagnostics" / f"{mode}_{stamp}.csv"


def run(args: argparse.Namespace) -> Path | None:
    if not args.execute:
        print("Preview only: no CR3 connection and no motion. Add --execute to run the diagnostic.")
        print(
            f"mode={args.mode} hz={args.hz:g} servo_t={args.servo_t:g} "
            f"lookahead={args.servo_lookahead_time:g} gain={args.servo_gain:g}"
        )
        if args.mode == "replay":
            actions = parse_cmd_log(args.replay_log)
            print(f"replay source: {args.replay_log} ({len(actions)} logged commands at {args.replay_source_hz:g}Hz)")
        return None

    config = DobotCR3Config(
        robot_ip=args.robot_ip,
        use_gripper=False,
        speed_factor=args.speed_factor,
        enable_robot_on_connect=True,
        use_opencv_camera=False,
    )
    robot = DobotCR3(config)
    output_path = args.output_csv or default_output_path(args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    robot.connect()
    try:
        start = np.asarray(robot.get_joints(), dtype=np.float32)
        targets = targets_for_mode(args, start)
        if args.mode == "replay":
            validate_replay_start(
                start,
                targets,
                tolerance_deg=args.replay_start_tolerance_deg,
            )
        print(f"Executing {args.mode}: {len(targets)} ServoJ targets at {args.hz:g}Hz")
        print(f"start={start.tolist()} final_target={targets[-1].tolist()}")

        period = 1.0 / args.hz
        next_tick = time.perf_counter()
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_s", *[f"target_q{i}" for i in range(1, 7)], *[f"actual_q{i}" for i in range(1, 7)]])
            for index, target in enumerate(targets):
                robot.move.ServoJ(
                    *[float(value) for value in target],
                    t=args.servo_t,
                    lookahead_time=args.servo_lookahead_time,
                    gain=args.servo_gain,
                )
                actual = robot.get_joints()
                writer.writerow([time.time(), *target.tolist(), *actual])
                if index % max(1, round(args.hz)) == 0:
                    print(f"step={index:04d} target={np.round(target, 3).tolist()} actual={np.round(actual, 3).tolist()}")
                next_tick += period
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
        print(f"CSV written: {output_path}")
        return output_path
    finally:
        robot.disconnect()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
