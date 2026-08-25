from __future__ import annotations

import numpy as np


# Simulation zero pose measured from the real robot's zero posture.
# In --use-joint-mapping mode, display/target [0, 0, 0, 0, 0, 0] maps to these MuJoCo angles.
DISPLAY_TO_SIM_JOINT_OFFSET_DEG = "0,-4.545,-97,8,-91,-287"
DISPLAY_TO_SIM_JOINT_SIGN = "1,1,1,-1,-1,-1"


def parse_six(value: str, *, default: float = 0.0) -> np.ndarray:
    if not value:
        return np.full(6, default, dtype=np.float32)
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(values)}: {value}")
    return np.asarray(values, dtype=np.float32)


def default_joint_sign() -> np.ndarray:
    return parse_six(DISPLAY_TO_SIM_JOINT_SIGN, default=1.0)


def default_joint_offset_deg() -> np.ndarray:
    return parse_six(DISPLAY_TO_SIM_JOINT_OFFSET_DEG)


def display_to_sim_deg(
    display_deg: np.ndarray,
    joint_sign: np.ndarray | None = None,
    joint_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    display_deg = np.asarray(display_deg, dtype=np.float32)
    if display_deg.shape[0] < 6:
        raise ValueError(f"Expected at least 6 joint values, got shape {display_deg.shape}.")
    joint_sign = default_joint_sign() if joint_sign is None else np.asarray(joint_sign, dtype=np.float32)
    joint_offset_deg = (
        default_joint_offset_deg() if joint_offset_deg is None else np.asarray(joint_offset_deg, dtype=np.float32)
    )

    return (display_deg[:6] * joint_sign + joint_offset_deg).astype(np.float32)


def sim_to_display_deg(
    sim_deg: np.ndarray,
    joint_sign: np.ndarray | None = None,
    joint_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    sim_deg = np.asarray(sim_deg, dtype=np.float32)
    if sim_deg.shape[0] < 6:
        raise ValueError(f"Expected at least 6 joint values, got shape {sim_deg.shape}.")
    joint_sign = default_joint_sign() if joint_sign is None else np.asarray(joint_sign, dtype=np.float32)
    joint_offset_deg = (
        default_joint_offset_deg() if joint_offset_deg is None else np.asarray(joint_offset_deg, dtype=np.float32)
    )

    return ((sim_deg[:6] - joint_offset_deg) / joint_sign).astype(np.float32)


GRIPPER_CLOSED = 1.03
GRIPPER_OPEN = 0.12
GRIPPER_CTRL_SCALES = np.asarray([GRIPPER_OPEN - GRIPPER_CLOSED, -(GRIPPER_OPEN - GRIPPER_CLOSED)], dtype=np.float32)


def gripper_to_finger_ctrl(gripper: float, actuator_ctrlrange: np.ndarray) -> np.ndarray:
    if actuator_ctrlrange.shape[0] < 8:
        raise ValueError(
            "Expected actuator ctrlrange for at least 8 actuators: J1-J6, Left_joint1, Right_joint."
        )
    gripper = float(np.clip(gripper, 0.0, 1.0))
    closed = np.asarray([GRIPPER_CLOSED, -GRIPPER_CLOSED], dtype=np.float32)
    targets = closed + gripper * GRIPPER_CTRL_SCALES
    return np.clip(targets, actuator_ctrlrange[6:8, 0], actuator_ctrlrange[6:8, 1]).astype(np.float32)


def policy_action_to_mujoco_ctrl(
    action: np.ndarray,
    actuator_ctrlrange: np.ndarray,
    *,
    current_ctrl: np.ndarray | None = None,
    joint_sign: np.ndarray | None = None,
    joint_offset_deg: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).flatten()
    if action.shape[0] != 7:
        raise ValueError(
            "Expected policy action shape (7,): J1-J6 in display degrees plus one gripper scalar in [0, 1]."
        )
    if actuator_ctrlrange.shape[0] < 8:
        raise ValueError("Expected at least 8 MuJoCo actuators: J1-J6, Left_joint1, Right_joint.")

    if current_ctrl is None:
        ctrl = np.zeros(actuator_ctrlrange.shape[0], dtype=np.float32)
    else:
        ctrl = np.asarray(current_ctrl, dtype=np.float32).copy()
        if ctrl.shape[0] < 8:
            raise ValueError(f"Expected current_ctrl with at least 8 values, got shape {ctrl.shape}.")

    ctrl[:6] = np.deg2rad(display_to_sim_deg(action[:6], joint_sign, joint_offset_deg))
    ctrl[6:8] = gripper_to_finger_ctrl(float(action[6]), actuator_ctrlrange)
    return ctrl
