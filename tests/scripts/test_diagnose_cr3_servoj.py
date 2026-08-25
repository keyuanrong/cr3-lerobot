"""Tests for the CR3 ServoJ diagnostic helper."""

import importlib.util
from pathlib import Path

import numpy as np


def load_diagnostic_module():
    path = Path(__file__).parents[2] / "scripts" / "diagnostics" / "diagnose_cr3_servoj.py"
    spec = importlib.util.spec_from_file_location("diagnose_cr3_servoj", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoothstep_targets_only_move_selected_joint():
    diagnostic = load_diagnostic_module()
    start = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)

    targets = diagnostic.smoothstep_targets(
        start, joint_index=0, delta_deg=3.0, duration_s=1.0, hz=4.0
    )

    np.testing.assert_allclose(targets[0], start)
    np.testing.assert_allclose(targets[-1], [4, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(targets[:, 1:], np.tile(start[1:], (len(targets), 1)))
    assert len(targets) == 5


def test_parse_cmd_log_ignores_model_first_cmd(tmp_path):
    diagnostic = load_diagnostic_module()
    log = tmp_path / "rollout.log"
    log.write_text(
        "request 0000 first_cmd=[1, 2, 3, 4, 5, 6, 1]\n"
        "0000 queue=31 cmd=[10, 11, 12, 13, 14, 15, 1]\n"
        "0001 queue=30 cmd=[20, 21, 22, 23, 24, 25, 0]\n",
        encoding="utf-8",
    )

    actions = diagnostic.parse_cmd_log(log)

    assert actions.shape == (2, 7)
    np.testing.assert_allclose(actions[0], [10, 11, 12, 13, 14, 15, 1])
    np.testing.assert_allclose(actions[1], [20, 21, 22, 23, 24, 25, 0])


def test_resample_actions_interpolates_joint_targets_and_keeps_last_gripper_state():
    diagnostic = load_diagnostic_module()
    actions = np.array(
        [[0, 0, 0, 0, 0, 0, 1], [2, 4, 6, 8, 10, 12, 0]], dtype=np.float32
    )

    replay = diagnostic.resample_actions(actions, source_hz=2.0, target_hz=4.0)

    assert replay.shape == (3, 7)
    np.testing.assert_allclose(replay[:, :6], [[0] * 6, [1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]])
    np.testing.assert_allclose(replay[:, 6], [1, 1, 0])


def test_validate_replay_start_rejects_a_distant_robot_pose():
    diagnostic = load_diagnostic_module()

    try:
        diagnostic.validate_replay_start(
            np.zeros(6, dtype=np.float32),
            np.array([[3, 0, 0, 0, 0, 0]], dtype=np.float32),
            tolerance_deg=2.0,
        )
    except ValueError as exc:
        assert "replay start" in str(exc)
    else:
        raise AssertionError("A distant replay start pose must be rejected.")
