"""Regression tests for paired RTC queues in the remote Pi0 rollout client."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


def load_runner_module():
    path = Path(__file__).parents[2] / "scripts" / "inference" / "run_remote_pi0_policy.py"
    spec = importlib.util.spec_from_file_location("run_remote_pi0_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rtc_snapshot_returns_the_same_unconsumed_original_actions_as_processed_actions():
    runner = load_runner_module()
    queue = runner.make_rtc_action_queue()
    original = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    processed = original + 100
    queue.merge(original, processed, real_delay=0)
    queue.get()
    queue.get()

    prefix, action_index = runner.snapshot_rtc_prefix(queue)

    assert action_index == 2
    torch.testing.assert_close(prefix, original[2:])
    torch.testing.assert_close(queue.get_processed_left_over(), processed[2:])


def test_rtc_merge_atomically_uses_the_actions_consumed_during_the_request():
    runner = load_runner_module()
    queue = runner.make_rtc_action_queue()
    previous = torch.zeros((3, 7), dtype=torch.float32)
    queue.merge(previous, previous, real_delay=0)
    request_action_index = queue.get_action_index()
    queue.get()
    queue.get()
    queue.get()

    original = torch.arange(42, dtype=torch.float32).reshape(6, 7)
    processed = np.arange(42, dtype=np.float32).reshape(6, 7) + 200

    actual_delay = runner.merge_rtc_response(
        queue,
        original,
        processed,
        request_action_index=request_action_index,
    )

    assert actual_delay == 3
    torch.testing.assert_close(queue.get_left_over(), original[3:])
    torch.testing.assert_close(queue.get_processed_left_over(), torch.from_numpy(processed[3:]))
