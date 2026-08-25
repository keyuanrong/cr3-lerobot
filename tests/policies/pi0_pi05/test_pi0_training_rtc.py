"""Unit tests for Pi0 training-time RTC and trajectory consistency losses."""

import pytest
import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import (
    _action_trajectory_consistency_losses,
    _build_training_rtc_inputs,
    _make_rtc_prefix_mask,
    _prepare_rtc_inference_prefix,
    _reduce_masked_flow_loss,
    _sample_training_rtc_prefix_lengths,
)


def test_action_trajectory_consistency_is_zero_for_matching_trajectory():
    actions = torch.tensor(
        [[[0.0, 0.0], [1.0, 2.0], [3.0, 3.0], [6.0, 5.0]]], dtype=torch.float32
    )

    velocity, acceleration = _action_trajectory_consistency_losses(actions, actions, joint_dims=2)

    assert velocity.item() == pytest.approx(0.0)
    assert acceleration.item() == pytest.approx(0.0)


def test_action_trajectory_consistency_ignores_gripper_dimension():
    target = torch.zeros((1, 4, 7), dtype=torch.float32)
    predicted = target.clone()
    predicted[:, :, 6] = torch.tensor([0.0, 1.0, 0.0, 1.0])

    velocity, acceleration = _action_trajectory_consistency_losses(predicted, target, joint_dims=6)

    assert velocity.item() == pytest.approx(0.0)
    assert acceleration.item() == pytest.approx(0.0)


def test_prefix_mask_excludes_known_prefix_from_flow_loss():
    losses = torch.ones((2, 5, 7), dtype=torch.float32)
    mask = _make_rtc_prefix_mask(torch.tensor([0, 2]), sequence_length=5)

    per_sample = _reduce_masked_flow_loss(losses, valid_token_mask=~mask)

    assert mask.tolist() == [
        [False, False, False, False, False],
        [True, True, False, False, False],
    ]
    assert per_sample.tolist() == pytest.approx([1.0, 1.0])


def test_training_rtc_uses_clean_prefix_and_clean_action_timestep():
    actions = torch.full((1, 4, 2), 2.0)
    noise = torch.full((1, 4, 2), 10.0)
    time = torch.tensor([0.75])

    x_t, token_time, prefix_mask = _build_training_rtc_inputs(
        actions, noise, time, prefix_lengths=torch.tensor([2])
    )

    assert prefix_mask.tolist() == [[True, True, False, False]]
    assert token_time.tolist() == [[0.0, 0.0, 0.75, 0.75]]
    torch.testing.assert_close(x_t[:, :2], actions[:, :2])
    torch.testing.assert_close(x_t[:, 2:], torch.full((1, 2, 2), 8.0))


def test_inference_prefix_is_padded_to_full_chunk_and_limited_by_delay():
    previous = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)

    prefix, mask = _prepare_rtc_inference_prefix(
        previous, inference_delay=2, chunk_size=5, action_dim=7
    )

    assert prefix.shape == (1, 5, 7)
    assert mask.tolist() == [[True, True, False, False, False]]
    torch.testing.assert_close(prefix[:, :2], previous[:, :2])
    torch.testing.assert_close(prefix[:, 2:], torch.zeros((1, 3, 7)))


def test_training_rtc_samples_official_delay_range():
    torch.manual_seed(0)

    delays = _sample_training_rtc_prefix_lengths(5, batch_size=256, device=torch.device("cpu"))

    assert delays.shape == (256,)
    assert int(delays.min()) >= 0
    assert int(delays.max()) < 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rtc_training_simulated_delay": -1}, "rtc_training_simulated_delay"),
        ({"rtc_training_simulated_delay": 50}, "rtc_training_simulated_delay"),
        ({"action_velocity_loss_weight": -0.1}, "action_velocity_loss_weight"),
        ({"action_acceleration_loss_weight": -0.1}, "action_acceleration_loss_weight"),
        ({"action_smoothness_joint_dims": 0}, "action_smoothness_joint_dims"),
    ],
)
def test_training_rtc_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PI0Config(**kwargs)
