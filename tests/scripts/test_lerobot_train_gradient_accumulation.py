from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from lerobot.utils.logging_utils import AverageMeter


class _ScalarPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, torch.Tensor]):
        loss = (self.weight - batch["target"]).square()
        return loss, {}


class _AccumulatingAccelerator:
    def __init__(self, sync_gradients: bool) -> None:
        self.sync_gradients = sync_gradients

    def autocast(self):
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        # Matches Accelerate's loss averaging for two accumulated microbatches.
        (loss / 2).backward()

    def clip_grad_norm_(self, parameters, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, policy, keep_fp32_wrapper: bool = True):
        return policy


class TestGradientAccumulation(unittest.TestCase):
    def test_formats_pi0_auxiliary_losses_for_console_logging(self) -> None:
        from lerobot.scripts.lerobot_train import format_policy_auxiliary_losses

        formatted = format_policy_auxiliary_losses(
            {
                "flow_loss": 0.123456,
                "action_velocity_loss": 0.012345,
                "action_acceleration_loss": 0.001234,
            }
        )

        self.assertEqual(
            formatted,
            "policy_losses flow_loss:0.123456 action_velocity_loss:0.012345 "
            "action_acceleration_loss:0.001234",
        )

    def test_microbatch_loss_mean_uses_meter_values(self) -> None:
        from lerobot.scripts.lerobot_train import mean_microbatch_losses

        first = AverageMeter("loss")
        first.update(0.25)
        second = AverageMeter("loss")
        second.update(0.75)

        self.assertEqual(mean_microbatch_losses([first, second]), 0.5)

    def test_only_sync_microbatch_updates_the_optimizer(self) -> None:
        from lerobot.scripts.lerobot_train import update_policy

        policy = _ScalarPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        metrics = SimpleNamespace()

        update_policy(
            metrics,
            policy,
            {"target": torch.tensor(0.0)},
            optimizer,
            grad_clip_norm=0.0,
            accelerator=_AccumulatingAccelerator(sync_gradients=False),
        )
        self.assertEqual(policy.weight.item(), 1.0)

        update_policy(
            metrics,
            policy,
            {"target": torch.tensor(0.0)},
            optimizer,
            grad_clip_norm=0.0,
            accelerator=_AccumulatingAccelerator(sync_gradients=True),
        )
        self.assertAlmostEqual(policy.weight.item(), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
