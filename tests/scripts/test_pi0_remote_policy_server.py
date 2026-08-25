"""Focused regression tests for remote Pi0 RTC request state."""

import importlib.util
from pathlib import Path

import pytest
import torch

from lerobot.policies.pi0.configuration_pi0 import PI0Config


def load_server_module():
    path = Path(__file__).parents[2] / "scripts" / "inference" / "pi0_remote_policy_server.py"
    spec = importlib.util.spec_from_file_location("pi0_remote_policy_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_rtc_sets_the_requested_pi0_runtime_config():
    server = load_server_module()
    config = PI0Config()

    server.configure_rtc(
        config,
        enabled=True,
        execution_horizon=11,
        guidance_weight=7.5,
        prefix_attention_schedule="exp",
    )

    assert config.rtc_config is not None
    assert config.rtc_config.execution_horizon == 11
    assert config.rtc_config.max_guidance_weight == 7.5
    assert config.rtc_config.prefix_attention_schedule.name == "EXP"


def test_configure_rtc_modes_enables_trained_prefix_without_v1_guidance():
    server = load_server_module()
    config = PI0Config(rtc_training_simulated_delay=5)

    server.configure_rtc_modes(
        config,
        rtc_enabled=False,
        rtc_trained_prefix=True,
        execution_horizon=10,
        guidance_weight=10.0,
        prefix_attention_schedule="exp",
    )

    assert config.rtc_training_inference_enabled is True
    assert config.rtc_config is None


def test_configure_rtc_modes_rejects_v1_and_trained_prefix_together():
    server = load_server_module()
    config = PI0Config(rtc_training_simulated_delay=5)

    with pytest.raises(ValueError, match="cannot be combined"):
        server.configure_rtc_modes(
            config,
            rtc_enabled=True,
            rtc_trained_prefix=True,
            execution_horizon=10,
            guidance_weight=10.0,
            prefix_attention_schedule="exp",
        )


def test_rtc_kwargs_use_the_client_prefix_without_server_history():
    server = load_server_module()
    prefix = torch.arange(28, dtype=torch.float32).reshape(4, 7).numpy()

    kwargs = server.rtc_kwargs_from_payload(
        {"rtc_prev_chunk": prefix, "rtc_estimated_delay_steps": 6},
        enabled=True,
        execution_horizon=10,
        device="cpu",
    )

    assert kwargs["inference_delay"] == 6
    assert kwargs["execution_horizon"] == 10
    torch.testing.assert_close(kwargs["prev_chunk_left_over"], torch.from_numpy(prefix).unsqueeze(0))


def test_reset_inference_rng_skips_torch_when_seed_is_none(monkeypatch):
    server = load_server_module()
    calls = []
    monkeypatch.setattr(server.torch, "manual_seed", lambda value: calls.append(value))

    server.reset_inference_rng(None)

    assert calls == []


def test_reset_inference_rng_resets_cpu_and_cuda(monkeypatch):
    server = load_server_module()
    calls = []
    monkeypatch.setattr(server.torch, "manual_seed", lambda value: calls.append(("cpu", value)))
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "manual_seed_all", lambda value: calls.append(("cuda", value)))

    server.reset_inference_rng(1234)

    assert calls == [("cpu", 1234), ("cuda", 1234)]
