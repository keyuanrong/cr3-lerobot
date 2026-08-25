import argparse
import os
import pickle  # nosec B403 - trusted client over SSH tunnel for local experiment.
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

LEROBOT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(LEROBOT_ROOT))
sys.path.insert(0, str(LEROBOT_ROOT / "src"))

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.configs import RTCAttentionSchedule


def rtc_kwargs_from_payload(
    payload: dict,
    *,
    enabled: bool,
    execution_horizon: int,
    device: str,
) -> dict:
    """Build RTC arguments from the client's unexecuted model-space prefix.

    The client owns the action queue because it is the only side that knows
    exactly which commands were sent to the robot. Keeping this state on the
    server caused a response that was dropped or latency-trimmed locally to
    desynchronize the next RTC prefix.
    """
    if not enabled:
        return {}

    prefix = payload.get("rtc_prev_chunk")
    previous_left_over = None
    if prefix is not None:
        prefix_array = np.asarray(prefix, dtype=np.float32)
        if prefix_array.ndim != 2 or prefix_array.shape[0] == 0:
            raise ValueError("rtc_prev_chunk must have shape (time_steps, action_dim).")
        previous_left_over = torch.from_numpy(np.ascontiguousarray(prefix_array)).to(device).unsqueeze(0)

    return {
        "inference_delay": max(0, int(payload.get("rtc_estimated_delay_steps", 0))),
        "prev_chunk_left_over": previous_left_over,
        "execution_horizon": execution_horizon,
    }


def configure_rtc(
    config,
    *,
    enabled: bool,
    execution_horizon: int,
    guidance_weight: float,
    prefix_attention_schedule: str,
) -> None:
    if not enabled:
        return
    config.rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[prefix_attention_schedule.upper()],
    )


def configure_rtc_modes(
    config,
    *,
    rtc_enabled: bool,
    rtc_trained_prefix: bool,
    execution_horizon: int,
    guidance_weight: float,
    prefix_attention_schedule: str,
) -> None:
    """Configure exactly one chunk-continuity mode for a loaded policy."""
    if rtc_enabled and rtc_trained_prefix:
        raise ValueError("RTC-V1 guidance and trained-prefix RTC cannot be combined")
    if rtc_trained_prefix:
        if getattr(config, "rtc_training_simulated_delay", 0) <= 0:
            raise ValueError(
                "--rtc-trained-prefix requires a checkpoint trained with rtc_training_simulated_delay > 0"
            )
        config.rtc_training_inference_enabled = True
        config.rtc_config = None
        return

    config.rtc_training_inference_enabled = False
    configure_rtc(
        config,
        enabled=rtc_enabled,
        execution_horizon=execution_horizon,
        guidance_weight=guidance_weight,
        prefix_attention_schedule=prefix_attention_schedule,
    )


def reset_inference_rng(seed: int | None) -> None:
    """Reset sampling RNGs when reproducible action chunks are requested."""
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recv_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(conn: socket.socket):
    header = recv_exact(conn, 4)
    (size,) = struct.unpack("!I", header)
    return pickle.loads(recv_exact(conn, size))  # nosec B301 - trusted tunnel.


def send_msg(conn: socket.socket, obj) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(data)) + data)


def image_to_tensor(frame: np.ndarray, image_width: int, image_height: int) -> torch.Tensor:
    import cv2

    if frame.shape[1] != image_width or frame.shape[0] != image_height:
        frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0


def payload_image(payload: dict, frame_name: str) -> np.ndarray:
    value = payload[frame_name]
    if isinstance(value, dict) and value.get("encoding") == "jpg":
        import cv2

        encoded = np.frombuffer(value["data"], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode jpeg payload for {frame_name}.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return value


def feature_hw(policy, key: str) -> tuple[int, int]:
    shape = policy.config.input_features[key].shape
    return int(shape[1]), int(shape[2])


def load_policy(
    policy_path: Path,
    device: str,
    base_policy_path: str | None,
    *,
    rtc_enabled: bool = False,
    rtc_trained_prefix: bool = False,
    rtc_execution_horizon: int = 10,
    rtc_guidance_weight: float = 10.0,
    rtc_prefix_attention_schedule: str = "exp",
):
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    config.device = device
    configure_rtc_modes(
        config,
        rtc_enabled=rtc_enabled,
        rtc_trained_prefix=rtc_trained_prefix,
        execution_horizon=rtc_execution_horizon,
        guidance_weight=rtc_guidance_weight,
        prefix_attention_schedule=rtc_prefix_attention_schedule,
    )
    policy_cls = get_policy_class(config.type)

    if getattr(config, "use_peft", False):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(policy_path))
        if base_policy_path:
            peft_config.base_model_name_or_path = str(base_policy_path)
        base_path = peft_config.base_model_name_or_path
        if not base_path:
            raise ValueError("LoRA adapter has no base_model_name_or_path; pass --base-policy-path.")
        print(f"Loading base policy from: {base_path}", flush=True)
        policy = policy_cls.from_pretrained(base_path, config=config, local_files_only=True)
        print(f"Loading LoRA adapter from: {policy_path}", flush=True)
        policy = PeftModel.from_pretrained(policy, str(policy_path), config=peft_config, is_trainable=False)
    else:
        policy = policy_cls.from_pretrained(policy_path, config=config, local_files_only=True)

    policy.eval()
    policy.to(device)
    return policy


def make_observation(payload: dict, policy) -> dict:
    obs = {"observation.state": torch.tensor(payload["state"], dtype=torch.float32)}
    for key, frame_name in [
        ("observation.images.front_rgb", "front_rgb"),
        ("observation.images.wrist_rgb", "wrist_rgb"),
    ]:
        if key not in policy.config.input_features:
            continue
        image_h, image_w = feature_hw(policy, key)
        obs[key] = image_to_tensor(payload_image(payload, frame_name), image_w, image_h)
    obs["task"] = payload.get("task", "")
    return obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument("--base-policy-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--return-chunk",
        action="store_true",
        help="Return a full action chunk instead of one select_action() result.",
    )
    parser.add_argument(
        "--chunk-actions",
        type=int,
        default=None,
        help="Limit returned chunk length. Defaults to the policy n_action_steps.",
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Reset PyTorch sampling RNG before every request for reproducible diagnostic chunks.",
    )
    parser.add_argument(
        "--rtc-enabled",
        action="store_true",
        help="Enable RTC-V1 guidance between consecutive Pi0 action chunks.",
    )
    parser.add_argument(
        "--rtc-trained-prefix",
        action="store_true",
        help="Use the fixed-prefix inference mode for a checkpoint trained with Training-Time RTC.",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
        help="Number of action frames RTC preserves as its execution prefix.",
    )
    parser.add_argument(
        "--rtc-guidance-weight",
        type=float,
        default=10.0,
        help="Maximum RTC-V1 guidance weight during flow denoising.",
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=["zeros", "ones", "linear", "exp"],
        default="exp",
        help="RTC prefix attention schedule.",
    )
    args = parser.parse_args()

    if (args.rtc_enabled or args.rtc_trained_prefix) and not args.return_chunk:
        parser.error("RTC chunk modes require --return-chunk.")
    if args.rtc_enabled and args.rtc_trained_prefix:
        parser.error("--rtc-enabled and --rtc-trained-prefix cannot be combined.")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    policy = load_policy(
        args.policy_path,
        args.device,
        args.base_policy_path,
        rtc_enabled=args.rtc_enabled,
        rtc_trained_prefix=args.rtc_trained_prefix,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_guidance_weight=args.rtc_guidance_weight,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
    )
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"pi0 remote policy server listening on {args.host}:{args.port}", flush=True)

    while True:
        conn, addr = server.accept()
        print(f"client connected: {addr}", flush=True)
        with conn:
            step = 0
            while True:
                try:
                    payload = recv_msg(conn)
                except Exception as exc:
                    print(f"client disconnected: {exc}", flush=True)
                    break
                t0 = time.perf_counter()
                raw_obs = make_observation(payload, policy)
                batch = preprocessor(raw_obs)
                with torch.no_grad():
                    reset_inference_rng(args.inference_seed)
                    if args.return_chunk:
                        rtc_kwargs = rtc_kwargs_from_payload(
                            payload,
                            enabled=args.rtc_enabled or args.rtc_trained_prefix,
                            execution_horizon=args.rtc_execution_horizon,
                            device=args.device,
                        )
                        actions = policy.predict_action_chunk(batch, **rtc_kwargs)
                        n_actions = args.chunk_actions or getattr(policy.config, "n_action_steps", actions.shape[1])
                        model_action = actions[:, :n_actions]
                    else:
                        model_action = policy.select_action(batch)
                action = postprocessor(model_action).detach().cpu().numpy()
                rtc_original_action = None
                if args.rtc_enabled or args.rtc_trained_prefix:
                    rtc_original_action = model_action.detach().cpu().numpy()
                if args.return_chunk:
                    action = action.reshape(action.shape[-2], action.shape[-1])
                    if rtc_original_action is not None:
                        rtc_original_action = rtc_original_action.reshape(
                            rtc_original_action.shape[-2], rtc_original_action.shape[-1]
                        )
                else:
                    action = action.reshape(-1)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                response = {"action": action, "latency_ms": dt_ms, "step": step}
                if rtc_original_action is not None:
                    response["rtc_original_action"] = rtc_original_action
                send_msg(conn, response)
                print(
                    f"step={step} latency_ms={dt_ms:.1f} action_shape={list(action.shape)} "
                    f"first_action={np.round(action[0] if args.return_chunk else action, 3).tolist()} "
                    f"rtc_v1={args.rtc_enabled} rtc_trained_prefix={args.rtc_trained_prefix}",
                    flush=True,
                )
                step += 1


if __name__ == "__main__":
    main()
