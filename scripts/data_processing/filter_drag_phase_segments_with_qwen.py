#!/usr/bin/env python
"""Three-stage quality control for CR3 phase-labelled demonstrations.

1. Local deterministic rules validate CSV, images, numeric actions and the
   gripper event expected for each phase.
2. A Qwen-compatible vision endpoint reviews only rule-passing segments.
3. The output JSONL files keep accept, reject and uncertain segments separate
   so only uncertain cases need manual review.

The source recordings and input manifest are never changed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PHASES = ("approach", "grasp", "carry", "place")
JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")
IMAGE_COLUMNS = ("front_rgb", "wrist_rgb")


class QwenAccessError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-first, Qwen-second quality filtering for CR3 phase segments.")
    parser.add_argument("--manifest", required=True, type=Path, help="Phase JSONL from the visual phase splitter.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phases", nargs="+", choices=PHASES, help="Optional subset, e.g. --phases grasp place.")
    parser.add_argument("--colors", nargs="+", choices=("red", "green", "yellow"), help="Optional color subset.")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument("--sample-frames", type=int, default=16, help="Temporal samples per phase sent to Qwen.")
    parser.add_argument("--panel-width", type=int, default=512, help="Width of each camera panel before composing a review image.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--gripper-threshold", type=float, default=50.0)
    parser.add_argument("--debounce-frames", type=int, default=5)
    parser.add_argument("--min-approach-frames", type=int, default=30)
    parser.add_argument("--min-grasp-frames", type=int, default=60)
    parser.add_argument("--min-carry-frames", type=int, default=15)
    parser.add_argument("--min-place-frames", type=int, default=30)
    parser.add_argument("--max-joint-step-deg", type=float, default=25.0)
    parser.add_argument("--verify-images", choices=("sample", "all"), default="sample")
    parser.add_argument(
        "--decode-rule-images",
        action="store_true",
        help="Also decode every rule-checked image. Slower; Qwen already decodes its sampled images.",
    )
    parser.add_argument("--keep-confidence", type=float, default=0.80)
    parser.add_argument("--reject-confidence", type=float, default=0.80)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 processes every segment.")
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic rules only. No API call is made.")
    parser.add_argument("--fresh", action="store_true", help="Discard an existing report in --output-dir.")
    parser.add_argument(
        "--retry-network-uncertain",
        action="store_true",
        help="Retry only previous Qwen uncertain results caused by a network/API request failure.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line_number, entry in enumerate(read_jsonl(path), 1):
        required = {"episode", "start", "end", "task", "color", "phase"}
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
        entries.append(entry)
    return entries


def segment_key(segment: dict[str, Any]) -> str:
    return "|".join((str(Path(segment["episode"]).resolve()), str(segment["start"]), str(segment["end"]), segment["task"]))


def read_rows(episode: Path) -> list[dict[str, str]]:
    with (episode / "data.csv").open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_events(rows: list[dict[str, str]], threshold: float, debounce: int) -> list[tuple[int, bool]]:
    values = [float(row["gripper"]) >= threshold for row in rows]
    if not values:
        return []
    state, candidate, count = values[0], None, 0
    events = []
    for index, value in enumerate(values[1:], 1):
        if value == state:
            candidate, count = None, 0
            continue
        if value != candidate:
            candidate, count = value, 1
        else:
            count += 1
        if count >= debounce:
            events.append((index - debounce + 1, value))
            state, candidate, count = value, None, 0
    return events


def min_frames_for_phase(args: argparse.Namespace, phase: str) -> int:
    return getattr(args, f"min_{phase}_frames")


def select_indices(start: int, end: int, count: int, events: list[tuple[int, bool]]) -> list[int]:
    length = end - start
    chosen: set[int] = {start, end - 1}
    for frame, _ in events:
        if start <= frame < end:
            chosen.add(frame)
    for index in range(min(max(1, count), length)):
        chosen.add(start + round(index * (length - 1) / max(1, min(count, length) - 1)))
    ordered = sorted(chosen)
    if len(ordered) <= count:
        return ordered
    # Keep boundaries/events first, then fill temporal coverage to the requested budget.
    required = sorted({start, end - 1, *(frame for frame, _ in events if start <= frame < end)})
    required = required[:count]
    for frame in ordered:
        if len(required) >= count:
            break
        if frame not in required:
            required.append(frame)
    return sorted(required)


def check_segment(
    segment: dict[str, Any], args: argparse.Namespace, rows_cache: dict[Path, list[dict[str, str]]]
) -> tuple[bool, list[str], list[dict[str, str]], list[int]]:
    reasons: list[str] = []
    episode = Path(segment["episode"])
    start, end, phase, color = int(segment["start"]), int(segment["end"]), segment["phase"], segment["color"]
    if phase not in PHASES:
        return False, [f"unknown_phase:{phase}"], [], []
    if color not in {"red", "green", "yellow"}:
        return False, [f"unknown_color:{color}"], [], []
    if color not in segment["task"].lower():
        reasons.append("task_color_mismatch")
    if not (episode / "data.csv").is_file():
        return False, ["missing_data_csv"], [], []
    try:
        if episode not in rows_cache:
            rows_cache[episode] = read_rows(episode)
        rows = rows_cache[episode]
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return False, [f"csv_read_error:{exc}"], [], []
    if not 0 <= start < end <= len(rows):
        return False, [f"invalid_frame_range:{start}:{end}:of:{len(rows)}"], rows, []
    if end - start < min_frames_for_phase(args, phase):
        reasons.append(f"segment_too_short:{end-start}")

    required_columns = set(JOINT_COLUMNS) | {"gripper", *IMAGE_COLUMNS}
    missing = required_columns - set(rows[0])
    if missing:
        return False, ["missing_columns:" + ",".join(sorted(missing))], rows, []
    segment_rows = rows[start:end]
    for row in segment_rows:
        action_text = (row.get("action") or "").lower()
        if "gripper_error" in action_text or "modbus" in action_text or "error" in action_text:
            reasons.append("recorded_gripper_or_control_error")
            break
    previous = None
    for offset, row in enumerate(segment_rows):
        try:
            joints = np.asarray([float(row[column]) for column in JOINT_COLUMNS], dtype=np.float32)
            gripper = float(row["gripper"])
        except (KeyError, ValueError):
            reasons.append(f"invalid_numeric_row:{start + offset}")
            break
        if not np.isfinite(joints).all() or not math.isfinite(gripper):
            reasons.append(f"non_finite_numeric_row:{start + offset}")
            break
        if previous is not None and float(np.max(np.abs(joints - previous))) > args.max_joint_step_deg:
            reasons.append(f"joint_jump_gt_{args.max_joint_step_deg:g}deg")
            break
        previous = joints

    events = stable_events(segment_rows, args.gripper_threshold, args.debounce_frames)
    event_states = [is_open for _, is_open in events]
    final_open = float(segment_rows[-1]["gripper"]) >= args.gripper_threshold
    starts_open = float(segment_rows[0]["gripper"]) >= args.gripper_threshold
    if phase == "grasp" and not ((starts_open or True in event_states) and False in event_states and not final_open):
        reasons.append("grasp_missing_open_then_close")
    elif phase == "carry" and final_open:
        reasons.append("carry_ends_open")
    elif phase == "place" and not (True in event_states and final_open):
        reasons.append("place_missing_release_open")

    image_indices = list(range(start, end)) if args.verify_images == "all" else select_indices(start, end, args.sample_frames, [(start + frame, state) for frame, state in events])
    for index in image_indices:
        row = rows[index]
        for column in IMAGE_COLUMNS:
            path = episode / row[column]
            if not path.is_file() or (args.decode_rule_images and cv2.imread(str(path), cv2.IMREAD_COLOR) is None):
                reasons.append(f"missing_or_unreadable_{column}:{index}")
                break
        if reasons and reasons[-1].startswith("missing_or_unreadable"):
            break
    return not reasons, reasons, rows, image_indices


def panel_as_data_url(episode: Path, row: dict[str, str], width: int, quality: int) -> str:
    images = []
    for column, label in zip(IMAGE_COLUMNS, ("front", "wrist"), strict=True):
        image = cv2.imread(str(episode / row[column]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"missing_or_unreadable_{column}")
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(image, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        images.append(image)
    max_height = max(image.shape[0] for image in images)
    images = [cv2.copyMakeBorder(image, 0, max_height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT) for image in images]
    ok, encoded = cv2.imencode(".jpg", cv2.hconcat(images), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg_encoding_failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_messages(segment: dict[str, Any], rows: list[dict[str, str]], indices: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    expected = {
        "approach": "Move toward the specified block and finish above or immediately beside it. Do not judge grasping here.",
        "grasp": "Open, accurately align with the specified block, close on it, and lift it from the table.",
        "carry": "Keep the specified block securely held and move it toward the black frame.",
        "place": "Move the held specified block into the black frame and release it there.",
    }[segment["phase"]]
    prompt = (
        "You are a strict robot-data quality inspector. Each image has the external camera on the left and wrist "
        "camera on the right, ordered in time. Review exactly one labelled phase. "
        f"Target color: {segment['color']}. Phase: {segment['phase']}. Expected behavior: {expected} "
        "Reject wrong-color actions, missed grasps, dropped blocks, obvious collision, or a phase that does not match "
        "its label. Use uncertain when visibility is insufficient. Return JSON only with keys: decision (accept/reject/uncertain), "
        "confidence (0 to 1), target_color_correct (true/false/null), phase_completed (true/false/null), "
        "grasp_confirmed (true/false/null), object_held (true/false/null), placement_confirmed (true/false/null), reason."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    episode = Path(segment["episode"])
    for index in indices:
        content.append({"type": "text", "text": f"raw frame {index}"})
        content.append({"type": "image_url", "image_url": {"url": panel_as_data_url(episode, rows[index], args.panel_width, args.jpeg_quality)}})
    return [{"role": "user", "content": content}]


def parse_json_response(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model_response_is_not_json")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model_response_is_not_object")
    return value


def request_qwen(messages: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise QwenAccessError(f"missing {args.api_key_env}")
    if not args.base_url:
        raise QwenAccessError("missing DASHSCOPE_BASE_URL or --base-url")
    base_url = args.base_url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    payload = {"model": args.model, "messages": messages, "temperature": 0, "enable_thinking": False, "response_format": {"type": "json_object"}}
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return parse_json_response(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code in {401, 403}:
                raise QwenAccessError(f"HTTP {exc.code}: {detail}") from exc
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < args.retries:
            time.sleep(2**attempt)
    raise RuntimeError(f"qwen_request_failed:{last_error}")


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def decide_qwen(answer: dict[str, Any], phase: str, args: argparse.Namespace) -> tuple[str, float, str, dict[str, bool | None]]:
    decision = str(answer.get("decision", "uncertain")).lower().strip()
    try:
        confidence = float(answer.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    checks = {key: as_bool(answer.get(key)) for key in ("target_color_correct", "phase_completed", "grasp_confirmed", "object_held", "placement_confirmed")}
    required = [checks["target_color_correct"], checks["phase_completed"]]
    if phase == "grasp":
        required.append(checks["grasp_confirmed"])
    elif phase == "carry":
        required.append(checks["object_held"])
    elif phase == "place":
        required.append(checks["placement_confirmed"])
    reason = str(answer.get("reason", "")).replace("\n", " ").strip()
    if decision == "accept" and all(value is True for value in required) and confidence >= args.keep_confidence:
        return "accept", confidence, reason, checks
    if decision == "reject" and any(value is False for value in required) and confidence >= args.reject_confidence:
        return "reject", confidence, reason, checks
    return "uncertain", confidence, reason or "low_confidence_or_incomplete_evidence", checks


def append_jsonl(path: Path, value: dict[str, Any]):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_decision_manifests(output_dir: Path, report_path: Path):
    grouped: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in ("rule_pass", "accept", "reject", "uncertain", "rule_reject")}
    for item in read_jsonl(report_path):
        segment = item["segment"]
        key = segment_key(segment)
        if item["stage"] == "rule" and item["decision"] == "pass":
            grouped["rule_pass"][key] = segment
        elif item["stage"] == "rule":
            grouped["rule_reject"][key] = segment
            grouped["reject"][key] = segment
        elif item["decision"] in {"accept", "reject", "uncertain"}:
            grouped[item["decision"]][key] = segment
    for name, values in grouped.items():
        ordered = sorted(values.values(), key=lambda item: (item["episode"], item["start"], item["end"]))
        (output_dir / f"phase_{name}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8"
        )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "phase_filter_report.jsonl"
    if args.fresh and report_path.exists():
        report_path.unlink()
    completed = set()
    if report_path.exists():
        for item in read_jsonl(report_path):
            is_network_uncertain = (
                item.get("stage") == "qwen"
                and item.get("decision") == "uncertain"
                and str(item.get("reason", "")).startswith("qwen_request_failed:")
            )
            if args.retry_network_uncertain and is_network_uncertain:
                continue
            if item.get("stage") == "qwen" or (item.get("stage") == "rule" and item.get("decision") == "reject"):
                completed.add(item["key"])
    segments = load_manifest(args.manifest)
    if args.phases:
        segments = [segment for segment in segments if segment["phase"] in args.phases]
    if args.colors:
        segments = [segment for segment in segments if segment["color"] in args.colors]
    if args.limit:
        segments = segments[: args.limit]
    print(f"selected segments: {len(segments)}, already completed: {len(completed)}")
    counts: Counter[str] = Counter()
    rows_cache: dict[Path, list[dict[str, str]]] = {}

    for number, segment in enumerate(segments, 1):
        key = segment_key(segment)
        if key in completed:
            continue
        ok, reasons, rows, image_indices = check_segment(segment, args, rows_cache)
        base = {"key": key, "segment": segment, "frames": int(segment["end"]) - int(segment["start"])}
        if not ok:
            append_jsonl(report_path, base | {"stage": "rule", "decision": "reject", "reason": "|".join(reasons)})
            counts["rule_reject"] += 1
            print(f"[{number}/{len(segments)}] RULE_REJECT {segment['color']}/{segment['phase']}: {'|'.join(reasons)}")
            continue
        append_jsonl(report_path, base | {"stage": "rule", "decision": "pass", "reason": "rule_pass"})
        if args.dry_run:
            counts["rule_pass"] += 1
            print(f"[{number}/{len(segments)}] RULE_PASS {segment['color']}/{segment['phase']}")
            continue
        try:
            answer = request_qwen(build_messages(segment, rows, image_indices, args), args)
            decision, confidence, reason, checks = decide_qwen(answer, segment["phase"], args)
            append_jsonl(report_path, base | {"stage": "qwen", "decision": decision, "confidence": confidence, "reason": reason, **checks})
            counts[decision] += 1
            print(f"[{number}/{len(segments)}] {decision.upper()} {segment['color']}/{segment['phase']} confidence={confidence:.2f}")
        except QwenAccessError as exc:
            raise SystemExit(f"Qwen access failed; no Qwen result was written for this segment. {exc}") from exc
        except Exception as exc:
            append_jsonl(report_path, base | {"stage": "qwen", "decision": "uncertain", "reason": str(exc)})
            counts["uncertain"] += 1
            print(f"[{number}/{len(segments)}] UNCERTAIN {segment['color']}/{segment['phase']}: {exc}")

    if report_path.exists():
        write_decision_manifests(args.output_dir, report_path)
    print("done: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
