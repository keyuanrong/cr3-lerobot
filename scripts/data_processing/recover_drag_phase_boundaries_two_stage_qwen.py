#!/usr/bin/env python
"""Recover reviewed-successful CR3 trajectories whose gripper CSV events are unusable.

This script is deliberately separate from the strict gripper-guardrail refiner.
It only considers records that failed with no_candidate_frames_after_phase_guardrails.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.data_processing.filter_drag_phase_segments_with_qwen import panel_as_data_url, read_rows, request_qwen
from scripts.data_processing.refine_drag_phase_boundaries_with_qwen import (
    BOUNDARY_NAMES,
    PHASES,
    action_key,
    build_messages,
    make_refined_segments,
    minimum_phase_frames,
    normalize_boundary_answer,
    write_jsonl,
)


RECOVERY_VERSION = 1
EMPTY_GUARDRAIL_REASON = "no_candidate_frames_after_phase_guardrails"
GLOBAL_ANCHOR_NAMES = ("lift", "arrival", "release")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage visual recovery for reviewed CR3 color-task phase segments.")
    parser.add_argument("--source-report", required=True, type=Path, help="Report containing empty-guardrail uncertain records.")
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument("--global-sample-frames", type=int, default=32)
    parser.add_argument("--local-window-frames", type=int, default=120)
    parser.add_argument("--local-sample-frames", type=int, default=64)
    parser.add_argument("--grasp-clip-count", type=int, default=8)
    parser.add_argument("--grasp-frames-per-clip", type=int, default=8)
    parser.add_argument("--approach-wrist-sample-frames", type=int, default=None)
    parser.add_argument("--max-boundary-snap-frames", type=int, default=5)
    parser.add_argument("--min-visual-confidence", type=float, default=0.75)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--min-approach-frames", type=int, default=20)
    parser.add_argument("--min-grasp-frames", type=int, default=20)
    parser.add_argument("--min-carry-frames", type=int, default=20)
    parser.add_argument("--min-place-frames", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="0 means all retryable color tasks.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing two-stage output records.")
    args = parser.parse_args()
    if args.approach_wrist_sample_frames is None:
        args.approach_wrist_sample_frames = args.local_sample_frames
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def snap_global_boundaries(answer: dict[str, Any], candidates: list[int]) -> dict[str, Any]:
    """Snap coarse global choices to the closest displayed sparse candidate.

    The global stage only locates a rough region, so its choices may be dozens
    of frames from a sparse panel. The local stage performs the strict <=5-frame
    snap once it receives dense candidates.
    """
    snapped = dict(answer)
    raw_boundaries = answer.get("boundaries") if isinstance(answer.get("boundaries"), dict) else {}
    snapped["boundaries"] = {
        name: min(candidates, key=lambda candidate: abs(candidate - value)) if (value := to_int(raw_boundaries.get(name))) is not None else None
        for name in BOUNDARY_NAMES
    }
    return snapped


def snap_global_anchors(answer: dict[str, Any], candidates: list[int]) -> dict[str, Any]:
    """Snap coarse three-event localization to the closest sparse panel."""
    snapped = dict(answer)
    raw_anchors = answer.get("anchors") if isinstance(answer.get("anchors"), dict) else {}
    snapped["anchors"] = {
        name: min(candidates, key=lambda candidate: abs(candidate - value)) if (value := to_int(raw_anchors.get(name))) is not None else None
        for name in GLOBAL_ANCHOR_NAMES
    }
    return snapped


def evenly_spaced_indices(start: int, end: int, count: int) -> list[int]:
    if count < 1 or end < start:
        raise ValueError("invalid_global_sample_range")
    length = end - start + 1
    actual_count = min(count, length)
    if actual_count == 1:
        return [start]
    return [start + round(i * (length - 1) / (actual_count - 1)) for i in range(actual_count)]


def grasp_clip_ranges(center: int, frame_count: int, window_frames: int, clips: int) -> list[tuple[int, int]]:
    """Partition the local lift window into chronological inclusive clip ranges."""
    if clips < 1:
        raise ValueError("grasp_clip_count_must_be_positive")
    start = max(0, center - window_frames)
    end = min(frame_count - 1, center + window_frames)
    length = end - start + 1
    actual_clips = min(clips, length)
    base_length = length // actual_clips
    ranges = []
    cursor = start
    for index in range(actual_clips):
        clip_length = base_length if index < actual_clips - 1 else end - cursor + 1
        ranges.append((cursor, cursor + clip_length - 1))
        cursor += clip_length
    return ranges


def grasp_clip_as_data_url(
    episode: Path,
    rows: list[dict[str, str]],
    start: int,
    end: int,
    frames_per_clip: int,
    panel_width: int,
    jpeg_quality: int,
) -> str:
    """Render ordered front+wrist frame pairs as a compact temporal contact sheet."""
    indices = evenly_spaced_indices(start, end, frames_per_clip)
    camera_width = max(96, panel_width // 4)
    tiles = []
    for index in indices:
        images = []
        for column, label in (("front_rgb", "front"), ("wrist_rgb", "wrist")):
            image = cv2.imread(str(episode / rows[index][column]), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"missing_or_unreadable_{column}")
            height = max(1, round(image.shape[0] * camera_width / image.shape[1]))
            image = cv2.resize(image, (camera_width, height), interpolation=cv2.INTER_AREA)
            cv2.putText(image, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(image, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            images.append(image)
        max_height = max(image.shape[0] for image in images)
        images = [
            cv2.copyMakeBorder(image, 0, max_height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            for image in images
        ]
        pair = cv2.hconcat(images)
        cv2.putText(pair, str(index), (6, pair.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(pair, str(index), (6, pair.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        tiles.append(pair)

    columns = 4
    blank = cv2.copyMakeBorder(tiles[-1], 0, 0, 0, 0, cv2.BORDER_CONSTANT)
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows_of_tiles = [cv2.hconcat(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    sheet = cv2.vconcat(rows_of_tiles)
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("jpeg_encoding_failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def grasp_clip_messages(action: dict[str, Any], rows: list[dict[str, str]], args: argparse.Namespace, center: int) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Ask Qwen to select the first clip that visibly contains the lift event."""
    episode = Path(action["episode"])
    clip_ranges = grasp_clip_ranges(center, len(rows), args.local_window_frames, args.grasp_clip_count)
    prompt = (
        "You inspect chronological contact sheets from one successful robot demonstration. "
        f"The target is the {action['color']} block. Each tile shows front view on the left and wrist view on the right. "
        "Choose the FIRST clip where the target is securely held and visibly lifted off the table. "
        "Do not choose a clip that only shows the fingers closing while the block remains on the table. "
        "If no clip visibly proves lift, return null and decision uncertain. "
        'Return JSON only: {"decision":"accept|uncertain","clip":1,"confidence":0.0,"reason":"..."}. '
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for clip_index, (start, end) in enumerate(clip_ranges, 1):
        content.append({"type": "text", "text": f"grasp clip {clip_index}: raw frames {start}..{end}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": grasp_clip_as_data_url(
                        episode, rows, start, end, args.grasp_frames_per_clip, args.panel_width, args.jpeg_quality
                    )
                },
            }
        )
    return [{"role": "user", "content": content}], clip_ranges


def recovery_actions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in records:
        if record.get("decision") != "uncertain" or record.get("reason") != EMPTY_GUARDRAIL_REASON:
            continue
        episode, color, coarse = record.get("episode"), record.get("color"), record.get("coarse")
        if not isinstance(episode, str) or not isinstance(color, str) or not isinstance(coarse, dict):
            continue
        if set(coarse) != set(PHASES):
            continue
        actions.append({"episode": episode, "color": color, "coarse": coarse})
    return sorted(actions, key=lambda item: (item["episode"], item["color"]))


def global_messages(action: dict[str, Any], rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[int]]:
    episode = Path(action["episode"])
    start = int(action["coarse"]["approach"][0])
    end = min(len(rows) - 1, int(action["coarse"]["place"][1]))
    indices = evenly_spaced_indices(start, end, args.global_sample_frames)
    prompt = (
        "You inspect one successful robot color-block demonstration in chronological order. "
        f"The target is the {action['color']} block. Locate only three coarse visual anchors: "
        "lift = FIRST frame where the target is securely held and visibly lifted off the table; "
        "arrival = FIRST frame where the held target arrives above or inside the black frame before release; "
        "release = FIRST frame after the gripper releases and the target remains inside the black frame. "
        "Each panel has front view on the left and wrist view on the right, with its real raw frame number. "
        "Use only displayed frame numbers. If any event is not visible, return null for it and decision uncertain. "
        "Do not infer an event from timing alone. Return JSON only: "
        '{"decision":"accept|uncertain","anchors":{"lift":123,"arrival":234,"release":345},'
        '"confidence":{"lift":0.0,"arrival":0.0,"release":0.0},"reason":"..."}.'
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index in indices:
        content.append({"type": "text", "text": f"global candidate raw frame {index}"})
        content.append({"type": "image_url", "image_url": {"url": panel_as_data_url(episode, rows[index], args.panel_width, args.jpeg_quality)}})
    return [{"role": "user", "content": content}], indices


def validate_visual_boundaries(
    answer: dict[str, Any], prior_end: int, frame_count: int, minimums: dict[str, int], min_confidence: float
) -> dict[str, int] | None:
    raw_boundaries = answer.get("boundaries")
    raw_confidence = answer.get("confidence")
    if str(answer.get("decision", "")).lower() != "accept" or not isinstance(raw_boundaries, dict) or not isinstance(raw_confidence, dict):
        return None
    boundaries = {name: to_int(raw_boundaries.get(name)) for name in BOUNDARY_NAMES}
    confidence = {name: to_float(raw_confidence.get(name)) for name in BOUNDARY_NAMES}
    if any(boundaries[name] is None or confidence[name] is None or confidence[name] < min_confidence for name in BOUNDARY_NAMES):
        return None
    values = [int(boundaries[name]) for name in BOUNDARY_NAMES]
    if not (prior_end < values[0] < values[1] < values[2] < values[3] < frame_count):
        return None
    points = [prior_end, *values]
    if any(end - start < minimums[phase] for phase, start, end in zip(PHASES, points[:-1], points[1:], strict=True)):
        return None
    return dict(zip(BOUNDARY_NAMES, values, strict=True))


def validate_global_anchors(answer: dict[str, Any], prior_end: int, frame_count: int, min_confidence: float) -> dict[str, int] | None:
    """Validate only coarse visual event order, never phase lengths.

    Global panels are intentionally sparse. They locate regions for the local
    pass and must not reject a good trajectory solely because two events are
    fewer than a final training phase's minimum length apart.
    """
    raw_anchors = answer.get("anchors")
    raw_confidence = answer.get("confidence")
    if str(answer.get("decision", "")).lower() != "accept" or not isinstance(raw_anchors, dict) or not isinstance(raw_confidence, dict):
        return None
    anchors = {name: to_int(raw_anchors.get(name)) for name in GLOBAL_ANCHOR_NAMES}
    confidence = {name: to_float(raw_confidence.get(name)) for name in GLOBAL_ANCHOR_NAMES}
    if any(anchors[name] is None or confidence[name] is None or confidence[name] < min_confidence for name in GLOBAL_ANCHOR_NAMES):
        return None
    values = [int(anchors[name]) for name in GLOBAL_ANCHOR_NAMES]
    if not (prior_end < values[0] < values[1] < values[2] < frame_count):
        return None
    return dict(zip(GLOBAL_ANCHOR_NAMES, values, strict=True))


def action_with_boundaries(action: dict[str, Any], boundaries: dict[str, int]) -> dict[str, Any]:
    start = int(action["coarse"]["approach"][0])
    points = [start, *(boundaries[name] for name in BOUNDARY_NAMES)]
    coarse = {phase: (phase_start, phase_end) for phase, phase_start, phase_end in zip(PHASES, points[:-1], points[1:], strict=True)}
    return {**action, "coarse": coarse}


def local_coarse_boundaries(start: int, anchors: dict[str, int], minimums: dict[str, int]) -> dict[str, int] | None:
    """Create local-window centers from global event anchors.

    The approach center is deliberately before the confirmed lift, while the
    other three centers use the visually confirmed lift/arrival/release events.
    """
    approach_end = anchors["lift"] - minimums["grasp"]
    boundaries = {
        "approach_end": approach_end,
        "grasp_end": anchors["lift"],
        "carry_end": anchors["arrival"],
        "place_end": anchors["release"],
    }
    values = [boundaries[name] for name in BOUNDARY_NAMES]
    if not (start < values[0] < values[1] < values[2] < values[3]):
        return None
    return boundaries


def process_action(action: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    episode = Path(action["episode"])
    try:
        rows = read_rows(episode)
        minimums = minimum_phase_frames(args)
        global_request, global_allowed = global_messages(action, rows, args)
        global_answer = request_qwen(global_request, args)
        normalized_global = snap_global_anchors(global_answer, global_allowed)
        global_anchors = validate_global_anchors(
            normalized_global,
            int(action["coarse"]["approach"][0]),
            len(rows),
            args.min_visual_confidence,
        )
        if global_anchors is None:
            return recovery_record(action, "uncertain", "global_visual_events_not_confirmed", [], global_answer=global_answer)

        local_boundaries = local_coarse_boundaries(int(action["coarse"]["approach"][0]), global_anchors, minimums)
        if local_boundaries is None:
            return recovery_record(action, "uncertain", "global_visual_anchors_invalid", [], global_answer=global_answer)
        local_action = action_with_boundaries(action, local_boundaries)
        messages_by_boundary, allowed = build_messages(
            local_action,
            rows,
            args,
            window_frames=args.local_window_frames,
            sample_frames=args.local_sample_frames,
        )
        local_answers: dict[str, dict[str, Any]] = {}
        for name in BOUNDARY_NAMES:
            allowed_for_boundary = allowed[name]
            clip_answer: dict[str, Any] | None = None
            if name == "grasp_end":
                clip_messages, clip_ranges = grasp_clip_messages(local_action, rows, args, global_anchors["lift"])
                clip_answer = request_qwen(clip_messages, args)
                clip_index = to_int(clip_answer.get("clip"))
                clip_confidence = to_float(clip_answer.get("confidence"))
                if (
                    str(clip_answer.get("decision", "")).lower() != "accept"
                    or clip_index is None
                    or not 1 <= clip_index <= len(clip_ranges)
                    or clip_confidence is None
                    or clip_confidence < args.min_visual_confidence
                ):
                    return recovery_record(
                        action,
                        "uncertain",
                        "local_grasp_clip_not_confirmed",
                        [],
                        global_answer,
                        {**local_answers, name: {"clip_answer": clip_answer}},
                    )
                clip_start, clip_end = clip_ranges[clip_index - 1]
                exact_boundaries = {**local_boundaries, "grasp_end": (clip_start + clip_end) // 2}
                exact_action = action_with_boundaries(action, exact_boundaries)
                exact_messages, exact_allowed = build_messages(
                    exact_action,
                    rows,
                    args,
                    boundary_names=(name,),
                    window_frames=max(1, clip_end - clip_start),
                    sample_frames=clip_end - clip_start + 1,
                    candidate_bounds={name: (clip_start, clip_end)},
                )
                answer = request_qwen(exact_messages[name], args)
                allowed_for_boundary = exact_allowed[name]
            else:
                answer = request_qwen(messages_by_boundary[name], args)
            local_answers[name] = {"raw_answer": answer, **({"clip_answer": clip_answer} if clip_answer else {})}
            if str(answer.get("decision", "")).lower() != "accept":
                return recovery_record(action, "uncertain", f"local_{name}_not_confirmed", [], global_answer, local_answers)
            normalized = normalize_boundary_answer(answer, allowed_for_boundary, args.max_boundary_snap_frames)
            confidence = to_float(answer.get("confidence"))
            if normalized.get("boundary") is None or confidence is None or confidence < args.min_visual_confidence:
                return recovery_record(action, "uncertain", f"local_{name}_not_confirmed", [], global_answer, local_answers)
            normalized["confidence"] = confidence
            normalized["raw_answer"] = answer
            if clip_answer:
                normalized["clip_answer"] = clip_answer
            local_answers[name] = normalized
            allowed[name] = allowed_for_boundary

        local_combined = {
            "decision": "accept",
            "boundaries": {name: local_answers[name]["boundary"] for name in BOUNDARY_NAMES},
            "confidence": {name: local_answers[name]["confidence"] for name in BOUNDARY_NAMES},
            "reason": json.dumps({name: local_answers[name].get("reason", "") for name in BOUNDARY_NAMES}, ensure_ascii=False),
        }
        local_boundaries = validate_visual_boundaries(
            local_combined,
            int(action["coarse"]["approach"][0]),
            len(rows),
            minimums,
            args.min_visual_confidence,
        )
        if local_boundaries is None:
            return recovery_record(action, "uncertain", "local_visual_boundaries_invalid", [], global_answer, local_answers)
        segments, errors = make_refined_segments(
            episode, action["color"], len(rows), int(action["coarse"]["approach"][0]), local_action["coarse"], local_combined, allowed
        )
        if errors:
            return recovery_record(action, "uncertain", ";".join(errors), [], global_answer, local_answers)
        for segment in segments:
            segment["boundary_source"] = "two_stage_visual_recovery"
        return recovery_record(action, "accept", "two_stage_visual_confirmed", segments, global_answer, local_answers)
    except Exception as exc:
        return recovery_record(action, "uncertain", f"request_or_parse_failed:{exc}", [])


def recovery_record(
    action: dict[str, Any],
    decision: str,
    reason: str,
    segments: list[dict[str, Any]],
    global_answer: dict[str, Any] | None = None,
    local_answers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "key": action_key(action),
        "recovery_version": RECOVERY_VERSION,
        "episode": action["episode"],
        "color": action["color"],
        "decision": decision,
        "reason": reason,
        "coarse": action["coarse"],
        "candidate_mode": "two_stage_visual_recovery",
        "global_answer": global_answer,
        "local_answers": local_answers,
        "segments": segments,
    }


def main() -> None:
    args = parse_args()
    source = read_jsonl(args.source_report)
    actions = recovery_actions(source)
    if args.limit:
        actions = actions[: args.limit]
    existing = [] if args.fresh else read_jsonl(args.report)
    completed = {item.get("key") for item in existing}
    pending = [action for action in actions if action_key(action) not in completed]
    report = [item for item in existing if item.get("key") in {action_key(action) for action in actions}]
    print(f"selected color tasks: {len(actions)}, already completed: {len(actions) - len(pending)}")
    for index, action in enumerate(pending, 1):
        record = process_action(action, args)
        report = [item for item in report if item.get("key") != record["key"]]
        report.append(record)
        segments = [segment for item in report if item.get("decision") == "accept" for segment in item.get("segments", [])]
        write_jsonl(args.report, report)
        write_jsonl(args.output_manifest, segments)
        print(f"[{index}/{len(pending)}] {record['decision'].upper()} {Path(action['episode']).name}/{action['color']}: {record['reason']}")
    accepted = sum(item.get("decision") == "accept" for item in report)
    segments = [segment for item in report if item.get("decision") == "accept" for segment in item.get("segments", [])]
    write_jsonl(args.report, report)
    write_jsonl(args.output_manifest, segments)
    print(f"done: accepted color tasks={accepted}, refined segments={len(segments)}")


if __name__ == "__main__":
    main()
