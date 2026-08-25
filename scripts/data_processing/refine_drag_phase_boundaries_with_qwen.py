#!/usr/bin/env python
"""Refine coarse CR3 task boundaries with dense, local Qwen visual evidence.

The first visual Qwen pass finds approximate approach/grasp/carry/place ends for
each color. This script only revisits the four local neighborhoods, using dense
front+wrist image panels, and writes a separate auditable refined manifest.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.data_processing.filter_drag_phase_segments_with_qwen import panel_as_data_url, read_rows, request_qwen, stable_events
from scripts.data_processing.segment_drag_trajectories_with_qwen import PHASES, number, phase_task


BOUNDARY_NAMES = tuple(f"{phase}_end" for phase in PHASES)
REFINEMENT_VERSION = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Densely refine Qwen visual phase boundaries for CR3 data.")
    parser.add_argument(
        "--input-manifest", required=True, type=Path, action="append", help="First-pass Qwen segments.jsonl; may be repeated."
    )
    parser.add_argument(
        "--fallback-report",
        type=Path,
        action="append",
        default=[],
        help="First-pass report JSONL whose visual answers can seed coarse candidates rejected only by boundary validation.",
    )
    parser.add_argument("--output-manifest", required=True, type=Path, help="Refined segments.jsonl")
    parser.add_argument("--report", required=True, type=Path, help="Per color-task audit report JSONL")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument("--window-frames", type=int, default=180, help="Frames before and after each coarse boundary.")
    parser.add_argument("--sample-frames-per-boundary", type=int, default=48)
    parser.add_argument(
        "--approach-wrist-sample-frames",
        type=int,
        default=96,
        help="Wrist-only candidate count for move_end; kept separate from the other three dual-view boundaries.",
    )
    parser.add_argument("--fallback-window-frames", type=int, default=180, help="Wider local window for an uncertain boundary.")
    parser.add_argument("--fallback-sample-frames", type=int, default=64, help="Dense samples for an uncertain-boundary retry.")
    parser.add_argument(
        "--max-boundary-snap-frames",
        type=int,
        default=5,
        help="Maximum Qwen-to-displayed-frame difference accepted before a boundary retry.",
    )
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--visual-fallback-on-empty-guardrails",
        action="store_true",
        help="When strict gripper guardrails leave no visual candidates, repair candidates inside their legal event bounds.",
    )
    parser.add_argument(
        "--visual-recovery-no-gripper-events",
        action="store_true",
        help="For an empty strict candidate set, use coarse visual windows without gripper-event bounds. Use only for reviewed-valid data.",
    )
    parser.add_argument(
        "--retry-empty-guardrails-only",
        action="store_true",
        help="Replace only prior uncertain records caused by no_candidate_frames_after_phase_guardrails.",
    )
    parser.add_argument(
        "--exclude-episode-list",
        type=Path,
        help="Optional text file of raw episode paths to exclude from processing, one path per line.",
    )
    parser.add_argument("--gripper-threshold", type=float, default=50.0)
    parser.add_argument("--gripper-debounce-frames", type=int, default=5)
    parser.add_argument("--event-grace-frames", type=int, default=15)
    parser.add_argument("--min-approach-frames", type=int, default=30)
    parser.add_argument("--min-grasp-frames", type=int, default=30)
    parser.add_argument("--min-carry-frames", type=int, default=30)
    parser.add_argument("--min-place-frames", type=int, default=30)
    parser.add_argument("--full-pre-grasp-window-frames", type=int, default=180)
    parser.add_argument("--full-post-release-window-frames", type=int, default=180)
    parser.add_argument("--full-event-context-frames", type=int, default=15)
    parser.add_argument(
        "--post-place-tail-frames",
        type=int,
        default=60,
        help="Keep this many frames after each place_end; cap at the episode end.",
    )
    parser.add_argument(
        "--full-final-place-tail-frames",
        type=int,
        default=30,
        help="For the final yellow place in a full task, keep this many frames before return_home.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Concurrent color tasks. Use 2 for the Qwen gateway.")
    parser.add_argument("--limit", type=int, default=0, help="0 means all color tasks.")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing report and call Qwen again.")
    parser.add_argument("--rebuild-only", action="store_true", help="Rebuild contiguous output segments from an existing report without Qwen calls.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def dense_indices(center: int, frame_count: int, window_frames: int, count: int) -> list[int]:
    """Return evenly spaced raw-frame indices in a clipped local window."""
    if frame_count <= 0:
        raise ValueError("empty_episode")
    if window_frames < 0 or count <= 0:
        raise ValueError("window_frames_and_count_must_be_positive")
    start = max(0, center - window_frames)
    end = min(frame_count - 1, center + window_frames)
    length = end - start + 1
    actual_count = min(count, length)
    if actual_count == 1:
        return [start]
    return [start + round(index * (length - 1) / (actual_count - 1)) for index in range(actual_count)]


def group_coarse_segments(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for segment in records:
        phase = segment.get("phase")
        if phase in PHASES and isinstance(segment.get("episode"), str) and isinstance(segment.get("color"), str):
            grouped[segment["episode"]][segment["color"]][phase] = segment

    actions: list[dict[str, Any]] = []
    for episode, colors in grouped.items():
        for color, phases in colors.items():
            if set(phases) != set(PHASES):
                continue
            coarse = {phase: (int(phases[phase]["start"]), int(phases[phase]["end"])) for phase in PHASES}
            actions.append({"episode": episode, "color": color, "coarse": coarse, "coarse_source": "manifest"})
    return sorted(actions, key=lambda item: (item["episode"], item["coarse"]["approach"][0]))


def actions_from_report(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover visual coarse candidates even when first-pass bounds were equal.

    Equal or misordered first-pass ends cannot be used for training, but they are
    valid centers for a dense local Qwen refinement window.
    """
    actions: list[dict[str, Any]] = []
    for record in records:
        episode = record.get("episode")
        answer = record.get("answer")
        if not isinstance(episode, str) or not isinstance(answer, dict):
            continue
        raw_segments = answer.get("segments")
        if not isinstance(raw_segments, list):
            continue
        cursor = 0
        for item in raw_segments:
            if not isinstance(item, dict) or not isinstance(item.get("color"), str):
                continue
            ends = [number(item.get(name)) for name in BOUNDARY_NAMES]
            if any(value is None for value in ends):
                continue
            values = [int(value) for value in ends]
            points = [cursor, *values]
            coarse = {phase: (start, end) for phase, start, end in zip(PHASES, points[:-1], values, strict=True)}
            actions.append({"episode": episode, "color": item["color"], "coarse": coarse, "coarse_source": "report_fallback"})
            cursor = max(cursor, values[-1])
    return actions


def action_key(action: dict[str, Any]) -> str:
    return f"{Path(action['episode']).resolve()}|{action['color']}"


def is_retryable_record(record: dict[str, Any]) -> bool:
    """Network/API failures should be retried on the next invocation."""
    return "request_or_parse_failed" in record.get("errors", []) or record.get("refinement_version") != REFINEMENT_VERSION


def is_empty_guardrail_record(record: dict[str, Any]) -> bool:
    return record.get("decision") == "uncertain" and record.get("reason") == "no_candidate_frames_after_phase_guardrails"


def actions_for_empty_guardrail_retry(actions: list[dict[str, Any]], report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retry_keys = {item.get("key") for item in report if is_empty_guardrail_record(item)}
    return [action for action in actions if action_key(action) in retry_keys]


def validate_workers(workers: int) -> int:
    if workers < 1:
        raise ValueError("workers_must_be_at_least_1")
    return workers


def minimum_phase_frames(args: argparse.Namespace) -> dict[str, int]:
    return {phase: max(1, int(getattr(args, f"min_{phase}_frames"))) for phase in PHASES}


def fallback_window_for_boundary(name: str, args: argparse.Namespace) -> int:
    """Keep pickup and arrival refinement local instead of drifting into later phases."""
    if name in {"grasp_end", "carry_end"}:
        return args.window_frames
    return args.fallback_window_frames


def is_full_task(action: dict[str, Any]) -> bool:
    return Path(action["episode"]).parent.name == "full"


def full_gripper_cycles(rows: list[dict[str, str]], threshold: float, debounce: int) -> list[tuple[int, int]]:
    """Return complete close/open cycles in chronological order."""
    cycles: list[tuple[int, int]] = []
    close: int | None = None
    for index, is_open in stable_events(rows, threshold=threshold, debounce=debounce):
        if not is_open:
            close = index
        elif close is not None and index > close:
            cycles.append((close, index))
            close = None
    return cycles


def full_task_cycle(
    action: dict[str, Any], rows: list[dict[str, str]], threshold: float, debounce: int
) -> tuple[int, int] | None:
    """Map red, green, yellow to the first, second, third complete gripper cycle."""
    colors = ("red", "green", "yellow")
    try:
        index = colors.index(action["color"])
    except ValueError:
        return None
    cycles = full_gripper_cycles(rows, threshold=threshold, debounce=debounce)
    return cycles[index] if len(cycles) > index else None


def range_indices(start: int, end: int, count: int, required: tuple[int, ...] = ()) -> list[int]:
    if start > end:
        return []
    length = end - start + 1
    actual_count = min(max(1, count), length)
    chosen = {start, end, *(index for index in required if start <= index <= end)}
    if actual_count == 1:
        return [start]
    for index in range(actual_count):
        chosen.add(start + round(index * (length - 1) / (actual_count - 1)))
    return sorted(chosen)


def full_boundary_candidate_indices(
    name: str,
    close: int,
    release: int,
    frame_count: int,
    pre_grasp_window: int,
    post_release_window: int,
    event_context_frames: int,
    min_grasp_frames: int,
    min_carry_frames: int,
    count: int,
) -> list[int]:
    """Generate full-task visual candidates from the relevant gripper cycle."""
    if name == "approach_end":
        return range_indices(
            max(0, close - pre_grasp_window),
            close,
            count,
            tuple(close - offset for offset in range(0, event_context_frames + 1, 5)),
        )
    if name == "grasp_end":
        return range_indices(close, release - min_carry_frames, count)
    if name == "carry_end":
        return range_indices(close + min_grasp_frames, release, count)
    if name == "place_end":
        return range_indices(release, min(frame_count - 1, release + post_release_window), count)
    raise ValueError(f"unknown_boundary:{name}")


def boundary_definition(name: str, full_task: bool) -> str:
    if name == "approach_end":
        return (
            "the FIRST frame where the target block first enters the still-open space between the two gripper fingers, "
            "starting near-field precise alignment; do not choose final descent alone, gripper closure, a later held-block, "
            "or a carry-to-frame image"
        )
    return {
        "approach_end": "the last clear pre-contact frame before the final descent or closing motion that leads to pickup",
        "grasp_end": "the FIRST frame where the block is securely grasped and visibly lifted off the table; never choose a later frame where it is merely still held or already moving toward the black frame",
        "carry_end": "the FIRST frame where the held block has arrived above or inside the black frame before release; never choose a later frame just because it remains above the frame",
        "place_end": "the gripper released and the block stays inside the black frame",
    }[name]


def wrist_as_data_url(episode: Path, row: dict[str, str], width: int, quality: int) -> str:
    """Encode one wrist-camera frame without shrinking it beside the front view."""
    image = cv2.imread(str(episode / row["wrist_rgb"]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("missing_or_unreadable_wrist_rgb")
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg_encoding_failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def gripper_phase_bounds(
    action: dict[str, Any],
    rows: list[dict[str, str]],
    threshold: float,
    debounce: int,
    event_grace_frames: int,
    min_carry_frames: int,
    min_grasp_frames: int = 30,
    full_task: bool = False,
) -> dict[str, tuple[int | None, int | None]]:
    """Return visual-candidate guardrails around one close/open gripper cycle.

    The gripper does not choose a boundary. It only prevents a visual answer
    from placing a completed grasp after release or collapsing carry to a few
    frames. Missing events leave Qwen unconstrained for that action.
    """
    coarse = action["coarse"]
    start = int(coarse["approach"][0])
    end = int(coarse["place"][1])
    events = stable_events(rows, threshold=threshold, debounce=debounce)
    close = next((index for index, is_open in events if start <= index <= end and not is_open), None)
    if close is None:
        return {name: (None, None) for name in BOUNDARY_NAMES}
    release = next((index for index, is_open in events if index > close and is_open), None)
    if release is None:
        return {name: (None, None) for name in BOUNDARY_NAMES}
    return {
        "approach_end": (None, close if full_task else close + max(0, event_grace_frames)),
        "grasp_end": (close, release - max(1, min_carry_frames)),
        # A boundary is the first frame of the following phase, so carry may
        # end on the first open/release frame while its final carry sample is
        # still held in the preceding half-open interval.
        "carry_end": (close + max(1, min_grasp_frames), release),
        "place_end": (release, None),
    }


def phase_bounds_from_cycle(
    close: int, release: int, event_grace_frames: int, min_grasp_frames: int, min_carry_frames: int, full_task: bool
) -> dict[str, tuple[int | None, int | None]]:
    return {
        "approach_end": (None, close if full_task else close + max(0, event_grace_frames)),
        "grasp_end": (close, release - max(1, min_carry_frames)),
        "carry_end": (close + max(1, min_grasp_frames), release),
        "place_end": (release, None),
    }


def apply_bounds(indices: list[int], bounds: tuple[int | None, int | None]) -> list[int]:
    lower, upper = bounds
    return [index for index in indices if (lower is None or index >= lower) and (upper is None or index <= upper)]


def repair_candidate_indices(
    sampled: list[int],
    bounds: tuple[int | None, int | None],
    after_frame: int,
    frame_count: int,
    count: int,
) -> list[int]:
    """Keep valid repair candidates, adding an exact legal edge when needed."""
    eligible = [index for index in apply_bounds(sampled, bounds) if index > after_frame]
    if eligible:
        return eligible
    lower, upper = bounds
    start = max(0, after_frame + 1, 0 if lower is None else lower)
    end = min(frame_count - 1, frame_count - 1 if upper is None else upper)
    if start > end:
        return []
    length = end - start + 1
    actual_count = min(max(1, count), length)
    if actual_count == 1:
        return [start]
    return [start + round(index * (length - 1) / (actual_count - 1)) for index in range(actual_count)]


def first_too_short_phase(
    prior_end: int, boundaries: list[int | None], minimum_frames: dict[str, int]
) -> str | None:
    points = [prior_end, *boundaries]
    for phase, start, end in zip(PHASES, points[:-1], points[1:], strict=True):
        if end is None or end - start < minimum_frames[phase]:
            return phase
    return None


def build_messages(
    action: dict[str, Any],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    boundary_names: tuple[str, ...] = BOUNDARY_NAMES,
    window_frames: int | None = None,
    sample_frames: int | None = None,
    force_choice: bool = False,
    after_frame: int | None = None,
    candidate_bounds: dict[str, tuple[int | None, int | None]] | None = None,
    repair_empty_candidates: bool = False,
    full_cycle: tuple[int, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
    """Build one bounded-size Qwen request per visual boundary.

    Sending all four dense windows together produced a very large request body
    that the compatible-mode gateway reset during upload. Each request now has
    one bounded local sequence; move_end uses dense wrist-only evidence.
    """
    episode = Path(action["episode"])
    color = action["color"]
    coarse: dict[str, tuple[int, int]] = action["coarse"]
    local_window = args.window_frames if window_frames is None else window_frames
    local_samples = args.sample_frames_per_boundary if sample_frames is None else sample_frames
    phase_by_boundary = dict(zip(BOUNDARY_NAMES, PHASES, strict=True))
    samples_by_boundary = {
        name: max(local_samples, args.approach_wrist_sample_frames) if name == "approach_end" else local_samples
        for name in boundary_names
    }
    if full_cycle is None:
        indices_by_boundary = {
            name: dense_indices(coarse[phase_by_boundary[name]][1], len(rows), local_window, samples_by_boundary[name])
            for name in boundary_names
        }
    else:
        close, release = full_cycle
        minimums = minimum_phase_frames(args)
        indices_by_boundary = {
            name: full_boundary_candidate_indices(
                name,
                close,
                release,
                len(rows),
                args.full_pre_grasp_window_frames,
                args.full_post_release_window_frames,
                args.full_event_context_frames,
                minimums["grasp"],
                minimums["carry"],
                samples_by_boundary[name],
            )
            for name in boundary_names
        }
    unbounded_indices = indices_by_boundary
    if candidate_bounds is not None:
        indices_by_boundary = {
            name: apply_bounds(indices, candidate_bounds.get(name, (None, None)))
            for name, indices in indices_by_boundary.items()
        }
    if repair_empty_candidates:
        if candidate_bounds is None:
            raise ValueError("repair_empty_candidates_requires_bounds")
        indices_by_boundary = {
            name: repair_candidate_indices(
                unbounded_indices[name],
                candidate_bounds.get(name, (None, None)),
                after_frame=-1,
                frame_count=len(rows),
                count=samples_by_boundary[name],
            )
            for name in boundary_names
        }
    if after_frame is not None:
        indices_by_boundary = {
            name: repair_candidate_indices(
                indices,
                (None, None) if candidate_bounds is None else candidate_bounds.get(name, (None, None)),
                after_frame,
                len(rows),
                local_samples,
            )
            for name, indices in indices_by_boundary.items()
        }
    if any(not indices for indices in indices_by_boundary.values()):
        raise ValueError("no_candidate_frames_after_phase_guardrails")
    full_task = full_cycle is not None
    messages_by_boundary: dict[str, list[dict[str, Any]]] = {}
    for name in boundary_names:
        _, end = coarse[phase_by_boundary[name]]
        uncertainty_instruction = (
            "This demonstration has already passed a task-success review. Select the closest displayed frame even if the event is not perfectly sharp; do not use null."
            if force_choice
            else "If this event is not visually clear, choose null and decision uncertain."
        )
        order_instruction = "" if after_frame is None else f" Your selected frame must be strictly later than raw frame {after_frame}."
        view_instruction = (
            "Each image is a wrist-camera time point. "
            if name == "approach_end"
            else "Each panel is a time point: front camera is left and wrist camera is right. "
        )
        evidence_instruction = "what the wrist camera shows" if name == "approach_end" else "what both cameras show"
        prompt = (
            "You precisely locate one visual boundary in a robot demonstration. "
            + view_instruction
            + "Panel labels are real raw frame numbers. Do not guess from fixed timing. "
            f"For the {color} block, choose the frame where {boundary_definition(name, full_task)}. "
            f"The gripper timing below is only a guardrail; determine the exact frame from {evidence_instruction}. "
            f"Choose only one frame number displayed below. {uncertainty_instruction}{order_instruction} "
            'Return JSON only: {"decision":"accept|uncertain","boundary":123,"confidence":0.0,"reason":"..."}. '
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt
                + f" Local window for {name}, centered on coarse frame {end}: "
                + f"raw range {max(0, end - local_window)}..{min(len(rows) - 1, end + local_window)}.",
            }
        ]
        for index in indices_by_boundary[name]:
            content.append({"type": "text", "text": f"{name} candidate raw frame {index}"})
            image_url = (
                wrist_as_data_url(episode, rows[index], args.panel_width, args.jpeg_quality)
                if name == "approach_end"
                else panel_as_data_url(episode, rows[index], args.panel_width, args.jpeg_quality)
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        messages_by_boundary[name] = [{"role": "user", "content": content}]
    return messages_by_boundary, indices_by_boundary


def build_initial_messages(
    action: dict[str, Any],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    strict_candidate_bounds: dict[str, tuple[int | None, int | None]],
    full_cycle: tuple[int, int] | None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[int]],
    dict[str, tuple[int | None, int | None]] | None,
    str,
]:
    """Build strict candidates first, using visual-only coarse candidates only when explicitly enabled."""
    try:
        messages, allowed = build_messages(
            action, rows, args, candidate_bounds=strict_candidate_bounds, full_cycle=full_cycle
        )
    except ValueError as exc:
        if str(exc) != "no_candidate_frames_after_phase_guardrails":
            raise
        if getattr(args, "visual_recovery_no_gripper_events", False):
            messages, allowed = build_messages(
                action,
                rows,
                args,
                force_choice=True,
            )
            return messages, allowed, None, "visual_recovery_no_gripper_events"
        if not args.visual_fallback_on_empty_guardrails:
            raise
        messages, allowed = build_messages(
            action,
            rows,
            args,
            candidate_bounds=strict_candidate_bounds,
            repair_empty_candidates=True,
            full_cycle=full_cycle,
        )
        return messages, allowed, strict_candidate_bounds, "event_anchored_visual_fallback"
    return messages, allowed, strict_candidate_bounds, "strict_gripper_guardrails"


def snap_boundary_to_displayed(value: int | None, allowed: list[int], max_distance: int) -> int | None:
    if value is None or not allowed:
        return None
    closest = min(allowed, key=lambda candidate: abs(candidate - value))
    return closest if abs(closest - value) <= max_distance else None


def normalize_boundary_answer(answer: dict[str, Any], allowed: list[int], max_distance: int) -> dict[str, Any]:
    normalized = dict(answer)
    reported = number(answer.get("boundary"))
    normalized["reported_boundary"] = reported
    normalized["boundary"] = snap_boundary_to_displayed(reported, allowed, max_distance)
    return normalized


def should_retry_visual_boundary(answer: dict[str, Any]) -> bool:
    """Re-ask only when Qwen is uncertain and did not provide any frame."""
    return str(answer.get("decision", "")).lower() == "uncertain" and number(answer.get("boundary")) is None


def is_usable_boundary_answer(answer: dict[str, Any], allowed: list[int], max_distance: int) -> bool:
    return (
        str(answer.get("decision", "")).lower() in {"accept", "uncertain"}
        and snap_boundary_to_displayed(number(answer.get("boundary")), allowed, max_distance) is not None
    )


def first_nonincreasing_boundary(values: list[int | None]) -> str | None:
    previous = -1
    for name, value in zip(BOUNDARY_NAMES, values, strict=True):
        if value is None or value <= previous:
            return name
        previous = value
    return None


def combine_boundary_answers(answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    boundaries = {name: answer.get("boundary") for name, answer in answers.items()}
    confidence = {name: answer.get("confidence") for name, answer in answers.items()}
    reasons = {name: str(answer.get("reason", "")) for name, answer in answers.items()}
    accepted = all(
        str(answer.get("decision", "")).lower() in {"accept", "uncertain"} and number(answer.get("boundary")) is not None
        for answer in answers.values()
    )
    return {
        "decision": "accept" if accepted else "uncertain",
        "boundaries": boundaries,
        "confidence": confidence,
        "reason": json.dumps(reasons, ensure_ascii=False),
        "boundary_answers": answers,
    }


def make_refined_segments(
    episode: Path,
    color: str,
    frame_count: int,
    prior_end: int,
    coarse: dict[str, tuple[int, int]],
    answer: dict[str, Any],
    allowed_indices: dict[str, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    boundaries = answer.get("boundaries")
    if not isinstance(boundaries, dict):
        return [], ["missing_boundaries_object"]
    ends = [number(boundaries.get(name)) for name in BOUNDARY_NAMES]
    if any(value is None for value in ends):
        return [], ["missing_visual_boundary"]
    chosen = [int(value) for value in ends]
    if allowed_indices is not None:
        invalid = [name for name, value in zip(BOUNDARY_NAMES, chosen, strict=True) if value not in allowed_indices[name]]
        if invalid:
            return [], [f"boundary_not_in_displayed_frames:{','.join(invalid)}"]
    if not (prior_end < chosen[0] < chosen[1] < chosen[2] < chosen[3] < frame_count):
        return [], [f"invalid_boundary_order:{chosen}"]

    points = [prior_end, *chosen]
    metadata = {
        "boundary_source": "qwen_dense_refine",
        "coarse_boundaries": {name: coarse[phase][1] for phase, name in zip(PHASES, BOUNDARY_NAMES, strict=True)},
        "refined_boundaries": dict(zip(BOUNDARY_NAMES, chosen, strict=True)),
        "qwen_confidence": answer.get("confidence", {}),
    }
    return [
        {
            "episode": str(episode),
            "start": start,
            "end": end,
            "task": phase_task(color, phase),
            "color": color,
            "phase": phase,
            **metadata,
        }
        for phase, start, end in zip(PHASES, points[:-1], chosen, strict=True)
    ], []


def make_segments_contiguous(
    action_segments: list[list[dict[str, Any]]],
    frame_counts: dict[str, int] | None = None,
    post_place_tail_frames: int = 0,
    full_final_place_tail_frames: int = 30,
) -> list[dict[str, Any]]:
    """Keep accepted phases continuous and retain a short post-release tail."""
    result: list[dict[str, Any]] = []
    successors = {"red": "green", "green": "yellow"}
    frame_counts = frame_counts or {}
    by_episode: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for segments in action_segments:
        if segments:
            by_episode[str(segments[0]["episode"])].append(segments)
    for episode, blocks in sorted(by_episode.items()):
        blocks = sorted(blocks, key=lambda segments: segments[0]["start"])
        colors = {str(block[0]["color"]) for block in blocks}
        is_full_episode = Path(episode).parent.name == "full"
        is_complete_full_task = is_full_episode and colors == {"red", "green", "yellow"}
        frame_count = frame_counts.get(episode)
        previous: list[dict[str, Any]] | None = None
        for block_index, block in enumerate(blocks):
            copied = [dict(segment) for segment in block]
            color = str(copied[0]["color"])
            next_block = blocks[block_index + 1] if block_index + 1 < len(blocks) else None
            has_direct_successor = next_block is not None and successors.get(color) == str(next_block[0]["color"])
            can_extend_place = frame_count is not None and (
                (not is_full_episode and len(blocks) == 1) or (is_complete_full_task and color == "yellow")
            )
            if can_extend_place and post_place_tail_frames > 0:
                place = copied[-1]
                tail_frames = (
                    full_final_place_tail_frames if is_complete_full_task and color == "yellow" else post_place_tail_frames
                )
                tail_end = min(int(place["end"]) + tail_frames, frame_count)
                place["end"] = tail_end
                place["post_place_tail_frames"] = tail_frames
            if (
                previous
                and successors.get(str(previous[0]["color"])) == str(copied[0]["color"])
                and previous[-1]["end"] < copied[0]["end"]
            ):
                copied[0]["start"] = previous[-1]["end"]
                copied[0]["continuity_adjusted_start"] = True
            for index in range(1, len(copied)):
                copied[index]["start"] = copied[index - 1]["end"]
            result.extend(copied)
            previous = copied
            if is_complete_full_task and color == "yellow" and frame_count is not None and copied[-1]["end"] < frame_count:
                return_home = dict(copied[-1])
                return_home.update(
                    {
                        "start": copied[-1]["end"],
                        "end": frame_count,
                        "phase": "return_home",
                        "task": "return the gripper to the home pose after completing the task",
                        "boundary_source": "post_place_return_home",
                    }
                )
                result.append(return_home)
    return result


def restore_qwen_segments(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Recover the four original Qwen phases before applying output-only tail rules."""
    coarse = item.get("coarse")
    segments = item.get("segments")
    if not isinstance(coarse, dict) or not isinstance(segments, list) or not segments:
        return None
    approach = coarse.get("approach")
    if not isinstance(approach, (list, tuple)) or not approach:
        return None
    answer = item.get("answer")
    boundaries = answer.get("boundaries") if isinstance(answer, dict) else None
    if not isinstance(boundaries, dict):
        boundaries = segments[0].get("refined_boundaries") if isinstance(segments[0], dict) else None
    if not isinstance(boundaries, dict):
        return None
    ends = [number(boundaries.get(name)) for name in BOUNDARY_NAMES]
    if any(end is None for end in ends):
        return None
    chosen = [int(end) for end in ends]
    start = int(approach[0])
    if not (start < chosen[0] < chosen[1] < chosen[2] < chosen[3]):
        return None
    templates = {str(segment.get("phase")): segment for segment in segments if segment.get("phase") in PHASES}
    color = str(item["color"])
    restored: list[dict[str, Any]] = []
    for phase, phase_start, phase_end in zip(PHASES, [start, *chosen[:-1]], chosen, strict=True):
        segment = dict(templates.get(phase, {}))
        segment.update(
            {
                "episode": str(item["episode"]),
                "start": phase_start,
                "end": phase_end,
                "task": phase_task(color, phase),
                "color": color,
                "phase": phase,
            }
        )
        segment.pop("continuity_adjusted_start", None)
        segment.pop("post_place_tail_frames", None)
        restored.append(segment)
    return restored


def rebuild_contiguous_report(
    report: list[dict[str, Any]],
    frame_counts: dict[str, int] | None = None,
    post_place_tail_frames: int = 0,
    full_final_place_tail_frames: int = 30,
) -> list[dict[str, Any]]:
    accepted = [item for item in report if item.get("decision") == "accept" and item.get("segments")]
    for item in accepted:
        restored = restore_qwen_segments(item)
        if restored is not None:
            item["segments"] = restored
            continue
        coarse = item.get("coarse")
        segments = item.get("segments")
        if not isinstance(coarse, dict) or not isinstance(segments, list) or not segments:
            continue
        approach = coarse.get("approach")
        if not isinstance(approach, (list, tuple)) or not approach:
            continue
        legacy_restored = [dict(segment) for segment in segments]
        legacy_restored[0]["start"] = int(approach[0])
        legacy_restored[0].pop("continuity_adjusted_start", None)
        item["segments"] = legacy_restored
    contiguous = make_segments_contiguous(
        [item["segments"] for item in accepted],
        frame_counts=frame_counts,
        post_place_tail_frames=post_place_tail_frames,
        full_final_place_tail_frames=full_final_place_tail_frames,
    )
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for segment in contiguous:
        by_key[(str(segment["episode"]), str(segment["color"]))].append(segment)
    for item in accepted:
        item["segments"] = by_key[(str(item["episode"]), str(item["color"]))]
    return contiguous


def process_action(action: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    episode = Path(action["episode"])
    color = action["color"]
    try:
        rows = read_rows(episode)
        minimums = minimum_phase_frames(args)
        full_cycle = (
            full_task_cycle(action, rows, threshold=args.gripper_threshold, debounce=args.gripper_debounce_frames)
            if is_full_task(action)
            else None
        )
        if is_full_task(action) and full_cycle is None:
            raise ValueError("missing_full_gripper_cycle")
        if full_cycle is None:
            candidate_bounds = gripper_phase_bounds(
                action,
                rows,
                threshold=args.gripper_threshold,
                debounce=args.gripper_debounce_frames,
                event_grace_frames=args.event_grace_frames,
                min_carry_frames=minimums["carry"],
                min_grasp_frames=minimums["grasp"],
            )
        else:
            candidate_bounds = phase_bounds_from_cycle(
                *full_cycle,
                event_grace_frames=args.event_grace_frames,
                min_carry_frames=minimums["carry"],
                min_grasp_frames=minimums["grasp"],
                full_task=True,
            )
        messages_by_boundary, allowed, effective_candidate_bounds, candidate_mode = build_initial_messages(
            action, rows, args, candidate_bounds, full_cycle
        )
        boundary_answers: dict[str, dict[str, Any]] = {}
        for name, messages in messages_by_boundary.items():
            answer = request_qwen(messages, args)
            if (
                not is_usable_boundary_answer(answer, allowed[name], args.max_boundary_snap_frames)
                and should_retry_visual_boundary(answer)
            ):
                fallback_messages, fallback_allowed = build_messages(
                action,
                rows,
                args,
                boundary_names=(name,),
                window_frames=fallback_window_for_boundary(name, args),
                    sample_frames=args.fallback_sample_frames,
                    force_choice=True,
                    candidate_bounds=effective_candidate_bounds,
                    full_cycle=full_cycle,
                )
                answer = request_qwen(fallback_messages[name], args)
                allowed[name] = fallback_allowed[name]
            if not is_usable_boundary_answer(answer, allowed[name], args.max_boundary_snap_frames):
                boundary_answers[name] = normalize_boundary_answer(answer, allowed[name], args.max_boundary_snap_frames)
                break
            boundary_answers[name] = normalize_boundary_answer(answer, allowed[name], args.max_boundary_snap_frames)

        # Qwen can identify both visual events but choose the same coarse frame
        # for adjacent phases. Re-ask only the first offending later boundary,
        # using candidates strictly after the prior chosen boundary.
        for _ in range(len(BOUNDARY_NAMES) * 2):
            chosen = [number(boundary_answers[name].get("boundary")) for name in BOUNDARY_NAMES]
            offending = first_nonincreasing_boundary(chosen)
            if offending is None:
                short_phase = first_too_short_phase(action["coarse"]["approach"][0], chosen, minimums)
                offending = None if short_phase is None else f"{short_phase}_end"
            if offending is None:
                break
            position = BOUNDARY_NAMES.index(offending)
            previous = action["coarse"]["approach"][0] if position == 0 else chosen[position - 1]
            if previous is None:
                break
            required_after = int(previous) + minimums[PHASES[position]] - 1
            repair_messages, repair_allowed = build_messages(
                action,
                rows,
                args,
                boundary_names=(offending,),
                window_frames=fallback_window_for_boundary(offending, args),
                sample_frames=args.fallback_sample_frames,
                force_choice=True,
                after_frame=required_after,
                candidate_bounds=effective_candidate_bounds,
                full_cycle=full_cycle,
            )
            repaired = request_qwen(repair_messages[offending], args)
            boundary_answers[offending] = normalize_boundary_answer(
                repaired, repair_allowed[offending], args.max_boundary_snap_frames
            )
            allowed[offending] = repair_allowed[offending]
        answer = combine_boundary_answers(boundary_answers)
        segments, errors = make_refined_segments(
            episode=episode,
            color=color,
            frame_count=len(rows),
            prior_end=action["coarse"]["approach"][0],
            coarse=action["coarse"],
            answer=answer,
            allowed_indices=allowed,
        )
        decision = "accept" if str(answer.get("decision", "")).lower() == "accept" and not errors else "uncertain"
        return {
            "key": action_key(action),
            "refinement_version": REFINEMENT_VERSION,
            "episode": str(episode),
            "color": color,
            "decision": decision,
            "reason": str(answer.get("reason", "")),
            "errors": errors,
            "coarse": action["coarse"],
            "gripper_phase_bounds": candidate_bounds,
            "candidate_mode": candidate_mode,
            "allowed_indices": allowed,
            "answer": answer,
            "segments": segments if decision == "accept" else [],
        }
    except Exception as exc:
        return {
            "key": action_key(action),
            "refinement_version": REFINEMENT_VERSION,
            "episode": str(episode),
            "color": color,
            "decision": "uncertain",
            "reason": str(exc),
            "errors": ["request_or_parse_failed"],
            "coarse": action["coarse"],
            "segments": [],
        }


def main() -> None:
    args = parse_args()
    workers = validate_workers(args.workers)
    actions = group_coarse_segments([record for path in args.input_manifest for record in read_jsonl(path)])
    fallback_actions = [action for path in args.fallback_report for action in actions_from_report(read_jsonl(path))]
    present = {action_key(action) for action in actions}
    actions.extend(action for action in fallback_actions if action_key(action) not in present)
    if args.exclude_episode_list:
        excluded = {
            line.strip()
            for line in args.exclude_episode_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        actions = [action for action in actions if str(action["episode"]) not in excluded]
        print(f"excluded raw episodes: {len(excluded)}")
    actions.sort(key=lambda item: (item["episode"], item["coarse"]["approach"][0]))
    prior = [] if args.fresh else read_jsonl(args.report)
    if args.retry_empty_guardrails_only:
        actions = actions_for_empty_guardrail_retry(actions, prior)
    if args.limit:
        actions = actions[: args.limit]
    frame_counts = {str(action["episode"]): len(read_rows(Path(action["episode"]))) for action in actions}
    retryable = (
        [item for item in prior if is_empty_guardrail_record(item)]
        if args.retry_empty_guardrails_only
        else [item for item in prior if is_retryable_record(item)]
    )
    report = [item for item in prior if not is_retryable_record(item)]
    completed = {item.get("key") for item in report}
    manifest = rebuild_contiguous_report(
        report, frame_counts, args.post_place_tail_frames, args.full_final_place_tail_frames
    )
    if args.rebuild_only:
        write_jsonl(args.report, report)
        write_jsonl(args.output_manifest, manifest)
        print(f"rebuilt: accepted color tasks={sum(item.get('decision') == 'accept' for item in report)}, refined segments={len(manifest)}")
        return
    pending = [action for action in actions if action_key(action) not in completed]
    print(
        f"selected color tasks: {len(actions)}, already completed: {len(actions) - len(pending)}, "
        f"retrying network failures: {len(retryable)}"
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_action, action, args): action for action in pending}
        for index, future in enumerate(as_completed(futures), 1):
            action = futures[future]
            record = future.result()
            episode = Path(action["episode"])
            color = action["color"]
            report.append(record)
            manifest = rebuild_contiguous_report(
                report, frame_counts, args.post_place_tail_frames, args.full_final_place_tail_frames
            )
            write_jsonl(args.report, report)
            write_jsonl(args.output_manifest, manifest)
            print(f"[{index}/{len(pending)}] {record['decision'].upper()} {episode.name}/{color}: {record['reason'][:120]}")

    write_jsonl(args.report, report)
    write_jsonl(args.output_manifest, manifest)
    accepted = sum(item.get("decision") == "accept" for item in report)
    print(f"done: accepted color tasks={accepted}, refined segments={len(manifest)}")


if __name__ == "__main__":
    main()
