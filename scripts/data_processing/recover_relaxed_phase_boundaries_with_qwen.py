#!/usr/bin/env python
"""Recover Qwen phase boundaries rejected by strict gripper guardrails.

This is intentionally separate from ``refine_drag_phase_boundaries_with_qwen``.
It only retries records marked ``no_candidate_frames_after_phase_guardrails``
and uses the first-pass boundaries as wide visual search anchors.  It never
overwrites an input manifest or report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.data_processing.filter_drag_phase_segments_with_qwen import panel_as_data_url, read_rows, request_qwen
from scripts.data_processing.segment_drag_trajectories_with_qwen import PHASES, number, phase_task


BOUNDARY_NAMES = tuple(f"{phase}_end" for phase in PHASES)
GUARDRAIL_REASON = "no_candidate_frames_after_phase_guardrails"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover only strict-guardrail Qwen rejections with wide visual candidate windows."
    )
    parser.add_argument("--coarse-manifest", required=True, type=Path, action="append")
    parser.add_argument("--source-report", required=True, type=Path, help="Original strict-refinement report JSONL.")
    parser.add_argument("--output-manifest", required=True, type=Path, help="New recovered segments JSONL.")
    parser.add_argument("--report", required=True, type=Path, help="New recovery report JSONL.")
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--sample-frames-per-boundary", type=int, default=48)
    parser.add_argument("--fallback-window-frames", type=int, default=360)
    parser.add_argument("--fallback-sample-frames", type=int, default=64)
    parser.add_argument("--max-boundary-snap-frames", type=int, default=5)
    parser.add_argument("--min-phase-frames", type=int, default=20)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--exclude-episode-list", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected color tasks.")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing recovery report.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def action_key(action: dict[str, Any]) -> str:
    return f"{action['episode']}::{action['color']}"


def select_guardrail_rejections(report_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only records rejected before Qwen by strict gripper candidates."""
    return [
        row
        for row in report_rows
        if row.get("decision") == "uncertain" and GUARDRAIL_REASON in str(row.get("reason", ""))
    ]


def match_rejections_to_actions(
    report_rows: list[dict[str, Any]], coarse_actions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve both current ``episode::color`` and historical ``episode|color`` keys."""
    matched: list[dict[str, Any]] = []
    for row in select_guardrail_rejections(report_rows):
        episode = row.get("episode")
        color = row.get("color")
        if not isinstance(episode, str) or not isinstance(color, str):
            continue
        action = coarse_actions.get(action_key({"episode": episode, "color": color}))
        if action is not None:
            matched.append(action)
    return matched


def group_coarse_segments(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for segment in records:
        if (
            segment.get("phase") not in PHASES
            or not isinstance(segment.get("episode"), str)
            or not isinstance(segment.get("color"), str)
        ):
            continue
        grouped.setdefault((segment["episode"], segment["color"]), {})[segment["phase"]] = segment

    actions: dict[str, dict[str, Any]] = {}
    for (episode, color), phases in grouped.items():
        if set(phases) != set(PHASES):
            continue
        coarse = {phase: (int(phases[phase]["start"]), int(phases[phase]["end"])) for phase in PHASES}
        action = {"episode": episode, "color": color, "coarse": coarse}
        actions[action_key(action)] = action
    return actions


def dense_indices(center: int, frame_count: int, window_frames: int, count: int) -> list[int]:
    if frame_count <= 0 or window_frames < 0 or count <= 0:
        raise ValueError("invalid_candidate_window")
    start = max(0, center - window_frames)
    end = min(frame_count - 1, center + window_frames)
    length = end - start + 1
    count = min(count, length)
    if count == 1:
        return [start]
    return [start + round(index * (length - 1) / (count - 1)) for index in range(count)]


def snap_to_candidate(value: int | None, candidates: list[int], max_distance: int) -> int | None:
    if value is None or not candidates:
        return None
    candidate = min(candidates, key=lambda frame: abs(frame - value))
    return candidate if abs(candidate - value) <= max_distance else None


def validate_relaxed_boundaries(
    boundaries: dict[str, Any],
    candidate_indices: list[int] | dict[str, list[int]],
    *,
    task_start: int,
    task_end: int,
    min_phase_frames: int,
    snap_frames: int,
) -> tuple[dict[str, int] | None, str | None]:
    """Validate visual choices without imposing gripper-event windows."""
    if isinstance(candidate_indices, list):
        candidate_indices = {name: candidate_indices for name in BOUNDARY_NAMES}
    chosen: dict[str, int] = {}
    for name in BOUNDARY_NAMES:
        value = number(boundaries.get(name))
        snapped = snap_to_candidate(value, candidate_indices.get(name, []), snap_frames)
        if snapped is None:
            return None, f"boundary_not_in_visual_candidates:{name}"
        chosen[name] = snapped

    points = [task_start, *(chosen[name] for name in BOUNDARY_NAMES)]
    for phase, start, end in zip(PHASES, points[:-1], points[1:], strict=True):
        if end <= start:
            return None, "invalid_boundary_order"
        if end - start < min_phase_frames:
            return None, f"phase_too_short:{phase}"
    if chosen["place_end"] >= task_end:
        return None, "place_end_outside_task"
    return chosen, None


def build_messages(
    action: dict[str, Any], rows: list[dict[str, str]], args: argparse.Namespace, *, fallback: bool = False
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
    episode = Path(action["episode"])
    window = args.fallback_window_frames if fallback else args.window_frames
    samples = args.fallback_sample_frames if fallback else args.sample_frames_per_boundary
    coarse = action["coarse"]
    candidates = {
        name: dense_indices(coarse[phase][1], len(rows), window, samples)
        for phase, name in zip(PHASES, BOUNDARY_NAMES, strict=True)
    }
    definitions = {
        "approach_end": "the colored block first enters the open space between the gripper fingers, before closing",
        "grasp_end": "the gripper has securely closed on the colored block and the block is visibly lifted clear of the table",
        "carry_end": "the still-held colored block arrives above or inside the black frame, before the gripper releases it",
        "place_end": "the gripper has released the colored block and the block remains inside the black frame",
    }
    messages: dict[str, list[dict[str, Any]]] = {}
    for name in BOUNDARY_NAMES:
        prompt = (
            "You locate one visual boundary in a successful robot pick-and-place demonstration. "
            "Each panel is a time point: front camera is on the left and wrist camera is on the right. "
            "Use the actual displayed raw frame number only. Do not infer a boundary from elapsed time. "
            f"For the {action['color']} block, select the frame where {definitions[name]}. "
            "If this event cannot be seen, return decision uncertain and boundary null. "
            'Return JSON only: {"decision":"accept|uncertain","boundary":123,"confidence":0.0,"reason":"..."}. '
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index in candidates[name]:
            content.append({"type": "text", "text": f"{name} candidate raw frame {index}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": panel_as_data_url(episode, rows[index], args.panel_width, args.jpeg_quality)
                    },
                }
            )
        messages[name] = [{"role": "user", "content": content}]
    return messages, candidates


def make_segments(action: dict[str, Any], boundaries: dict[str, int]) -> list[dict[str, Any]]:
    ends = [boundaries[name] for name in BOUNDARY_NAMES]
    starts = [int(action["coarse"]["approach"][0]), *ends[:-1]]
    return [
        {
            "episode": action["episode"],
            "start": start,
            "end": end,
            "task": phase_task(action["color"], phase),
            "color": action["color"],
            "phase": phase,
            "boundary_source": "qwen_relaxed_visual_recovery",
            "coarse_boundaries": {
                name: action["coarse"][phase][1] for phase, name in zip(PHASES, BOUNDARY_NAMES, strict=True)
            },
            "refined_boundaries": boundaries,
        }
        for phase, start, end in zip(PHASES, starts, ends, strict=True)
    ]


def process_action(action: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(Path(action["episode"]))
    if not rows:
        return {"key": action_key(action), "episode": action["episode"], "color": action["color"], "decision": "uncertain", "reason": "empty_episode", "segments": []}
    task_start = int(action["coarse"]["approach"][0])
    task_end = int(action["coarse"]["place"][1])
    try:
        messages, candidates = build_messages(action, rows, args)
        answers: dict[str, dict[str, Any]] = {}
        for name in BOUNDARY_NAMES:
            answer = request_qwen(messages[name], args)
            if number(answer.get("boundary")) is None:
                fallback_messages, fallback_candidates = build_messages(action, rows, args, fallback=True)
                answer = request_qwen(fallback_messages[name], args)
                candidates[name] = fallback_candidates[name]
            answers[name] = answer
        raw_boundaries = {name: answers[name].get("boundary") for name in BOUNDARY_NAMES}
        boundaries, reason = validate_relaxed_boundaries(
            raw_boundaries,
            candidates,
            task_start=task_start,
            task_end=task_end,
            min_phase_frames=args.min_phase_frames,
            snap_frames=args.max_boundary_snap_frames,
        )
        if boundaries is None:
            return {
                "key": action_key(action), "episode": action["episode"], "color": action["color"],
                "decision": "uncertain", "reason": reason, "coarse": action["coarse"],
                "allowed_indices": candidates, "boundary_answers": answers, "segments": [],
            }
        segments = make_segments(action, boundaries)
        return {
            "key": action_key(action), "episode": action["episode"], "color": action["color"],
            "decision": "accept", "reason": "", "coarse": action["coarse"],
            "allowed_indices": candidates, "boundary_answers": answers, "boundaries": boundaries, "segments": segments,
        }
    except Exception as exc:
        return {
            "key": action_key(action), "episode": action["episode"], "color": action["color"],
            "decision": "uncertain", "reason": str(exc), "coarse": action["coarse"], "segments": [],
        }


def main() -> None:
    args = parse_args()
    if args.window_frames < 0 or args.sample_frames_per_boundary <= 0 or args.min_phase_frames <= 0:
        raise SystemExit("window and sample/minimum phase counts must be positive")
    coarse = group_coarse_segments([row for path in args.coarse_manifest for row in read_jsonl(path)])
    actions = match_rejections_to_actions(read_jsonl(args.source_report), coarse)
    if args.exclude_episode_list:
        excluded = {line.strip() for line in args.exclude_episode_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        actions = [action for action in actions if action["episode"] not in excluded]
    actions.sort(key=lambda action: (action["episode"], action["coarse"]["approach"][0]))
    if args.limit:
        actions = actions[: args.limit]

    previous = [] if args.fresh else read_jsonl(args.report)
    completed = {row.get("key") for row in previous}
    pending = [action for action in actions if action_key(action) not in completed]
    report = [row for row in previous if row.get("key") in {action_key(action) for action in actions}]
    print(f"selected guardrail rejections: {len(actions)}, already completed: {len(actions) - len(pending)}")
    for index, action in enumerate(pending, 1):
        record = process_action(action, args)
        report.append(record)
        manifest = [segment for row in report if row.get("decision") == "accept" for segment in row.get("segments", [])]
        write_jsonl(args.report, report)
        write_jsonl(args.output_manifest, manifest)
        print(f"[{index}/{len(pending)}] {record['decision'].upper()} {Path(action['episode']).name}/{action['color']}: {record['reason'][:120]}")
    manifest = [segment for row in report if row.get("decision") == "accept" for segment in row.get("segments", [])]
    write_jsonl(args.report, report)
    write_jsonl(args.output_manifest, manifest)
    print(f"done: accepted color tasks={sum(row.get('decision') == 'accept' for row in report)}, recovered segments={len(manifest)}")


if __name__ == "__main__":
    main()
