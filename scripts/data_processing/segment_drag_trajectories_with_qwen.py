#!/usr/bin/env python
"""Create visually grounded CR3 task phases with a Qwen vision model.

Qwen sees a temporal sequence of front/wrist image pairs and identifies visual
task milestones. No gripper-event or fixed-frame-window heuristic is used to
choose a phase boundary. The result is a candidate manifest which must still
pass phase-quality filtering before training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.data_processing.filter_drag_phase_segments_with_qwen import (
    panel_as_data_url,
    read_rows,
    request_qwen,
    select_indices,
)
COLORS = ("red", "green", "yellow")


PHASES = ("approach", "grasp", "carry", "place")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Qwen visual milestones to split CR3 demonstrations.")
    parser.add_argument("--episode-list", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="qwen3.6-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument("--sample-frames", type=int, default=24)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means all episodes.")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def phase_task(color: str, phase: str) -> str:
    return {
        "approach": f"move above the {color} block",
        "grasp": f"align, grasp, and lift the {color} block",
        "carry": f"move the {color} block above the black frame",
        "place": f"place the {color} block into the black frame",
    }[phase]


def read_episode_list(path: Path) -> list[Path]:
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        episode = Path(line).expanduser()
        if not episode.is_absolute():
            episode = (path.parent.parent.parent / episode).resolve()
        episodes.append(episode)
    return episodes


def task_colors(task: str, episode: Path) -> list[str]:
    found = [color for color in COLORS if re.search(rf"\b{color}\b", task.lower())]
    if len(found) >= 2 or episode.parent.name.lower() == "full":
        return [color for color in COLORS if color in found] or list(COLORS)
    if episode.parent.name.lower() in COLORS:
        return [episode.parent.name.lower()]
    return found


def build_messages(episode: Path, colors: list[str], rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    indices = select_indices(0, len(rows), args.sample_frames, [])
    color_text = ", ".join(colors)
    prompt = (
        "You segment a robot demonstration from temporally ordered image pairs. Every image pair has front camera "
        "on the left and wrist camera on the right. Use visual evidence, not guessed fixed frame offsets. "
        f"Expected target order: {color_text}. For EACH target color, identify four END boundaries as one of the listed raw "
        "frame numbers: approach_end (gripper is aligned above/beside target before final pickup), grasp_end (target is securely "
        "grasped AND visibly lifted), carry_end (held target is above/inside the black frame before release), place_end (target was "
        "released and remains inside black frame). Boundaries must be strictly increasing and colors must follow the expected order. "
        "If the sampled images cannot support a boundary, set that value to null and decision to uncertain. Return JSON only: "
        '{"decision":"accept|reject|uncertain","segments":[{"color":"red","approach_end":123,"grasp_end":234,'
        '"carry_end":345,"place_end":456}],"reason":"..."}. '
        "Do not mark carry incomplete just because placement happens in the next phase."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index in indices:
        content.append({"type": "text", "text": f"raw frame {index}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": panel_as_data_url(episode, rows[index], args.panel_width, args.jpeg_quality)},
            }
        )
    return [{"role": "user", "content": content}]


def number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_and_make_segments(
    episode: Path, colors: list[str], rows: list[dict[str, str]], answer: dict[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_segments = answer.get("segments")
    if not isinstance(raw_segments, list):
        return [], ["missing_segments_array"]
    by_color = {item.get("color"): item for item in raw_segments if isinstance(item, dict)}
    errors: list[str] = []
    result: list[dict[str, Any]] = []
    cursor = 0
    for color in colors:
        item = by_color.get(color)
        if not item:
            errors.append(f"missing_color:{color}")
            continue
        boundaries = [number(item.get(name)) for name in ("approach_end", "grasp_end", "carry_end", "place_end")]
        if any(value is None for value in boundaries):
            errors.append(f"missing_visual_boundary:{color}")
            continue
        ends = [int(value) for value in boundaries]
        if not (cursor < ends[0] < ends[1] < ends[2] < ends[3] <= len(rows)):
            errors.append(f"invalid_boundary_order:{color}:{ends}")
            continue
        points = [cursor, *ends]
        for phase, start, end in zip(PHASES, points[:-1], ends, strict=True):
            result.append(
                {"episode": str(episode), "start": start, "end": end, "task": phase_task(color, phase), "color": color, "phase": phase,
                 "boundary_source": "qwen_visual"}
            )
        cursor = ends[-1]
    if errors:
        return [], errors

    return ([] if errors else result), errors


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    episodes = read_episode_list(args.episode_list)
    if args.limit:
        episodes = episodes[: args.limit]
    prior = [] if args.fresh or not args.report.exists() else [json.loads(line) for line in args.report.read_text(encoding="utf-8").splitlines() if line]
    report = []
    manifest: list[dict[str, Any]] = []
    for item in prior:
        # A previous run may have obtained a valid Qwen answer but failed while
        # writing segments. Revalidate locally instead of charging another API call.
        if not item.get("segments") and isinstance(item.get("answer"), dict):
            try:
                episode = Path(item["episode"])
                rows = read_rows(episode)
                colors = task_colors(rows[0].get("task", ""), episode)
                segments, errors = validate_and_make_segments(episode, colors, rows, item["answer"], args)
                item["segments"] = segments
                item["errors"] = errors
                item["decision"] = "accept" if str(item["answer"].get("decision", "")).lower() == "accept" and not errors else "uncertain"
            except Exception as exc:
                item["errors"] = [f"local_revalidation_failed:{exc}"]
                item["decision"] = "uncertain"
        report.append(item)
        manifest.extend(item.get("segments", []))
    completed = {item["episode"] for item in report}

    for index, episode in enumerate(episodes, 1):
        if str(episode) in completed:
            continue
        try:
            rows = read_rows(episode)
            colors = task_colors(rows[0].get("task", ""), episode)
            if not colors or any(color not in COLORS for color in colors):
                raise ValueError("could_not_infer_expected_colors")
            answer = request_qwen(build_messages(episode, colors, rows, args), args)
            segments, errors = validate_and_make_segments(episode, colors, rows, answer, args)
            decision = "accept" if str(answer.get("decision", "")).lower() == "accept" and not errors else "uncertain"
            record = {"episode": str(episode), "decision": decision, "reason": str(answer.get("reason", "")), "errors": errors, "answer": answer, "segments": segments}
        except Exception as exc:
            record = {"episode": str(episode), "decision": "uncertain", "reason": str(exc), "errors": ["request_or_parse_failed"], "segments": []}
        report.append(record)
        manifest.extend(record["segments"])
        write_jsonl(args.report, report)
        write_jsonl(args.output_manifest, manifest)
        print(f"[{index}/{len(episodes)}] {record['decision'].upper()} {episode.name}: {record['reason'][:100]}")

    write_jsonl(args.report, report)
    write_jsonl(args.output_manifest, manifest)
    print(f"done: accepted episodes={sum(item['decision'] == 'accept' for item in report)}, segments={len(manifest)}")


if __name__ == "__main__":
    main()
