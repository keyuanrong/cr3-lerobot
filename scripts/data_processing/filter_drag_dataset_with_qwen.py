#!/usr/bin/env python
"""Rule-first, Qwen2.5-VL second-stage filtering for raw CR3 drag episodes.

The script keeps deterministic checks in ``auto_filter_drag_dataset.py`` as the
first gate. Only episodes that pass those checks are sent to a Qwen-compatible
vision API for a visual quality decision.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.data_processing.auto_filter_drag_dataset import check_episode, read_rows


DEFAULT_TASKS = ("red", "green", "yellow", "full")
REPORT_FIELDS = (
    "episode",
    "task",
    "stage",
    "decision",
    "confidence",
    "frames",
    "close_events",
    "open_events",
    "reason",
    "target_color_correct",
    "grasp_success",
    "placement_success",
)


class QwenAccessError(RuntimeError):
    """An API key, endpoint, or model permission error that needs user action."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply hard rules first, then use Qwen2.5-VL to filter CR3 drag episodes."
    )
    parser.add_argument("--input-root", default="data/cr3_real_drag_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-pattern", default="drag_episode_*")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--episode-list",
        type=Path,
        help="Optional text file of episode paths. When set, it replaces root/task discovery.",
    )
    parser.add_argument(
        "--exclude-list",
        type=Path,
        action="append",
        default=[],
        help="Text file of already-reviewed episode paths to skip. May be passed more than once.",
    )
    parser.add_argument("--single-min-frames", type=int, default=30)
    parser.add_argument("--full-min-frames", type=int, default=120)
    parser.add_argument("--single-min-close-events", type=int, default=1)
    parser.add_argument("--single-min-open-events", type=int, default=1)
    parser.add_argument("--full-min-close-events", type=int, default=3)
    parser.add_argument("--full-min-open-events", type=int, default=3)
    parser.add_argument("--allow-gripper-error", action="store_true")
    parser.add_argument(
        "--model",
        default="qwen3.7-plus",
        help="Model Studio vision model used for the second-stage visual review.",
    )
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base-url", default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=16,
        help="Maximum review frames for a single-block task. Key gripper frames are prioritized.",
    )
    parser.add_argument(
        "--full-sample-frames",
        type=int,
        default=24,
        help="Maximum review frames for a red-green-yellow full task.",
    )
    parser.add_argument(
        "--event-context-frames",
        type=int,
        default=5,
        help="Frames before and after each GRIPPER_OPEN/CLOSE transition to inspect.",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--keep-confidence", type=float, default=0.85)
    parser.add_argument("--reject-confidence", type=float, default=0.85)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means process every selected episode.")
    parser.add_argument("--dry-run", action="store_true", help="Run only deterministic rules; do not call Qwen.")
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing report and start a new run.")
    return parser.parse_args()


def discover_episodes(args: argparse.Namespace) -> list[Path]:
    preserve_input_order = bool(args.episode_list)
    if args.episode_list:
        paths = []
        for line in args.episode_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                paths.append(Path(line))
        episodes = paths
    else:
        episodes = []
        for task in args.tasks:
            task_root = Path(args.input_root) / task
            episodes.extend(path for path in task_root.glob(args.episode_pattern) if path.is_dir())

    excluded = set()
    for list_path in args.exclude_list:
        for line in list_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                excluded.add(str(Path(line).resolve()))
    if excluded:
        before = len(episodes)
        episodes = [episode for episode in episodes if str(episode.resolve()) not in excluded]
        print(f"excluded already-reviewed episodes: {before - len(episodes)}")
    return episodes if preserve_input_order else sorted(episodes)


def infer_task(episode: Path) -> str:
    task = episode.parent.name
    return task if task in DEFAULT_TASKS else "unknown"


def read_completed(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    with report_path.open(newline="", encoding="utf-8") as file:
        return {row["episode"] for row in csv.DictReader(file) if row.get("episode")}


def append_report(report_path: Path, row: dict[str, object]) -> None:
    exists = report_path.exists()
    with report_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})
        file.flush()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = min(max(count, 1), length)
    if count == 1:
        return [length - 1]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def action_transition_indices(rows: list[dict[str, str]], token: str) -> list[int]:
    indices = []
    was_active = False
    for index, row in enumerate(rows):
        active = token in (row.get("action") or "")
        if active and not was_active:
            indices.append(index)
        was_active = active
    return indices


def select_review_frames(episode: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> list[tuple[int, str]]:
    """Keep task boundaries and gripper transitions before adding coverage frames."""
    length = len(rows)
    max_frames = args.full_sample_frames if infer_task(episode) == "full" else args.sample_frames
    max_frames = min(max(max_frames, 1), length)
    selected: dict[int, str] = {}

    def add(index: int, label: str) -> None:
        if 0 <= index < length and len(selected) < max_frames:
            selected.setdefault(index, label)

    add(0, "start")
    if length > 1:
        add(length - 1, "end")

    transitions: list[tuple[int, str]] = []
    for token, label in (("GRIPPER_CLOSE", "close"), ("GRIPPER_OPEN", "open")):
        transitions.extend((index, label) for index in action_transition_indices(rows, token))
    transitions.sort(key=lambda item: item[0])

    # The event frame is essential; reserve it before its surrounding context.
    for index, label in transitions:
        add(index, label)
    for index, label in transitions:
        add(index - args.event_context_frames, f"before_{label}")
        add(index + args.event_context_frames, f"after_{label}")

    # Fill the remaining budget with broad trajectory coverage.
    for index in evenly_spaced_indices(length, max_frames * 4):
        add(index, "coverage")
    for index in range(length):
        add(index, "coverage")
    return [(index, selected[index]) for index in sorted(selected)]


def load_panel(episode: Path, row: dict[str, str], width: int, jpeg_quality: int) -> str:
    images = []
    for column, label in (("front_rgb", "front"), ("wrist_rgb", "wrist")):
        image = cv2.imread(str(episode / row[column]))
        if image is None:
            raise ValueError(f"missing {label} image")
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        images.append(image)

    max_height = max(image.shape[0] for image in images)
    padded = [
        cv2.copyMakeBorder(image, 0, max_height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for image in images
    ]
    ok, encoded = cv2.imencode(".jpg", cv2.hconcat(padded), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("jpeg_encoding_failed")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def build_messages(episode: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, object]]:
    task = rows[0].get("task", "")
    prompt = (
        "You are reviewing a robot demonstration. Each image shows front camera on the left and wrist camera "
        "on the right. Judge the whole sequence for this requested task: "
        f"{task}. Return JSON only with keys decision, confidence, target_color_correct, grasp_success, "
        "placement_success, reason. decision must be keep, reject, or uncertain. confidence is 0 to 1. "
        "Use keep only when the requested colored block is handled correctly and the demonstration is usable. "
        "Use reject for a wrong target, failed grasp, failed placement, obvious collision, or incomplete trajectory."
    )
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for index, label in select_review_frames(episode, rows, args):
        content.append({"type": "text", "text": f"sequence frame {index}: {label}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": load_panel(episode, rows[index], args.image_width, args.jpeg_quality)},
            }
        )
    return [{"role": "user", "content": content}]


def extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    marker = chr(96) * 3
    if text.startswith(marker):
        text = text.split("\n", 1)[-1]
        if marker in text:
            text = text.rsplit(marker, 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model_response_is_not_json")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model_response_is_not_an_object")
    return parsed


def request_qwen(messages: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing environment variable: {args.api_key_env}")
    if not args.base_url:
        raise RuntimeError("missing --base-url or DASHSCOPE_BASE_URL")
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    base_url = args.base_url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    error: Exception | None = None
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return extract_json(body["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code in {401, 403}:
                raise QwenAccessError(f"HTTP {exc.code}: {detail}") from exc
            error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if attempt + 1 < args.retries:
                time.sleep(2**attempt)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < args.retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"qwen_request_failed: {error}")


def as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"true", "yes", "1"}:
            return True
        if value in {"false", "no", "0"}:
            return False
    return None


def qwen_result(decision: dict[str, object], args: argparse.Namespace) -> tuple[str, float, str, bool | None, bool | None, bool | None]:
    raw_decision = str(decision.get("decision", "uncertain")).strip().lower()
    confidence = float(decision.get("confidence", 0) or 0)
    color = as_bool(decision.get("target_color_correct"))
    grasp = as_bool(decision.get("grasp_success"))
    placement = as_bool(decision.get("placement_success"))
    reason = str(decision.get("reason", "")).replace("\n", " ").strip()
    if raw_decision == "keep" and color is True and grasp is True and placement is True and confidence >= args.keep_confidence:
        return "keep", confidence, reason, color, grasp, placement
    if raw_decision == "reject" and any(value is False for value in (color, grasp, placement)) and confidence >= args.reject_confidence:
        return "reject", confidence, reason, color, grasp, placement
    return "uncertain", confidence, reason or "low_confidence_or_inconsistent_fields", color, grasp, placement


def write_decision_lists(output_dir: Path, report_path: Path) -> None:
    grouped = {"keep": [], "reject": [], "uncertain": [], "rule_reject": []}
    with report_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            decision = row.get("decision", "uncertain")
            grouped.setdefault(decision, []).append(row["episode"])
            if row.get("stage") == "rule" and decision == "reject":
                grouped["rule_reject"].append(row["episode"])
    for decision in ("keep", "reject", "uncertain"):
        episodes = grouped[decision]
        (output_dir / f"final_{decision}.txt").write_text("\n".join(episodes) + ("\n" if episodes else ""), encoding="utf-8")
    rule_reject = grouped["rule_reject"]
    (output_dir / "rule_reject.txt").write_text("\n".join(rule_reject) + ("\n" if rule_reject else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "qwen_filter_report.csv"
    if args.fresh and report_path.exists():
        report_path.unlink()
    completed = read_completed(report_path)
    episodes = discover_episodes(args)
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"selected episodes: {len(episodes)}, already completed: {len(completed)}")

    counts = {"keep": 0, "reject": 0, "uncertain": 0, "rule_reject": 0}
    for number, episode in enumerate(episodes, start=1):
        key = str(episode)
        if key in completed:
            continue
        task = infer_task(episode)
        check = check_episode(episode, task, args)
        base_row: dict[str, object] = {
            "episode": key,
            "task": task,
            "frames": check.frames,
            "close_events": check.close_events,
            "open_events": check.open_events,
        }
        if not check.ok:
            append_report(
                report_path,
                base_row | {"stage": "rule", "decision": "reject", "reason": "|".join(check.reasons)},
            )
            counts["rule_reject"] += 1
            print(f"[{number}/{len(episodes)}] RULE_REJECT {episode.name}: {','.join(check.reasons)}")
            continue
        if args.dry_run:
            append_report(report_path, base_row | {"stage": "rule", "decision": "uncertain", "reason": "rule_pass_dry_run"})
            counts["uncertain"] += 1
            continue
        try:
            rows = read_rows(episode)
            answer = request_qwen(build_messages(episode, rows, args), args)
            decision, confidence, reason, color, grasp, placement = qwen_result(answer, args)
            append_report(
                report_path,
                base_row
                | {
                    "stage": "qwen",
                    "decision": decision,
                    "confidence": confidence,
                    "reason": reason,
                    "target_color_correct": color,
                    "grasp_success": grasp,
                    "placement_success": placement,
                },
            )
            counts[decision] += 1
            print(f"[{number}/{len(episodes)}] {decision.upper()} {episode.name} confidence={confidence:.2f}")
        except QwenAccessError as exc:
            raise SystemExit(f"Qwen access configuration failed. No result was saved for this episode. {exc}") from exc
        except Exception as exc:
            append_report(report_path, base_row | {"stage": "qwen", "decision": "uncertain", "reason": str(exc)})
            counts["uncertain"] += 1
            print(f"[{number}/{len(episodes)}] UNCERTAIN {episode.name}: {exc}")

    if report_path.exists():
        write_decision_lists(output_dir, report_path)
    print("done: " + ", ".join(f"{name}={value}" for name, value in counts.items()))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
