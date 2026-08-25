#!/usr/bin/env python

"""Plot LeRobot training metrics from a log file or exported CSV.

Examples:

    python scripts/training/plot_training_metrics.py \
        --log-file /root/autodl-tmp/kyr_outputs/smolvla_subtask_finetune/train.log \
        --output-dir /root/autodl-tmp/kyr_outputs/smolvla_subtask_finetune/analysis

    python scripts/training/plot_training_metrics.py \
        --csv outputs/train/cr3_smolvla_100k_v6_bs8/analysis/loss_log.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METRIC_LINE_RE = re.compile(r"\bstep:(?P<step>\S+).*?\bloss:(?P<loss>[-+0-9.eE]+)")
KV_RE = re.compile(r"\b(?P<key>step|smpl|ep|epch|loss|grdn|lr|updt_s|data_s):(?P<value>\S+)")


def parse_compact_number(value: str) -> float:
    """Parse LeRobot compact numbers such as 2K, 1.5M, or normal floats."""
    value = value.strip().replace(",", "")
    if not value:
        return math.nan
    suffix = value[-1].upper()
    multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix)
    if multiplier is not None:
        return float(value[:-1]) * multiplier
    return float(value)


def parse_log(log_file: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not METRIC_LINE_RE.search(line):
                continue

            row: dict[str, float | str] = {}
            datetime_match = re.search(r"INFO\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
            if datetime_match:
                row["datetime"] = datetime_match.group(1)

            for match in KV_RE.finditer(line):
                key = match.group("key")
                value = match.group("value")
                try:
                    parsed_value = parse_compact_number(value)
                except ValueError:
                    continue
                column = {
                    "step": "step",
                    "smpl": "samples",
                    "ep": "episode",
                    "epch": "epoch",
                    "loss": "loss",
                    "grdn": "grad_norm",
                    "lr": "lr",
                    "updt_s": "update_s",
                    "data_s": "data_s",
                }[key]
                row[column] = parsed_value

            if "step" in row and "loss" in row:
                rows.append(row)
    return rows


def parse_csv(csv_file: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row: dict[str, float | str] = {}
            for key, value in raw_row.items():
                if value is None or value == "":
                    continue
                if key == "datetime":
                    row[key] = value
                    continue
                try:
                    row[key] = parse_compact_number(value)
                except ValueError:
                    row[key] = value
            if "step" in row and "loss" in row:
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float | str]], output_csv: Path) -> None:
    columns = ["index", "datetime", "step", "samples", "episode", "epoch", "loss", "grad_norm", "lr", "update_s", "data_s"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow({"index": index, **{column: row.get(column, "") for column in columns if column != "index"}})


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    averaged: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        averaged.append(total / len(queue))
    return averaged


def plot_metrics(rows: list[dict[str, float | str]], output_base: Path, smooth_window: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required. Install it with: pip install matplotlib") from exc

    steps = [float(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    smoothed_losses = moving_average(losses, smooth_window)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(steps, losses, label="Training Loss", linewidth=1.2, alpha=0.55)
    if smooth_window > 1:
        axes[0].plot(steps, smoothed_losses, label=f"Smoothed Loss ({smooth_window})", linewidth=2.0)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if all("lr" in row for row in rows):
        lrs = [float(row["lr"]) for row in rows]
        axes[1].plot(steps, lrs, label="Learning Rate", color="tab:orange", linewidth=1.5)
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate")
    elif all("grad_norm" in row for row in rows):
        grad_norms = [float(row["grad_norm"]) for row in rows]
        axes[1].plot(steps, grad_norms, label="Grad Norm", color="tab:green", linewidth=1.5)
        axes[1].set_ylabel("Grad Norm")
        axes[1].set_title("Gradient Norm")
    else:
        axes[1].axis("off")

    if axes[1].has_data():
        axes[1].set_xlabel("Step")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LeRobot training loss curves.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log-file", type=Path, help="Path to a lerobot-train log file.")
    source.add_argument("--csv", type=Path, help="Path to an existing metrics CSV.")
    parser.add_argument("--output-dir", type=Path, help="Directory for loss_log.csv and loss_curve files.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for loss smoothing.")
    args = parser.parse_args()

    rows = parse_log(args.log_file) if args.log_file else parse_csv(args.csv)
    if not rows:
        raise SystemExit("No training metrics found. Check that the log contains lines with step:... loss:...")

    source_path = args.log_file or args.csv
    output_dir = args.output_dir or source_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "loss_log.csv"
    output_base = output_dir / "loss_curve"
    write_csv(rows, output_csv)
    plot_metrics(rows, output_base, args.smooth_window)

    print(f"parsed rows: {len(rows)}")
    print(f"csv written: {output_csv}")
    print(f"png written: {output_base.with_suffix('.png')}")
    print(f"svg written: {output_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
