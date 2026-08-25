#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/kyr/miniconda3/envs/lerobot/bin/python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/lerobot_data}"
LIST_ROOT="$ROOT_DIR/data/episode_lists/pi0_unified_goal_v1"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--force]" >&2
  exit 2
fi

convert_one() {
  local name="$1"
  local manifest="$LIST_ROOT/${name}_train.jsonl"
  local repo_id="local/cr3_pi0_unified_goal_v1_${name}"
  local output="$DATA_ROOT/$repo_id"

  [[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 1; }
  if [[ -e "$output" ]]; then
    if [[ "$FORCE" -ne 1 ]]; then
      echo "Output already exists: $output (rerun with --force to replace it)" >&2
      exit 1
    fi
    rm -rf "$output"
  fi

  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" -m scripts.datasets.convert_drag_to_lerobot \
      --segment-manifest "$manifest" \
      --output-root "$DATA_ROOT" \
      --repo-id "$repo_id" \
      --fps 30 \
      --gripper-action-semantics close_high \
      --video-codec h264
  )
}

convert_one complete_goal
convert_one goal_grasp_event
convert_one goal_place_event
convert_one atomic_assist

echo "Converted four datasets under: $DATA_ROOT/local"
