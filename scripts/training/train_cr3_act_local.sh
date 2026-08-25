#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/train_cr3_act_local.sh --dataset-root PATH [options]

Required:
  --dataset-root PATH       LeRobot-format dataset directory on the server.

Options:
  --repo-id ID              Dataset repo_id used by LeRobot. Default: read from meta/info.json if possible.
  --output-dir PATH         Training output directory. Default: outputs/train/cr3_act_local
  --job-name NAME           Training job name. Default: cr3_act_local
  --steps N                 Training steps. Default: 70000
  --batch-size N            Batch size. Default: 8
  --num-workers N           Dataloader workers. Default: 4
  --device DEVICE           cpu/cuda. Default: cuda
  --no-wandb                Disable Weights & Biases. Default: disabled already, kept for clarity.
  --dry-run                 Print the command but do not run it.
  -h, --help                Show this help.

Example:
  bash scripts/training/train_cr3_act_local.sh \
    --dataset-root /data/lerobot/cr3_drag_lerobot_new \
    --output-dir outputs/train/cr3_act_new_70k \
    --steps 70000
EOF
}

DATASET_ROOT=""
REPO_ID=""
OUTPUT_DIR="outputs/train/cr3_act_local"
JOB_NAME="cr3_act_local"
STEPS=70000
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --no-wandb)
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "--dataset-root is required" >&2
  usage
  exit 2
fi

DATASET_ROOT="$(realpath "${DATASET_ROOT}")"
if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "Not a LeRobot dataset root: ${DATASET_ROOT}" >&2
  echo "Expected: ${DATASET_ROOT}/meta/info.json" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${REPO_ID}" ]]; then
  REPO_ID="$(uv run python - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
print(info.get("repo_id") or "local/cr3_drag")
PY
)"
fi

echo "Dataset root: ${DATASET_ROOT}"
echo "Dataset repo_id: ${REPO_ID}"
echo "Output dir: ${OUTPUT_DIR}"

CMD=(
  uv run lerobot-train
  --dataset.repo_id="${REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --policy.type=act
  --policy.device="${DEVICE}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${JOB_NAME}"
  --steps="${STEPS}"
  --batch_size="${BATCH_SIZE}"
  --num_workers="${NUM_WORKERS}"
  --wandb.enable=false
)

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 0 ]]; then
  "${CMD[@]}"
fi
