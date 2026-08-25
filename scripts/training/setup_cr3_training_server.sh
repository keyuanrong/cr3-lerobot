#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/training/setup_cr3_training_server.sh [options]

Options:
  --extras "training"        uv extras to install. Default: training
  --index-url URL            Python package index for uv/pip. Default: Tsinghua PyPI mirror.
  --no-index-mirror          Do not set uv/pip mirror environment variables.
  --allow-lock-update        Run uv sync without --locked if uv.lock is out of date.
  --torch-index-url URL      Reinstall torch/torchvision from a specific PyTorch index after uv sync.
                             Example: https://download.pytorch.org/whl/cu128
  --hf-endpoint URL          Hugging Face endpoint mirror. Default: keep existing environment.
  --hf-login                Run Hugging Face login prompt. Default: off.
  --warmup-act-backbone     Pre-download torchvision ResNet18 ImageNet weights for ACT. Default: off.
  --check-only              Only print environment checks after setup.
  -h, --help                Show this help.

Examples:
  bash scripts/training/setup_cr3_training_server.sh
  bash scripts/training/setup_cr3_training_server.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
  bash scripts/training/setup_cr3_training_server.sh --torch-index-url https://download.pytorch.org/whl/cu128
EOF
}

EXTRAS="training"
INDEX_URL="${UV_INDEX_URL:-${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}}"
USE_INDEX_MIRROR=1
LOCKED=1
TORCH_INDEX_URL=""
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-}"
HF_LOGIN=0
WARMUP_ACT_BACKBONE=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extras)
      EXTRAS="$2"
      shift 2
      ;;
    --index-url)
      INDEX_URL="$2"
      USE_INDEX_MIRROR=1
      shift 2
      ;;
    --no-index-mirror)
      USE_INDEX_MIRROR=0
      shift
      ;;
    --allow-lock-update)
      LOCKED=0
      shift
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --hf-endpoint)
      HF_ENDPOINT_VALUE="$2"
      shift 2
      ;;
    --hf-login)
      HF_LOGIN=1
      shift
      ;;
    --warmup-act-backbone)
      WARMUP_ACT_BACKBONE=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Repo: ${REPO_ROOT}"

if [[ "${USE_INDEX_MIRROR}" -eq 1 && -n "${INDEX_URL}" ]]; then
  export UV_INDEX_URL="${INDEX_URL}"
  export PIP_INDEX_URL="${INDEX_URL}"
  echo "Python package index: ${INDEX_URL}"
else
  echo "Python package index: default"
fi

if [[ -n "${HF_ENDPOINT_VALUE}" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
  echo "Hugging Face endpoint: ${HF_ENDPOINT}"
fi

if ! command -v uv >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    echo "uv not found. Installing uv with pip..."
    python3 -m pip install --user uv
  elif command -v python >/dev/null 2>&1; then
    echo "uv not found. Installing uv with pip..."
    python -m pip install --user uv
  else
    echo "uv not found. Installing uv with the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "uv: $(uv --version)"

if command -v git >/dev/null 2>&1; then
  git lfs install >/dev/null 2>&1 || true
fi

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
  UV_ARGS=(sync)
  if [[ "${LOCKED}" -eq 1 ]]; then
    UV_ARGS+=(--locked)
  fi
  IFS=',' read -ra EXTRA_LIST <<< "${EXTRAS}"
  for extra in "${EXTRA_LIST[@]}"; do
    extra="$(echo "${extra}" | xargs)"
    if [[ -n "${extra}" ]]; then
      UV_ARGS+=(--extra "${extra}")
    fi
  done

  echo "Running: uv ${UV_ARGS[*]}"
  uv "${UV_ARGS[@]}"

if [[ -n "${TORCH_INDEX_URL}" ]]; then
    echo "Reinstalling torch/torchvision from: ${TORCH_INDEX_URL}"
    uv pip install --force-reinstall torch torchvision --index-url "${TORCH_INDEX_URL}"
  fi
fi

if [[ "${WARMUP_ACT_BACKBONE}" -eq 1 ]]; then
  echo "Pre-downloading ACT default ResNet18 ImageNet weights..."
  uv run python - <<'PY'
from torchvision.models import ResNet18_Weights, resnet18
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("ResNet18 weights ready")
PY
fi

echo
echo "Environment check:"
uv run python - <<'PY'
import importlib.util
import shutil
import sys

print("python:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", repr(exc))

for name in ["lerobot", "datasets", "accelerate", "huggingface_hub"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}:", "ok" if spec is not None else "missing")

for cmd in ["lerobot-train", "hf", "ffmpeg"]:
    print(f"{cmd}:", shutil.which(cmd) or "missing")
PY

if [[ "${HF_LOGIN}" -eq 1 ]]; then
  echo
  if uv run hf auth whoami >/dev/null 2>&1; then
    uv run hf auth whoami
  else
    echo "Hugging Face is not logged in. Starting login..."
    uv run hf auth login
  fi
fi

cat <<'EOF'

Setup finished.

This script only prepares the server environment.

For local-dataset ACT training, put your LeRobot-format dataset on the server and run:
  bash scripts/training/train_cr3_act_local.sh --dataset-root /path/to/lerobot_dataset
EOF
