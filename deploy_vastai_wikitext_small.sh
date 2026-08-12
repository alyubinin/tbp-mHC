#!/usr/bin/env bash
# End-to-end Vast.ai deployment for WikiText-103 small-model training.
#
# Usage (on the rented instance):
#   export WANDB_API_KEY=...          # optional but recommended
#   export VAST_API_KEY=...           # required to stop the instance
#   bash deploy_vastai_wikitext_small.sh <instance_id>
#
# Or:
#   VAST_INSTANCE_ID=12345 bash deploy_vastai_wikitext_small.sh
#
# Set SKIP_VAST_STOP=1 to keep the instance running after training completes.

set -euo pipefail

INSTANCE_ID="${1:-${VAST_INSTANCE_ID:-}}"
REPO_DIR="/workspace/tbp-mHC"
TRAINING_SCRIPT="${REPO_DIR}/train_local_wikitext_small.sh"
DATA_DIR="${REPO_DIR}/data/wikitext-103-raw-v1"
GDOWN_ID="1FdCBv9LOb8--BosHhtajc7zk_1HpjnwZ"
ARCHIVE_NAME="wikitext-103-raw-v1.7z"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <vast_instance_id>"
  echo "   or: VAST_INSTANCE_ID=<id> $0"
  exit 1
fi

stop_instance() {
  if [[ "${SKIP_VAST_STOP:-0}" == "1" ]]; then
    echo "SKIP_VAST_STOP=1 — leaving instance ${INSTANCE_ID} running."
    return 0
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "Warning: vastai CLI not found; cannot stop instance ${INSTANCE_ID}."
    return 0
  fi
  echo "Stopping vast.ai instance ${INSTANCE_ID}..."
  vastai stop instance "${INSTANCE_ID}" || echo "Warning: failed to stop instance ${INSTANCE_ID}."
}

echo "=== Installing system packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq p7zip-full mc git

echo "=== Installing Python packages ==="
pip install -q --upgrade pip
pip install -q numpy transformers datasets tiktoken wandb tqdm einops gdown vastai

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "=== Configuring Weights & Biases ==="
  wandb login --relogin "${WANDB_API_KEY}"
fi

echo "=== Cloning repository ==="
mkdir -p /workspace
if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone https://github.com/alyubinin/tbp-mHC "${REPO_DIR}"
fi

echo "=== Downloading WikiText-103 parquet archive ==="
mkdir -p "${DATA_DIR}/parquet"
cd "${DATA_DIR}"
if [[ ! -f "${ARCHIVE_NAME}" ]]; then
  gdown "${GDOWN_ID}" -O "${ARCHIVE_NAME}"
fi

echo "=== Extracting parquet files ==="
7z x "${ARCHIVE_NAME}" -y
if compgen -G "${DATA_DIR}/*.parquet" > /dev/null; then
  mv -f "${DATA_DIR}"/*.parquet "${DATA_DIR}/parquet/"
fi
if ! compgen -G "${DATA_DIR}/parquet/*.parquet" > /dev/null; then
  echo "Error: no parquet files found under ${DATA_DIR}/parquet after extraction."
  exit 1
fi

echo "=== Preparing tokenized dataset ==="
cd "${REPO_DIR}"
if [[ ! -f "${DATA_DIR}/train.bin" ]]; then
  python "${DATA_DIR}/prepare.py"
else
  echo "train.bin already exists — skipping prepare.py"
fi

echo "=== GPU check ==="
nvidia-smi
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo "=== Starting training ==="
set +e
bash "${TRAINING_SCRIPT}"
training_exit=$?
set -e

echo "=== Training finished (exit code: ${training_exit}) ==="
stop_instance
exit "${training_exit}"
