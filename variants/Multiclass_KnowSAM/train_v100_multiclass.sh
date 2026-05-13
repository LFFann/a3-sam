#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_multiclass}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/Multiclass_KnowSAM_V100_106_117_13_13}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

BATCH_SIZE="${BATCH_SIZE:-12}"
LABELED_BS="${LABELED_BS:-6}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LR="${LR:-1e-4}"
UNET_LR="${UNET_LR:-0.0025}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
MIXED_ITERATIONS="${MIXED_ITERATIONS:-240}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
CONSISTENCY="${CONSISTENCY:-0.1}"
CONSISTENCY_RAMPUP="${CONSISTENCY_RAMPUP:-200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "Missing SAM checkpoint: ${SAM_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting Multiclass KnowSAM training with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  NUM_CLASSES=3"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" train_semi_SAM.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --num_classes 3 \
  --labeled_num 1 \
  --batch_size "${BATCH_SIZE}" \
  --labeled_bs "${LABELED_BS}" \
  --image_size "${IMAGE_SIZE}" \
  -lr "${LR}" \
  -UNet_lr "${UNET_LR}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --mixed_iterations "${MIXED_ITERATIONS}" \
  --val_interval "${VAL_INTERVAL}" \
  --consistency "${CONSISTENCY}" \
  --consistency_rampup "${CONSISTENCY_RAMPUP}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --snapshot_path "${SNAPSHOT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
echo "  ${SNAPSHOT_PATH}/SGDL_best_model.pth"
