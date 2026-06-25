#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_labeled30pct}"
SPLIT="${SPLIT:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MEASUREMENT_CLASS="${MEASUREMENT_CLASS:-1}"
PIXEL_SPACING="${PIXEL_SPACING:-}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/Multiclass_KnowSAM_labeled30pct_260624_kd3_tuned}"
MODEL_PATH="${MODEL_PATH:-${SNAPSHOT_PATH}/SGDL_best_model.pth}"
SAVE_DIR="${SAVE_DIR:-${SNAPSHOT_PATH}/prediction_${SPLIT}}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing model checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "Starting tuned Multiclass KnowSAM evaluation with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  SAVE_DIR=${SAVE_DIR}"
echo "  MEASUREMENT_CLASS=${MEASUREMENT_CLASS}"
echo "  PIXEL_SPACING=${PIXEL_SPACING}"

"${PYTHON_BIN}" ./prediction_multiclass.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --num_classes 3 \
  --image_size "${IMAGE_SIZE}" \
  --SGDL_model_path "${MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  --measurement_class "${MEASUREMENT_CLASS}" \
  --pixel_spacing "${PIXEL_SPACING}" \
  "$@"

echo "Evaluation finished. Outputs:"
echo "  ${SAVE_DIR}"
