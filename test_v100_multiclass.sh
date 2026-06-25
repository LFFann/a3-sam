#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_multiclass}"
SPLIT="${SPLIT:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/Multiclass_KnowSAM_V100_bs32_10k_106_117_13_13}"
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

echo "Starting Multiclass KnowSAM evaluation with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SPLIT=${SPLIT}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  SAVE_DIR=${SAVE_DIR}"

"${PYTHON_BIN}" ./prediction_multiclass.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --image_size "${IMAGE_SIZE}" \
  --SGDL_model_path "${MODEL_PATH}" \
  --save_dir "${SAVE_DIR}" \
  --num_workers "${NUM_WORKERS}" \
  "$@"

echo "Evaluation finished. Outputs:"
echo "  ${SAVE_DIR}"
