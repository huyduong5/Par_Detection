#!/usr/bin/env bash
# Thesis matrix C0→C4 (+ optional C5). Uses aq_visibility package.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO}/scripts/sol1/env.sh" ]; then
  # shellcheck source=/dev/null
  source "${REPO}/scripts/sol1/env.sh"
fi
PAR_ROOT="${PAR_ROOT:-$REPO}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
ulimit -n 65536 2>/dev/null || true

PKL="${PAR_ROOT}/MSP60k/SUBMIT/dataset_random.pkl"
IMG="${PAR_ROOT}/MSP60k/SUBMIT/images"
OCC="${PAR_ROOT}/data/occlusion"
OUT_ROOT="${PAR_ROOT}/aq_visibility/output"
LOG_DIR="${PAR_ROOT}/aq_visibility/output/matrix_logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
RUN_C5="${RUN_C5:-0}"
BUILD_DATA="${BUILD_DATA:-1}"
MAX_PER_SPLIT="${MAX_PER_SPLIT:-0}"

cd "${PAR_ROOT}"

if [ "${BUILD_DATA}" = "1" ]; then
  MAX_PER_SPLIT="${MAX_PER_SPLIT}" bash "${SCRIPT_DIR}/build_data.sh"
fi

train_aq() {
  local VAR="$1"
  local ODIR="${OUT_ROOT}/${VAR}"
  mkdir -p "${ODIR}"
  echo "[$(date)] TRAIN ${VAR}"
  python aq_visibility/train/train.py \
    --config_variant "${VAR}" \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --output_dir "${ODIR}" \
    --no_ddp --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee -a "${ODIR}/train.log"
  # Point occlusion paths for C3/C4
}

# Fix: pass occlusion paths via env / config defaults (config __post_init__)
export AQ_OCC_DIR="${OCC}"

mkdir -p "${OUT_ROOT}/C0" "${OUT_ROOT}/C1"
echo "[$(date)] TRAIN C0"
python aq_visibility/train/train_c0_c1.py --variant C0 \
  --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --pkl_path "${PKL}" --img_dir "${IMG}" \
  --output_dir "${OUT_ROOT}/C0" \
  2>&1 | tee -a "${OUT_ROOT}/C0/train.log"

echo "[$(date)] TRAIN C1"
python aq_visibility/train/train_c0_c1.py --variant C1 \
  --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --pkl_path "${PKL}" --img_dir "${IMG}" --occ_aug_prob 0.5 \
  --output_dir "${OUT_ROOT}/C1" \
  2>&1 | tee -a "${OUT_ROOT}/C1/train.log"

for VAR in C2 C3 C4; do
  ODIR="${OUT_ROOT}/${VAR}"
  mkdir -p "${ODIR}"
  echo "[$(date)] TRAIN ${VAR}"
  # Override occlusion paths in config via CLI if supported; else symlink defaults
  python - <<PY
import os, sys
sys.path.insert(0, "aq_visibility")
from config import get_thesis_variant
cfg = get_thesis_variant("${VAR}")
print("lambda_v", cfg.lambda_v, "lambda_c", cfg.lambda_c, "occ", cfg.use_occlusion_pairs)
print("occ_pkl", cfg.occlusion_pkl_path)
PY
  EXTRA=()
  if [ "${VAR}" != "C2" ]; then
    # ensure config paths — train.py reads config.occlusion_*
    :
  fi
  python aq_visibility/train/train.py \
    --config_variant "${VAR}" \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --occlusion_pkl_path "${OCC}/occlusion_pairs_train.pkl" \
    --output_dir "${ODIR}" \
    --no_ddp --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee -a "${ODIR}/train.log"
done

if [ "${RUN_C5}" = "1" ]; then
  echo "[$(date)] TRAIN C5 (background ablation)"
  python aq_visibility/train/train.py \
    --config_variant C5 \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --occlusion_pkl_path "${OCC}/occlusion_pairs_train.pkl" \
    --output_dir "${OUT_ROOT}/C5" \
    --no_ddp --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee -a "${OUT_ROOT}/C5/train.log"
fi

echo "[$(date)] EVAL ALL"
bash "${SCRIPT_DIR}/eval_all.sh"

echo "[$(date)] AGGREGATE"
python "${SCRIPT_DIR}/aggregate_results.py" \
  --root "${PAR_ROOT}" \
  --output_md "${PAR_ROOT}/docs/aq_visibility/results.md"

echo "[$(date)] ANALYZE"
python "${SCRIPT_DIR}/analyze_gaps.py" \
  --root "${PAR_ROOT}" \
  --output_md "${PAR_ROOT}/docs/aq_visibility/gap_analysis.md"

echo "[$(date)] AQ_MATRIX_DONE"
