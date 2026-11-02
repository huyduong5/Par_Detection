#!/usr/bin/env bash
# Train C0–C4 on 2 GPUs in parallel (independent variants).
#   GPU0: C0 → C2 → C4
#   GPU1: C1 → C3
# Then single-process eval + aggregate.
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
LOG_DIR="${OUT_ROOT}/matrix_logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
RUN_C5="${RUN_C5:-0}"
BUILD_DATA="${BUILD_DATA:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

cd "${PAR_ROOT}"

if [ "${BUILD_DATA}" = "1" ]; then
  bash "${SCRIPT_DIR}/build_data.sh"
fi

train_base() {
  local VAR="$1"
  local GPU="$2"
  local ODIR="${OUT_ROOT}/${VAR}"
  mkdir -p "${ODIR}"
  echo "[$(date)] TRAIN ${VAR} on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/train/train_c0_c1.py \
    --variant "${VAR}" \
    --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --occ_aug_prob 0.5 \
    --output_dir "${ODIR}" \
    2>&1 | tee -a "${ODIR}/train.log"
}

train_aq() {
  local VAR="$1"
  local GPU="$2"
  local ODIR="${OUT_ROOT}/${VAR}"
  mkdir -p "${ODIR}"
  echo "[$(date)] TRAIN ${VAR} on GPU${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/train/train.py \
    --config_variant "${VAR}" \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --occlusion_pkl_path "${OCC}/occlusion_pairs_train.pkl" \
    --output_dir "${ODIR}" \
    --no_ddp --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    2>&1 | tee -a "${ODIR}/train.log"
}

worker0() {
  train_base C0 "${GPU0}"
  train_aq C2 "${GPU0}"
  train_aq C4 "${GPU0}"
  if [ "${RUN_C5}" = "1" ]; then
    train_aq C5 "${GPU0}"
  fi
  echo "[$(date)] WORKER0_DONE"
}

worker1() {
  train_base C1 "${GPU1}"
  train_aq C3 "${GPU1}"
  echo "[$(date)] WORKER1_DONE"
}

echo "[$(date)] PARALLEL start GPU0=${GPU0} (C0→C2→C4) GPU1=${GPU1} (C1→C3) BS=${BATCH_SIZE} EP=${EPOCHS}"
worker0 > >(tee -a "${LOG_DIR}/worker0.log") 2>&1 &
PID0=$!
worker1 > >(tee -a "${LOG_DIR}/worker1.log") 2>&1 &
PID1=$!
echo "PID0=${PID0} PID1=${PID1}"

ec0=0
ec1=0
wait "${PID0}" || ec0=$?
wait "${PID1}" || ec1=$?
echo "[$(date)] workers finished ec0=${ec0} ec1=${ec1}"
if [ "${ec0}" -ne 0 ] || [ "${ec1}" -ne 0 ]; then
  echo "ERROR: a worker failed"; exit 1
fi

FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU0}")
FREE1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU1}")
if [ "${FREE0}" -ge "${FREE1}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU0}"
else
  export CUDA_VISIBLE_DEVICES="${GPU1}"
fi
echo "[$(date)] EVAL ALL on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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
