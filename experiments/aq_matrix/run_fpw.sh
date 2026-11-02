#!/usr/bin/env bash
# C4b_MS with fixed dataset-level pos_weight = sqrt(neg/pos) (batch 16x4, seed 42).
#   setsid nohup bash experiments/aq_matrix/run_fpw.sh > /tmp/aq_fpw.out 2>&1 &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PAR_ROOT}"
if [ -f /mnt/data8tb/huydq/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /mnt/data8tb/huydq/miniconda3/etc/profile.d/conda.sh
  conda activate pedestrian-par
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536 2>/dev/null || true

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-30}"; NUM_WORKERS="${NUM_WORKERS:-4}"
PKL="${PAR_ROOT}/MSP60k/SUBMIT/dataset_random.pkl"
IMG="${PAR_ROOT}/MSP60k/SUBMIT/images"
OCC="${PAR_ROOT}/data/occlusion"
OUT_ROOT="${PAR_ROOT}/aq_visibility/output"
JSON_DIR="${OUT_ROOT}/eval_json"
LOG_DIR="${OUT_ROOT}/ext_logs"
MARK="${MARK:-/tmp/aq_fpw_markers.out}"
WAIT_MARK="/tmp/aq_bs16_markers.out"
mkdir -p "${LOG_DIR}" "${JSON_DIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${MARK}"; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
wait_gpu_mem() {
  local gpu="$1" need="$2" n=0
  while [ "$(free_mb "${gpu}")" -lt "${need}" ]; do
    [ $((n % 10)) -eq 0 ] && log "WAIT_GPU${gpu} free=$(free_mb "${gpu}")MiB need=${need}MiB"
    n=$((n + 1)); sleep 60
  done
}

claim() {
  local lock="${OUT_ROOT}/$1.lock"
  if mkdir "${lock}" 2>/dev/null; then echo "${BASHPID}" > "${lock}/pid"; return 0; fi
  local pid; pid="$(cat "${lock}/pid" 2>/dev/null)"
  if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
    rm -rf "${lock}"
    mkdir "${lock}" 2>/dev/null && { echo "${BASHPID}" > "${lock}/pid"; return 0; }
  fi
  return 1
}

train_var() {
  local NAME="$1" GPU="$2" CFG="${1%_s${SEED}}"
  local ODIR="${OUT_ROOT}/${NAME}"
  mkdir -p "${ODIR}"
  echo "16 4" > "${ODIR}/batch.cfg"
  [ -f "${ODIR}/DONE" ] && { log "SKIP_TRAIN ${NAME} (DONE)"; return 0; }
  local attempt
  for attempt in 1 2 3; do
    wait_gpu_mem "${GPU}" 10500
    log "TRAIN ${NAME} (config ${CFG}, seed ${SEED}) GPU${GPU} attempt=${attempt}"
    CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/train/train.py \
      --config_variant "${CFG}" --seed "${SEED}" \
      --pkl_path "${PKL}" --img_dir "${IMG}" \
      --occlusion_pkl_path "${OCC}/occlusion_pairs_train.pkl" \
      --output_dir "${ODIR}" --no_ddp \
      --batch_size 16 --grad_accum_steps 4 --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" >> "${LOG_DIR}/train_${NAME}.log" 2>&1
    local rc=$?
    if [ "${rc}" -eq 0 ] && [ -f "${ODIR}/best_model.pth" ]; then
      touch "${ODIR}/DONE"; log "SEED_TRAIN_${NAME}_DONE"; return 0
    fi
    log "SEED_TRAIN_${NAME}_RETRY rc=${rc} attempt=${attempt}"; sleep 120
  done
  log "SEED_TRAIN_${NAME}_FAIL"; return 1
}

eval_var() {
  local NAME="$1" GPU="$2" CFG="${1%_s${SEED}}"
  local CKPT="${OUT_ROOT}/${NAME}/best_model.pth" OUTJ="${JSON_DIR}/${NAME}_test.json"
  [ -f "${OUTJ}" ] && [ "${OUTJ}" -nt "${CKPT}" ] && { log "SKIP_EVAL ${NAME}"; return 0; }
  local attempt
  for attempt in 1 2 3; do
    wait_gpu_mem "${GPU}" 6000
    log "EVAL ${NAME} GPU${GPU} attempt=${attempt}"
    CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/eval/evaluate.py \
      --checkpoint "${CKPT}" --config_variant "${CFG}" --split test \
      --pkl_path "${PKL}" --img_dir "${IMG}" \
      --occlusion_pkl_path "${OCC}/occlusion_pairs_test.pkl" \
      --batch_size 8 --num_workers "${NUM_WORKERS}" \
      --output_json "${OUTJ}" >> "${LOG_DIR}/eval_${NAME}.log" 2>&1 \
      && [ -f "${OUTJ}" ] && { log "SEED_EVAL_${NAME}_DONE"; return 0; }
    log "SEED_EVAL_${NAME}_RETRY attempt=${attempt}"; sleep 120
  done
  log "SEED_EVAL_${NAME}_FAIL"; return 1
}

run_one() {
  local NAME="$1" GPU="$2"
  if ! claim "${NAME}"; then log "SKIP ${NAME} on GPU${GPU} (claimed)"; return 0; fi
  train_var "${NAME}" "${GPU}" && eval_var "${NAME}" "${GPU}"
}

worker0() {
  run_one "C4b_MS_fpw" "${GPU0}"
  log "FPW_WORKER0_DONE"
}

log "FPW_START seed=${SEED}"
[ -f data/attr_pos_weight_sqrt.json ] || python tools/make_pos_weight.py --mode sqrt >> "${MARK}" 2>&1
rm -rf "${OUT_ROOT}/C4b_MS_fpw.lock"
worker0

# Probabilities for post-hoc threshold calibration (same procedure as run_calib.sh).
PRED_DIR="${OUT_ROOT}/calib"; mkdir -p "${PRED_DIR}"
for SPLIT in val test; do
  NPZ="${PRED_DIR}/C4b_MS_fpw_${SPLIT}.npz"
  [ -f "${NPZ}" ] && continue
  wait_gpu_mem "${GPU0}" 6000
  log "PRED C4b_MS_fpw ${SPLIT}"
  CUDA_VISIBLE_DEVICES="${GPU0}" python aq_visibility/eval/evaluate.py \
    --checkpoint "${OUT_ROOT}/C4b_MS_fpw/best_model.pth" --config_variant C4b_MS_fpw --split "${SPLIT}" \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --occlusion_pkl_path "${OCC}/occlusion_pairs_${SPLIT}.pkl" \
    --batch_size 8 --num_workers "${NUM_WORKERS}" \
    --output_json "${PRED_DIR}/C4b_MS_fpw_${SPLIT}_eval.json" --save_preds "${NPZ}" \
    >> "${LOG_DIR}/calib_C4b_MS_fpw.log" 2>&1 && log "PRED_C4b_MS_fpw_${SPLIT}_DONE"
done
python experiments/aq_matrix/calibrate_thresholds.py --pred_dir "${PRED_DIR}" \
  --variants C3,C3_MS,C4b_MS_bs16,C4b_MS_fpw,C3_s43,C3_MS_s43,C4b_MS_s43,C4_reaudit,C4b_bs16,C4b_s43 \
  --output_md docs/aq_visibility/threshold_calibration.md \
  --output_json "${PRED_DIR}/calibration.json" >> "${MARK}" 2>&1

python experiments/aq_matrix/build_ext_report.py --root "${PAR_ROOT}" \
  --output_md docs/aq_visibility/results_ext_report.md >> "${MARK}" 2>&1
log "FPW_DONE"
