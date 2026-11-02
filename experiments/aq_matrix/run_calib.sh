#!/usr/bin/env bash
# Save val/test probabilities for trained models, then fit per-attribute thresholds on val.
# No training. GPU0 starts now; GPU1 joins after run_seed3.sh is done (SEED3_DONE).
#   setsid nohup bash experiments/aq_matrix/run_calib.sh > /tmp/aq_calib.out 2>&1 &
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

GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; NUM_WORKERS="${NUM_WORKERS:-4}"
PKL="${PAR_ROOT}/MSP60k/SUBMIT/dataset_random.pkl"
IMG="${PAR_ROOT}/MSP60k/SUBMIT/images"
OCC="${PAR_ROOT}/data/occlusion"
OUT_ROOT="${PAR_ROOT}/aq_visibility/output"
PRED_DIR="${OUT_ROOT}/calib"
LOG_DIR="${OUT_ROOT}/ext_logs"
MARK="${MARK:-/tmp/aq_calib_markers.out}"
WAIT_MARK="/tmp/aq_seed3_markers.out"
VARIANTS=(C3_MS C4b_MS_bs16 C3 C3_MS_s43 C4b_MS_s43 C3_s43 C4_reaudit C4b_bs16 C4b_s43)
mkdir -p "${PRED_DIR}" "${LOG_DIR}"

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
  local lock="${PRED_DIR}/$1.lock"
  mkdir "${lock}" 2>/dev/null
}

# Checkpoint dir name -> config preset name.
cfg_of() { local n="${1%_s43}"; echo "${n%_bs16}"; }

predict() {
  local VAR="$1" GPU="$2" SPLIT="$3"
  local CKPT="${OUT_ROOT}/${VAR}/best_model.pth" NPZ="${PRED_DIR}/${VAR}_${SPLIT}.npz"
  [ -f "${NPZ}" ] && [ "${NPZ}" -nt "${CKPT}" ] && { log "SKIP ${VAR} ${SPLIT}"; return 0; }
  [ -f "${CKPT}" ] || { log "NO_CKPT ${VAR}"; return 1; }
  local attempt
  for attempt in 1 2 3; do
    wait_gpu_mem "${GPU}" 6000
    log "PRED ${VAR} ${SPLIT} GPU${GPU} attempt=${attempt}"
    CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/eval/evaluate.py \
      --checkpoint "${CKPT}" --config_variant "$(cfg_of "${VAR}")" --split "${SPLIT}" \
      --pkl_path "${PKL}" --img_dir "${IMG}" \
      --occlusion_pkl_path "${OCC}/occlusion_pairs_${SPLIT}.pkl" \
      --batch_size 8 --num_workers "${NUM_WORKERS}" \
      --output_json "${PRED_DIR}/${VAR}_${SPLIT}_eval.json" --save_preds "${NPZ}" \
      >> "${LOG_DIR}/calib_${VAR}.log" 2>&1 && [ -f "${NPZ}" ] && { log "PRED_${VAR}_${SPLIT}_DONE"; return 0; }
    log "PRED_${VAR}_${SPLIT}_RETRY"; sleep 60
  done
  log "PRED_${VAR}_${SPLIT}_FAIL"; return 1
}

worker() {
  local GPU="$1" VAR
  for VAR in "${VARIANTS[@]}"; do
    claim "${VAR}" || continue
    predict "${VAR}" "${GPU}" val && predict "${VAR}" "${GPU}" test
  done
  log "CALIB_WORKER_GPU${GPU}_DONE"
}

log "CALIB_START"
rm -rf "${PRED_DIR}"/*.lock
worker "${GPU0}" & P0=$!
( until grep -q "SEED3_DONE" "${WAIT_MARK}" 2>/dev/null; do sleep 300; done; worker "${GPU1}" ) & P1=$!
wait "${P0}"; wait "${P1}"

python experiments/aq_matrix/calibrate_thresholds.py --pred_dir "${PRED_DIR}" \
  --variants "$(IFS=,; echo "${VARIANTS[*]}")" \
  --output_md docs/aq_visibility/threshold_calibration.md \
  --output_json "${PRED_DIR}/calibration.json" >> "${MARK}" 2>&1
log "CALIB_DONE"
