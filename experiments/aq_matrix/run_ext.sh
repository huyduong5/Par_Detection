#!/usr/bin/env bash
# Extension matrix on 2 GPUs: C4b (sg_bce consistency), C3_MS, C4b_MS.
#   GPU0: C4b -> eval -> C4b_MS -> eval
#   GPU1: C3_MS -> eval
# Then aggregate + gap/small-attr analysis + 2-image comparison.
#
#   MODE=smoke bash experiments/aq_matrix/run_ext.sh   # 1 epoch, 30 batches, 200 eval samples
#   nohup bash experiments/aq_matrix/run_ext.sh > /tmp/aq_ext.out 2>&1 &
#
# Re-running is safe: finished variants are skipped (DONE marker), unfinished
# ones resume from latest_checkpoint.pth.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAR_ROOT="${PAR_ROOT:-$REPO}"
cd "${PAR_ROOT}"

if [ -f /mnt/data8tb/huydq/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /mnt/data8tb/huydq/miniconda3/etc/profile.d/conda.sh
  conda activate pedestrian-par
fi
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536 2>/dev/null || true

MODE="${MODE:-full}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PKL="${PAR_ROOT}/MSP60k/SUBMIT/dataset_random.pkl"
IMG="${PAR_ROOT}/MSP60k/SUBMIT/images"
OCC="${PAR_ROOT}/data/occlusion"

if [ "${MODE}" = "smoke" ]; then
  EPOCHS=1; MAX_TRAIN_BATCHES=30; MAX_VAL_BATCHES=10; EVAL_MAX=200
  OUT_ROOT="${PAR_ROOT}/aq_visibility/output/_smoke"
  JSON_DIR="${OUT_ROOT}/eval_json"
  MARK="${MARK:-/tmp/aq_ext_smoke.out}"
  [ "${KEEP:-0}" = "1" ] || rm -rf "${OUT_ROOT}"
else
  EPOCHS="${EPOCHS:-30}"; MAX_TRAIN_BATCHES=0; MAX_VAL_BATCHES=0; EVAL_MAX=0
  OUT_ROOT="${PAR_ROOT}/aq_visibility/output"
  JSON_DIR="${OUT_ROOT}/eval_json"
  MARK="${MARK:-/tmp/aq_ext_markers.out}"
fi
LOG_DIR="${OUT_ROOT}/ext_logs"
mkdir -p "${LOG_DIR}" "${JSON_DIR}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "${MARK}"; }

free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

# Wait until GPU $1 has at least $2 MiB free (other users share sol1).
wait_gpu_mem() {
  local gpu="$1" need="$2" n=0
  while true; do
    local free; free="$(free_mb "${gpu}")"
    if [ "${free:-0}" -ge "${need}" ]; then return 0; fi
    if [ $((n % 10)) -eq 0 ]; then
      log "WAIT_GPU${gpu} free=${free}MiB need=${need}MiB"
    fi
    n=$((n + 1)); sleep 60
  done
}

# Batch size is fixed at first launch and reused on resume (scheduler stays valid).
# Effective batch stays 64: 16x4 if VRAM allows, else 8x8.
pick_batch() {
  local gpu="$1" odir="$2"
  if [ -f "${odir}/batch.cfg" ]; then cat "${odir}/batch.cfg"; return; fi
  local bs=16 ga=4
  if [ "${BATCH_SIZE:-}" != "" ]; then
    bs="${BATCH_SIZE}"; ga="${GRAD_ACCUM:-$((64 / bs))}"
  elif [ "$(free_mb "${gpu}")" -lt 11000 ]; then
    bs=8; ga=8
  fi
  echo "${bs} ${ga}" > "${odir}/batch.cfg"
  echo "${bs} ${ga}"
}

train_var() {
  local VAR="$1" GPU="$2"
  local ODIR="${OUT_ROOT}/${VAR}"
  mkdir -p "${ODIR}"
  if [ -f "${ODIR}/DONE" ]; then log "SKIP_TRAIN ${VAR} (DONE)"; return 0; fi
  local attempt
  for attempt in 1 2 3; do
    local bs ga
    read -r bs ga < <(pick_batch "${GPU}" "${ODIR}")
    wait_gpu_mem "${GPU}" $(( bs >= 16 ? 10500 : 7000 ))
    log "TRAIN ${VAR} GPU${GPU} attempt=${attempt} batch=${bs} accum=${ga} epochs=${EPOCHS}"
    CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/train/train.py \
      --config_variant "${VAR}" \
      --pkl_path "${PKL}" --img_dir "${IMG}" \
      --occlusion_pkl_path "${OCC}/occlusion_pairs_train.pkl" \
      --output_dir "${ODIR}" --no_ddp \
      --batch_size "${bs}" --grad_accum_steps "${ga}" --num_workers "${NUM_WORKERS}" \
      --epochs "${EPOCHS}" \
      --max_train_batches "${MAX_TRAIN_BATCHES}" --max_val_batches "${MAX_VAL_BATCHES}" \
      >> "${LOG_DIR}/train_${VAR}.log" 2>&1
    local rc=$?
    if [ "${rc}" -eq 0 ] && [ -f "${ODIR}/best_model.pth" ]; then
      touch "${ODIR}/DONE"
      log "EXT_TRAIN_${VAR}_DONE"
      return 0
    fi
    log "EXT_TRAIN_${VAR}_RETRY rc=${rc} attempt=${attempt}"
    sleep 120
  done
  log "EXT_TRAIN_${VAR}_FAIL"
  return 1
}

eval_var() {
  local VAR="$1" GPU="$2"
  local CKPT="${OUT_ROOT}/${VAR}/best_model.pth"
  local OUTJ="${JSON_DIR}/${VAR}_test.json"
  if [ ! -f "${CKPT}" ]; then log "SKIP_EVAL ${VAR}: no ckpt"; return 1; fi
  if [ -f "${OUTJ}" ] && [ "${OUTJ}" -nt "${CKPT}" ]; then log "SKIP_EVAL ${VAR} (fresh JSON)"; return 0; fi
  local attempt
  for attempt in 1 2 3; do
    wait_gpu_mem "${GPU}" 6000
    log "EVAL ${VAR} GPU${GPU} attempt=${attempt}"
    CUDA_VISIBLE_DEVICES="${GPU}" python aq_visibility/eval/evaluate.py \
      --checkpoint "${CKPT}" --config_variant "${VAR}" --split test \
      --pkl_path "${PKL}" --img_dir "${IMG}" \
      --occlusion_pkl_path "${OCC}/occlusion_pairs_test.pkl" \
      --batch_size 8 --num_workers "${NUM_WORKERS}" --max_samples "${EVAL_MAX}" \
      --output_json "${OUTJ}" \
      >> "${LOG_DIR}/eval_${VAR}.log" 2>&1
    if [ $? -eq 0 ] && [ -f "${OUTJ}" ]; then
      log "EXT_EVAL_${VAR}_DONE"
      return 0
    fi
    log "EXT_EVAL_${VAR}_RETRY attempt=${attempt}"
    sleep 120
  done
  log "EXT_EVAL_${VAR}_FAIL"
  return 1
}

# A variant is claimed by whichever GPU worker reaches it first (atomic mkdir lock),
# so C4b_MS goes to the GPU that frees up first.
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

run_queue() {
  local GPU="$1"; shift
  local VAR
  for VAR in "$@"; do
    if ! claim "${VAR}"; then log "SKIP ${VAR} on GPU${GPU} (claimed by other worker)"; continue; fi
    train_var "${VAR}" "${GPU}" && eval_var "${VAR}" "${GPU}"
  done
}

worker0() {
  run_queue "${GPU0}" C4b C4b_MS
  log "WORKER0_DONE"
}

worker1() {
  run_queue "${GPU1}" C3_MS C4b_MS
  log "WORKER1_DONE"
}

log "EXT_START mode=${MODE} epochs=${EPOCHS} out=${OUT_ROOT}"
rm -rf "${OUT_ROOT}"/*.lock
worker0 & P0=$!
sleep 30
worker1 & P1=$!
wait "${P0}"; wait "${P1}"

if [ "${MODE}" = "smoke" ]; then
  python experiments/aq_matrix/check_smoke.py --out_root "${OUT_ROOT}" \
    --variants C4b,C3_MS,C4b_MS 2>&1 | tee -a "${MARK}"
  log "EXT_SMOKE_DONE"
  exit 0
fi

VARS=C0,C1,C2,C3,C4_reaudit,C4b,C3_MS,C4b_MS
python experiments/aq_matrix/aggregate_results.py --root "${PAR_ROOT}" \
  --output_md docs/aq_visibility/results_ext.md --variants "${VARS}" >> "${MARK}" 2>&1
python experiments/aq_matrix/analyze_gaps.py --root "${PAR_ROOT}" \
  --output_md docs/aq_visibility/gap_analysis.md --variants "${VARS}" >> "${MARK}" 2>&1
python experiments/aq_matrix/small_attr_gate.py --root "${PAR_ROOT}" \
  --output_md docs/aq_visibility/small_attr_gate.md \
  --variants C0,C3,C4_reaudit,C4b,C3_MS,C4b_MS >> "${MARK}" 2>&1

IDX=$(python - << 'PY'
import json, os
p = 'aq_visibility/output/compare_5c_images/meta.json'
if os.path.isfile(p):
    print(' '.join(str(i['global_idx']) for i in json.load(open(p))['images'][:2]))
PY
)
wait_gpu_mem "${GPU1}" 6000
CUDA_VISIBLE_DEVICES="${GPU1}" python aq_visibility/tools/compare_c0_c4_images.py \
  --out_dir aq_visibility/output/compare_ext_images \
  --variants C0 C1 C2 C3 C4_reaudit C4b C3_MS C4b_MS \
  ${IDX:+--indices ${IDX}} >> "${LOG_DIR}/compare_ext.log" 2>&1 \
  && log "EXT_COMPARE_DONE" || log "EXT_COMPARE_FAIL"
log "EXT_DONE"
