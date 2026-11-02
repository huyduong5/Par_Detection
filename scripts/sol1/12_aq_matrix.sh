#!/usr/bin/env bash
# Launch thesis AQ matrix on sol1.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/env.sh"
export PYTHONUNBUFFERED=1
ulimit -n 65536 2>/dev/null || true

FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
FREE1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
if [ "${FREE0}" -ge "${FREE1}" ]; then export CUDA_VISIBLE_DEVICES=0; else export CUDA_VISIBLE_DEVICES=1; fi
echo "GPU=${CUDA_VISIBLE_DEVICES} free0=${FREE0} free1=${FREE1}"

export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EPOCHS="${EPOCHS:-30}"
export MAX_PER_SPLIT="${MAX_PER_SPLIT:-0}"
export BUILD_DATA="${BUILD_DATA:-1}"
export RUN_C5="${RUN_C5:-0}"

cd "${PAR_ROOT}"
MODE="${MODE:-parallel}"   # parallel | sequential
if [ "${MODE}" = "sequential" ]; then
  nohup bash experiments/aq_matrix/run_all.sh > /tmp/aq_matrix.out 2>&1 &
else
  # 2-GPU parallel: GPU0 C0→C2→C4, GPU1 C1→C3
  export BUILD_DATA="${BUILD_DATA:-0}"
  nohup bash experiments/aq_matrix/run_parallel_2gpu.sh > /tmp/aq_matrix.out 2>&1 &
fi
echo PID=$!
echo "mode=${MODE} log=/tmp/aq_matrix.out"
