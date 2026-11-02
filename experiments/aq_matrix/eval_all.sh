#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO}/scripts/sol1/env.sh" ]; then
  # shellcheck source=/dev/null
  source "${REPO}/scripts/sol1/env.sh"
fi
PAR_ROOT="${PAR_ROOT:-$REPO}"
PKL="${PAR_ROOT}/MSP60k/SUBMIT/dataset_random.pkl"
IMG="${PAR_ROOT}/MSP60k/SUBMIT/images"
OCC="${PAR_ROOT}/data/occlusion"
OUT_ROOT="${PAR_ROOT}/aq_visibility/output"
JSON_DIR="${OUT_ROOT}/eval_json"
mkdir -p "${JSON_DIR}"
cd "${PAR_ROOT}"

eval_aq() {
  local VAR="$1"
  local CKPT="${OUT_ROOT}/${VAR}/best_model.pth"
  if [ ! -f "${CKPT}" ]; then
    echo "SKIP ${VAR}: missing ${CKPT}"
    return 0
  fi
  echo "[$(date)] EVAL ${VAR}"
  python aq_visibility/eval/evaluate.py \
    --checkpoint "${CKPT}" \
    --config_variant "${VAR}" \
    --split test \
    --pkl_path "${PKL}" --img_dir "${IMG}" \
    --occlusion_pkl_path "${OCC}/occlusion_pairs_test.pkl" \
    --output_json "${JSON_DIR}/${VAR}_test.json" \
    || true
}

# C0/C1 use _my_model_8 eval if available
for VAR in C0 C1; do
  CKPT="${OUT_ROOT}/${VAR}/best_model.pth"
  if [ -f "${CKPT}" ] && [ -f "${PAR_ROOT}/_my_model_8/eval_test_json.py" ]; then
    echo "[$(date)] EVAL ${VAR} (my_model_8)"
    python "${PAR_ROOT}/_my_model_8/eval_test_json.py" \
      --checkpoint "${CKPT}" \
      --pkl_path "${PKL}" --img_dir "${IMG}" \
      --output_json "${JSON_DIR}/${VAR}_test.json" || true
  fi
done

for VAR in C2 C3 C4 C5; do
  eval_aq "${VAR}"
done

echo "[$(date)] EVAL_ALL_DONE"
