#!/usr/bin/env bash
# Build occlusion pairs AFTER fixed split (dataset_random.pkl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
if [ -f "${REPO}/scripts/sol1/env.sh" ]; then
  source "${REPO}/scripts/sol1/env.sh"
fi
PAR_ROOT="${PAR_ROOT:-$REPO}"
PKL="${PAR_ROOT}/MSP60k/SUBMIT/dataset_random.pkl"
IMG="${PAR_ROOT}/MSP60k/SUBMIT/images"
OUT="${PAR_ROOT}/data/occlusion"
MAP="${OUT}/body_region_map.json"
MAX="${MAX_PER_SPLIT:-0}"  # 0 = full

mkdir -p "${OUT}/occlusion_images"
if [ ! -f "${MAP}" ]; then
  if [ -f "${PAR_ROOT}/data/body_region_map.json" ]; then
    cp "${PAR_ROOT}/data/body_region_map.json" "${MAP}"
  else
    echo "Missing body_region_map.json"; exit 1
  fi
fi

echo "[$(date)] build_synth_pairs max_per_split=${MAX}"
cd "${PAR_ROOT}"
python tools/occlusion/build_synth_pairs.py \
  --pkl_path "${PKL}" \
  --img_dir "${IMG}" \
  --region_map "${MAP}" \
  --output_dir "${OUT}" \
  --occ_img_dir "${OUT}/occlusion_images" \
  --splits train,val,test \
  --max_per_split "${MAX}" \
  --seed 42

echo "[$(date)] BUILD_DATA_DONE"
ls -la "${OUT}"/occlusion_pairs_*.pkl
