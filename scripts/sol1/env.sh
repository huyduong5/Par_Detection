#!/usr/bin/env bash
# Shared paths on sol1 — adjust if project root differs.
export PAR_ROOT="${PAR_ROOT:-/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition}"
export PYTHONPATH="${PAR_ROOT}/aq_visibility:${PAR_ROOT}/_my_model_8:${PAR_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ -f /mnt/data8tb/huydq/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /mnt/data8tb/huydq/miniconda3/etc/profile.d/conda.sh
  conda activate pedestrian-par
fi
