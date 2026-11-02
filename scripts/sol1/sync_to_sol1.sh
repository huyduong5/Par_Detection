#!/usr/bin/env bash
# Sync thesis packages + sol1 scripts to sol1 (run from dev machine with SSH).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-huydq@sol1.swin.edu.vn}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition}"
SSH_OPTS="${SSH_OPTS:--p 8000 -i $HOME/.ssh/id_ed25519}"

RSYNC=(rsync -az -e "ssh ${SSH_OPTS}")

"${RSYNC[@]}" "${ROOT}/aq_visibility/" "${REMOTE_HOST}:${REMOTE_ROOT}/aq_visibility/"
"${RSYNC[@]}" "${ROOT}/_my_model_8/" "${REMOTE_HOST}:${REMOTE_ROOT}/_my_model_8/"
"${RSYNC[@]}" "${ROOT}/scripts/sol1/" "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/sol1/"
"${RSYNC[@]}" "${ROOT}/experiments/" "${REMOTE_HOST}:${REMOTE_ROOT}/experiments/"
if [ -d "${ROOT}/docs" ]; then
  "${RSYNC[@]}" "${ROOT}/docs/" "${REMOTE_HOST}:${REMOTE_ROOT}/docs/"
fi
if [ -d "${ROOT}/data" ]; then
  "${RSYNC[@]}" "${ROOT}/data/" "${REMOTE_HOST}:${REMOTE_ROOT}/data/"
fi
if [ -d "${ROOT}/tools/occlusion" ]; then
  "${RSYNC[@]}" "${ROOT}/tools/occlusion/" "${REMOTE_HOST}:${REMOTE_ROOT}/tools/occlusion/"
fi

echo "Synced thesis packages. On sol1:"
echo "  bash scripts/sol1/12_aq_matrix.sh"
