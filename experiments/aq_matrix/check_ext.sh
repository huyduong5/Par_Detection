#!/usr/bin/env bash
# Quick status of the extension matrix (run on sol1).
OUT="${OUT:-/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition/aq_visibility/output}"
MARK="${MARK:-/tmp/aq_ext_markers.out}"
EPOCHS="${EPOCHS:-30}"

echo "=== $(date) ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- markers ---"
tail -n 12 "${MARK}" 2>/dev/null || echo "(no markers)"
for VAR in C4b C3_MS C4b_MS; do
  LOG="${OUT}/ext_logs/train_${VAR}.log"
  echo "--- ${VAR} ---"
  [ -f "${OUT}/${VAR}/DONE" ] && echo "  train DONE"
  [ -f "${OUT}/${VAR}/batch.cfg" ] && echo "  batch/accum: $(cat "${OUT}/${VAR}/batch.cfg")"
  if [ -f "${LOG}" ]; then
    last=$(grep -E "^Epoch [0-9]+/${EPOCHS} —" "${LOG}" | tail -n 1)
    best=$(grep "Best model saved" "${LOG}" | tail -n 1 | sed 's/.*mA=//')
    step=$(grep -E "\[Epoch [0-9]+\] Batch" "${LOG}" | tail -n 1 | cut -c1-140)
    echo "  last epoch: ${last:-none} | best val mA: ${best:-n/a}"
    echo "  ${step}"
    ep=$(echo "${last}" | sed -E 's/^Epoch ([0-9]+)\/.*/\1/')
    avg=$(grep -E "^Epoch [0-9]+/${EPOCHS} —" "${LOG}" | sed -E 's/.*— ([0-9.]+)s/\1/' \
      | awk '{s+=$1; n++} END {if (n) printf "%.0f", s/n}')
    if [ -n "${ep}" ] && [ -n "${avg}" ]; then
      left=$(( (EPOCHS - ep) * avg / 3600 ))
      echo "  avg epoch ${avg}s, ETA ~${left}h (+ val/eval)"
    fi
    grep -E "Traceback|out of memory" "${LOG}" | tail -n 2
  else
    echo "  not started"
  fi
done
ls -la "${OUT}/eval_json/"C4b*_test.json "${OUT}/eval_json/"C3_MS_test.json 2>/dev/null
