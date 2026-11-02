#!/usr/bin/env bash
# Check C4 train progress + ETA on sol1 (or any host with the run).
# Usage: bash experiments/aq_matrix/check_c4.sh
set -euo pipefail
PAR_ROOT="${PAR_ROOT:-/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition}"
LOG="${PAR_ROOT}/aq_visibility/output/C4/train.log"

echo "=== $(date) ==="
if [ ! -f "${LOG}" ]; then
  echo "Missing log: ${LOG}"; exit 1
fi

pgrep -afu "${USER:-huydq}" "config_variant C4" | grep -v pgrep | head -3 || echo "(C4 process not found)"

echo "--- last log ---"
tail -n 8 "${LOG}"

echo "--- epoch times ---"
grep -E "Epoch [0-9]+/30 —" "${LOG}" | tail -n 10 || true

python3 - <<'PY'
import os, re
from datetime import datetime, timedelta

log_path = os.environ.get(
    "C4_LOG",
    "/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition/aq_visibility/output/C4/train.log",
)
text = open(log_path, encoding="utf-8", errors="ignore").read()
pairs = [(int(a), float(b)) for a, b in re.findall(r"Epoch (\d+)/30 — ([0-9.]+)s", text)]
trips = re.findall(r"\[Epoch (\d+)\] Batch (\d+)/(\d+)", text)
if not trips:
    print("No batch lines yet.")
    raise SystemExit
ep, b, tot = map(int, trips[-1])
frac = (ep - 1) + b / tot
remain_ep = 30 - frac
avg = (sum(t for _, t in pairs) / len(pairs)) if pairs else 1720.0
if pairs:
    recent = pairs[-3:]
    avg = sum(t for _, t in recent) / len(recent)
sec = remain_ep * avg
eta = datetime.now() + timedelta(seconds=sec)
done = "YES" if ("Training complete" in text) else "NO"
print(f"progress: epoch {ep}/30 batch {b}/{tot} ({100*frac/30:.1f}%)")
print(f"avg_epoch: {avg/60:.1f} min  remain: {sec/3600:.1f}h")
print(f"ETA: {eta.strftime('%Y-%m-%d %H:%M')}  finished={done}")
best = re.findall(r"Best model saved! mA=([0-9.]+)", text)
if best:
    print(f"best_val_mA: {best[-1]}")
PY
