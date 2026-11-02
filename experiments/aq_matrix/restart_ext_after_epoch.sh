#!/usr/bin/env bash
# One-off: wait until the running C4b and C3_MS jobs have saved latest_checkpoint.pth,
# then restart run_ext.sh with the shared queue (C4b_MS goes to the first free GPU).
# Training resumes from the saved epoch.
set -uo pipefail
PAR=/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition
OUT="${PAR}/aq_visibility/output"
cd "${PAR}"

lock_selftest() {
  local T; T="$(mktemp -d)"
  OUT_ROOT="${T}"
  # shellcheck disable=SC1090
  source <(sed -n '/^claim() {/,/^}/p' experiments/aq_matrix/run_ext.sh)
  ( claim X && echo A; sleep 3 ) > "${T}/a" &
  sleep 1
  ( claim X && echo B ) > "${T}/b"
  wait
  mkdir -p "${T}/Y.lock"; echo 999999 > "${T}/Y.lock/pid"
  local stale=no; claim Y && stale=yes
  [ "$(cat "${T}/a")" = "A" ] && [ ! -s "${T}/b" ] && [ "${stale}" = yes ]
}

if ! lock_selftest; then echo "[$(date)] LOCK_SELFTEST_FAIL"; exit 1; fi
echo "[$(date)] LOCK_SELFTEST_OK"

ready() {
  local f="${OUT}/$1/latest_checkpoint.pth"
  [ -f "${f}" ] && [ $(( $(date +%s) - $(stat -c %Y "${f}") )) -ge 60 ]
}
until ready C4b && ready C3_MS; do sleep 60; done
echo "[$(date)] both checkpoints saved; restarting"

pkill -f "experiments/aq_matrix/run_ext.sh" || true
pkill -f "output/C4b --no_ddp" || true
pkill -f "output/C3_MS --no_ddp" || true
sleep 30
echo "--- restart with shared queue $(date '+%F %T') ---" >> /tmp/aq_ext_markers.out
setsid nohup bash experiments/aq_matrix/run_ext.sh > /tmp/aq_ext.out 2>&1 < /dev/null &
echo "[$(date)] RESTARTED pid=$!"
