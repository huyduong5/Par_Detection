#!/usr/bin/env bash
set -euo pipefail
par_project=/mnt/data8tb/huydq/projects/Pedestrain-Attribute-Recognition
par_python=/mnt/data8tb/huydq/miniconda3/envs/pedestrian-par/bin/python
par_reports="$par_project/experiments/par_resolution/reports/release_v5"
par_output="$par_project/experiments/par_resolution/outputs/release_v5"
mkdir -p "$par_output"
for par_attempt in $(seq 1 120); do
    if [[ -f "$par_reports/release_ready.json" ]]; then
        cd "$par_project/experiments/releases/par_resolution_v5/repo"
        exec "$par_python" -u -m experiments.par_resolution.worker --gpu "$1" --registry "$par_reports/checkpoint_registry.json" --manifest "$par_project/data/robustness/par_resolution_v1/manifests/index.json" --cpu-report "$par_reports/cpu_verification.json" --output "$par_output"
    fi
    sleep 15
done
printf 'Preflight readiness timeout; evaluation was not launched.\n' >&2
exit 1
