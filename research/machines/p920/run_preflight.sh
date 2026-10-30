#!/usr/bin/env bash
set -euo pipefail
cd /lost+found/nvme/dexmimic
source superdex/activate_mjw.sh
run_dir="superdex/runs/preflight_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"
timeout 480 python superdex/scripts/mjw_preflight.py --scene cube --worlds 64 --output "$run_dir/mjw_cube_64.json" > "$run_dir/mjw_cube_64.log" 2>&1
for n in 64 256 1024; do
    timeout 480 python superdex/scripts/mjw_preflight.py --scene wuji --worlds "$n" --output "$run_dir/mjw_wuji_${n}.json" > "$run_dir/mjw_wuji_${n}.log" 2>&1
done
source activate.sh
export CUDA_VISIBLE_DEVICES=0
python superdex/scripts/verify_gradients.py --output "$run_dir/gradient_checks.json" > "$run_dir/gradient_checks.log" 2>&1
if grep -Ein 'warning|error|traceback|limit reached|overflow' "$run_dir"/*.log; then
    printf 'Review diagnostic messages before accepting this run: %s\n' "$run_dir"
    exit 2
fi
printf 'All bounded preflight checks passed. Results: %s\n' "$run_dir"

