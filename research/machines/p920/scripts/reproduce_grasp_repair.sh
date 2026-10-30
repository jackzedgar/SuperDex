#!/usr/bin/env bash
# CPU first; GPU work is serial and confined to the installation's GPU-0 UUID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source activate_mjw.sh
MODE="${1:-cpu}"
RUN_PATH="${2:-$ROOT/runs/grasp_repair_$(date -u +%Y%m%dT%H%M%SZ)}"
export SUPERDEX_GRASP_RUN="$RUN_PATH"
export MUJOCO_GL=disable
if [[ "$MODE" == cpu ]]; then
  if [[ -e "$RUN_PATH" ]]; then echo "Use a NEW result directory; refusing to overwrite $RUN_PATH"; exit 2; fi
  mkdir -p "$RUN_PATH"
  nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv > "$RUN_PATH/resources_start.txt"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv >> "$RUN_PATH/resources_start.txt"
  python scripts/grasp_lab.py --variants baseline contact4 wrist_ff wrist_ff35 zero_velocity --tasks A B > "$RUN_PATH/cpu_ablation.log" 2>&1
  python scripts/grasp_lab.py --variants baseline --tasks C > "$RUN_PATH/cpu_policy.log" 2>&1
  python scripts/fit_grip_distribution.py > "$RUN_PATH/grip_fit.log" 2>&1
  python scripts/grasp_lab.py --variants grip_balance --tasks A B > "$RUN_PATH/grip_balance.log" 2>&1
  python scripts/complete_contact_target.py > "$RUN_PATH/contact_completion.log" 2>&1
  python scripts/grasp_lab.py --variants index_contact --tasks A B > "$RUN_PATH/index_contact.log" 2>&1
  python scripts/grasp_lab.py --variants grip_balance_index --tasks A B > "$RUN_PATH/grip_balance_index.log" 2>&1
  python scripts/grasp_wrench_audit.py > "$RUN_PATH/wrench.log" 2>&1
  python scripts/audit_grasp_geometry.py > "$RUN_PATH/geometry.log" 2>&1
  python scripts/audit_initialization.py > "$RUN_PATH/initialization.log" 2>&1
  python scripts/grasp_microperturb.py > "$RUN_PATH/microperturb.log" 2>&1
  echo "CPU diagnosis saved: $RUN_PATH"
  echo "Next: bash scripts/reproduce_grasp_repair.sh gpu '$RUN_PATH'"
elif [[ "$MODE" == gpu ]]; then
  test -f "$RUN_PATH/fixed_initials.npz"
  # This all-in-one path includes GPU actor runs, so reserve a larger margin than
  # the separately usable 340-MiB physics-only diagnostics.
  python -c 'import subprocess; s=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True); free=int(s.splitlines()[0].split(",")[1]); assert free>=1024, "Less than 1 GiB free on GPU 0: GPU tests NOT RUN, not an algorithm failure"'
  for n in 1 2 8; do python scripts/grasp_batch_audit.py --worlds "$n" > "$RUN_PATH/batch$n.log" 2>&1; done
  for n in 2 8; do python scripts/grasp_reset_markers.py --worlds "$n" > "$RUN_PATH/reset_markers_$n.log" 2>&1; done
  python scripts/grasp_gpu_microperturb.py > "$RUN_PATH/gpu_microperturb.log" 2>&1
  for variant in baseline grip_balance; do python scripts/grasp_gpu_trials.py --variant "$variant" > "$RUN_PATH/gpu_$variant.log" 2>&1; done
  python scripts/grasp_checkpoint_pair.py --policy zero > "$RUN_PATH/gpu_original_zero.log" 2>&1
  python scripts/grasp_checkpoint_pair.py --policy checkpoint > "$RUN_PATH/gpu_original_checkpoint.log" 2>&1
  python scripts/grasp_checkpoint_pair.py --variant grip_balance --policy checkpoint > "$RUN_PATH/gpu_original_candidate.log" 2>&1
  echo "GPU diagnosis saved: $RUN_PATH"
  echo "Next: bash scripts/reproduce_grasp_repair.sh report '$RUN_PATH'"
elif [[ "$MODE" == report ]]; then
  test -f "$RUN_PATH/gpu_original_graph/grip_balance_checkpoint/summary.json"
  python scripts/analyze_grasp_repair.py > "$RUN_PATH/analysis.log" 2>&1
  python scripts/report_grasp_repair.py > "$RUN_PATH/report.log" 2>&1
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python scripts/visualize_grasp_repair.py > "$RUN_PATH/visualization.log" 2>&1
  echo "Report: $RUN_PATH/report.html"
else
  echo "Usage: bash scripts/reproduce_grasp_repair.sh {cpu|gpu|report} [absolute-new-run-directory]"; exit 2
fi
