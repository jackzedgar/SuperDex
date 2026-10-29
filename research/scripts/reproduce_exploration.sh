#!/usr/bin/env bash
# Fresh, bounded replay of the two learning experiments on the existing station.
# A resource error or budget interruption stops with preserved logs; it is not
# silently converted into an algorithm score. No unrelated process is touched.
set -euo pipefail
cd "$(dirname "$0")/.."
source activate_mjw.sh
export SUPERDEX_EXPLORE_RUN="$PWD/runs/explore_repro_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$SUPERDEX_EXPLORE_RUN"
printf 'New result directory: %s\n' "$SUPERDEX_EXPLORE_RUN"
python -c 'import rtree; assert rtree.__version__ == "1.4.1", "Install rtree==1.4.1 only in .venv-mjw before reproduction"'
sha256sum runs/real_track_20261026/ppo/checkpoint.pt | cut -d ' ' -f1 > "$SUPERDEX_EXPLORE_RUN/baseline_checkpoint_sha256.txt"
cp configs/exploration_budget.json "$SUPERDEX_EXPLORE_RUN/budget.json"
nvidia-smi > "$SUPERDEX_EXPLORE_RUN/resources_before.txt"
# GNU timeout targets only these newly launched experiment process groups.
# Outer caps sum to 3480 s; the remainder of the 3600 s GPU-process allowance
# covers small orchestration overhead. CPU-only geometry/report work is separate.
timeout --signal=TERM --kill-after=5s 1200s python -u scripts/train_exploratory_ppo.py --arm uniform_rsi --steps 40960 --seconds 1180 > "$SUPERDEX_EXPLORE_RUN/ppo_uniform.log" 2>&1
timeout --signal=TERM --kill-after=5s 1200s python -u scripts/train_exploratory_ppo.py --arm start50 --steps 40960 --seconds 1180 > "$SUPERDEX_EXPLORE_RUN/ppo_start50.log" 2>&1
test -f "$SUPERDEX_EXPLORE_RUN/uniform_rsi/checkpoint_040960.pt"
test -f "$SUPERDEX_EXPLORE_RUN/start50/checkpoint_040960.pt"
python -u scripts/prepare_wm_geometry.py > "$SUPERDEX_EXPLORE_RUN/geometry.log" 2>&1
python scripts/validate_wm_geometry.py > "$SUPERDEX_EXPLORE_RUN/geometry_validation.log" 2>&1
timeout --signal=TERM --kill-after=5s 300s python -u scripts/collect_wm_rollouts.py > "$SUPERDEX_EXPLORE_RUN/collection.log" 2>&1
python scripts/build_wm_dataset.py > "$SUPERDEX_EXPLORE_RUN/dataset.log" 2>&1
# Reproduce the completed 590-update comparison, not the original 1000-update cap.
timeout --signal=TERM --kill-after=5s 600s python -u scripts/train_world_models_mini.py --seconds 580 --updates 590 > "$SUPERDEX_EXPLORE_RUN/world_models.log" 2>&1
timeout --signal=TERM --kill-after=5s 180s python -u scripts/evaluate_exploratory_ppo.py > "$SUPERDEX_EXPLORE_RUN/canonical_eval.log" 2>&1
python scripts/analyze_exploration.py > "$SUPERDEX_EXPLORE_RUN/analysis.log" 2>&1
python scripts/validate_exploration.py > "$SUPERDEX_EXPLORE_RUN/artifact_validation.log" 2>&1
python scripts/visualize_exploration.py
nvidia-smi > "$SUPERDEX_EXPLORE_RUN/resources_after.txt"
printf 'Complete. Frozen metrics, figures and logs: %s\n' "$SUPERDEX_EXPLORE_RUN"
# report_exploration.py documents the original experiment's interpretation;
# a new seed/numerical replay needs fresh interpretation, not copied conclusions.
