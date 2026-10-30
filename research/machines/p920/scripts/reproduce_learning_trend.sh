#!/usr/bin/env bash
# Bounded reproduction from the preserved b93903f experiment artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
source activate_mjw.sh
export SUPERDEX_EXPLORE_RUN="$PWD/runs/learning_trend_repro_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SUPERDEX_EXPLORE_RUN"
printf 'Fresh result directory: %s\n' "$SUPERDEX_EXPLORE_RUN"
python scripts/prepare_learning_trend.py
python -u scripts/run_learning_trend.py --only training
PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}" python -c 'from exploration_common import RUN,read_json; assert all(x["returncode"]==0 for x in read_json(RUN/"process_ledger.json")["processes"]), "A training job failed; inspect preserved logs before evaluation."'
python -u scripts/run_learning_trend.py --only evaluation
python scripts/analyze_learning_trend.py > "$SUPERDEX_EXPLORE_RUN/analysis.log"
python scripts/validate_learning_trend.py > "$SUPERDEX_EXPLORE_RUN/validation.log"
python scripts/visualize_learning_trend.py
nvidia-smi > "$SUPERDEX_EXPLORE_RUN/resources_final.txt"
printf 'Done. No automatic extra experiments. Fresh curves and metrics: %s\n' "$SUPERDEX_EXPLORE_RUN"
# Interpretation is intentionally not copied from a previous run.
