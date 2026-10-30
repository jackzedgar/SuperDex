#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source activate_mjw.sh
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export SUPERDEX_EXPLORE_RUN="$PWD/runs/core_iteration_reproduce_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/prepare_core_iteration.py
python scripts/check_core_objective.py
python scripts/run_core_iteration.py development
python scripts/run_core_iteration.py freeze
python - <<'PY'
from exploration_common import RUN,read_json,write_json
c=read_json(RUN/'budget.json');r=read_json(RUN/'frozen_recipe.json')
write_json(RUN/'resource_adaptation.json',dict(revised_allocation_seconds=c['allocation_seconds'],revised_max_common_target_per_arm=r['target_steps_per_arm'],confirmation_target_per_arm=131072,source='Fresh run: measured development rate, no historical resource events copied.'))
PY
python scripts/continue_core_iteration.py --stage core
# The follow-up decision uses only seen development outcomes and training diagnostics.
python scripts/choose_core_confirmation.py
python - <<'PY'
from exploration_common import RUN
assert (RUN/'confirmation_plan.json').exists(), 'A refresh trigger needs a separately labeled follow-up design; do not silently pool models or use reserved results.'
PY
python scripts/plan_confirmation_budget.py
python scripts/continue_core_iteration.py --stage confirmation
python - <<'PY'
from run_core_iteration import launch,remaining
for stage,category,cap in [('core','core',600),('confirmation','confirmation',400)]:
    launch(stage+'_heldout',category,['scripts/evaluate_core_iteration.py','--stage',stage],min(cap,remaining(category)-5))
launch('comparison_delivery','confirmation',['scripts/render_core_comparison.py','--output','videos_delivery'],90)
PY
python scripts/analyze_core_iteration.py
python scripts/summarize_core_findings.py
python scripts/validate_core_iteration.py
python scripts/report_core_iteration.py
printf 'Completed bounded reproduction: %s\nInspect the new curves and retained failures before writing an interpretation. No extra experiments are launched.\n' "$SUPERDEX_EXPLORE_RUN"
