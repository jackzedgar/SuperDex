#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source activate_mjw.sh
export SUPERDEX_EXPLORE_RUN="$PWD/runs/structure_gradient_reproduce_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/prepare_structure_gradient.py
python scripts/check_residual_parameterization.py
python scripts/run_structure_gradient.py structure
python scripts/run_structure_gradient.py gradient
python scripts/analyze_structure_gradient.py
python scripts/validate_structure_gradient.py
printf 'Completed bounded reproduction: %s\nInterpret the new results before writing interpretation.json and report; do not reuse an old conclusion automatically.\n' "$SUPERDEX_EXPLORE_RUN"
