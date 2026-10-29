#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source activate_mjw.sh
# This pilot is deliberately one clip and one physical environment.
# Each training invocation gets a NEW directory; existing checkpoints are retained.
PILOT_RUN="${1:-runs/pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
test ! -e "$PILOT_RUN"
mkdir -p runs/real_track_20261026
python scripts/prepare_real.py --start 118 --end 149
python scripts/qualify_real.py
python scripts/diagnose_grasp.py
python scripts/real_env.py
python -u scripts/train_real_ppo.py --worlds 1 --updates 160 --seconds 900 --output "$PILOT_RUN"
python scripts/visualize_real.py --ppo "$PILOT_RUN"
