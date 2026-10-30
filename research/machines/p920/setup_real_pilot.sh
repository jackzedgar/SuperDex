#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export UV_CACHE_DIR="$PWD/cache/uv"
# The existing Isaac Gym interpreter is never a dependency installation target.
test -x .venv-mjw/bin/python
uv pip sync --python .venv-mjw/bin/python requirements-training.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
if [ ! -d references/hand_tracking_toolkit/.git ]; then
  git clone https://github.com/facebookresearch/hand_tracking_toolkit.git references/hand_tracking_toolkit
  git -C references/hand_tracking_toolkit checkout --detach c521930d966c871977aaa4d15f33bfd4b3781f09
fi
test "$(git -C references/hand_tracking_toolkit rev-parse HEAD)" = c521930d966c871977aaa4d15f33bfd4b3781f09
source activate_mjw.sh
python scripts/fetch_minimal.py --clip 0 --object 4
