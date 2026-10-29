#!/usr/bin/env bash
# Source this file; never changes system CUDA or any other environment.
export SUPERDEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(hostname | tr '[:upper:]' '[:lower:]')" == lambda ]]; then
    source "$SUPERDEX_ROOT/configs/workstations/lambda.env"
fi
source "${SUPERDEX_ENV:-$SUPERDEX_ROOT/.venv-mjw}/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SUPERDEX_GPU_INDEX="${SUPERDEX_GPU_INDEX:-0}"
export SUPERDEX_GPU_UUID="$(nvidia-smi -i "$SUPERDEX_GPU_INDEX" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ "$SUPERDEX_GPU_UUID" != GPU-* ]]; then
    echo 'Cannot resolve selected GPU UUID' >&2
    return 1
fi
export CUDA_VISIBLE_DEVICES="$SUPERDEX_GPU_UUID"
export WARP_CACHE_PATH="$SUPERDEX_ROOT/cache/warp"
export UV_CACHE_DIR="$SUPERDEX_ROOT/cache/uv"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
