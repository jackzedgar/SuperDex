#!/usr/bin/env bash
# Source this file. Existing Isaac Gym venv and driver remain untouched.
export SUPERDEX_ROOT=/lost+found/nvme/dexmimic/superdex
source "$SUPERDEX_ROOT/.venv-mjw/bin/activate"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-810d1bb5-dcdc-3d20-c3c5-22bffb1fcf9d
export WARP_CACHE_PATH="$SUPERDEX_ROOT/cache/warp"
export UV_CACHE_DIR="$SUPERDEX_ROOT/cache/uv"
export OMP_NUM_THREADS=4
