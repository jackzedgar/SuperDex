"""Workstation-specific locations/devices; frozen research inputs stay unchanged."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('SUPERDEX_ROOT', Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get('SUPERDEX_DATA_DIR', ROOT / 'data/processed/clip000000_obj000004_right')).resolve()
GPU = os.environ.get('SUPERDEX_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', ''))
if not GPU.startswith('GPU-') or ',' in GPU:
    raise RuntimeError('Source activate_mjw.sh to select exactly one physical GPU by its current UUID.')
