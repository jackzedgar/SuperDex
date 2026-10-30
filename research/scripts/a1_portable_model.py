"""Opt-in loading of hash-verified, numerically validated frozen MuJoCo binaries."""
import hashlib,json,os
from pathlib import Path
import mujoco

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_task_model(path):
    path=Path(path).resolve();setting=os.environ.get('A1_PORTABLE_MODEL_MANIFEST')
    if not setting:return mujoco.MjModel.from_xml_path(str(path/'scene.xml'))
    manifest_path=(ROOT/setting).resolve();assert manifest_path.is_relative_to(ROOT)
    manifest=json.loads(manifest_path.read_text());assert manifest['status']=='complete' and manifest['mujoco_version']==mujoco.__version__
    relative=str(path.relative_to(ROOT));rows=[r for r in manifest['tasks'] if r['source_task_path']==relative];assert len(rows)==1
    row=rows[0];assert row['numeric_model_arrays_bitwise_equal']>400 and row['three_phase_forward_max_difference']<=1e-10
    assert sha(path/'scene.xml')==row['source_xml_sha256'] and sha(path/'reference.npz')==row['source_reference_sha256']
    binary=(ROOT/row['compiled_file']).resolve();assert binary.is_relative_to(ROOT)
    assert sha(binary)==row['compiled_sha256'] and binary.stat().st_size==row['compiled_bytes']
    return mujoco.MjModel.from_binary_path(str(binary))
