"""Freeze the prior experiment's inputs without rebuilding data or physics."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from exploration_common import RUN,ROOT,GPU,read_json,write_json,resource_check

cfg=read_json(ROOT/'configs/learning_trend_budget.json');old=ROOT/cfg['source_run']
assert RUN.resolve()!=old.resolve() and not (RUN/'input_manifest.json').exists()
(RUN/'resources_before.txt').write_text(resource_check(2000)+'\n'+subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True))
(RUN/'wm').mkdir(exist_ok=False)
files=['dataset.npz','dataset_manifest.json','validation_sample_8.npy','validation_sample_32.npy','sdf_grid.npz','kinematics.npz','geometry_manifest.json','geometry_validation.json']
frozen=[]
for name in files:
    src=old/'wm'/name;dst=RUN/'wm'/name;shutil.copy2(src,dst);digest=hashlib.sha256(src.read_bytes()).hexdigest();assert digest==hashlib.sha256(dst.read_bytes()).hexdigest()
    frozen.append(dict(name=name,source=str(src.relative_to(ROOT)),bytes=src.stat().st_size,sha256=digest))
source_files=[old/arm/'checkpoint_final.pt' for arm in ['uniform_rsi','start50']]+[ROOT/'runs/grasp_repair_20261026/fixed_initials.npz',ROOT/'data/processed/clip000000_obj000004_right/trajectory.npz',ROOT/'data/processed/clip000000_obj000004_right/controller_reference.npz',ROOT/'data/processed/clip000000_obj000004_right/scene.xml',ROOT/'scripts/real_env.py',ROOT/'scripts/world_model_mini.py']
write_json(RUN/'input_manifest.json',dict(baseline_commit=cfg['baseline_commit'],source_run=cfg['source_run'],frozen_wm_files=frozen,source_files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_files],new_PPO_logs_in_WM=False,geometry_rebuilt=False,physics_reconfigured=False))
write_json(RUN/'budget.json',cfg)
print('Prepared new run:',RUN,'; frozen dataset, split, validation samples and SDF copied byte-for-byte.')
