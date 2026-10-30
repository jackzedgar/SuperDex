"""Freeze requested inputs and preregister probes before any new learning."""
import hashlib,shutil,subprocess
from exploration_common import RUN,ROOT,read_json,write_json,resource_check

cfg=read_json(ROOT/'configs/structure_gradient_budget.json');old=ROOT/cfg['source_run'];assert not (RUN/'input_manifest.json').exists()
(RUN/'resources_before.txt').write_text(resource_check(2000)+'\n'+subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True))
(RUN/'wm').mkdir(exist_ok=False);(RUN/'gradient').mkdir();files=[]
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sha=hashlib.sha256(src.read_bytes()).hexdigest();assert sha==hashlib.sha256(dst.read_bytes()).hexdigest();files.append(dict(source=str(src.relative_to(ROOT)),copy=str(dst.relative_to(RUN)),sha256=sha,bytes=src.stat().st_size))
for n in ['dataset.npz','dataset_manifest.json','validation_sample_8.npy','validation_sample_32.npy','sdf_grid.npz','kinematics.npz','geometry_manifest.json','geometry_validation.json','normalization.npz']:
    copy(old/'wm'/n,RUN/'wm'/n)
for src in sorted((old/'wm').glob('W1_*.pt')):copy(src,RUN/'wm/A'/src.name)
for src in sorted((old/'wm/validation').rglob('prediction_W1_*.npz')):copy(src,RUN/'wm/A/validation'/src.relative_to(old/'wm/validation'))
for n in ['training.csv','config.json']:copy(old/'wm'/n,RUN/'wm/A'/n)
for kind in ['persistence','constant_velocity','ballistic']:
    for h in [8,32]:copy(old/'wm'/f'prediction_{kind}_{h:02d}.npz',RUN/'wm'/f'prediction_{kind}_{h:02d}.npz')
copy(ROOT/cfg['gradient']['policy'],RUN/'gradient/policy.pt');copy(ROOT/'runs/grasp_repair_20261026/fixed_initials.npz',RUN/'gradient/fixed_initials.npz')
for i in cfg['gradient']['initial_ids']:copy(ROOT/cfg['gradient']['prefix_source']/f'trial_{i:02d}.npz',RUN/'gradient/prefixes'/f'trial_{i:02d}.npz')
sources=[ROOT/'scripts/real_env.py',ROOT/'scripts/world_model_mini.py',ROOT/'data/processed/clip000000_obj000004_right/scene.xml',ROOT/'data/processed/clip000000_obj000004_right/trajectory.npz',ROOT/'data/processed/clip000000_obj000004_right/controller_reference.npz']
write_json(RUN/'input_manifest.json',dict(source_commit=cfg['source_commit'],files=files,unchanged_sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],A_reused_no_new_A_training=True,new_PPO_data_used=False))
write_json(RUN/'budget.json',cfg)
print('Frozen inputs and gradient design:',RUN,flush=True)
