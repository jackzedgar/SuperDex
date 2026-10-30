"""Prepare separate seen-development and previously unused held-out conditions."""
import hashlib,shutil,subprocess
import numpy as np
from exploration_common import RUN,ROOT,DATA,read_json,write_json,resource_check
cfg=read_json(ROOT/'configs/core_iteration_budget.json');assert not (RUN/'budget.json').exists()
write_json(RUN/'budget.json',cfg);(RUN/'resources_before.txt').write_text(resource_check(2400)+'\n'+subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv'],text=True))
initial=np.load(ROOT/'runs/grasp_repair_20261026/fixed_initials.npz')
np.savez(RUN/'development_initials.npz',qpos=initial['qpos'],qvel=initial['qvel'])
ref=np.load(DATA/'trajectory.npz');n=cfg['evaluation']['heldout_count'];rng=np.random.default_rng(cfg['evaluation']['heldout_seed']);q=np.repeat(ref['qpos'][0][None],n,axis=0);v=np.repeat(ref['qvel'][0][None],n,axis=0);q[:,26:29]+=rng.normal(0,.001,(n,3));q[:,6:26]+=rng.normal(0,.002,(n,20));np.savez(RUN/'heldout_initials.npz',qpos=q,qvel=v)
files=[ROOT/cfg['policy_source'],ROOT/cfg['model_source'],DATA/'trajectory.npz',DATA/'controller_reference.npz',DATA/'scene.xml',ROOT/'scripts/real_env.py',ROOT/'scripts/world_model_mini.py',RUN/'development_initials.npz',RUN/'heldout_initials.npz']
write_json(RUN/'input_manifest.json',dict(source_commit=cfg['source_commit'],files=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],heldout_generated_before_selection=True,heldout_not_training_data=True,existing_eight_are_seen_development=True))
print('Prepared full iteration',RUN)
