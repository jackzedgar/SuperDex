"""Common single-world evaluator for all saved PPO checkpoints, fixed eight states."""
import time,hashlib
import torch
from train_exploratory_ppo import evaluate
from train_real_ppo import Policy
from real_env import RealEnv
from exploration_common import RUN,ROOT,write_json,resource_check,Contacts,finish_timing

started=time.perf_counter();torch.set_num_threads(1);out=RUN/'canonical_eval';out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check())
env=RealEnv(1);contacts=Contacts(env);policy=Policy(118).to(env.device);policy.eval()
original=ROOT/'runs/real_track_20261026/ppo/checkpoint.pt';jobs=[('original',0,original)]
for step in [10240,20480,30720,40960]:
    for arm in ['uniform_rsi','start50']:
        p=RUN/arm/f'checkpoint_{step:06d}.pt'
        if p.exists():jobs.append((arm,step,p))
jobs.append(('original_repeat',0,original));rows=[]
for arm,step,p in jobs:
    if time.perf_counter()-started>170:break
    c=torch.load(p,map_location='cpu',weights_only=True);policy.load_state_dict(c['policy']);base=out/arm;base.mkdir(exist_ok=True)
    result=evaluate(env,policy,base,contacts,step);result.update(arm=arm,checkpoint=str(p.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest());rows.append(result)
write_json(out/'summary.json',dict(results=rows,one_common_physics_graph=True,original_repeat='Identical weights and eight initial states at start/end; quantifies numerical replay variation, not independent seeds.',all_requested_evaluations_completed=len(rows)==len(jobs)))
print('CANONICAL EVAL',finish_timing('canonical_evaluation',started,dict(evaluations=len(rows),episodes=8*len(rows))),flush=True)
