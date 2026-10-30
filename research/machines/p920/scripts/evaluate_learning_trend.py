"""One serial physics graph evaluates the complete historical/current curve."""
import time,hashlib
import torch
from train_exploratory_ppo import evaluate
from train_real_ppo import Policy
from real_env import RealEnv
from exploration_common import RUN,ROOT,read_json,write_json,resource_check,Contacts,finish_timing

started=time.perf_counter();torch.set_num_threads(1);out=RUN/'canonical_eval';out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check())
env=RealEnv(1);contacts=Contacts(env);policy=Policy(118).to(env.device);policy.eval()
old=ROOT/read_json(RUN/'budget.json')['source_run'];jobs=[('original',20480,ROOT/'runs/real_track_20261026/ppo/checkpoint.pt','original')]
for step in [10240,20480,30720,40960]:
    for arm in ['uniform_rsi','start50']:jobs.append((arm,20480+step,old/arm/f'checkpoint_{step:06d}.pt','previous'))
current_steps=set();final={}
for arm in ['uniform_rsi','start50']:
    summary=read_json(RUN/arm/'summary.json');final[arm]=summary['additional_steps'];current_steps.update([int(p.stem.split('_')[1]) for p in (RUN/arm).glob('checkpoint_0*.pt')]);current_steps.add(final[arm])
for step in sorted(current_steps):
    for arm in ['uniform_rsi','start50']:
        p=RUN/arm/f'checkpoint_{step:06d}.pt'
        if step==final[arm]:p=RUN/arm/'checkpoint_final.pt'
        if p.exists():jobs.append((arm,61440+step,p,'current'))
for arm in ['uniform_rsi','start50']:jobs.append((arm+'_repeat',61440+final[arm],RUN/arm/'checkpoint_final.pt','repeat'))
rows=[]
for arm,total,p,stage in jobs:
    if time.perf_counter()-started>275:break
    c=torch.load(p,map_location='cpu',weights_only=True);policy.load_state_dict(c['policy']);base=out/arm;base.mkdir(exist_ok=True)
    result=evaluate(env,policy,base,contacts,total-20480);result.update(arm=arm,total_training_steps=total,stage=stage,checkpoint=str(p.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest());rows.append(result)
    write_json(out/'summary.json',dict(results=rows,jobs_planned=len(jobs),one_common_physics_graph=True,repeat='Each final checkpoint repeated on the same eight states after the full curve; numerical replay, not an independent training seed.',all_requested_evaluations_completed=len(rows)==len(jobs)))
print('CANONICAL EVAL',finish_timing('canonical_evaluation',started,dict(evaluations=len(rows),episodes=8*len(rows))),flush=True)
