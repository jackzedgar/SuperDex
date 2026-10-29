"""Bounded single-world physical collection, including post-task-failure dynamics.

Task success/termination definitions are unchanged. Collection episodes have a
fixed 64-control duration, retaining task failure flags without resetting early.
"""
import time,json
import numpy as np,torch
from real_env import RealEnv
from train_real_ppo import Policy
from exploration_common import RUN,write_json,read_json,resource_check,Contacts,EpisodeLog,finish_timing

started=time.perf_counter();torch.set_num_threads(1);out=RUN/'collection';out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check())
env=RealEnv(1);contacts=Contacts(env);policies={}
for arm in ['uniform_rsi','start50']:
    c=torch.load(RUN/arm/'checkpoint_final.pt',map_location='cpu',weights_only=True);p=Policy(118).to(env.device);p.load_state_dict(c['policy']);p.eval();policies[arm]=p
torch.manual_seed(403);rows=[]
for i in range(64):
    if time.perf_counter()-started>285:break
    mode=['reference','uniform_rsi','start50'][i%3];env.reset(torch.tensor([0],device=env.device),False,noise=True)
    log=EpisodeLog(out/'episodes'/f'episode_{i:04d}.npz',env,dict(kind='collection',mode=mode,episode_id=i,continued_after_task_failure=True,seed=403));first_failure=None;first_strict=None
    for t in range(64):
        with torch.no_grad():
            a=torch.zeros(1,26,device=env.device) if mode=='reference' else policies[mode].actor(env.observe())+torch.randn(1,26,device=env.device)*.05
            phase=int(env.phase[0]);_,r,done,info=env.step(a);assert bool(info['finite'].all());c=contacts.read();log.add(env,a,phase,r,info,c)
        if first_failure is None and info['failure'][0]:first_failure=t+1
        if first_strict is None and (float(info['pos_error'][0])>.03 or float(info['rot_error'][0])>np.pi/6):first_strict=t+1
    log.meta.update(first_task_failure_step=first_failure,first_strict_failure_step=first_strict);log.save('fixed_64_step_collection_horizon',True)
    rows.append(dict(episode=i,mode=mode,first_task_failure_step=first_failure,first_strict_failure_step=first_strict))
write_json(out/'summary.json',dict(episodes=len(rows),control_steps=64*len(rows),horizon=64,extra_action_noise_std=.05,reset_noise='original 1 mm object / 0.002 rad fingers',task_success_claim=False,rows=rows))
print('COLLECTION',finish_timing('collection',started,dict(episodes=len(rows),steps=64*len(rows))),flush=True)
