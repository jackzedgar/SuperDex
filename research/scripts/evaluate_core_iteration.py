"""Fixed post-training evaluation; held-out states never select configurations."""
import argparse,time
import numpy as np,torch
from exploration_common import RUN,ROOT,read_json,write_json,resource_check,Contacts
from real_env import RealEnv
from train_real_ppo import Policy
from train_core_policy import evaluate_policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);p.add_argument('--stage',choices=['core','confirmation'],required=True);args=p.parse_args()
    started=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    completion=read_json(RUN/(args.stage+'_completion.json'));recipe=read_json(RUN/'frozen_recipe.json')
    out=RUN/(args.stage+'_heldout');out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check(1600))
    # The manifest is written before loading any held-out state or executing a policy.
    jobs=[]
    if args.stage=='core':jobs.append(dict(arm='shared_start',steps=0,checkpoint=recipe['source_policy'],repeat=0))
    progress=read_json(RUN/(args.stage+'_progress.json'))['checkpoints']
    for k in range(completion['paired_checkpoints']):
        for arm in (['P','M'] if k%2==0 else ['M','P']):
            if 'folders' in progress[k]:folder=(ROOT/progress[k]['folders'][arm]).parent.parent;steps=progress[k]['steps']
            else:
                folder=RUN/args.stage/arm/f'block_{k:02d}';c=read_json(folder/'config.json');steps=c['offset']+c['steps']
            jobs.append(dict(arm=arm,steps=steps,checkpoint=str((folder/'checkpoint_final.pt').relative_to(ROOT)),repeat=0))
    # Two additional endpoint replays of every initial condition quantify solver variability.
    # They are repeated observations of the same conditions, not extra independent samples.
    for repeat in [1,2]:
        for arm in (['M','P'] if repeat==1 else ['P','M']):
            jobs.append(dict(arm=arm,steps=completion['steps_per_arm'],checkpoint=completion['final_checkpoints'][arm],repeat=repeat))
    write_json(out/'manifest.json',dict(jobs=jobs,initials='heldout_initials.npz',selection='All completed paired checkpoints; endpoint repeat count fixed at three. No performance-based selection.',used_for_training_or_tuning=False))
    initials=dict(np.load(RUN/'heldout_initials.npz'));env=RealEnv(1);contacts=Contacts(env);policy=Policy(118).to('cuda');results=[]
    for job in jobs:
        if time.perf_counter()-started+30>args.seconds:break
        checkpoint=torch.load(ROOT/job['checkpoint'],map_location='cpu',weights_only=True);policy.load_state_dict(checkpoint['policy']);policy.eval()
        dest=out/f"{job['arm']}_step_{job['steps']:06d}_repeat_{job['repeat']}"
        result=evaluate_policy(env,policy,dest,initials,contacts);results.append(dict(**job,folder=str(dest.relative_to(ROOT)),**result));write_json(out/'results.json',results)
        print(args.stage,job['arm'],job['steps'],'repeat',job['repeat'],'strict',result['successes'],'/16',flush=True)
    write_json(out/'summary.json',dict(completed_jobs=len(results),planned_jobs=len(jobs),complete=len(results)==len(jobs),physical_controls=sum(r['physical_controls'] for r in results),process_seconds=time.perf_counter()-started,heldout_used_for_training=False))

if __name__=='__main__':main()
