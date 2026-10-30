"""Resume budget-ended chunks under measured shared-GPU load, never change the loss."""
import argparse,math,csv,shutil
from pathlib import Path
import numpy as np
from exploration_common import RUN,ROOT,read_json,write_json
from run_core_iteration import launch,remaining,common,ledger

def chunks(stage,arm,block=None):
    prefix=f'block_{block:02d}' if block is not None else 'block_'
    return [p for p in sorted((RUN/stage/arm).glob(prefix+'*')) if (p/'summary.json').exists()]

def latest(stage,arm):
    cs=chunks(stage,arm)
    if not cs:return None,0
    last=cs[-1];s=read_json(last/'summary.json');c=read_json(last/'config.json')
    return last,c['offset']+s['training_interactions']

def recent_rate(stage):
    rates=[]
    # Recent within-process rate excludes initial evaluation/compilation and
    # reacts to a new external task without interpreting it as learning quality.
    for arm in ['P','M']:
        folder,_=latest(stage,arm)
        if folder is None:continue
        rows=list(csv.DictReader((folder/'training.csv').open()))
        if len(rows)>=3:
            tail=rows[-min(12,len(rows)):];ds=float(tail[-1]['steps'])-float(tail[0]['steps']);dt=float(tail[-1]['process_seconds'])-float(tail[0]['process_seconds'])
            if ds>0:rates.append(dt/ds)
    if not rates and stage!='core':return recent_rate('core')
    return max(rates+[read_json(RUN/'frozen_recipe.json')['development_seconds_per_pair_step']/2])

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['core','confirmation'],required=True);a=p.parse_args();stage=a.stage;cfg=read_json(RUN/'budget.json');recipe=read_json(RUN/'frozen_recipe.json');adapt=read_json(RUN/'resource_adaptation.json')
    if 'temporary_core_supervisor_ceiling_seconds' in cfg:
        cfg['allocation_seconds']=adapt['revised_allocation_seconds'];del cfg['temporary_core_supervisor_ceiling_seconds'];write_json(RUN/'budget.json',cfg)
    recovery=read_json(RUN/'resource_recovery.json') if (RUN/'resource_recovery.json').exists() else {}
    if stage=='core':seed=recipe['seed'];target=recovery.get('restored_core_target_per_arm',adapt['revised_max_common_target_per_arm']);rho=recipe['rho']
    else:
        # The post-core plan already incorporates measured remaining budget.
        # Do not overwrite its frozen common target with an older resource ceiling.
        choice=read_json(RUN/'confirmation_plan.json');seed=choice['seed'];rho=choice['rho'];target=choice['steps_per_arm']
    progress_path=RUN/(stage+'_progress.json');progress=read_json(progress_path)['checkpoints'] if progress_path.exists() else [];done=progress[-1]['steps'] if progress else 0
    category=stage;last_sources={arm:str((latest(stage,arm)[0]/'checkpoint_final.pt').relative_to(ROOT)) for arm in ['P','M'] if latest(stage,arm)[0] is not None};administrative=read_json(RUN/(stage+'_administrative_resets.json')) if (RUN/(stage+'_administrative_resets.json')).exists() else []
    initial_completion=RUN/(stage+'_completion.json')
    if initial_completion.exists() and not (RUN/(stage+'_initial_supervisor_completion.json')).exists():shutil.copy2(initial_completion,RUN/(stage+'_initial_supervisor_completion.json'))
    if initial_completion.exists() and done<target:initial_completion.rename(RUN/(stage+f'_intermediate_completion_{done:06d}.json'))
    write_json(RUN/(stage+'_resource_schedule.json'),dict(target_per_arm=target,seed=seed,rho=rho,source='Frozen recipe; external load changes only time allocation and checkpoint chunking.',allocation_seconds=cfg['allocation_seconds'],extra_budget_resets_disclosed=True))
    for block in range(done//65536,math.ceil(target/65536)):
        endpoint=min((block+1)*65536,target);rate=recent_rate(stage);needed={}
        for arm in ['P','M']:
            _,position=latest(stage,arm);needed[arm]=max(0,endpoint-position)
        estimate=sum(needed.values())*rate*1.05+80*sum(v>0 for v in needed.values())
        if any(needed.values()) and remaining(category)<estimate+100:
            write_json(RUN/(stage+'_budget_stop.json'),dict(matched_steps=done,needed_controls=needed,seconds_per_control_estimate=rate,estimated_pair_seconds=estimate,remaining_seconds=remaining(category)));break
        for arm in (['P','M'] if block%2==0 else ['M','P']):
            folder,position=latest(stage,arm)
            while position<endpoint:
                existing=chunks(stage,arm,block);suffix='' if not existing else '_resume_'+str(len(existing)).zfill(2);base=f'{stage}/{arm}/block_{block:02d}{suffix}';name=base;used={r['name'] for r in ledger()['processes']};attempt=2
                while name in used:name=base+'_attempt_'+str(attempt).zfill(2);attempt+=1
                source=str((folder/'checkpoint_final.pt').relative_to(ROOT)) if folder else cfg['policy_source'];count=endpoint-position
                # Leave a measured reserve for the other member of this pair.
                other='M' if arm=='P' else 'P';_,other_position=latest(stage,other);other_remaining=max(0,endpoint-other_position);rate=recent_rate(stage)
                cap=min(count*rate*1.35+100,remaining(category)-other_remaining*rate*1.05-70)
                if cap<100:raise RuntimeError('Preserve partial evidence: insufficient reserve for a matched pair.')
                command=['scripts/train_core_policy.py','--name',name,'--rho',str(0 if arm=='P' else rho),'--steps',str(count),'--seed',str(seed),'--source',source,'--offset',str(position),'--interval',str(count)]
                if folder:command+=['--resume-rng','--skip-initial-eval']
                launch(name,category,command,cap);new=RUN/name;s=read_json(new/'summary.json');new_position=position+s['training_interactions'];assert new_position>position
                if s['stop_reason'].startswith('external_signal_'):raise RuntimeError('External stop honored and checkpoint saved; review before another restart.')
                if existing:administrative.append(dict(arm=arm,at_steps=position,new_chunk=name,reason='Prior owned child reached wall cap after external GPU load appeared; optimizer and RNG restored, one administrative physical reset. Same reset distribution, no loss/physics change.'))
                folder,position=new,new_position
            last_sources[arm]=str((folder/'checkpoint_final.pt').relative_to(ROOT))
        folders={arm:latest(stage,arm)[0]/f'evaluation/step_{endpoint:06d}' for arm in ['P','M']};summaries={arm:read_json(folders[arm]/'summary.json') for arm in folders};degraded={}
        for arm in folders:
            base=RUN/stage/arm/'block_00/evaluation/step_000000';initial=read_json(base/'summary.json');cp=common(base,folders[arm]);s=summaries[arm]
            degraded[arm]=bool(s['successes']==0 and (s['median_first_failure_s'] or 1.02)<.8*(initial['median_first_failure_s'] or 1.02) and cp['b_position_mm']>1.2*cp['a_position_mm'] and cp['b_rotation_deg']>1.2*cp['a_rotation_deg'] and s['mean_length']<.8*initial['mean_length'])
        done=endpoint;progress.append(dict(steps=done,summaries=summaries,folders={k:str(v.relative_to(ROOT)) for k,v in folders.items()},paired_common=common(folders['P'],folders['M']),degraded_vs_start=degraded));write_json(RUN/(stage+'_progress.json'),dict(completed_steps_per_arm=done,requested_target=target,checkpoints=progress));write_json(RUN/(stage+'_administrative_resets.json'),administrative)
        print('MATCHED',stage,done,'successes',summaries['P']['successes'],summaries['M']['successes'],'remaining',remaining(category),flush=True)
        if done>=262144 and len(progress)>=3 and all(all(r['degraded_vs_start'].values()) for r in progress[-3:]):break
    write_json(RUN/(stage+'_completion.json'),dict(steps_per_arm=done,target=target,original_target=recipe['target_steps_per_arm'] if stage=='core' else 131072,final_checkpoints=last_sources,paired_checkpoints=len(progress),stop='resource_adjusted_target' if done==target else 'budget_or_predeclared_degradation',seed=seed,rho=rho,administrative_resets=administrative))

if __name__=='__main__':main()
