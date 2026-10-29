"""Budgeted child-process orchestration; paired core blocks and dev-only rules."""
import argparse,time,subprocess,sys,os,signal,math,json
import numpy as np
from exploration_common import RUN,ROOT,read_json,write_json

def ledger():
    p=RUN/'process_ledger.json'
    return read_json(p) if p.exists() else dict(processes=[])
def remaining(category):
    cfg=read_json(RUN/'budget.json');rows=ledger()['processes']
    return min(cfg['allocation_seconds'][category]-sum(r['wall_seconds'] for r in rows if r['category']==category),21600-sum(r['wall_seconds'] for r in rows))
def launch(name,category,command,cap):
    l=ledger();assert not any(r['name']==name for r in l['processes']),'No silent rerun'
    limit=min(cap,remaining(category)-2);assert limit>50
    full=[sys.executable,'-u',*command,'--seconds',str(limit-20)];start=time.perf_counter();timeout=False;signal_trace=None
    (RUN/'process_logs').mkdir(exist_ok=True)
    if os.environ.pop('SUPERDEX_TRACE_NEXT_GPU_CHILD',None)=='1':
        signal_trace=str(RUN/'process_logs'/(name.replace('/','_')+'_signals.txt'))
        full=['strace','-qq','-f','-e','trace=none','-e','signal=SIGTERM,SIGINT,SIGHUP','-o',signal_trace,*full]
    with (RUN/'process_logs'/(name.replace('/','_')+'.log')).open('x') as f:
        proc=subprocess.Popen(full,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=1)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    row=dict(name=name,category=category,command=full,wall_seconds=time.perf_counter()-start,hard_cap_seconds=limit,returncode=code,hard_timeout=timeout,pid=proc.pid,signal_trace=signal_trace);l['processes'].append(row);write_json(RUN/'process_ledger.json',l);print(json.dumps(row),flush=True)
    if code!=0:raise RuntimeError('Corresponding experiment paused after implementation/resource failure; failed attempt retained.')
    return row

def common(folder_a,folder_b,n=8):
    rows=[]
    for i in range(n):
        a=np.load(folder_a/f'trial_{i:02d}.npz');b=np.load(folder_b/f'trial_{i:02d}.npz');k=min(len(a['reward']),len(b['reward']))
        rows.append([k,float(a['position_error'][:k].mean()*1000),float(b['position_error'][:k].mean()*1000),float(np.rad2deg(a['rotation_error'][:k].mean())),float(np.rad2deg(b['rotation_error'][:k].mean())),float(a['reward'][:k].sum()),float(b['reward'][:k].sum())])
    return dict(mean_common_steps=float(np.mean(rows,0)[0]),a_position_mm=float(np.mean(rows,0)[1]),b_position_mm=float(np.mean(rows,0)[2]),a_rotation_deg=float(np.mean(rows,0)[3]),b_rotation_deg=float(np.mean(rows,0)[4]),a_return_common=float(np.mean(rows,0)[5]),b_return_common=float(np.mean(rows,0)[6]),per_initial=rows)

def freeze():
    assert not (RUN/'frozen_recipe.json').exists()
    cfg=read_json(RUN/'budget.json');rows=[]
    for label,rho in [('rho10',.1),('rho30',.3)]:
        folder=RUN/'development'/label;summary=read_json(folder/'summary.json');curve=read_json(folder/'development_curve.json');final=curve[-1];pair=common(folder/'evaluation/step_000000',folder/f'evaluation/step_{final["steps"]:06d}')
        score=pair['b_position_mm']/30+pair['b_rotation_deg']/30
        rows.append(dict(candidate=label,rho=rho,successes=final['successes'],first_failure=final['median_first_failure_s'] or 1.02,common_score=score,common=pair,summary=summary))
    ranked=sorted(rows,key=lambda r:(-r['successes'],-r['first_failure'],r['common_score']))
    chosen=ranked[0]
    if rows[0]['successes']==rows[1]['successes'] and abs(rows[0]['first_failure']-rows[1]['first_failure'])<=.02 and abs(rows[0]['common_score']-rows[1]['common_score'])<=.05*min(r['common_score'] for r in rows):chosen=rows[0]
    pure=read_json(RUN/'development/P/summary.json');seconds_per_pair_step=pure['process_seconds']/pure['training_interactions']+chosen['summary']['process_seconds']/chosen['summary']['training_interactions'];target=min(524288,int(.85*cfg['allocation_seconds']['core']/seconds_per_pair_step)//65536*65536);assert target>=65536
    value=dict(source_policy=cfg['policy_source'],source_model=cfg['model_source'],rho=chosen['rho'],selected_candidate=chosen['candidate'],development_candidates=rows,shared_ppo=cfg['ppo'],seed=cfg['core']['seed'],target_steps_per_arm=target,interval=65536,development_seconds_per_pair_step=seconds_per_pair_step,expected_core_seconds=target*seconds_per_pair_step,heldout_accessed_for_selection=False,frozen_before_core=True,stop_rule=cfg['stopping'],model_refresh_during_core=False,matching='Alternate 65,536-step blocks, fresh physical reset at each block boundary; resume policy, optimizer and all RNG streams.')
    write_json(RUN/'frozen_recipe.json',value);print('FROZEN RECIPE',json.dumps(value),flush=True)

def paired(stage,seed,target,rho,category):
    cfg=read_json(RUN/'budget.json');decisions=[];last_sources={'P':cfg['policy_source'],'M':cfg['policy_source']};done=0;duration={'P':None,'M':None}
    for block in range(math.ceil(target/65536)):
        count=min(65536,target-done);estimate=read_json(RUN/'frozen_recipe.json')['development_seconds_per_pair_step']*count
        if remaining(category)<estimate*1.1+120:break
        # Alternate order to avoid systematically allocating a busier GPU period to one arm.
        for arm in (['P','M'] if block%2==0 else ['M','P']):
            name=f'{stage}/{arm}/block_{block:02d}';command=['scripts/train_core_policy.py','--name',name,'--rho',str(0 if arm=='P' else rho),'--steps',str(count),'--seed',str(seed),'--source',last_sources[arm],'--offset',str(done),'--interval',str(count)]
            if block:command+=['--resume-rng','--skip-initial-eval']
            # Both arms share a conservative cap; retain enough budget for the pair.
            cap=min(max(estimate*.8+90,180),remaining(category)-60)
            job=launch(name,category,command,cap);summary=read_json(RUN/name/'summary.json')
            assert summary['training_interactions']==count,'Partial block: retain results, do not call unequal budgets matched.'
            last_sources[arm]=str((RUN/name/'checkpoint_final.pt').relative_to(ROOT));duration[arm]=job['wall_seconds']
        done+=count
        folders={a:RUN/stage/a/f'block_{block:02d}'/f'evaluation/step_{done:06d}' for a in ['P','M']}
        summaries={a:read_json(folders[a]/'summary.json') for a in folders};degraded={}
        for a in folders:
            base=RUN/stage/a/'block_00/evaluation/step_000000';initial=read_json(base/'summary.json');cp=common(base,folders[a]);degraded[a]=bool(summaries[a]['successes']==0 and (summaries[a]['median_first_failure_s'] or 1.02)<.8*(initial['median_first_failure_s'] or 1.02) and cp['b_position_mm']>1.2*cp['a_position_mm'] and cp['b_rotation_deg']>1.2*cp['a_rotation_deg'] and summaries[a]['mean_length']<.8*initial['mean_length'])
        record=dict(steps=done,summaries=summaries,paired_common=common(folders['P'],folders['M']),degraded_vs_start=degraded);decisions.append(record);write_json(RUN/(stage+'_progress.json'),dict(completed_steps_per_arm=done,requested_target=target,checkpoints=decisions))
        print('MATCHED',stage,done,'P/M successes',summaries['P']['successes'],summaries['M']['successes'],'remaining seconds',remaining(category),flush=True)
        if done>=262144 and len(decisions)>=3 and all(all(x['degraded_vs_start'].values()) for x in decisions[-3:]):
            write_json(RUN/(stage+'_stop.json'),dict(reason='Predeclared sustained degradation of BOTH arms',steps=done));break
    write_json(RUN/(stage+'_completion.json'),dict(steps_per_arm=done,target=target,final_checkpoints=last_sources,paired_checkpoints=len(decisions),stop='target' if done==target else 'predeclared_degradation_or_budget',seed=seed,rho=rho))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['development','freeze','core','confirmation']);a=p.parse_args();cfg=read_json(RUN/'budget.json')
    if a.phase=='development':
        for name,rho in [('P',0),('rho10',.1),('rho30',.3)]:
            if (RUN/'development'/name/'summary.json').exists():continue
            job='development/'+name;attempt=sum(r['name']==job or r['name'].startswith(job+'_attempt') for r in ledger()['processes'])
            if attempt:job+='_attempt'+str(attempt+1)
            launch(job,'development',['scripts/train_core_policy.py','--name','development/'+name,'--rho',str(rho),'--steps','16384','--seed','41','--interval','16384'],min(1100,remaining('development')-5))
    elif a.phase=='freeze':freeze()
    elif a.phase=='core':
        recipe=read_json(RUN/'frozen_recipe.json');paired('core',recipe['seed'],recipe['target_steps_per_arm'],recipe['rho'],'core')
    else:
        choice=read_json(RUN/'confirmation_plan.json');assert choice['kind']=='second_seed_replication';paired('confirmation',choice['seed'],choice['steps_per_arm'],choice['rho'],'confirmation')

if __name__=='__main__':main()
