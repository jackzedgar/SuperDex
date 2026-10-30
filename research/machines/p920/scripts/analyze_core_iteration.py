"""CPU analysis preserving all checkpoints, failures and common execution prefixes."""
import csv,json,collections
from pathlib import Path
import numpy as np
from exploration_common import RUN,ROOT,read_json,write_json

def evaluation(folder):
    original=read_json(folder/'summary.json');trials=[]
    for i,old in enumerate(original['trials']):
        a=np.load(folder/f'trial_{i:02d}.npz');p=a['position_error'];r=a['rotation_error'];n=len(p)
        bad=(p>.03)|(r>np.pi/6);ix=np.flatnonzero(bad);strict=bool(n==51 and not bad.any() and old['reason']==0)
        assert strict==old['success'],(folder,i,'Stored strict result disagrees with raw states')
        first=int(ix[0]) if len(ix) else None
        cause=('position+orientation' if p[first]>.03 and r[first]>np.pi/6 else 'position' if p[first]>.03 else 'orientation') if first is not None else ('success' if strict else 'physical_termination')
        trials.append(dict(**old,first_failure_type=cause,restricted_failure_time_s=(first+1)*.02 if first is not None else n*.02,strict_recomputed=strict,hand_contact_fraction=float(np.mean(a['hand_contact']>0)),floor_contact_fraction=float(np.mean(a['floor_contact']>0)),max_position_mm=float(p.max()*1000),max_rotation_deg=float(np.rad2deg(r.max()))))
    return dict(successes=sum(x['strict_recomputed'] for x in trials),n=len(trials),mean_return=original['mean_return'],mean_length=original['mean_length'],restricted_mean_failure_s=float(np.mean([x['restricted_failure_time_s'] for x in trials])),failure_types=dict(collections.Counter(x['first_failure_type'] for x in trials)),mean_position_mm=float(np.mean([x['position_mm'] for x in trials])),mean_rotation_deg=float(np.mean([x['rotation_deg'] for x in trials])),component_returns=np.mean([x['component_returns'] for x in trials],0).tolist(),trials=trials)

def common(a,b,n):
    rows=[]
    for i in range(n):
        x=np.load(a/f'trial_{i:02d}.npz');y=np.load(b/f'trial_{i:02d}.npz');k=min(len(x['reward']),len(y['reward']))
        rows.append(dict(trial=i,steps=k,P_position_mm=float(x['position_error'][:k].mean()*1000),M_position_mm=float(y['position_error'][:k].mean()*1000),P_rotation_deg=float(np.rad2deg(x['rotation_error'][:k].mean())),M_rotation_deg=float(np.rad2deg(y['rotation_error'][:k].mean())),P_return=float(x['reward'][:k].sum()),M_return=float(y['reward'][:k].sum())))
    return dict(**{('mean_common_steps' if k=='steps' else k):float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='trial'},per_initial=rows)

def load_csv(paths):
    rows=[]
    for p in paths:
        for r in csv.DictReader(p.open()):rows.append({k:float(v) for k,v in r.items()})
    return rows

def main():
    result=dict(evaluations=[],pairs=[],training={},model={},aborted_branches=[],cost={},scope='One wooden-spoon task; fixed seed43 core, optional seed59 confirmation; reused eight development conditions and sixteen held-out conditions. Replays are not independent conditions.')
    for name in ['P','rho10','rho30']:
        folder=RUN/'development'/name
        for step in sorted((folder/'evaluation').glob('step_*')):
            result['evaluations'].append(dict(stage='development',split='seen8',arm=name,steps=int(step.name.split('_')[1]),repeat=0,folder=str(step.relative_to(ROOT)),**evaluation(step)))
    for stage in ['core','confirmation']:
        if not (RUN/stage).exists():continue
        for arm in ['P','M']:
            folders=sorted((RUN/stage/arm).glob('block_*'))
            result['training'][stage+'/'+arm]=load_csv([f/'training.csv' for f in folders if (f/'training.csv').exists()])
            result['model'][stage+'/'+arm]=[dict(row,block=f.name) for f in folders if (f/'model_diagnostics.json').exists() for row in read_json(f/'model_diagnostics.json')]
            for folder in folders:
                for step in sorted((folder/'evaluation').glob('step_*')):
                    result['evaluations'].append(dict(stage=stage,split='seen8',arm=arm,steps=int(step.name.split('_')[1]),repeat=0,folder=str(step.relative_to(ROOT)),**evaluation(step)))
        heldout=RUN/(stage+'_heldout/results.json')
        if heldout.exists():
            for r in read_json(heldout):result['evaluations'].append(dict(stage=stage,split='heldout16',arm=r['arm'],steps=r['steps'],repeat=r['repeat'],folder=r['folder'],**evaluation(ROOT/r['folder'])))
        for split,n in [('seen8',8),('heldout16',16)]:
            es=[r for r in result['evaluations'] if r['stage']==stage and r['split']==split]
            for step,repeat in sorted(set((r['steps'],r['repeat']) for r in es)):
                arms={r['arm']:r for r in es if r['steps']==step and r['repeat']==repeat}
                if 'P' in arms and 'M' in arms:result['pairs'].append(dict(stage=stage,split=split,steps=step,repeat=repeat,**common(ROOT/arms['P']['folder'],ROOT/arms['M']['folder'],n)))
    led=read_json(RUN/'process_ledger.json')['processes'];costs=[]
    for proc in led:
        path=RUN/proc['name']/'summary.json'
        # The first P attempt had no training and is retained in its failed folder.
        if proc['name']=='development/P_attempt2':path=RUN/'development/P/summary.json'
        s=read_json(path) if path.exists() and proc['returncode']==0 else {}
        if proc.get('artifact_path'):
            folder=RUN/proc['artifact_path'];result['aborted_branches'].append(dict(name=proc['name'],reason='SIGTERM; policy updates after the source checkpoint were not retained',training=load_csv([folder/'training.csv']),model=read_json(folder/'model_diagnostics.json') if (folder/'model_diagnostics.json').exists() else []))
        costs.append(dict(name=proc['name'],category=proc['category'],wall_seconds=proc['wall_seconds'],returncode=proc['returncode'],training_interactions=s.get('training_interactions',0),evaluation_interactions=s.get('evaluation_interactions',s.get('physical_controls',0)),auxiliary_model_predictions=s.get('auxiliary_model_predictions',0),diagnostic_model_predictions=s.get('diagnostic_model_predictions',0),ppo_optimizer_updates=s.get('ppo_optimizer_updates',0),aborted_interactions_lower_bound=proc.get('aborted_interactions_lower_bound',0),aborted_interactions_upper_bound=proc.get('aborted_interactions_upper_bound',0)))
        costs[-1].update(aborted_auxiliary_predictions_lower_bound=proc.get('aborted_auxiliary_predictions_bounds',[0,0])[0],aborted_auxiliary_predictions_upper_bound=proc.get('aborted_auxiliary_predictions_bounds',[0,0])[1],aborted_optimizer_updates_lower_bound=proc.get('aborted_ppo_optimizer_updates_bounds',[0,0])[0],aborted_optimizer_updates_upper_bound=proc.get('aborted_ppo_optimizer_updates_bounds',[0,0])[1])
    result['cost']=dict(total_gpu_process_seconds=sum(r['wall_seconds'] for r in costs),cap_seconds=21600,processes=costs,category_seconds={cat:sum(r['wall_seconds'] for r in costs if r['category']==cat) for cat in set(r['category'] for r in costs)})
    write_json(RUN/'iteration_analysis.json',result)
    for stage in ['core','confirmation']:
        p=RUN/(stage+'_administrative_resets.json')
        if p.exists():
            rows=read_json(p)
            for row in rows:
                c=read_json(RUN/row['new_chunk']/'config.json');previous=(ROOT/c['source']).parent/'summary.json'
                if previous.exists():
                    row['previous_stop_reason']=read_json(previous)['stop_reason'];row['reason']='Previous child stopped with '+row['previous_stop_reason']+'; policy/Adam/RNG restored with one additional physical reset. No loss or physics change.'
            write_json(p,rows)
    flat=[]
    for r in result['evaluations']:flat.append({k:v for k,v in r.items() if k not in ['trials','failure_types','component_returns']})
    if flat:
        with (RUN/'evaluation_table.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
    with (RUN/'compute_table.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=costs[0]);w.writeheader();w.writerows(costs)
    print('Analyzed',len(result['evaluations']),'evaluations;',len(result['pairs']),'paired checkpoints; GPU seconds',result['cost']['total_gpu_process_seconds'])

if __name__=='__main__':main()
