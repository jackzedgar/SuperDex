"""Paired PPO prefixes, prediction strata and a conservative compute ledger."""
import json,csv,hashlib,collections
import numpy as np
from exploration_common import RUN,ROOT,read_json,write_json


def eval_path(row):return RUN/'canonical_eval'/row['arm']/f'eval_{row["additional_steps"]:06d}'
def paired(rows):
    result=[]
    for trial in range(8):
        records=[np.load(eval_path(r)/f'trial_{trial:02d}.npz') for r in rows];n=min(len(a['reward']) for a in records);d=dict(trial=trial,common_steps=n,common_seconds=.02*n)
        for row,a in zip(rows,records):
            name=row['arm'];d[name]=dict(mean_position_mm=float(a['position_error'][:n].mean()*1000),mean_rotation_deg=float(np.rad2deg(a['rotation_error'][:n].mean())),return_common=float(a['reward'][:n].sum()),total_steps=len(a['reward']))
        result.append(d)
    return dict(trials=result,means={r['arm']:{k:float(np.mean([p[r['arm']][k] for p in result])) for k in ['mean_position_mm','mean_rotation_deg','return_common']} for r in rows},mean_common_seconds=float(np.mean([p['common_seconds'] for p in result])),definition='Each initial condition uses the same actually executed interval for ALL listed policies, then eight initial conditions receive equal weight.')


def weighted_quantile(x,w,q):
    order=np.argsort(x);x=x[order];w=w[order];c=(np.cumsum(w)-.5*w)/w.sum();return float(np.interp(q,c,x))


def main():
    canonical=read_json(RUN/'canonical_eval/summary.json');assert canonical['all_requested_evaluations_completed'];rows=canonical['results'];original=next(x for x in rows if x['arm']=='original');final={arm:next(x for x in rows if x['arm']==arm and x['additional_steps']==40960) for arm in ['uniform_rsi','start50']}
    trajectory=[]
    for row in rows:
        first=[t['first_strict_failure_s'] for t in row['trials'] if t['first_strict_failure_s'] is not None]
        r=dict(arm=row['arm'],additional_steps=row['additional_steps'],total_steps=20480+row['additional_steps'],successes=row['successes'],n=8,mean_return=row['mean_return'],mean_length=row['mean_length'],first_failure_median_s=float(np.median(first)) if first else None,first_failure_min_s=min(first) if first else None,first_failure_max_s=max(first) if first else None,failure_counts=dict(collections.Counter(str(t['reason']) for t in row['trials'])))
        if row['arm'] in final:
            common=paired([original,row]);r['versus_original_common_prefix']=common
        trajectory.append(r)
    comparisons=dict(final_arms_full_common=paired(list(final.values())),original_and_both_final=paired([original,*final.values()]),original_repeat=paired([original,next(x for x in rows if x['arm']=='original_repeat')]))
    write_json(RUN/'ppo_comparison.json',dict(learning_curve=trajectory,comparisons=comparisons,one_training_seed=17,matched_additional_steps=40960,unchanged_success_rule='All 51 controls <=30mm and <=30deg, original failure rules.',training_diagnostics=['reward six components','GAE advantage mean/std before normalization','approximate KL','PPO clip fraction','raw-action clip fraction','explained variance','preclip gradient norm']))
    with (RUN/'fixed_evaluation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm','additional_steps','trial','steps','return_','success','first_strict_failure_s','termination_s','reason']);w.writeheader()
        for row in rows:
            for t in row['trials']:w.writerow({k:row[k] if k in ['arm','additional_steps'] else t[k] for k in w.fieldnames})
    data=np.load(RUN/'wm/dataset.npz');state=data['states'];mask=data['contact_mode'];manifest=read_json(RUN/'wm/dataset_manifest.json');evaluations=read_json(RUN/'wm/evaluation.json');wm=[];distribution=[]
    for split in ['train','validation']:
        ids=np.concatenate([np.arange(e['controls_start'],e['controls_start']+e['steps']) for e in manifest['episodes'] if e['split']==split]);frames=np.concatenate([np.arange(e['states_start'],e['states_start']+e['steps']) for e in manifest['episodes'] if e['split']==split]);v=(state[frames+1,59:62]-state[frames,59:62])/.02;ballistic=np.linalg.norm(v-[0,0,-9.81],axis=1)<1.;none=mask[ids]==0
        distribution.append(dict(split=split,transitions=len(ids),fractions={str(i):float((mask[ids]==i).mean()) for i in range(4)},freefall_like_fraction_all=float((ballistic&none).mean()),freefall_like_fraction_of_no_contact=float(ballistic[none].mean()) if none.any() else None,definition='Freefall-like: sampled contact mode zero and 50Hz object linear acceleration within 1m/s^2 of gravity; contact labels sample the last physics substep.'))
    for name in evaluations:
        for h in [8,32]:
            a=np.load(RUN/'wm'/f'prediction_{name}_{h:02d}.npz');ix=a['windows'];err=a['errors'][:,-1];assert np.isfinite(err).all();classes=ix[:,3];pop=manifest['window_counts'][f'{h}_validation'];weights=np.array([pop[str(c)]/max(1,(classes==c).sum()) for c in classes]);hand=np.array([bool((mask[c:c+h]&1).any()) for c in ix[:,1]])
            for label,select in [('natural_validation_weighted',np.ones(len(ix),dtype=bool)),('hand_contact_in_window',hand),('no_hand_contact_in_window',~hand)]:
                x=err[select];w=weights[select]
                if not len(x):continue
                wm.append(dict(model=name,horizon=h,subset=label,sampled_windows=len(x),episodes=len(set(ix[select,2].tolist())),position_median_mm=weighted_quantile(x[:,0],w,.5)*1000,position_p90_mm=weighted_quantile(x[:,0],w,.9)*1000,position_mean_mm=float(np.average(x[:,0],weights=w)*1000),rotation_median_deg=float(np.rad2deg(weighted_quantile(x[:,1],w,.5))),rotation_p90_deg=float(np.rad2deg(weighted_quantile(x[:,1],w,.9))),definition='Stratum population/sample-count weights recover the natural validation-window mixture; windows overlap only within held-out episodes.'))
    write_json(RUN/'wm/prediction_analysis.json',dict(weighted_metrics=wm,data_distribution=distribution,regime_results=evaluations,missing='No 32-step no-contact validation windows; report N/A. Only 10 no-contact 8-step windows from existing logs; no broad no-contact generalization claim.',no_window_leakage=manifest['group_sets_disjoint']))
    with (RUN/'wm/prediction_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(wm[0]));w.writeheader();w.writerows(wm)
    timings={name:read_json(RUN/(name+'_timing.json')) for name in ['ppo_uniform_rsi','ppo_start50','collection','world_models','canonical_evaluation']}
    failed=read_json(RUN/'wm_preload_repair.json')['elapsed_process_seconds_charged'] if (RUN/'wm_preload_repair.json').exists() else 0
    startup=read_json(RUN/'startup_repair.json')['gpu_time_reserved_seconds'] if (RUN/'startup_repair.json').exists() else 0
    charged=dict(ppo=sum(timings['ppo_'+a]['gpu_process_seconds'] for a in final),additional_collection=timings['collection']['gpu_process_seconds'],world_models=failed+timings['world_models']['gpu_process_seconds'],evaluation_and_verification=startup+timings['canonical_evaluation']['gpu_process_seconds']);total=sum(charged.values());limits=read_json(ROOT/'configs/exploration_budget.json')['allocation_seconds'];assert all(charged[k]<=limits[k] for k in charged) and total<=3600
    baseline=ROOT/'runs/real_track_20261026/ppo/checkpoint.pt';assert hashlib.sha256(baseline.read_bytes()).hexdigest()==(RUN/'baseline_checkpoint_sha256.txt').read_text().strip()
    write_json(RUN/'compute_ledger.json',dict(budget_seconds=3600,allocation_seconds=limits,charged_seconds=charged,total_charged_seconds=total,gpu_hours=total/3600,remaining_seconds=3600-total,details=timings,failed_preprocessing_seconds=failed,startup_error_conservative_allowance_seconds=startup,cpu_sdf_build_seconds=read_json(RUN/'wm/geometry_manifest.json')['cpu_seconds'],definition='GPU-process wall time, including CPU I/O/optimization inside those processes; conservative 30s allowance for startup error. Not GPU kernel-only time, FLOPs, energy or dedicated-device throughput. CPU-only SDF construction is reported separately.',baseline_checkpoint_unchanged=True,all_budgets_respected=True))
    print('PPO paired',json.dumps(comparisons['final_arms_full_common']['means']));print('Compute seconds',total);print('WM hand-involving',json.dumps([x for x in wm if x['model'] in ['W0','W1'] and x['subset']=='hand_contact_in_window']))


if __name__=='__main__':main()
