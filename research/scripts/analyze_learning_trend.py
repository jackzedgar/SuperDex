"""Frozen evaluation analysis; compare matched prefixes and held-out predictions."""
import csv,json,hashlib,collections
import numpy as np,torch
from exploration_common import RUN,ROOT,DATA,read_json,write_json
from analyze_exploration import weighted_quantile
from world_model_mini import encode


def evalfile(r,i):return RUN/'canonical_eval'/r['arm']/f'eval_{r["total_training_steps"]-20480:06d}'/f'trial_{i:02d}.npz'
def common_prefix(rows):
    trials=[]
    for i in range(8):
        logs=[np.load(evalfile(r,i)) for r in rows];n=min(len(a['reward']) for a in logs);values=[]
        for r,a in zip(rows,logs):values.append(dict(arm=r['arm'],total_training_steps=r['total_training_steps'],position_mean_mm=float(a['position_error'][:n].mean()*1000),rotation_mean_deg=float(np.rad2deg(a['rotation_error'][:n].mean())),return_common=float(a['reward'][:n].sum())))
        trials.append(dict(trial=i,common_controls=n,common_seconds=.02*n,policies=values))
    means=[]
    for j,r in enumerate(rows):means.append(dict(arm=r['arm'],total_training_steps=r['total_training_steps'],**{k:float(np.mean([t['policies'][j][k] for t in trials])) for k in ['position_mean_mm','rotation_mean_deg','return_common']}))
    return dict(mean_common_seconds=float(np.mean([t['common_seconds'] for t in trials])),trials=trials,means=means)


def wm_metrics(name,h,update,file,data,manifest,norm):
    a=np.load(file);ix=a['windows'];err=a['errors'][:,-1];assert np.isfinite(a['errors']).all();classes=ix[:,3];pop=manifest['window_counts'][f'{h}_validation'];weights=np.array([pop[str(c)]/max(1,(classes==c).sum()) for c in classes]);mask=data['contact_mode'];hand=np.array([bool((mask[c:c+h]&1).any()) for c in ix[:,1]])
    metrics=[]
    for label,sel in [('natural',np.ones(len(ix),bool)),('hand',hand),('no_hand',~hand)]+[(key,classes==j) for j,key in enumerate(['no_contact','sustained_contact','contact_switch'])]:
        x=err[sel];ww=weights[sel] if label in ['natural','hand','no_hand'] else np.ones(sel.sum())
        def quantile(v,q):return weighted_quantile(v,ww,q) if label in ['natural','hand','no_hand'] else float(np.quantile(v,q))
        metrics.append(dict(model=name,update=update,horizon=h,subset=label,windows=int(sel.sum()),episodes=len(set(ix[sel,2].tolist())),position_p50_mm=quantile(x[:,0],.5)*1000 if len(x) else None,position_p90_mm=quantile(x[:,0],.9)*1000 if len(x) else None,position_mean_mm=float(np.average(x[:,0],weights=ww)*1000) if len(x) else None,rotation_p50_deg=float(np.rad2deg(quantile(x[:,1],.5))) if len(x) else None,rotation_mean_deg=float(np.rad2deg(np.average(x[:,1],weights=ww))) if len(x) else None,finger_rmse_mean_rad=float(np.average(x[:,2],weights=ww)) if len(x) else None,object_velocity_mean_m_s=float(np.average(x[:,4],weights=ww)) if len(x) else None,object_angular_velocity_mean_rad_s=float(np.average(x[:,5],weights=ww)) if len(x) else None))
    # Match the 8-step normalized, object-weighted fitting objective on validation.
    # Each contact class has equal weight, as in the training batch sampler.
    loss=None
    if h==8:
        pred=a['predicted_states'][:,1:];truth=data['states'][ix[:,0,None]+np.arange(1,h+1),:65];pad=np.zeros((*pred.shape[:2],78),np.float32)
        with torch.no_grad():
            pe=encode(torch.tensor(np.concatenate([pred,pad],-1)))[...,:67];te=encode(torch.tensor(np.concatenate([truth,pad],-1)))[...,:67];e=((pe-te)/torch.tensor(norm['state_std'])).square()*torch.tensor(norm['loss_weight']);v=e.mean(-1).mean(-1).numpy()
        loss=float(np.mean([v[classes==j].mean() for j in range(3) if (classes==j).any()]))
    return metrics,loss


def main():
    torch.set_num_threads(1);old=ROOT/read_json(RUN/'budget.json')['source_run'];canonical=read_json(RUN/'canonical_eval/summary.json');assert canonical['all_requested_evaluations_completed'];rows=canonical['results'];curves=[];detail=[]
    for r in rows:
        ts=[]
        for t in r['trials']:
            a=np.load(evalfile(r,t['trial']));bad=np.flatnonzero((a['position_error']>.03)|(a['rotation_error']>np.pi/6));kind=None
            if len(bad):
                b=bad[0];kind='both' if a['position_error'][b]>.03 and a['rotation_error'][b]>np.pi/6 else 'position' if a['position_error'][b]>.03 else 'orientation'
            ts.append(t['first_strict_failure_s']);detail.append(dict(arm=r['arm'],total_training_steps=r['total_training_steps'],trial=t['trial'],steps=t['steps'],success=t['success'],first_strict_failure_s=t['first_strict_failure_s'],first_strict_failure_type=kind,termination_reason=t['reason'],return_=t['return_']))
        observed=[v for v in ts if v is not None];curves.append(dict(arm=r['arm'],total_training_steps=r['total_training_steps'],stage=r['stage'],successes=r['successes'],mean_return=r['mean_return'],mean_length=r['mean_length'],first_failure_median_s=float(np.median(observed)) if observed else None,failed_trials=len(observed),right_censored_successes=sum(v is None for v in ts),first_failure_type_counts=dict(collections.Counter(d['first_strict_failure_type'] or 'none' for d in detail[-8:]))))
    arms=['uniform_rsi','start50'];matched=[]
    for total in sorted({r['total_training_steps'] for r in rows if r['arm'] in arms}):
        pair=[r for r in rows if r['arm'] in arms and r['total_training_steps']==total]
        if len(pair)==2:matched.append(dict(total_training_steps=total,comparison=common_prefix(pair)))
    final=[max([r for r in rows if r['arm']==a],key=lambda r:r['total_training_steps']) for a in arms];start=[next(r for r in rows if r['arm']==a and r['total_training_steps']==61440) for a in arms]
    repetitions=[common_prefix([r,next(x for x in rows if x['arm']==r['arm']+'_repeat')]) for r in final]
    write_json(RUN/'ppo_analysis.json',dict(curve=curves,matched_checkpoints=matched,final_common=common_prefix(final),start_and_final_common=common_prefix(start+final),repetitions=repetitions,first_failure_note='Median over trials that cross strict thresholds; successes are right-censored at 1.02 s and counted separately. No success assigned an invented failure time.',shared_prefix_note='Within each initial condition use the minimum executed length across compared policies, then equally average eight conditions.'))
    with (RUN/'fixed_evaluation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=detail[0]);w.writeheader();w.writerows(detail)
    with np.load(RUN/'wm/dataset.npz') as f:data={k:f[k] for k in f.files}
    manifest=read_json(RUN/'wm/dataset_manifest.json');norm=dict(np.load(RUN/'wm/normalization.npz'));wm=[];val_loss=[]
    jobs=[]
    for folder in sorted((RUN/'wm/validation').glob('update_*')):
        update=int(folder.name.split('_')[1])
        for name in ['W0','W1']:
            for h in [8,32]:
                file=folder/f'prediction_{name}_{h:02d}.npz'
                if file.exists():jobs.append((name,h,update,file))
    cfg=read_json(RUN/'wm/config.json');last=cfg['paired_updates']
    for name in ['W0','W1','persistence','constant_velocity','ballistic']:
        for h in [8,32]:
            if name in ['W0','W1'] and any(n==name and hh==h and u==last for n,hh,u,p in jobs):continue
            jobs.append((name,h,last if name in ['W0','W1'] else 0,RUN/'wm'/f'prediction_{name}_{h:02d}.npz'))
    for name,h,u,file in jobs:
        metrics,loss=wm_metrics(name,h,u,file,data,manifest,norm);wm.extend(metrics)
        if loss is not None:val_loss.append(dict(model=name,update=u,normalized_8step_validation_loss=loss))
    same590=[]
    for name in ['W0','W1']:
        p=RUN/'wm'/f'{name}_0590.pt'
        if not p.exists():continue
        new=torch.load(p,map_location='cpu',weights_only=False);previous=torch.load(old/'wm'/f'{name}_final.pt',map_location='cpu',weights_only=False)
        diff=max(float((new['model'][k]-v).abs().max()) for k,v in previous['model'].items());same590.append(dict(model=name,max_state_dict_difference_at_same_590_updates=diff))
    write_json(RUN/'wm/trend_analysis.json',dict(metrics=wm,validation_loss=val_loss,previous_590_reproduction=same590,updates=last,data_statistics=manifest['statistics'],windows=manifest['window_counts'],frozen_data_no_new_PPO=True,meaning='Held-out endpoint error with known executed future controls. Natural/hand subsets have stratum population weights. Windows overlap within episodes; not independent replication.'))
    with (RUN/'wm_prediction_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=wm[0]);w.writeheader();w.writerows(wm)
    ledger=read_json(RUN/'process_ledger.json');caps=read_json(RUN/'budget.json')['allocation_seconds'];charged={k:sum(p['wall_seconds'] for p in ledger['processes'] if p['category']==k) for k in caps};assert all(charged[k]<=caps[k] for k in caps) and sum(charged.values())<=3600
    write_json(RUN/'compute_ledger.json',dict(cap_seconds=3600,allocation_seconds=caps,charged_seconds=charged,total_charged_seconds=sum(charged.values()),gpu_hours=sum(charged.values())/3600,processes=ledger['processes'],world_model_component_training_seconds=read_json(RUN/'world_models_timing.json')['component_training_seconds'],all_budgets_respected=True,definition=ledger['definition']))
    print('PPO final:',json.dumps(final));print('WM updates:',last,'; source-590 comparison:',same590);print('Charged seconds:',sum(charged.values()))


if __name__=='__main__':main()
