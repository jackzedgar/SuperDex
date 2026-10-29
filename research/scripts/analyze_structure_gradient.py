"""All predeclared probes retained, with replay-spread inconclusive category."""
import csv,collections,hashlib
import numpy as np,torch
from exploration_common import RUN,ROOT,read_json,write_json
from analyze_learning_trend import wm_metrics
from world_model_mini import qlog,qmul,qconj

def main():
    torch.set_num_threads(1)
    with np.load(RUN/'wm/dataset.npz') as f:data={k:f[k] for k in f.files}
    manifest=read_json(RUN/'wm/dataset_manifest.json');norm=dict(np.load(RUN/'wm/normalization.npz'));metrics=[];loss=[];final=read_json(RUN/'wm/B/summary.json')['matched_comparison_update']
    for name,label in [('A','W1'),('B','B')]:
        for folder in sorted((RUN/'wm'/name/'validation').glob('update_*')):
            update=int(folder.name.split('_')[1])
            for h in [8,32]:
                mm,ll=wm_metrics(name,h,update,folder/f'prediction_{label}_{h:02d}.npz',data,manifest,norm);metrics.extend(mm)
                if ll is not None:loss.append(dict(model=name,update=update,normalized_8step_validation_loss=ll))
    for name in ['persistence','constant_velocity','ballistic']:
        for h in [8,32]:
            mm,ll=wm_metrics(name,h,0,RUN/'wm'/f'prediction_{name}_{h:02d}.npz',data,manifest,norm);metrics.extend(mm)
    write_json(RUN/'structure_analysis.json',dict(metrics=metrics,validation_loss=loss,matched_update=final,data_statistics=manifest['statistics'],windows=manifest['window_counts'],A_reused=True))
    with (RUN/'prediction_curve.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=metrics[0]);w.writeheader();w.writerows(metrics)
    probe=read_json(RUN/'gradient/summary.json');details=[];base_spreads=[]
    for case in probe['plans']:
        rows=[r for r in probe['records'] if r['case']==case['case']];base=[r for r in rows if r['label']=='baseline'];bv=np.array([r['return_'] for r in base]);spread=float(np.ptp(bv));base_spreads.append(spread)
        actual=[dict(np.load(RUN/r['file'])) for r in base];lastq=np.array([a['qpos'][-1] for a in actual]);modes=[(a['contacts'][:,0]>0).astype(int)+2*(a['contacts'][:,1]>0) for a in actual]
        labels=['no_contact' if np.all(m==0) else 'sustained_contact' if m[0]!=0 and np.all(m==m[0]) else 'contact_switch' for m in modes];regime=collections.Counter(labels).most_common(1)[0][0]
        for model in ['A','B']:
            proposal=dict(np.load(RUN/'gradient'/f'case_{case["case"]:03d}'/f'proposal_{model}.npz'));predq=proposal['baseline_predicted_states'][-1,:33];poserr=np.linalg.norm(lastq[:,26:29]-predq[26:29],axis=1).mean()*1000
            with torch.no_grad():rot=torch.linalg.vector_norm(qlog(qmul(qconj(torch.tensor(lastq[:,29:33])),torch.tensor(predq[29:33]).expand(len(lastq),4))),dim=-1).mean().item()*180/np.pi
            for sign in ['plus','minus']:
                rr=[r for r in rows if r['label']==model+'_'+sign];rv=np.array([r['return_'] for r in rr]);actual_delta=float(rv.mean()-bv.mean());noise=max(1e-6,spread,.5*spread+.5*float(np.ptp(rv)));pred_delta=case['proposals'][model]['predicted_'+sign+'_improvement'];status='inconclusive' if abs(actual_delta)<=noise else 'improves' if actual_delta>0 else 'harms';agreement=None if status=='inconclusive' or abs(pred_delta)<1e-9 else bool(np.sign(actual_delta)==np.sign(pred_delta))
                directional=[dict(np.load(RUN/r['file'])) for r in rr];controls=[a['ctrl'] for a in directional];base_ctrl=actual[0]['ctrl'];change=np.array(controls)-base_ctrl
                bc=np.array([a['reward_parts'].sum(0) for a in actual]);dc=np.array([a['reward_parts'].sum(0) for a in directional]);component_change=dc.mean(0)-bc.mean(0);pred_components=proposal[sign+'_reward_components'].sum(0)-proposal['baseline_reward_components'].sum(0)
                tracking_change=float(component_change[:4].sum());tracking_noise=max(1e-6,float(np.ptp(bc[:,:4].sum(1))),.5*float(np.ptp(bc[:,:4].sum(1)))+.5*float(np.ptp(dc[:,:4].sum(1))))
                record=dict(case=case['case'],initial_id=case['initial_id'],prefix_controls=case['prefix_controls'],horizon=case['horizon'],model=model,direction=sign,predicted_change=pred_delta,true_change=actual_delta,noise_threshold=noise,baseline_range=spread,direction_range=float(np.ptp(rv)),baseline_mean=float(bv.mean()),direction_mean=float(rv.mean()),status=status,direction_agreement=agreement,baseline_position_error_mm=float(poserr),baseline_orientation_error_deg=rot,control_delta_max=float(abs(change).max()),wrist_translation_control_delta_max_m=float(abs(change[...,:3]).max()),wrist_rotation_control_delta_max_rad=float(abs(change[...,3:6]).max()),finger_control_delta_max_rad=float(abs(change[...,6:26]).max()),action_delta_max=float(abs(proposal[sign]-proposal['nominal']).max()),baseline_contact_regime=regime,baseline_hand_involved=bool(any((m&1).any() for m in modes)),any_original_failure=any(r['any_original_failure'] for r in rr),max_anchor_position_difference_mm=max(r['anchor_position_difference_mm'] for r in rows),family_held_out=case['initial_id'] in [6,7],predicted_tracking_change=float(pred_components[:4].sum()),true_tracking_change=tracking_change,true_action_penalty_change=float(component_change[4:].sum()),tracking_noise_threshold=tracking_noise,tracking_change_status='inconclusive' if abs(tracking_change)<=tracking_noise else 'improves' if tracking_change>0 else 'harms');details.append(record)
    aggregate=[]
    for model in ['A','B']:
        for horizon in [1,4]:
            for sign in ['plus','minus']:
                rr=[r for r in details if r['model']==model and r['horizon']==horizon and r['direction']==sign];counts=collections.Counter(r['status'] for r in rr);determined=[r for r in rr if r['direction_agreement'] is not None]
                aggregate.append(dict(model=model,horizon=horizon,direction=sign,cases=len(rr),improves=counts['improves'],harms=counts['harms'],inconclusive=counts['inconclusive'],direction_agrees=sum(r['direction_agreement'] for r in determined),direction_judgable=len(determined),mean_predicted_change=float(np.mean([r['predicted_change'] for r in rr])) if rr else None,mean_true_change=float(np.mean([r['true_change'] for r in rr])) if rr else None))
    inversions=[]
    for r in details:
        if r['direction']!='plus':continue
        other=next(x for x in details if x['case']==r['case'] and x['direction']=='plus' and x['model']!=r['model'])
        if r['baseline_position_error_mm']<other['baseline_position_error_mm'] and r['status']=='harms':inversions.append(dict(case=r['case'],model=r['model'],horizon=r['horizon'],position_error_mm=r['baseline_position_error_mm'],other_position_error_mm=other['baseline_position_error_mm'],true_change=r['true_change'],noise=r['noise_threshold'],predicted_change=r['predicted_change'],orientation_error_deg=r['baseline_orientation_error_deg'],other_orientation_error_deg=other['baseline_orientation_error_deg']))
    write_json(RUN/'gradient_analysis.json',dict(aggregate=aggregate,details=details,lower_position_error_but_harmful_proposal=inversions,completed_cases=probe['completed_cases'],predeclared_cases=32,baseline_return_spread=dict(min=min(base_spreads),median=float(np.median(base_spreads)),max=max(base_spreads)),replay_count=len(probe['records']),uncertainty='Empirical full ranges from 5 baseline and 3 directional repeats, not confidence intervals or independent seed replication.',all_cases_retained=True,model_parameters_frozen=probe['model_parameters_frozen'],policy_parameters_frozen=probe['policy_parameters_frozen']))
    with (RUN/'gradient_comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=details[0]);w.writeheader();w.writerows(details)
    ledger=read_json(RUN/'process_ledger.json');cfg=read_json(RUN/'budget.json');charged={k:sum(r['wall_seconds'] for r in ledger['processes'] if r['category']==k) for k in cfg['allocation_seconds']};assert all(charged[k]<=cap for k,cap in cfg['allocation_seconds'].items()) and sum(charged.values())<=3600
    write_json(RUN/'compute_ledger.json',dict(allocation_seconds=cfg['allocation_seconds'],charged_seconds=charged,total_charged_seconds=sum(charged.values()),gpu_hours=sum(charged.values())/3600,processes=ledger['processes'],A_reused=True,A_historical_training_not_charged_again=True,definition='Observed child-process wall time incl. import, CPU/I/O and validation. Shared GPU load is retained, not interpreted as algorithm efficiency. CPU-only artifact work excluded.'))
    print('Matched update:',final,'; probe aggregates:',aggregate,'; lower-error harmful:',len(inversions),'seconds:',sum(charged.values()),flush=True)

if __name__=='__main__':main()
