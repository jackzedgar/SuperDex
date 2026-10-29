"""Paired physical evidence; raw conditions, no smoothing or seed pooling."""
import json,csv,time
from pathlib import Path
import numpy as np
from bptt_common import RUN,write
def load(p):return json.loads(p.read_text())
def curvepath(arm,setname,step):
    return RUN/arm/(f'development/step_{step:06d}' if setname=='development8' else f'fixed_evaluation/{setname}/step_{step:06d}')
def metrics(folder):
    result=load(folder/'summary.json');trials=[]
    for t in result['trials']:
        a=np.load(folder/f'trial_{t["trial"]:02d}.npz')
        ep=a['position_error'];er=a['rotation_error'];bad=(ep>.03)|(er>np.pi/6);ix=np.flatnonzero(bad)
        strict='none'
        if len(ix):
            k=ix[0];strict='position+orientation' if ep[k]>.03 and er[k]>np.pi/6 else ('position' if ep[k]>.03 else 'orientation')
        effective_first=t['first_failure_s'] if t['first_failure_s'] is not None else (len(ep)*.02 if t['reason'] else None)
        if strict=='none' and t['reason']:strict={3:'joint_limit',4:'nonfinite'}.get(t['reason'],'physical_failure')
        trials.append(dict(**t,effective_first_failure_s=effective_first,strict_failure_type=strict,physical_termination_time_s=len(ep)*.02,hand_contact_fraction=float((a['hand_contact']>0).mean()),support_contact_fraction=float((a['floor_contact']>0).mean()),mean_normal_force_N=float(a['normal_force'].mean()),wrist_translation_saturation=float((abs(a['raw_action'][:,:3])>=1).mean()),wrist_rotation_saturation=float((abs(a['raw_action'][:,3:6])>=1).mean()),finger_saturation=float((abs(a['raw_action'][:,6:])>=1).mean())))
    return result,trials
def main():
    out=RUN/'analysis';out.mkdir(exist_ok=True);rows=[];paired=[];pertrial=[];stages=[]
    for seed in [43,59]:
      for method in ['P','S']:
        arm=f'core_{method}_{seed}'
        if not (RUN/arm/'development_curve.json').exists():continue
        curve=load(RUN/arm/'development_curve.json')
        if (RUN/arm/'summary.json').exists():
            final=load(RUN/arm/'summary.json')
            if final['real_controls'] not in [r['steps'] for r in curve]:curve.append(dict(steps=final['real_controls'],process_seconds=final['seconds']))
        timeat={r['steps']:r['process_seconds'] for r in curve}
        for setname in ['development8','regression16','new32']:
          for step in [r['steps'] for r in curve]:
            folder=curvepath(arm,setname,step)
            if not (folder/'summary.json').exists():continue
            r,trials=metrics(folder)
            row=dict(seed=seed,method=method,set=setname,steps=step,process_seconds=timeat[step],successes=r['successes'],n=r['n'],mean_return=r['mean_return'],mean_length=r['mean_length'],censored_mean_first_failure_s=float(np.mean([t['effective_first_failure_s'] or 1.02 for t in trials])),position_mm=float(np.mean([t['position_mm'] for t in trials])),rotation_deg=float(np.mean([t['rotation_deg'] for t in trials])),hand_contact_fraction=float(np.mean([t['hand_contact_fraction'] for t in trials])),support_contact_fraction=float(np.mean([t['support_contact_fraction'] for t in trials])))
            row.update({key:float(np.mean([t[key] for t in trials])) for key in ['wrist_translation_saturation','wrist_rotation_saturation','finger_saturation']})
            rows.append(row);pertrial.extend([dict(seed=seed,method=method,set=setname,checkpoint_steps=step,**t) for t in trials])
      for setname in ['development8','regression16','new32']:
        p={r['steps']:r for r in rows if r['seed']==seed and r['method']=='P' and r['set']==setname}
        s={r['steps']:r for r in rows if r['seed']==seed and r['method']=='S' and r['set']==setname}
        for step in sorted(p.keys()&s.keys()):
            a=curvepath(f'core_P_{seed}',setname,step);b=curvepath(f'core_S_{seed}',setname,step);x=[]
            for i in range(p[step]['n']):
                pa=np.load(a/f'trial_{i:02d}.npz');sa=np.load(b/f'trial_{i:02d}.npz');n=min(len(pa['position_error']),len(sa['position_error']))
                x.append(dict(trial=i,common_controls=n,P_pos_mm=float(pa['position_error'][:n].mean()*1000),S_pos_mm=float(sa['position_error'][:n].mean()*1000),P_rot_deg=float(np.rad2deg(pa['rotation_error'][:n]).mean()),S_rot_deg=float(np.rad2deg(sa['rotation_error'][:n]).mean()),P_return_common=float(pa['reward'][:n].sum()),S_return_common=float(sa['reward'][:n].sum())))
            paired.append(dict(seed=seed,set=setname,steps=step,common_mean_controls=float(np.mean([z['common_controls'] for z in x])),P_position_mm=float(np.mean([z['P_pos_mm'] for z in x])),S_position_mm=float(np.mean([z['S_pos_mm'] for z in x])),P_rotation_deg=float(np.mean([z['P_rot_deg'] for z in x])),S_rotation_deg=float(np.mean([z['S_rot_deg'] for z in x])),trials=x))
    write(out/'results.json',rows);write(out/'paired_common_interval.json',paired);write(out/'trials.json',pertrial)
    if rows:
        with (out/'results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ledger=load(RUN/'ledger.json');now=time.time();jobs=[dict(**j,observed_seconds=j.get('seconds',now-j['started'])) for j in ledger['jobs']]
    cost=dict(total_gpu_process_seconds=sum(j['observed_seconds'] for j in jobs),cap=86400,jobs=jobs,waiting_CPU_not_GPU=True,shared_historical_pretraining='Policy61440 real controls; initialW1 2000 updates plus prior dataset collection, historical sunk costs outside NEW24h but not free pretraining.')
    write(out/'cost.json',cost)
    budgetrows=[]
    for seed in [43,59]:
        pc=sorted([r for r in rows if r['seed']==seed and r['method']=='P' and r['set']=='new32'],key=lambda r:r['process_seconds'])
        sc=sorted([r for r in rows if r['seed']==seed and r['method']=='S' and r['set']=='new32'],key=lambda r:r['process_seconds'])
        if pc and sc:
            t=min(pc[-1]['process_seconds'],sc[-1]['process_seconds'])
            budgetrows.append(dict(seed=seed,common_time_ceiling_seconds=t,P=next(r for r in pc[::-1] if r['process_seconds']<=t),S=next(r for r in sc[::-1] if r['process_seconds']<=t),interpretation='Last actually evaluated checkpoint at/below common compute ceiling; unequal checkpoint timing disclosed, no interpolation/extrapolation. Includes collection, WM fitting, actor/critic updates and in-run dev evaluation.'))
    write(out/'compute_budget_comparison.json',budgetrows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
      fig,axs=plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
      for ri,seed in enumerate([43,59]):
        for method,color,mark,style in [('P','#0072B2','o','-'),('S','#D55E00','s','--')]:
          rr=sorted([r for r in rows if r['seed']==seed and r['method']==method and r['set']=='new32'],key=lambda r:r['steps'])
          if not rr:rr=sorted([r for r in rows if r['seed']==seed and r['method']==method and r['set']=='development8'],key=lambda r:r['steps']);label=method+' (dev8; new32 pending)'
          else:label=method+' (new32)'
          if not rr:continue
          axs[ri,0].plot([r['steps']/1000 for r in rr],[100*r['successes']/r['n'] for r in rr],color=color,marker=mark,linestyle=style,label=label)
          axs[ri,1].plot([r['process_seconds']/3600 for r in rr],[100*r['successes']/r['n'] for r in rr],color=color,marker=mark,linestyle=style,label=label)
          axs[ri,2].plot([r['steps']/1000 for r in rr],[r['censored_mean_first_failure_s'] for r in rr],color=color,marker=mark,linestyle=style,label=label)
        for ci in [0,1]:axs[ri,ci].set_ylim(-2,102);axs[ri,ci].set_ylabel(f'Seed {seed}: strict success (%)')
        axs[ri,0].set_xlabel('Additional real controls (thousands)');axs[ri,1].set_xlabel('Run GPU-process hours incl. model fitting')
        axs[ri,2].set(xlabel='Additional real controls (thousands)',ylabel=f'Seed {seed}: first strict failure (s)',ylim=(0,1.06))
        for ax in axs[ri]:ax.grid(alpha=.2);ax.legend(fontsize=8)
      fig.suptitle('P versus H32 model-gradient S — full 51-control spoon task\nTwo continuation seeds from common pretrained policy; lines connect measured checkpoints only',fontsize=12)
      fig.savefig(out/'learning_curves.png',dpi=180);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    write(out/'figure_manifest.json',dict(audience='Research iteration report, provisional general format; no journal submission',raw='results.csv, per-condition physical NPZ files',transformations='No smoothing; per-seed raw checkpoint proportions, first failure right-censored at1.02s; no pooled confidence interval for two seeds',missing='No missing point treated as zero; pending new eval explicitly labeled development only',colors='Blue circles solid=P; vermilion squares dashed=S',citation='Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.2026. https://doi.org/10.48550/arXiv.2609.00065 (current arXiv record verified2026-10-27).'))
    print(json.dumps(dict(rows=len(rows),paired=len(paired),gpu_hours=cost['total_gpu_process_seconds']/3600)))
if __name__=='__main__':main()

