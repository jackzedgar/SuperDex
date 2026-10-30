"""CPU audit of frozen training/evaluation artifacts, not a grasp admission gate."""
import csv,hashlib,collections
import numpy as np,torch,mujoco
from exploration_common import RUN,ROOT,DATA,read_json,write_json
from world_model_mini import Geometry,WorldModel
from build_wm_dataset import classify


def main():
    torch.set_num_threads(1);checks={};details={}
    baseline=ROOT/'runs/real_track_20261026/ppo/checkpoint.pt'
    checks['original_checkpoint_unchanged']=hashlib.sha256(baseline.read_bytes()).hexdigest()==(RUN/'baseline_checkpoint_sha256.txt').read_text().strip()
    initial=np.load(ROOT/'runs/grasp_repair_20261026/fixed_initials.npz');ref=np.load(DATA/'trajectory.npz')['qpos'].astype('float32');target=np.load(DATA/'controller_reference.npz')['ctrl'].astype('float32')
    model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));lo=model.actuator_ctrlrange[:,0];hi=model.actuator_ctrlrange[:,1];scale=np.array([.025]*3+[.18]*3+[.25]*20)
    canonical=read_json(RUN/'canonical_eval/summary.json');n=0;max_reward=max_control=0.;failures=[]
    for row in canonical['results']:
        folder=RUN/'canonical_eval'/row['arm']/f'eval_{row["additional_steps"]:06d}'
        for t in row['trials']:
            a=np.load(folder/f'trial_{t["trial"]:02d}.npz');q=a['qpos'];v=a['qvel'];act=np.clip(a['raw_action'],-1,1);phase=a['phase']+1;k=len(act)
            assert np.array_equal(q[0],initial['qpos'][t['trial']].astype(q.dtype)) and np.array_equal(v[0],initial['qvel'][t['trial']].astype(v.dtype))
            assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(act).all()
            expected=np.clip(target[phase]+act*scale,lo,hi);max_control=max(max_control,float(np.max(abs(expected-a['ctrl']))))
            ep=a['position_error'];er=a['rotation_error'];eq=np.mean((q[1:,6:26]-ref[phase,6:26])**2,axis=1);ew=np.sum((q[1:,:3]-ref[phase,:3])**2,axis=1);last=np.vstack([np.zeros(26),act[:-1]])
            reward=.02*(5*np.exp(-(ep/.04)**2)+2*np.exp(-(er/.5)**2)+.5*np.exp(-eq/.1)+.5*np.exp(-ew/.001)-.03*np.mean(act**2,axis=1)-.05*np.mean((act-last)**2,axis=1))
            max_reward=max(max_reward,float(np.max(abs(reward-a['reward']))));assert abs(a['reward'].sum()-t['return_'])<2e-5
            assert bool(t['success'])==bool(k==51 and t['reason']==0 and (ep<=.03).all() and (er<=np.pi/6).all())
            bad=np.flatnonzero((ep>.03)|(er>np.pi/6));first=float((bad[0]+1)*.02) if len(bad) else None;assert first==t['first_strict_failure_s']
            if len(bad):
                b=bad[0];kind='both' if ep[b]>.03 and er[b]>np.pi/6 else 'position' if ep[b]>.03 else 'orientation'
            else:kind=None
            failures.append(dict(arm=row['arm'],additional_steps=row['additional_steps'],trial=t['trial'],first_strict_failure_s=first,first_strict_failure_type=kind,termination_reason=t['reason'],completed_controls=k))
            n+=1
    assert max_reward<1e-6 and max_control<1e-6
    checks.update(fixed_initial_states_exact=True,unchanged_success_recomputed=True,finite_evaluation_states=True,logged_reward_and_control_recomputed=True)
    details.update(evaluation_episodes=n,max_reward_difference=max_reward,max_control_difference=max_control)
    write_json(RUN/'failure_types.json',failures)
    ppo=[]
    for arm in ['uniform_rsi','start50']:
        rows=list(csv.DictReader((RUN/arm/'training.csv').open()));assert len(rows)==320 and int(rows[-1]['additional_steps'])==40960
        for r in rows:assert all(np.isfinite(float(x)) for x in r.values() if x!='')
        diag={k:dict(mean=float(np.mean([float(r[k]) for r in rows])),max=float(np.max([float(r[k]) for r in rows])),last40_mean=float(np.mean([float(r[k]) for r in rows[-40:]]))) for k in ['approx_kl','ppo_clip_fraction','action_clip_fraction','advantage_std','explained_variance_before_update','contact_fraction','reward_obj_pos','reward_obj_rot','reward_fingers','reward_wrist','reward_action','reward_action_change']}
        resets=[read_json(p) for p in (RUN/arm/'episodes').glob('*.json')];diag['reset_distribution']=dict(episodes=len(resets),phase_zero=sum(r['start_phase']==0 for r in resets),phase_zero_fraction=float(np.mean([r['start_phase']==0 for r in resets])),start_branch_fraction=float(np.mean([r['start_branch'] for r in resets])))
        ppo.append(dict(arm=arm,diagnostics=diag))
    write_json(RUN/'training_diagnostic_summary.json',ppo);checks['both_PPO_arms_320_finite_updates']=True
    with np.load(RUN/'wm/dataset.npz') as f:data={k:f[k] for k in f.files}
    manifest=read_json(RUN/'wm/dataset_manifest.json');eps=manifest['episodes'];groups={s:{e['group'] for e in eps if e['split']==s} for s in ['train','validation']};assert groups['train'].isdisjoint(groups['validation']);assert len({e['fingerprint'] for e in eps})==len(eps)
    assert not any('/canonical_eval/' in e['path'] or '/eval_' in e['path'] for e in eps)
    assert np.isfinite(data['states']).all() and np.isfinite(data['controls']).all()
    for h in [8,32]:
        for split in ['train','validation']:
            ix=data[f'windows_{h}_{split}'];assert len(ix)==sum(max(0,e['steps']-h+1) for e in eps if e['split']==split)
            for s,c,eid,label in ix:
                e=eps[eid];offset=s-e['states_start'];assert e['split']==split and 0<=offset<=e['steps']-h and c==e['controls_start']+offset and classify(data['contact_mode'][c:c+h])==label
    checks.update(whole_episode_groups_disjoint=True,windows_within_episode_bounds=True,no_fixed_PPO_evaluations_in_WM=True,finite_model_dataset=True)
    reload=[]
    for variant in ['W0','W1']:
        ckpt=torch.load(RUN/'wm'/f'{variant}_final.pt',map_location='cpu',weights_only=False)  # Own trusted artifact; normalization contains numpy arrays.
        wm=WorldModel(variant,Geometry(RUN/'wm'),ckpt['normalization']);wm.load_state_dict(ckpt['model']);wm.eval();assert ckpt['update']==590
        ix=np.load(RUN/'wm'/f'prediction_{variant}_08.npz')['windows'][:8];s=torch.tensor(data['states'][ix[:,0]])
        with torch.no_grad():
            for j in range(8):s=wm(s,torch.tensor(data['controls'][ix[:,1]+j]));assert torch.isfinite(s).all()
        assert torch.allclose(torch.linalg.vector_norm(s[:,29:33],dim=-1),torch.ones(8),atol=1e-5)
        saved=np.load(RUN/'wm'/f'prediction_{variant}_08.npz')['predicted_states'][:8,-1];delta=float(np.max(abs(s[:,:65].numpy()-saved)));assert delta<.002
        reload.append(dict(model=variant,updates=590,cpu_vs_saved_GPU_max_state_coordinate_difference=delta))
    checks['world_model_checkpoint_reload_and_8step_prediction']=True;details['model_reload']=reload
    ledger=read_json(RUN/'compute_ledger.json');assert ledger['all_budgets_respected'] and ledger['total_charged_seconds']<=3600;checks['budget_respected']=True
    assert all(checks.values());write_json(RUN/'artifact_validation.json',dict(checks=checks,details=details,scope='CPU artifact audit; no new physics or training run. Original physics/reset checks remain historical evidence. No grasp success prerequisite or formal G1 claim.'))
    print('Frozen artifact audit passed:',len(checks),'checks; reward/control max error:',max_reward,max_control)


if __name__=='__main__':main()
