"""Targeted CPU audit of continuation, frozen inputs and recorded evaluations."""
import csv,hashlib
import numpy as np,torch,mujoco
from exploration_common import RUN,ROOT,DATA,read_json,write_json
from analyze_learning_trend import evalfile
from world_model_mini import Geometry,WorldModel


def main():
    torch.set_num_threads(1);manifest=read_json(RUN/'input_manifest.json');checks={};details={}
    for r in manifest['source_files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    for r in manifest['frozen_wm_files']:
        assert hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest()==r['sha256']
        assert hashlib.sha256((RUN/'wm'/r['name']).read_bytes()).hexdigest()==r['sha256']
    checks.update(source_checkpoints_reference_and_physics_unchanged=True,world_model_inputs_frozen_byte_for_byte=True,no_new_PPO_in_model_data=True)
    training=[]
    for arm in ['uniform_rsi','start50']:
        config=read_json(RUN/arm/'config.json');summary=read_json(RUN/arm/'summary.json');rows=list(csv.DictReader((RUN/arm/'training.csv').open()));steps=summary['additional_steps'];assert steps<=81920 and steps%128==0 and len(rows)==steps//128
        assert config['cuda_rng_restored'] and config['source_training_steps']==61440 and config['start_probability']==(0 if arm=='uniform_rsi' else .5)
        source=torch.load(config['source_checkpoint'],map_location='cpu',weights_only=True);final=torch.load(RUN/arm/'checkpoint_final.pt',map_location='cpu',weights_only=True)
        before={float(s['step']) for s in source['optimizer']['state'].values()};after={float(s['step']) for s in final['optimizer']['state'].values()};assert before=={config['optimizer_step_before']} and after=={config['optimizer_step_before']+4*len(rows)}
        assert source['optimizer']['param_groups']==final['optimizer']['param_groups']
        for r in rows:assert all(np.isfinite(float(v)) for v in r.values() if v!='')
        summary_diag={k:dict(first40_mean=float(np.mean([float(r[k]) for r in rows[:40]])),last40_mean=float(np.mean([float(r[k]) for r in rows[-40:]])),maximum=float(np.max([float(r[k]) for r in rows]))) for k in ['approx_kl','ppo_clip_fraction','action_clip_fraction','advantage_std','explained_variance_before_update','contact_fraction','reward_obj_pos','reward_obj_rot','reward_fingers','reward_wrist','reward_action','reward_action_change']}
        training.append(dict(arm=arm,steps=steps,updates=len(rows),optimizer_steps_before=list(before),optimizer_steps_after=list(after),diagnostics=summary_diag))
    checks.update(optimizer_progress_and_hyperparameters_retained=True,PPO_step_caps_and_finite_updates=True);write_json(RUN/'training_diagnostics.json',training)
    initial=np.load(ROOT/'runs/grasp_repair_20261026/fixed_initials.npz');ref=np.load(DATA/'trajectory.npz')['qpos'].astype('float32');target=np.load(DATA/'controller_reference.npz')['ctrl'].astype('float32');m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));lo=m.actuator_ctrlrange[:,0];hi=m.actuator_ctrlrange[:,1];scale=np.array([.025]*3+[.18]*3+[.25]*20)
    max_reward=max_control=0.;count=0
    canonical=read_json(RUN/'canonical_eval/summary.json');assert canonical['all_requested_evaluations_completed']
    for row in canonical['results']:
        for t in row['trials']:
            a=np.load(evalfile(row,t['trial']));q=a['qpos'];v=a['qvel'];act=np.clip(a['raw_action'],-1,1);phase=a['phase']+1;n=len(act)
            assert np.array_equal(q[0],initial['qpos'][t['trial']].astype(q.dtype)) and np.array_equal(v[0],initial['qvel'][t['trial']].astype(v.dtype))
            assert np.isfinite(q).all() and np.isfinite(v).all();expected=np.clip(target[phase]+act*scale,lo,hi);max_control=max(max_control,float(np.max(abs(expected-a['ctrl']))))
            ep=a['position_error'];er=a['rotation_error'];eq=np.mean((q[1:,6:26]-ref[phase,6:26])**2,axis=1);ew=np.sum((q[1:,:3]-ref[phase,:3])**2,axis=1);last=np.vstack([np.zeros(26),act[:-1]])
            reward=.02*(5*np.exp(-(ep/.04)**2)+2*np.exp(-(er/.5)**2)+.5*np.exp(-eq/.1)+.5*np.exp(-ew/.001)-.03*np.mean(act**2,axis=1)-.05*np.mean((act-last)**2,axis=1));max_reward=max(max_reward,float(np.max(abs(reward-a['reward']))))
            assert bool(t['success'])==bool(n==51 and t['reason']==0 and (ep<=.03).all() and (er<=np.pi/6).all());bad=np.flatnonzero((ep>.03)|(er>np.pi/6));assert t['first_strict_failure_s']==(float((bad[0]+1)*.02) if len(bad) else None);count+=1
    assert max_reward<1e-6 and max_control<1e-6;checks.update(fixed_states_success_and_failure_times_recomputed=True,reward_and_executed_control_recomputed=True);details.update(evaluation_episodes=count,max_reward_difference=max_reward,max_control_difference=max_control)
    with np.load(RUN/'wm/dataset.npz') as f:data={k:f[k] for k in f.files}
    ds=read_json(RUN/'wm/dataset_manifest.json');train={e['group'] for e in ds['episodes'] if e['split']=='train'};validation={e['group'] for e in ds['episodes'] if e['split']=='validation'};assert train.isdisjoint(validation)
    cfg=read_json(RUN/'wm/config.json');updates=cfg['paired_updates'];assert updates<=2000;rows=list(csv.DictReader((RUN/'wm/training.csv').open()));assert len([r for r in rows if r['training_loss']])==2*updates
    for r in rows:assert all(np.isfinite(float(r[k])) for k in ['training_loss','preclip_gradient_norm'] if r[k]!='')
    windows=data['windows_8_train'];pools=[np.flatnonzero(windows[:,3]==i) for i in range(3)];available=[i for i,p in enumerate(pools) if len(p)];rng=np.random.default_rng(23);digest=hashlib.sha256();checkpoints={int(p.stem.split('_')[1]) for p in (RUN/'wm').glob('W0_[0-9]*.pt')}|{updates};batch_matches=[]
    for update in range(1,updates+1):
        labels=rng.choice(available,64);ix=np.array([rng.choice(pools[int(k)]) for k in labels]);digest.update(ix.astype('<i8').tobytes())
        if update in checkpoints:
            for name in ['W0','W1']:
                path=RUN/'wm'/f'{name}_{update:04d}.pt'
                if not path.exists():path=RUN/'wm'/f'{name}_final.pt'
                cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['update']==update and cp['batch_sequence_sha256']==digest.hexdigest()
            batch_matches.append(update)
    reload=[]
    for name in ['W0','W1']:
        cp=torch.load(RUN/'wm'/f'{name}_final.pt',map_location='cpu',weights_only=False);model=WorldModel(name,Geometry(RUN/'wm'),cp['normalization']);model.load_state_dict(cp['model']);model.eval();pred=np.load(RUN/'wm'/f'prediction_{name}_08.npz');ix=pred['windows'][:8];s=torch.tensor(data['states'][ix[:,0]])
        with torch.no_grad():
            for j in range(8):s=model(s,torch.tensor(data['controls'][ix[:,1]+j]));assert torch.isfinite(s).all()
        delta=float(np.max(abs(s[:,:65].numpy()-pred['predicted_states'][:8,-1])));assert delta<.002;reload.append(dict(model=name,cpu_vs_saved_GPU_max_coordinate_difference=delta))
    checks.update(whole_episode_split_retained=True,matched_seed23_batch_sequence=True,WM_updates_finite_and_capped=True,checkpoint_reload_8step_prediction=True);details.update(model_reload=reload,batch_hash_checked_at_updates=batch_matches)
    ledger=read_json(RUN/'compute_ledger.json');assert ledger['all_budgets_respected'] and all(p['returncode']==0 for p in ledger['processes']);checks['budget_and_process_completion']=True
    write_json(RUN/'artifact_validation.json',dict(checks=checks,details=details,scope='CPU audit of newly produced learning artifacts; no static-grasp gate and no repeated physics qualification.'))
    print('Continuation and frozen-data artifact checks passed; evaluated episodes:',count)


if __name__=='__main__':main()
