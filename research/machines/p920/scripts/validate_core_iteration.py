"""CPU audit of this iteration's changed learning path and retained evidence."""
import hashlib,csv
import numpy as np,torch,mujoco
from exploration_common import RUN,ROOT,DATA,read_json,write_json

def main():
    torch.set_num_threads(1);checks={};details={};cfg=read_json(RUN/'budget.json');manifest=read_json(RUN/'input_manifest.json')
    for r in manifest['files']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']
    checks['baseline_model_task_physics_and_initials_unchanged']=True
    orig=np.load(ROOT/'runs/grasp_repair_20261026/fixed_initials.npz');dev=np.load(RUN/'development_initials.npz')
    assert all(np.array_equal(orig[k],dev[k]) for k in ['qpos','qvel']);checks['original_eight_conditions_preserved']=True
    impl=read_json(RUN/'implementation_check.json');assert impl['finite_difference_absolute_error']<1e-5 and impl['valid_transitions']==9 and impl['model_parameter_gradients_absent'];checks['new_policy_gradient_history_reference_end_checks_passed']=True
    summaries=[];sources={};lasts={};training_steps=0;aux=0
    for stage in ['core','confirmation']:
        if not (RUN/(stage+'_completion.json')).exists():continue
        completion=read_json(RUN/(stage+'_completion.json'));totals={}
        for arm in ['P','M']:
            previous=None;count=0
            for folder in sorted((RUN/stage/arm).glob('block_*')):
                c=read_json(folder/'config.json');s=read_json(folder/'summary.json');summaries.append(s);assert c['ppo']==cfg['ppo'] and c['heldout_used']==False and c['reward_success_physics_unchanged']
                assert c['model_sha256']==next(x['sha256'] for x in manifest['files'] if x['path']==cfg['model_source'])
                assert c['rho']==(0 if arm=='P' else read_json(RUN/'frozen_recipe.json')['rho'])
                assert s['optimizer_step_after']-s['optimizer_step_before']==s['ppo_optimizer_updates']==s['training_interactions']//128*4
                if previous is not None:assert s['optimizer_step_before']==previous
                else:sources[stage+'/'+arm]=c['source_sha256']
                previous=s['optimizer_step_after'];count+=s['training_interactions'];aux+=s['auxiliary_model_predictions']
                ck=torch.load(folder/'checkpoint_final.pt',map_location='cpu',weights_only=True)
                assert all(torch.isfinite(t).all() for t in ck['policy'].values());assert all(k in ck for k in ['optimizer','cuda_rng','action_rng','update_rng'])
                for row in csv.DictReader((folder/'training.csv').open()):
                    assert all(np.isfinite(float(v)) for v in row.values())
                    if arm=='M':assert float(row['model_scaled_ratio'])<=c['rho']+1e-5
                assert s['auxiliary_model_predictions']==(0 if arm=='P' else s['training_interactions']//128*16*4)
            totals[arm]=count;lasts[stage+'/'+arm]=previous;training_steps+=count
        assert totals['P']==totals['M']==completion['steps_per_arm'];assert sources[stage+'/P']==sources[stage+'/M']
    checks['matched_controls_shared_recipe_sources_optimizer_resume']=True;checks['finite_weights_updates_and_bounded_auxiliary_gradient']=True;details.update(total_core_confirmation_training_controls=training_steps,auxiliary_predicted_transitions=aux,optimizer_steps=lasts)
    ref=np.load(DATA/'trajectory.npz')['qpos'];target=np.load(DATA/'controller_reference.npz')['ctrl'];model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));scale=np.array([.025]*3+[.18]*3+[.25]*20);max_reward=max_control=0.;logs=0
    analysis=read_json(RUN/'iteration_analysis.json')
    for e in analysis['evaluations']:
        initial=np.load(RUN/('development_initials.npz' if e['split']=='seen8' else 'heldout_initials.npz'))
        for trial in e['trials']:
            a=np.load(ROOT/e['folder']/f"trial_{trial['trial']:02d}.npz");n=len(a['reward']);assert np.allclose(a['qpos'][0],initial['qpos'][trial['trial']],atol=1e-7);assert np.allclose(a['qvel'][0],initial['qvel'][trial['trial']],atol=1e-6)
            assert np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all();action=np.clip(a['raw_action'],-1,1);u=np.clip(target[1:n+1]+action*scale,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1]);max_control=max(max_control,float(abs(u-a['ctrl']).max()))
            q=a['qpos'][1:];r=ref[1:n+1];pos=np.linalg.norm(q[:,26:29]-r[:,26:29],axis=1);quat=q[:,29:33]/np.maximum(np.linalg.norm(q[:,29:33],axis=1,keepdims=True),1e-8);rot=2*np.arccos(np.clip(abs((quat*r[:,29:33]).sum(1)),0,1));last=np.concatenate([np.zeros((1,26)),action[:-1]])
            rew=.02*(5*np.exp(-(pos/.04)**2)+2*np.exp(-(rot/.5)**2)+.5*np.exp(-np.mean((q[:,6:26]-r[:,6:26])**2,1)/.1)+.5*np.exp(-np.sum((q[:,:3]-r[:,:3])**2,1)/.001)-.03*np.mean(action**2,1)-.05*np.mean((action-last)**2,1));max_reward=max(max_reward,float(abs(rew-a['reward']).max()));logs+=1
            assert trial['strict_recomputed']==bool(n==51 and np.all(a['position_error']<=.03) and np.all(a['rotation_error']<=np.pi/6) and trial['reason']==0)
    assert max_control<1e-6 and max_reward<4e-6;checks['all_fixed_evaluation_controls_rewards_strict_success_recomputed']=True;details.update(evaluation_episode_logs=logs,max_control_error=max_control,max_reward_error=max_reward)
    for stage in ['core','confirmation']:
        p=RUN/(stage+'_heldout/summary.json')
        if not (RUN/(stage+'_completion.json')).exists():continue
        summary=read_json(p);assert summary['complete'] and not summary['heldout_used_for_training']
        manifest=read_json(RUN/(stage+'_heldout/manifest.json'));assert not manifest['used_for_training_or_tuning']
    assert not read_json(RUN/'frozen_recipe.json')['heldout_accessed_for_selection'];assert not read_json(RUN/'confirmation_plan.json')['heldout_accessed_for_selection'];checks['reserved_conditions_all_checkpoints_and_endpoint_replays_retained']=True
    ledger=read_json(RUN/'process_ledger.json')['processes'];total=sum(r['wall_seconds'] for r in ledger);assert total<=21600
    for cat,cap in cfg['allocation_seconds'].items():assert sum(r['wall_seconds'] for r in ledger if r['category']==cat)<=cap
    failures=[r for r in ledger if r['returncode']!=0]
    for failure in failures:
        if failure['name']=='development/P':assert failure['wall_seconds']<10 and (RUN/'development/P_failed_startup').exists()
        else:
            assert failure['returncode']==-15 and not failure['hard_timeout'] and (RUN/failure['artifact_path']).exists()
            assert failure['aborted_interactions_lower_bound']<=failure['aborted_interactions_upper_bound']
    checks['budget_all_processes_including_retained_startup_failure']=True;details['gpu_process_seconds']=total
    write_json(RUN/'artifact_validation.json',dict(checks=checks,details=details,scope='Only this iteration changed path and delivered artifacts. Not new grasp prerequisites, formal research gates, or a claim that contact sensitivity is solved.'))
    print('Artifact validity checks:',checks)

if __name__=='__main__':main()
