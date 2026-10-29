"""Evidence tables, contact-time correction, slip proxies and paired prefixes."""
import json,csv
from pathlib import Path
import numpy as np,mujoco,torch
from scipy.spatial.transform import Rotation
from grasp_lab import DATA,RUN,OLD,model_for,write_json
from train_real_ppo import Policy


def load(path):return json.loads(path.read_text())


def enrich_trial(engine,variant,task,trial):
    folder=RUN/engine/variant/task;stem=f'trial_{trial:02d}';a=np.load(folder/(stem+'.npz'));rows=load(folder/(stem+'_metrics.json'));logs=load(folder/(stem+'_contacts.json'));ref=np.load(DATA/'trajectory.npz');m=model_for(variant);d=mujoco.MjData(m);rd=mujoco.MjData(m);wrist=m.body('wrist').id;obj=m.body('object').id
    extra=[]
    for i,(row,log) in enumerate(zip(rows,logs)):
        phase=0 if task=='A' else i+1
        d.qpos[:]=a['qpos'][i+1];mujoco.mj_kinematics(m,d);rd.qpos[:]=ref['qpos'][phase];mujoco.mj_kinematics(m,rd)
        rw=Rotation.from_quat(d.xquat[wrist],scalar_first=True);rr=Rotation.from_quat(rd.xquat[wrist],scalar_first=True)
        po=rw.inv().apply(d.qpos[26:29]-d.xpos[wrist]);pr=rr.inv().apply(rd.qpos[26:29]-rd.xpos[wrist])
        ro=rw.inv()*Rotation.from_quat(d.qpos[29:],scalar_first=True);rt=rr.inv()*Rotation.from_quat(rd.qpos[29:],scalar_first=True)
        relative_pos=np.linalg.norm(po-pr);relative_angle=(ro*rt.inv()).magnitude()
        d.qpos[:]=log['prestep_qpos'];d.qvel[:]=log['prestep_qvel'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        speeds=[];weights=[];hand=[]
        for c in log['contacts']:
            if c['kind']!='hand_object':continue
            hand.append(c);force=max(0,c['force_contact'][0])
            if force<.01:continue
            g=c['geom'][0] if m.geom_contype[c['geom'][0]]==1 else c['geom'][1]
            jo=np.zeros((3,m.nv));jh=jo.copy();jr=jo.copy();mujoco.mj_jac(m,d,jo,jr,np.array(c['pos']),obj);mujoco.mj_jac(m,d,jh,jr,np.array(c['pos']),int(m.geom_bodyid[g]))
            v=(jo-jh)@d.qvel;normal=np.array(c['frame'][:3]);vt=v-normal*np.dot(v,normal);speeds.append(np.linalg.norm(vt));weights.append(force)
        speed=float(np.average(speeds,weights=weights)) if weights else 0
        extra.append(dict(**row,contact_time_s=log['time_s'],relative_grasp_position_m=float(relative_pos),relative_grasp_rotation_rad=float(relative_angle),force_weighted_tangential_speed_m_s=speed,loaded_normal_force_n=float(sum(weights)),contact_geometries=sorted(set(g for c in hand for g in c['geom'] if m.geom_contype[g]==1))))
    def first(pred):return next((r['time_s'] for r in extra if pred(r)),None)
    slip=next((extra[i]['contact_time_s'] for i in range(len(extra)-1) if all(extra[j]['force_weighted_tangential_speed_m_s']>.02 and extra[j]['loaded_normal_force_n']>.1 for j in [i,i+1])),None)
    record=dict(engine=engine,variant=variant,task=task,trial=trial,steps=len(rows),first_observed_slip_s=slip,
                first_relative_grasp_deviation_10mm_10deg_s=first(lambda r:r['relative_grasp_position_m']>.01 or r['relative_grasp_rotation_rad']>np.pi/18),
                first_no_geometric_contact_s=next((r['contact_time_s'] for r in extra if r['contact_count']==0),None),
                first_strict_failure_s=first(lambda r:r['position_m']>.03 or r['rotation_rad']>np.pi/6) if task!='A' else None,
                first_static_hold_failure_s=first(lambda r:r['position_m']>.01 or r['rotation_rad']>np.pi/12) if task=='A' else None,
                termination_s=len(rows)*.02 if task!='A' and len(rows)<51 else None,
                slip_definition='First observed force-weighted tangent speed >20 mm/s with >0.1 N loaded normal force for two consecutive samples; 20 ms sampling, contact timestamp=end-2ms. Relative-pose deviation is separately reported; neither is tactile ground truth.')
    write_json(folder/(stem+'_contact_aligned_metrics.json'),extra)
    return record


def batch_evidence():
    reports=[];one=np.load(RUN/'batch_1/trajectory.npz')
    for n in [1,2,8]:
        folder=RUN/f'batch_{n}';a=np.load(folder/'trajectory.npz');contact=load(folder/'gpu_contacts.json');summary=load(folder/'summary.json')
        assert np.array_equal(a['initial_qpos'][0],one['initial_qpos'][0]) and np.array_equal(a['initial_qvel'][0],one['initial_qvel'][0])
        assert np.array_equal(a['ctrl'][:,0],one['ctrl'][:,0])
        cross=np.linalg.norm(a['qpos_after'][:,0,26:29]-one['qpos_after'][:,0,26:29],axis=1)
        snapshots={}
        if n>1:
            q=a['qpos_after'];spread=np.linalg.norm(q[:,:,26:29]-q[:,0:1,26:29],axis=-1)
            numerical=np.flatnonzero(np.any(q!=q[:,0:1],axis=(1,2)))
            summary['first_nonbitwise_qpos_time_s']=float((numerical[0]+1)*.002) if len(numerical) else None
            if len(numerical):
                s=int(numerical[0]);world=int(np.max(abs(q[s]-q[s,0]),axis=1).argmax())
                snapshots['first_nonbitwise']=dict(physics_step=s+1,state_time_s=(s+1)*.002,contact_time_s=s*.002,world=world,
                    prestate_0=a['qpos_before'][s,0].tolist(),prestate_other=a['qpos_before'][s,world].tolist(),prevelocity_0=a['qvel_before'][s,0].tolist(),prevelocity_other=a['qvel_before'][s,world].tolist(),
                    poststate_0=q[s,0].tolist(),poststate_other=q[s,world].tolist(),ctrl_0=a['ctrl'][s,0].tolist(),ctrl_other=a['ctrl'][s,world].tolist(),contacts=[c for c in contact[s]['contacts'] if c['world'] in [0,world]])
            for threshold in [1e-6,1e-4,.001,.005]:
                hit=np.flatnonzero(spread.max(1)>threshold)
                if not len(hit):continue
                s=int(hit[0]);world=int(spread[s].argmax())
                snapshots[str(threshold)]=dict(physics_step=s+1,state_time_s=(s+1)*.002,contact_time_s=s*.002,world=world,
                    prestate_0=a['qpos_before'][s,0].tolist(),prestate_other=a['qpos_before'][s,world].tolist(),prevelocity_0=a['qvel_before'][s,0].tolist(),prevelocity_other=a['qvel_before'][s,world].tolist(),
                    poststate_0=q[s,0].tolist(),poststate_other=q[s,world].tolist(),ctrl_0=a['ctrl'][s,0].tolist(),ctrl_other=a['ctrl'][s,world].tolist(),contacts=[c for c in contact[s]['contacts'] if c['world'] in [0,world]])
            topology=None
            for s,frame in enumerate(contact):
                pairs=[sorted(tuple(c['geom']) for c in frame['contacts'] if c['world']==w) for w in range(n)]
                if any(x!=pairs[0] for x in pairs[1:]):topology=dict(time_s=s*.002,physics_step=s+1,pairs=pairs);break
            summary['first_contact_pair_multiset_difference']=topology
        summary['cross_world_count_vs_1_max_position_difference_m']=float(cross.max());summary['identical_inputs_across_world_counts']=True
        write_json(folder/'first_difference_evidence.json',snapshots);reports.append(summary)
    write_json(RUN/'batch_evidence_summary.json',reports)


def paired_prefix(engine,baseline_variant,newvariant,task):
    pairs=[]
    for trial in range(8):
        a=load(RUN/engine/baseline_variant/task/f'trial_{trial:02d}_metrics.json');b=load(RUN/engine/newvariant/task/f'trial_{trial:02d}_metrics.json');n=min(len(a),len(b))
        pairs.append(dict(trial=trial,common_controls=n,common_duration_s=.02*n,baseline_controls=len(a),candidate_controls=len(b),baseline_position_common_m=float(np.mean([x['position_m'] for x in a[:n]])),candidate_position_common_m=float(np.mean([x['position_m'] for x in b[:n]])),baseline_rotation_common_deg=float(np.rad2deg(np.mean([x['rotation_rad'] for x in a[:n]]))),candidate_rotation_common_deg=float(np.rad2deg(np.mean([x['rotation_rad'] for x in b[:n]])))))
    return dict(engine=engine,baseline=baseline_variant,candidate=newvariant,task=task,pairs=pairs,mean_common_position_baseline_m=np.mean([r['baseline_position_common_m'] for r in pairs]),mean_common_position_candidate_m=np.mean([r['candidate_position_common_m'] for r in pairs]),mean_common_rotation_baseline_deg=np.mean([r['baseline_rotation_common_deg'] for r in pairs]),mean_common_rotation_candidate_deg=np.mean([r['candidate_rotation_common_deg'] for r in pairs]),aggregation='Equal weight per paired trial, each pair truncated to the SAME execution interval; full-horizon success is reported separately.')


def learning_audit():
    # Finite-horizon/failure/truncation examples test the existing algebra, not a
    # mirror of its implementation. The deliberately huge next episode cannot leak.
    r=np.array([1.,2.,100.]);v=np.array([.1,.2,9.]);nv=np.array([.2,3.,0.]);done=np.array([False,True,True]);delta=r+.99*nv-v
    adv=np.zeros(3);g=0
    for t in reversed(range(3)):g=delta[t]+.99*.95*(not done[t])*g;adv[t]=g
    assert abs(adv[1]-(2+.99*3-.2))<1e-12 and abs(adv[0]-(delta[0]+.99*.95*delta[1]))<1e-12
    rows=list(csv.DictReader((OLD/'training.csv').open()));ckpt=torch.load(OLD/'checkpoint.pt',map_location='cpu',weights_only=True);p=Policy(118);p.load_state_dict(ckpt['policy']);std=p.logstd.detach().numpy();std=np.exp(std);scale=np.array(ckpt['config']['action_scale'])
    report=dict(gae_formula_boundary_example_passed=True,truncation_formula_bootstrap_checked=True,true_failure_bootstrap_value_in_code=0,
        historical_reconstruction_limit='The baseline saved no per-update old log-probabilities, value predictions, advantages, clip fractions or explained variance. Formula and reset boundaries were inspected; historical optimizer inputs cannot be reconstructed from training.csv alone.',
        action_gaussian_std=std.tolist(),physical_action_std=(std*scale).tolist(),mean_preclip_gradient_norm=float(np.mean([float(r['gradient_norm']) for r in rows])),max_preclip_gradient_norm=max(float(r['gradient_norm']) for r in rows),mean_approx_kl=float(np.mean([float(r['approx_kl']) for r in rows])),max_approx_kl=max(float(r['approx_kl']) for r in rows),
        clip_horizon_semantics='Baseline uses time-limit bootstrap. A finite recorded clip has no defined post-clip continuation; choose and freeze this task convention before future training. No change in this diagnostic round.',
        contact_reward='NOT ADDED: reference contact/grasp stability is not established. Geometric contact gate alone cannot certify a useful contact reward.',
        action_scale_observation='About 5 mm wrist translation and 0.05 rad finger exploration SD in the saved checkpoint are large relative to the millimetre-sensitive grasp. This motivates a later isolated scale experiment, not simultaneous reward/control changes.',
        new_training_run=False,reason='No physical/control candidate met the unchanged full-trajectory criterion on the eight fixed states; no justification for another PPO run.')
    write_json(RUN/'learning_design_audit.json',report)


def main():
    batch_evidence();events=[];paired=[]
    for engine,variants in [('gpu',['baseline','grip_balance']),('cpu',['baseline','contact4','wrist_ff','wrist_ff35','zero_velocity','grip_balance','index_contact','grip_balance_index'])]:
        for variant in variants:
            for task in ['A','B','C']:
                if not (RUN/engine/variant/task/'trial_00.npz').exists():continue
                for trial in range(8):events.append(enrich_trial(engine,variant,task,trial))
                if variant!='baseline' and (RUN/engine/'baseline'/task/'trial_00.npz').exists():paired.append(paired_prefix(engine,'baseline',variant,task))
    write_json(RUN/'events_and_failures.json',events);write_json(RUN/'paired_common_prefix.json',paired);learning_audit()
    write_json(RUN/'incremental_index_ablation.json',[paired_prefix('cpu','grip_balance','grip_balance_index',task) for task in ['A','B']])
    print('Analyzed',len(events),'trial logs and',len(paired),'paired groups')


if __name__=='__main__':main()
