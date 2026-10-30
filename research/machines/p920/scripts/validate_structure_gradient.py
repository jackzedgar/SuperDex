"""CPU-only artifact and derivative validity audit, no physical success gate."""
import hashlib,csv
from types import SimpleNamespace
import numpy as np,torch,mujoco
from exploration_common import RUN,ROOT,DATA,read_json,write_json
from world_model_mini import Geometry,WorldModel
from world_model_residual import ResidualWorldModel
from probe_model_gradient import prediction

def main():
    torch.set_num_threads(1);checks={};detail={};manifest=read_json(RUN/'input_manifest.json')
    for r in manifest['files']:assert hashlib.sha256((RUN/r['copy']).read_bytes()).hexdigest()==r['sha256'] and hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest()==r['sha256']
    for r in manifest['unchanged_sources']:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    checks['frozen_inputs_sources_and_A_byte_identical']=True
    bnorm=dict(np.load(RUN/'wm/B/normalization.npz'));anorm=dict(np.load(RUN/'wm/normalization.npz'));slots=list(range(26,32))+list(range(58,64));other=[i for i in range(64) if i not in slots]
    for key in anorm:assert np.array_equal(anorm[key][other] if key in ['delta_mean','delta_std'] else anorm[key],bnorm[key][other] if key in ['delta_mean','delta_std'] else bnorm[key])
    checks['normalization_changes_only_declared_object_outputs']=True
    g=Geometry(RUN/'wm');torch.manual_seed(23);w0=WorldModel('W0',g,anorm);a=WorldModel('W1',g,anorm)
    with torch.no_grad():
        params=dict(a.named_parameters())
        for name,p in w0.named_parameters():
            if name=='net.0.weight':params[name].zero_();params[name][:,:p.shape[1]].copy_(p)
            else:params[name].copy_(p)
    initial=torch.load(RUN/'wm/B/initial.pt',map_location='cpu',weights_only=False)
    for key,v in a.net.state_dict().items():assert torch.equal(v,initial['net'][key])
    b=ResidualWorldModel(g,bnorm);assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())==213568
    checks['same_seed_initial_network_and_parameter_count']=True
    updates=[]
    for p in sorted((RUN/'wm/B').glob('B_[0-9]*.pt')):
        c=torch.load(p,map_location='cpu',weights_only=False);ref=torch.load(RUN/'wm/A'/f'W1_{c["update"]:04d}.pt',map_location='cpu',weights_only=False);assert c['batch_sequence_sha256']==ref['batch_sequence_sha256'];assert all(torch.isfinite(t).all() for t in c['model'].values());updates.append(c['update'])
    checks['same_batch_sequence_and_finite_checkpoints']=True;detail['matched_updates']=updates
    rows=list(csv.DictReader((RUN/'wm/B/training.csv').open()));assert len(rows)<=2000 and all(np.isfinite(float(r['training_loss'])) for r in rows);checks['finite_bounded_training']=True
    data_manifest=read_json(RUN/'wm/dataset_manifest.json');tr={r['group'] for r in data_manifest['episodes'] if r['split']=='train'};va={r['group'] for r in data_manifest['episodes'] if r['split']=='validation'};assert tr.isdisjoint(va);checks['original_episode_disjoint_split']=True
    probe=read_json(RUN/'gradient/summary.json');cfg=read_json(RUN/'budget.json')['gradient'];pairs=[(i,k,h) for i in cfg['initial_ids'] for k in cfg['prefix_controls'] for h in cfg['horizons']];observed=[(p['initial_id'],p['prefix_controls'],p['horizon']) for p in probe['plans']];assert observed==pairs[:len(observed)] and len(set(observed))==len(observed)
    assert probe['completed_cases']==32 and len(probe['records'])==32*17;checks['all_32_predeclared_cases_and_544_replays_retained']=True
    target=np.load(DATA/'controller_reference.npz')['ctrl'];model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));lo=model.actuator_ctrlrange[:,0];hi=model.actuator_ctrlrange[:,1];scale=np.array([.025]*3+[.18]*3+[.25]*20);maxcontrol=0.;maxaction=0.
    for case in probe['plans']:
        folder=RUN/'gradient'/f'case_{case["case"]:03d}'
        for name in ['A','B']:
            pp=np.load(folder/f'proposal_{name}.npz')
            for sign in ['plus','minus']:maxaction=max(maxaction,float(abs(pp[sign]-pp['nominal']).max()))
        rr=[r for r in probe['records'] if r['case']==case['case']];assert sum(r['label']=='baseline' for r in rr)==5
        for label in ['A_plus','A_minus','B_plus','B_minus']:assert sum(r['label']==label for r in rr)==3
        for r in rr:
            log=np.load(RUN/r['file']);u=np.clip(target[case['prefix_controls']+1:case['prefix_controls']+case['horizon']+1]+np.clip(log['action'],-1,1)*scale,lo,hi);maxcontrol=max(maxcontrol,float(abs(u-log['ctrl']).max()));assert np.isfinite(log['qpos']).all() and np.isfinite(log['reward']).all()
    assert maxcontrol<1e-6 and maxaction<=.050001;checks['executed_controls_action_bounds_and_finite_logs']=True;detail.update(max_control_reconstruction_error=maxcontrol,max_action_perturbation=maxaction)
    assert probe['max_original_reward_difference']<2e-6 and probe['model_parameters_frozen'] and probe['policy_parameters_frozen'];checks['original_reward_and_frozen_model_policy']=True
    # Double precision finite differences at both horizons on saved anchors.
    ref=np.load(DATA/'trajectory.npz')['qpos'];fake=SimpleNamespace(lo=torch.tensor(lo),hi=torch.tensor(hi),scale=torch.tensor(scale),target=torch.tensor(target,dtype=torch.float64),ref=torch.tensor(ref,dtype=torch.float64));fd=[]
    for name in ['A','B']:
        step=probe['model_update'];path=RUN/'wm'/name/(f'W1_{step:04d}.pt' if name=='A' else f'B_{step:04d}.pt');ck=torch.load(path,map_location='cpu',weights_only=False);gm=Geometry(RUN/'wm');wm=WorldModel('W1',gm,ck['normalization']) if name=='A' else ResidualWorldModel(gm,ck['normalization']);wm.load_state_dict(ck['model']);wm=wm.double().requires_grad_(False)
        for case_id in [0,1]:
            pp=np.load(RUN/'gradient'/f'case_{case_id:03d}'/f'proposal_{name}.npz');state=torch.tensor(pp['anchor_state'],dtype=torch.float64);last=torch.tensor(pp['previous_action'],dtype=torch.float64);aa=torch.tensor(pp['nominal'],dtype=torch.float64,requires_grad=True);val=prediction(wm,state,aa,0,last,fake)[0];grad=torch.autograd.grad(val,aa)[0];direction=grad/grad.abs().max().clamp_min(1e-12);eps=1e-5
            with torch.no_grad():
                ap=(aa+eps*direction).clamp(-1,1);am=(aa-eps*direction).clamp(-1,1);vp=prediction(wm,state,ap,0,last,fake)[0];vm=prediction(wm,state,am,0,last,fake)[0];actual=(vp-vm)/(2*eps);expected=(grad*(ap-am)/(2*eps)).sum()
            error=float(abs(actual-expected));assert error<1e-5;fd.append(dict(model=name,case=case_id,autograd=float(expected),finite_difference=float(actual),absolute_error=error))
    detail['cpu_double_derivative_checks']=fd;checks['model_action_gradient_finite_difference']=True
    budget=read_json(RUN/'compute_ledger.json');assert budget['total_charged_seconds']<=3600 and all(budget['charged_seconds'][k]<=v for k,v in budget['allocation_seconds'].items()) and all(r['returncode']==0 for r in budget['processes']);checks['budget_and_process_completion']=True
    checks['object_integration_algebra_and_quaternion_convention']=all(read_json(RUN/'parameterization_check.json')[k] for k in ['gravity_prior_check','all_object_output_channels_have_gradient'])
    write_json(RUN/'artifact_validation.json',dict(checks=checks,details=detail,scope='Validity of this single-seed experiment; no static-hold, grasp-success, formal G1/G2 or throughput qualification.'));print('ARTIFACT CHECKS',checks,flush=True)

if __name__=='__main__':main()
