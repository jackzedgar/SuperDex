"""Predeclared frozen-model short-horizon action-direction checks, no learning."""
import argparse,time,hashlib
import numpy as np,torch,warp as wp
from exploration_common import RUN,read_json,write_json,resource_check,Contacts
from world_model_mini import Geometry,WorldModel,qlog,qmul,qconj
from world_model_residual import ResidualWorldModel
from real_env import RealEnv
from train_real_ppo import Policy

def reward_from_state(s,ref,action,last):
    ep=torch.linalg.vector_norm(s[...,26:29]-ref[...,26:29],dim=-1)
    # atan2/log form is mathematically the same angle as the environment's acos,
    # with a finite derivative at zero angle. Forward equality is audited below.
    er=torch.linalg.vector_norm(qlog(qmul(qconj(ref[...,29:33]),s[...,29:33])),dim=-1)
    eq=((s[...,6:26]-ref[...,6:26])**2).mean(-1);ew=((s[...,:3]-ref[...,:3])**2).sum(-1)
    parts=.02*torch.stack([5*torch.exp(-(ep/.04)**2),2*torch.exp(-(er/.5)**2),.5*torch.exp(-eq/.1),.5*torch.exp(-ew/.001),-.03*action.square().mean(-1),-.05*(action-last).square().mean(-1)],-1)
    return parts.sum(-1),parts

def prediction(model,state,actions,phase,last,env):
    s=state;total=0.;ss=[];controls=[];parts=[]
    for t,action in enumerate(actions):
        a=action[None].clamp(-1,1);u=torch.maximum(env.lo,torch.minimum(env.hi,env.target[phase+t+1]+a*env.scale));s=model(s,u);r,rp=reward_from_state(s,env.ref[phase+t+1][None],a,last);total=total+r.sum();last=a;ss.append(s[0]);controls.append(u[0]);parts.append(rp[0])
    return total,torch.stack(ss),torch.stack(controls),torch.stack(parts)

def main():
    started=time.perf_counter();p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);args=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);out=RUN/'gradient';cfg=read_json(RUN/'budget.json')['gradient'];assert not (out/'summary.json').exists();(out/'resources_start.txt').write_text(resource_check(2000))
    update=read_json(RUN/'wm/B/summary.json')['matched_comparison_update'];assert update>0;geometry=Geometry(RUN/'wm').to('cuda');models={};checkpoints={}
    for name,path in [('A',RUN/'wm/A'/f'W1_{update:04d}.pt'),('B',RUN/'wm/B'/f'B_{update:04d}.pt')]:
        c=torch.load(path,map_location='cpu',weights_only=False);m=WorldModel('W1',geometry,c['normalization']) if name=='A' else ResidualWorldModel(geometry,c['normalization']);m.load_state_dict(c['model']);models[name]=m.to('cuda').eval().requires_grad_(False);checkpoints[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    policy=Policy(118).to('cuda');pc=torch.load(out/'policy.pt',map_location='cpu',weights_only=True);policy.load_state_dict(pc['policy']);policy.eval().requires_grad_(False)
    env=RealEnv(1);contacts=Contacts(env);initial=dict(np.load(out/'fixed_initials.npz'));prefixes={i:dict(np.load(out/'prefixes'/f'trial_{i:02d}.npz')) for i in cfg['initial_ids']};ids=torch.tensor([0],device='cuda');records=[];planned=[];max_reward_difference=0.;fd=[]
    def replay(i,k):
        env.reset(ids,False);env.qpos[0]=torch.tensor(initial['qpos'][i],device='cuda');env.qvel[0]=torch.tensor(initial['qvel'][i],device='cuda');history=env.ctrl.repeat(3,1)
        for j in range(k):
            action=torch.tensor(prefixes[i]['raw_action'][j],device='cuda',dtype=torch.float32)[None];env.step(action);history=torch.cat([history[1:],env.ctrl.clone()],0)
            assert torch.max(abs(env.ctrl[0]-torch.tensor(prefixes[i]['ctrl'][j],device='cuda')))<1e-6
        wp.synchronize();state=torch.cat([env.qpos[0],env.qvel[0],history.flatten()])[None].detach().clone();return state,env.last_action.clone(),wp.to_torch(env.wd.qacc_warmstart).clone()
    stop='all_predeclared_cases';case_id=0
    for i in cfg['initial_ids']:
        for k in cfg['prefix_controls']:
            for horizon in cfg['horizons']:
                if time.perf_counter()-started>args.seconds-25:stop='wall_budget';break
                folder=out/f'case_{case_id:03d}';folder.mkdir(exist_ok=False);state,last,warm=replay(i,k)
                with torch.no_grad():nominal=policy.actor(env.observe())[0].clamp(-1,1).repeat(horizon,1).detach()
                proposals={'baseline':nominal};pred={};plans={}
                for name,model in models.items():
                    aa=nominal.clone().requires_grad_(True);value,states,controls,parts=prediction(model,state,aa,k,last,env);gradient=torch.autograd.grad(value,aa)[0];assert torch.isfinite(gradient).all();direction=gradient/gradient.abs().max().clamp_min(1e-12);delta=cfg['perturbation_linf']*direction
                    plus=(nominal+delta).clamp(-1,1).detach();minus=(nominal-delta).clamp(-1,1).detach();proposals[name+'_plus']=plus;proposals[name+'_minus']=minus
                    with torch.no_grad():
                        vp,sp,up,rp=prediction(model,state,plus,k,last,env);vm,sm,um,rm=prediction(model,state,minus,k,last,env)
                    pred[name]=dict(baseline=float(value.detach()),plus=float(vp),minus=float(vm));plans[name]=dict(predicted_plus_improvement=float(vp-value.detach()),predicted_minus_improvement=float(vm-value.detach()),gradient_linf=float(gradient.abs().max()),delta_linf=float((plus-nominal).abs().max()),control_plus_delta_linf=float((up-controls.detach()).abs().max()),control_minus_delta_linf=float((um-controls.detach()).abs().max()))
                    # Numerical directional derivative only, not physical validity.
                    if case_id in [0,1]:
                        with torch.no_grad():
                            ap=(nominal+.001*direction).clamp(-1,1);am=(nominal-.001*direction).clamp(-1,1);fp=prediction(model,state,ap,k,last,env)[0];fm=prediction(model,state,am,k,last,env)[0]
                        fd.append(dict(case=case_id,model=name,autograd=float((gradient*(ap-am)/.002).sum()),finite_difference=float((fp-fm)/.002),note='Projected central difference; uses actual displacement at action bounds.'))
                    np.savez_compressed(folder/f'proposal_{name}.npz',anchor_state=state.cpu().numpy(),previous_action=last.cpu().numpy(),warmstart=warm.cpu().numpy(),nominal=nominal.cpu().numpy(),gradient=gradient.cpu().numpy(),plus=plus.cpu().numpy(),minus=minus.cpu().numpy(),baseline_predicted_states=states.detach().cpu().numpy(),plus_predicted_states=sp.cpu().numpy(),minus_predicted_states=sm.cpu().numpy(),baseline_controls=controls.detach().cpu().numpy(),plus_controls=up.cpu().numpy(),minus_controls=um.cpu().numpy(),baseline_reward_components=parts.detach().cpu().numpy(),plus_reward_components=rp.cpu().numpy(),minus_reward_components=rm.cpu().numpy())
                meta=dict(case=case_id,initial_id=i,prefix_controls=k,anchor_time_s=k*.02,horizon=horizon,model_update=update,predictions=pred,proposals=plans,proposals_fixed_before_future_physics=True);write_json(folder/'plan.json',meta);planned.append(meta)
                jobs=[('baseline',r) for r in range(cfg['baseline_repeats'])]+[(name+'_'+sign,r) for name in ['A','B'] for sign in ['plus','minus'] for r in range(cfg['direction_repeats'])];np.random.default_rng(cfg['replay_order_seed']+case_id).shuffle(jobs);rows=[]
                for order,(label,rep) in enumerate(jobs):
                    st,prev,wa=replay(i,k);qs=[env.qpos[0].cpu().numpy().copy()];vs=[env.qvel[0].cpu().numpy().copy()];cc=[];rr=[];pp=[];flags=[];ct=[]
                    for action in proposals[label]:
                        previous=env.last_action.clone();_,r,done,info=env.step(action[None]);assert bool(info['finite'][0]);actual=torch.cat([env.qpos,env.qvel],-1);reconstructed,parts=reward_from_state(actual,env.ref[env.phase],action[None],previous);err=float(abs(reconstructed-r).max());max_reward_difference=max(max_reward_difference,err);assert err<2e-6
                        qs.append(env.qpos[0].cpu().numpy().copy());vs.append(env.qvel[0].cpu().numpy().copy());cc.append(env.ctrl[0].cpu().numpy().copy());rr.append(float(r[0]));pp.append(parts[0].cpu().numpy());flags.append(int(info['reason'][0]));ct.append(contacts.read()[:3])
                    filename=f'{order:02d}_{label}_{rep}.npz';np.savez_compressed(folder/filename,qpos=qs,qvel=vs,ctrl=cc,action=proposals[label].cpu().numpy(),reward=rr,reward_parts=pp,termination_reason=flags,contacts=ct,anchor_state=st.cpu().numpy(),anchor_warmstart=wa.cpu().numpy())
                    record=dict(case=case_id,initial_id=i,prefix_controls=k,horizon=horizon,label=label,repeat=rep,order=order,return_=sum(rr),file=str((folder/filename).relative_to(RUN)),anchor_max_abs_difference=float((st-state).abs().max()),anchor_position_difference_mm=float(torch.linalg.vector_norm(st[:,26:29]-state[:,26:29])*1000),anchor_warmstart_difference=float((wa-warm).abs().max()),any_original_failure=bool(any(flags)),hand_contact_steps=int(sum(x[0]>0 for x in ct)),floor_contact_steps=int(sum(x[1]>0 for x in ct)))
                    rows.append(record);records.append(record)
                write_json(folder/'replays.json',rows);case_id+=1;write_json(out/'progress.json',dict(completed_cases=case_id,total_predeclared_cases=32,elapsed_seconds=time.perf_counter()-started));print('CASE COMPLETE',case_id,'elapsed',time.perf_counter()-started,flush=True)
            if stop=='wall_budget':break
        if stop=='wall_budget':break
    assert all(p.grad is None for m in models.values() for p in m.parameters());assert all(p.grad is None for p in policy.parameters())
    write_json(out/'summary.json',dict(completed_cases=case_id,total_predeclared_cases=32,stop_reason=stop,process_seconds=time.perf_counter()-started,model_update=update,checkpoints_sha256=checkpoints,policy_parameters_frozen=True,model_parameters_frozen=True,max_original_reward_difference=max_reward_difference,numerical_directional_derivative_checks=fd,records=records,plans=planned,missing_cases=32-case_id,selection='IDs/times/horizons fixed in budget.json before training. All observed cases, all repeats and both directions retained.'))
    print('GRADIENT PROBE COMPLETE',case_id,time.perf_counter()-started,flush=True)

if __name__=='__main__':main()
