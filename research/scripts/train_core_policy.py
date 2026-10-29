"""Real PPO and PPO plus frozen-W1 differentiable policy-rollout loss."""
import argparse,time,csv,json,hashlib,signal
import numpy as np,torch
from exploration_common import RUN,ROOT,read_json,write_json,resource_check,Contacts,EpisodeLog
from real_env import RealEnv
from train_real_ppo import Policy
from core_policy_common import load_model,physical_state,observe_state,model_policy_objective,model_diagnostics
from probe_model_gradient import reward_from_state

@torch.no_grad()
def evaluate_policy(env,policy,out,initials,contacts):
    out.mkdir(parents=True,exist_ok=False);rows=[];steps=0
    for i in range(len(initials['qpos'])):
        env.reset(torch.tensor([0],device='cuda'),False)
        env.qpos[0]=torch.tensor(initials['qpos'][i],device='cuda',dtype=torch.float32);env.qvel[0]=torch.tensor(initials['qvel'][i],device='cuda',dtype=torch.float32)
        log=EpisodeLog(out/f'trial_{i:02d}.npz',env,dict(kind='evaluation_only',trial=i));first=None;components=[]
        for k in range(env.length-1):
            action=policy.actor(env.observe());previous=env.last_action.clone()
            _,r,done,info=env.step(action);c=contacts.read();log.add(env,action,k,r,info,c)
            _,part=reward_from_state(torch.cat([env.qpos,env.qvel],-1),env.ref[env.phase],action.clamp(-1,1),previous);components.append(part[0].cpu().numpy())
            if first is None and (float(info['pos_error'][0])>.03 or float(info['rot_error'][0])>np.pi/6):first=(k+1)*.02
            steps+=1
            if bool(done[0]):break
        log.save(int(info['reason'][0]),True);np.save(out/f'trial_{i:02d}_reward_parts.npy',np.array(components))
        rows.append(dict(trial=i,steps=k+1,success=bool(info['success'][0]),first_failure_s=first,reason=int(info['reason'][0]),return_=float(info['episode_return'][0]),position_mm=float(info['mean_pos'][0]*1000),rotation_deg=float(info['mean_rot'][0]*180/np.pi),component_returns=np.sum(components,0).tolist()))
    failures=[r['first_failure_s'] for r in rows if r['first_failure_s'] is not None]
    result=dict(successes=sum(r['success'] for r in rows),n=len(rows),mean_return=float(np.mean([r['return_'] for r in rows])),mean_length=float(np.mean([r['steps'] for r in rows])),median_first_failure_s=float(np.median(failures)) if failures else None,physical_controls=steps,trials=rows)
    write_json(out/'summary.json',result);return result

def main():
    start=time.perf_counter();p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--rho',type=float,required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--source');p.add_argument('--model');p.add_argument('--interval',type=int,default=65536);p.add_argument('--offset',type=int,default=0);p.add_argument('--resume-rng',action='store_true');p.add_argument('--skip-initial-eval',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);cfg=read_json(RUN/'budget.json');pcfg=cfg['ppo'];out=RUN/args.name;out.mkdir(parents=True,exist_ok=False);(out/'resources_start.txt').write_text(resource_check(2400))
    source=ROOT/(args.source or cfg['policy_source']);modelpath=ROOT/(args.model or cfg['model_source']);checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    env=RealEnv(1);contact=Contacts(env);model=load_model(modelpath);policy=Policy(118).to('cuda');policy.load_state_dict(checkpoint['policy']);optimizer=torch.optim.Adam(policy.parameters(),lr=pcfg['learning_rate'],eps=1e-5);optimizer.load_state_dict(checkpoint['optimizer'])
    for group in optimizer.param_groups:group['lr']=pcfg['learning_rate']
    torch.manual_seed(args.seed+3);action_rng=torch.Generator(device='cuda').manual_seed(args.seed+1);update_rng=torch.Generator(device='cuda').manual_seed(args.seed+2)
    if args.resume_rng:
        torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu());action_rng.set_state(checkpoint['action_rng'].cpu());update_rng.set_state(checkpoint['update_rng'].cpu())
    actor_params=list(policy.actor.parameters());start_actor=[v.detach().clone() for v in actor_params]
    before_opt=float(next(iter(optimizer.state.values()))['step'])
    config=vars(args).copy();config.update(source=str(source.relative_to(ROOT)),model=str(modelpath.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),ppo=pcfg,model_horizon=4,model_batch=16,model_frozen=True,optimizer_step_before=before_opt,reset_distribution='50% phase zero + 50% uniform 0..41',only_method_difference='rho; no auxiliary term for P',reward_success_physics_unchanged=True,heldout_used=False)
    write_json(out/'config.json',config)
    initial=dict(np.load(RUN/'development_initials.npz'));evals=[];evaluation_steps=0
    if not args.skip_initial_eval:
        result=evaluate_policy(env,policy,out/f'evaluation/step_{args.offset:06d}',initial,contact);evals.append(dict(steps=args.offset,**result));evaluation_steps+=result['physical_controls']
    episode_id=0;sample=None;history=None;sample_step=0;episode_stats=[];all_new_steps=0;aux_predictions=aux_valid=diagnostic_predictions=0;last_seconds=2.;stop='step_budget';diagnostics=[];received_signal=None
    def request_stop(signum,frame):
        nonlocal received_signal
        received_signal=signum
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def reset_training():
        nonlocal history,sample,sample_step
        env.reset(torch.tensor([0],device='cuda'),random_phase=bool(torch.rand((),device='cuda')>=pcfg['start_probability']))
        history=env.ctrl[:,None,:].expand(1,3,26).clone();sample_step=0
        sample=EpisodeLog(out/'sample_episodes'/f'episode_{episode_id:06d}.npz',env,dict(kind='training_only',episode_id=episode_id,start_phase=int(env.phase[0]))) if episode_id%16==0 else None
    reset_training();obs=env.observe()
    def save_checkpoint(step,name):
        torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),additional_steps=args.offset+step,config=config,cuda_rng=torch.cuda.get_rng_state(),action_rng=action_rng.get_state(),update_rng=update_rng.get_state(),auxiliary_model_predictions=aux_predictions+checkpoint.get('auxiliary_model_predictions',0),diagnostic_model_predictions=diagnostic_predictions+checkpoint.get('diagnostic_model_predictions',0)),out/name)
    csvfile=(out/'training.csv').open('w',newline='');writer=None
    for update in range(1,args.steps//128+1):
        if received_signal is not None:stop='external_signal_'+str(received_signal);break
        if time.perf_counter()-start+last_seconds+40>args.seconds:stop='wall_budget';break
        t0=time.perf_counter();ob=[];ac=[];lp=[];vv=[];rr=[];dd=[];nv=[];ss=[];ns=[];uu=[];ph=[];la=[];parts=[];errors=[];ep_rows=[]
        for t in range(128):
            with torch.no_grad():
                state=physical_state(env,history);phase=env.phase.clone();last=env.last_action.clone()
                if update==1 and t<4:assert torch.allclose(observe_state(state,phase,last,env),obs,atol=1e-6,rtol=1e-5)
                dist=policy.distribution(obs);action=dist.loc+dist.scale*torch.randn(dist.loc.shape,device='cuda',generator=action_rng);logp=dist.log_prob(action).sum(-1);value=policy.value(obs)
                nextobs,reward,done,info=env.step(action);bootstrap=policy.value(nextobs)*(~info['failure'])
                control=env.ctrl.clone();history=torch.cat([history[:,1:],control[:,None]],1);nextstate=physical_state(env,history)
                reconstructed,component=reward_from_state(nextstate,env.ref[env.phase],action.clamp(-1,1),last)
                if update==1 and t<4:assert torch.allclose(reconstructed,reward,atol=2e-6,rtol=1e-5)
                if sample is not None:sample.add(env,action,int(phase[0]),reward,info,contact.read())
            ob.append(obs);ac.append(action);lp.append(logp);vv.append(value);rr.append(reward);dd.append(done);nv.append(bootstrap);ss.append(state[0]);ns.append(nextstate[0]);uu.append(control[0]);ph.append(phase[0]);la.append(last[0]);parts.append(component[0]);errors.append(torch.stack([info['pos_error'][0],info['rot_error'][0],(action.abs()>1).float().mean()]))
            if bool(done[0]):
                if not bool(info['finite'][0]):raise RuntimeError('Nonfinite real state invalidates training.')
                ep_rows.append([float(info['episode_return'][0]),int(info['episode_length'][0]),int(info['reason'][0])])
                if sample is not None:sample.save(int(info['reason'][0]),True)
                episode_id+=1;reset_training()
            obs=env.observe();all_new_steps+=1
        rewards=torch.stack(rr);values=torch.stack(vv);dones=torch.stack(dd);boot=torch.stack(nv);adv=torch.zeros_like(rewards);gae=torch.zeros(1,device='cuda')
        for t in reversed(range(128)):
            delta=rewards[t]+pcfg['gamma']*boot[t]-values[t];gae=delta+pcfg['gamma']*pcfg['gae_lambda']*(~dones[t])*gae;adv[t]=gae
        ret=(adv+values).flatten();rawadv=adv.flatten();normalized=(rawadv-rawadv.mean())/(rawadv.std()+1e-8)
        obsbuf=torch.cat(ob);actions=torch.cat(ac);oldlp=torch.cat(lp);oldval=values.flatten();states=torch.stack(ss);nextstates=torch.stack(ns);controls=torch.stack(uu);phases=torch.stack(ph);lasts=torch.stack(la);episode_stats.extend(ep_rows)
        epoch_rows=[];gdiag=dict(ppo_actor_norm=0.,model_actor_norm=0.,model_scaled_ratio=0.,gradient_cosine=0.,aux_lambda=0.,predicted_aux_return=0.)
        for epoch in range(pcfg['epochs']):
            ix=torch.randperm(128,device='cuda',generator=update_rng);dist=policy.distribution(obsbuf[ix]);newlp=dist.log_prob(actions[ix]).sum(-1);logratio=newlp-oldlp[ix];ratio=logratio.exp()
            ploss=torch.maximum(-normalized[ix]*ratio,-normalized[ix]*ratio.clamp(1-pcfg['clip'],1+pcfg['clip'])).mean()
            newv=policy.value(obsbuf[ix]);vloss=.5*torch.maximum((newv-ret[ix]).square(),(oldval[ix]+(newv-oldval[ix]).clamp(-pcfg['value_clip'],pcfg['value_clip'])-ret[ix]).square()).mean();entropy=dist.entropy().sum(-1).mean()
            loss=ploss+pcfg['value_weight']*vloss-pcfg['entropy']*entropy
            if args.rho>0 and epoch==pcfg['epochs']-1:
                mx=torch.arange(0,128,8,device='cuda');aux,imagined,count=model_policy_objective(policy,model,states[mx].detach(),phases[mx],lasts[mx].detach(),env)
                gp=torch.autograd.grad(loss,actor_params,retain_graph=True);gm=torch.autograd.grad(aux,actor_params,retain_graph=True)
                pn=torch.sqrt(sum(x.square().sum() for x in gp));mn=torch.sqrt(sum(x.square().sum() for x in gm));cos=sum((a*b).sum() for a,b in zip(gp,gm))/(pn*mn).clamp_min(1e-12);scale=(args.rho*pn/mn.clamp_min(1e-8)).clamp(max=100).detach()
                assert torch.isfinite(aux) and torch.isfinite(mn) and torch.isfinite(scale)
                loss=loss+scale*aux;aux_predictions+=16*4;aux_valid+=int(count)
                gdiag=dict(ppo_actor_norm=float(pn),model_actor_norm=float(mn),model_scaled_ratio=float(scale*mn/pn.clamp_min(1e-12)),gradient_cosine=float(cos),aux_lambda=float(scale),predicted_aux_return=float(imagined.mean().detach()))
            optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),pcfg['max_gradient_norm']);assert torch.isfinite(grad);optimizer.step()
            epoch_rows.append([float(ploss.detach()),float(vloss.detach()),float(entropy.detach()),float(((ratio-1)-logratio).mean().detach()),float(((ratio-1).abs()>.2).float().mean()),float(grad)])
        assert all(p.grad is None for p in model.parameters())
        # Recovery-only writes do not reset physics, consume RNG or change losses.
        # One completed PPO update is the durable accounting granularity.
        progress=out/'progress.tmp';write_json(progress,dict(training_interactions=all_new_steps,additional_steps=args.offset+all_new_steps,ppo_optimizer_updates=update*pcfg['epochs'],auxiliary_model_predictions=aux_predictions,process_seconds=time.perf_counter()-start));progress.replace(out/'progress.json')
        if all_new_steps%8192==0:save_checkpoint(all_new_steps,'recovery_latest.pt')
        if update==1 or update%16==0:
            md=model_diagnostics(model,states,nextstates,controls,actions,phases,lasts,rewards.flatten(),dones.flatten(),env);md.update(update=update+args.offset//128,steps=args.offset+all_new_steps);diagnostic_predictions+=md.get('predictions',0);diagnostics.append(md);write_json(out/'model_diagnostics.json',diagnostics)
        if update==1 or update%8==0:
            e=torch.stack(errors).mean(0).cpu().numpy();rp=torch.stack(parts).mean(0).cpu().numpy();d=np.array(epoch_rows).mean(0)
            row=dict(update=update+args.offset//128,steps=args.offset+all_new_steps,process_seconds=time.perf_counter()-start,mean_reward=float(rewards.mean()),position_mean_mm=e[0]*1000,rotation_mean_deg=e[1]*180/np.pi,action_clip_fraction=e[2],policy_loss=d[0],value_loss=d[1],entropy=d[2],approx_kl=d[3],ppo_clip_fraction=d[4],preclip_grad_norm=d[5],advantage_std=float(rawadv.std()),explained_variance=float(1-(ret-oldval).var()/ret.var().clamp_min(1e-8)),**gdiag)
            row.update({k:float(v) for k,v in zip(['reward_obj_pos','reward_obj_rot','reward_fingers','reward_wrist','reward_action','reward_action_change'],rp)})
            if writer is None:writer=csv.DictWriter(csvfile,fieldnames=row.keys());writer.writeheader()
            writer.writerow(row);csvfile.flush()
        last_seconds=time.perf_counter()-t0
        if all_new_steps%args.interval==0:
            if sample is not None:sample.save('checkpoint_boundary_partial',False);sample=None
            save_checkpoint(all_new_steps,f'step_{args.offset+all_new_steps:06d}.pt')
            result=evaluate_policy(env,policy,out/f'evaluation/step_{args.offset+all_new_steps:06d}',initial,contact);evals.append(dict(steps=args.offset+all_new_steps,**result));evaluation_steps+=result['physical_controls'];write_json(out/'development_curve.json',evals)
            print(json.dumps(dict(name=args.name,steps=args.offset+all_new_steps,dev_success=result['successes'],dev_return=result['mean_return'],dev_first_failure=result['median_first_failure_s'],elapsed=time.perf_counter()-start)),flush=True)
            if all_new_steps<args.steps:episode_id+=1;reset_training();obs=env.observe()
        if update%128==0:print('PROGRESS',args.name,all_new_steps,round(time.perf_counter()-start,2),flush=True)
    if sample is not None:sample.save('budget_end_partial',False)
    csvfile.close();save_checkpoint(all_new_steps,'checkpoint_final.pt')
    if not stop.startswith('external_signal_') and (not evals or evals[-1]['steps']!=args.offset+all_new_steps):
        result=evaluate_policy(env,policy,out/f'evaluation/step_{args.offset+all_new_steps:06d}',initial,contact);evals.append(dict(steps=args.offset+all_new_steps,**result));evaluation_steps+=result['physical_controls']
    write_json(out/'development_curve.json',evals);np.save(out/'training_episode_summaries.npy',np.array(episode_stats))
    elapsed=time.perf_counter()-start;write_json(out/'summary.json',dict(name=args.name,rho=args.rho,seed=args.seed,training_interactions=all_new_steps,evaluation_interactions=evaluation_steps,auxiliary_model_predictions=aux_predictions,auxiliary_valid_transitions=aux_valid,diagnostic_model_predictions=diagnostic_predictions,ppo_optimizer_updates=all_new_steps//128*pcfg['epochs'],model_optimizer_updates=0,optimizer_step_before=before_opt,optimizer_step_after=float(next(iter(optimizer.state.values()))['step']),actor_parameter_l2_change=float(torch.sqrt(sum((a-b).square().sum() for a,b in zip(actor_params,start_actor)))),stop_reason=stop,process_seconds=elapsed,training_controls_per_process_second=all_new_steps/elapsed,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,model_frozen=True,heldout_used=False))
    print('COMPLETE',args.name,all_new_steps,elapsed,flush=True)

if __name__=='__main__':main()
