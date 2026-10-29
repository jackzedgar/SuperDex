"""Single-world matched P or primary-model-gradient S training."""
import argparse,time,json,signal,os,hashlib,faulthandler
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
from pathlib import Path
import numpy as np,torch
from real_env import RealEnv
from train_real_ppo import Policy
from train_core_policy import evaluate_policy
from exploration_common import Contacts,EpisodeLog
from core_policy_common import physical_state,observe_state,model_diagnostics
from probe_model_gradient import reward_from_state
from bptt_common import ROOT,RUN,write,load_wm,imagine,Replay,model_fit,model_valid
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--method',choices=['P','S'],required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--horizon',type=int,default=32);p.add_argument('--interval',type=int,default=65536);p.add_argument('--wm-interval',type=int,default=4096);p.add_argument('--wm-updates',type=int,default=25);p.add_argument('--eval-only',action='store_true');p.add_argument('--checkpoint');p.add_argument('--eval-set',default='new_initials.npz');args=p.parse_args()
    started=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=RUN/args.name;out.mkdir(parents=True,exist_ok=False);write(out/'config.json',vars(args))
    torch.manual_seed(args.seed);print("BUILD_ENV",flush=True);env=RealEnv(1);print("ENV_READY",env.compile_seconds,flush=True);contact=Contacts(env);policy=Policy(118).cuda()
    source=ROOT/(args.checkpoint or 'runs/explore_20261026/start50/checkpoint_final.pt')
    ck=torch.load(source,map_location='cpu',weights_only=False);policy.load_state_dict(ck['policy'])
    if args.eval_only:
        result=evaluate_policy(env,policy,out/'evaluation',dict(np.load(RUN/args.eval_set)),contact)
        write(out/'summary.json',dict(**result,seconds=time.time()-started,source=str(source)));return
    model,mopt,norm=load_wm();model.requires_grad_(False)
    popt=torch.optim.Adam(policy.parameters(),lr=1e-4,eps=1e-5)
    if args.method=='P':
        popt.load_state_dict(ck['optimizer'])
        for g in popt.param_groups:g['lr']=1e-4
    aopt=torch.optim.Adam(policy.actor.parameters(),lr=args.lr,betas=(.9,.95))
    vopt=torch.optim.Adam(policy.critic.parameters(),lr=1e-4,eps=1e-5)
    replay=Replay(args.seed+17);faulthandler.cancel_dump_traceback_later()
    action_rng=torch.Generator(device='cuda').manual_seed(args.seed+1)
    update_rng=torch.Generator(device='cuda').manual_seed(args.seed+2)
    torch.manual_seed(args.seed+3)
    counts=dict(real_controls=0,evaluation_controls=0,actor_updates=0,critic_updates=0,wm_updates=0,wm_training_samples=0,imagined_transitions=0,computed_model_transitions=0,diagnostic_predictions=0)
    times=dict(collect=0.,actor=0.,critic=0.,model_fit=0.,evaluation=0.)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    dev=dict(np.load(ROOT/'runs/core_iteration_20261027/development_initials.npz'));curve=[];episodes=[];wm_versions=[];training=(out/'training.jsonl').open('w');diags=(out/'model_diagnostics.jsonl').open('w')
    history=None;es=[];eu=[];ea=[];ep=[];sample=None;episode=0
    def reset():
        nonlocal history,es,eu,ea,ep,sample
        env.reset(torch.tensor([0],device='cuda'),bool(torch.rand((),device='cuda')>=.5))
        history=env.ctrl[:,None,:].expand(1,3,26).clone()
        es=[physical_state(env,history)[0]];eu=[];ea=[];ep=[]
        sample=EpisodeLog(out/'sample_episodes'/f'episode_{episode:06d}.npz',env,dict(kind='training_only',episode_id=episode,start_phase=int(env.phase[0]))) if episode%128==0 else None
    def checkpoint(name):
        data=dict(policy=policy.state_dict(),optimizer=popt.state_dict() if args.method=='P' else aopt.state_dict(),actor_optimizer=aopt.state_dict(),critic_optimizer=vopt.state_dict(),counts=counts,config=vars(args),additional_steps=counts['real_controls'],cuda_rng=torch.cuda.get_rng_state(),action_rng=action_rng.get_state(),update_rng=update_rng.get_state())
        tmp=out/(name+'.tmp');torch.save(data,tmp);tmp.replace(out/name)
    def evaluate(step):
        nonlocal episode
        t=time.time();checkpoint(f'step_{step:06d}.pt')
        result=evaluate_policy(env,policy,out/f'development/step_{step:06d}',dev,contact)
        counts['evaluation_controls']+=result['physical_controls'];times['evaluation']+=time.time()-t
        curve.append(dict(steps=step,process_seconds=time.time()-started,**result));write(out/'development_curve.json',curve)
        print(json.dumps(dict(name=args.name,step=step,success=result['successes'],length=result['mean_length'],time=time.time()-started)),flush=True)
        episode+=1;reset()
    evaluate(0)
    for update in range(1,args.steps//128+1):
        if stop[0]:break
        t0=time.time();ob=[];ac=[];lp=[];vv=[];rr=[];dd=[];nv=[];ss=[];ns=[];uu=[];ph=[];la=[];parts=[];errors=[]
        for t in range(128):
            with torch.no_grad():
                obs=env.observe();state=physical_state(env,history);phase=env.phase.clone();last=env.last_action.clone()
                dist=policy.distribution(obs)
                # Same inherited exploration scale initially; S keeps it frozen,
                # P learns logstd with its likelihood objective.
                action=dist.loc+dist.scale*torch.randn(dist.loc.shape,device='cuda',generator=action_rng)
                logp=dist.log_prob(action).sum(-1);value=policy.value(obs)
                nxt,r,done,info=env.step(action);boot=policy.value(nxt)*(~done)
                control=env.ctrl.clone();history=torch.cat([history[:,1:],control[:,None]],1);nextstate=physical_state(env,history)
                rc,rp=reward_from_state(nextstate,env.ref[env.phase],action.clamp(-1,1),last)
                if update==1 and t<4:
                    assert torch.allclose(rc,r,atol=2e-6,rtol=1e-5)
                    assert torch.allclose(observe_state(state,phase,last,env),obs,atol=1e-6)
                if sample is not None:sample.add(env,action,int(phase[0]),r,info,contact.read())
                es.append(nextstate[0]);eu.append(control[0]);ea.append(action[0].clamp(-1,1));ep.append(int(phase[0]))
            ob.append(obs);ac.append(action);lp.append(logp);vv.append(value);rr.append(r);dd.append(done);nv.append(boot);ss.append(state[0]);ns.append(nextstate[0]);uu.append(control[0]);ph.append(phase[0]);la.append(last[0]);parts.append(rp[0]);errors.append(torch.stack([info['pos_error'][0],info['rot_error'][0],(action.abs()>1).float().mean()]))
            if bool(done[0]):
                assert bool(info['finite'][0]),'Nonfinite physical state'
                replay.add(es,eu,ep,ea)
                episodes.append(dict(id=episode,return_=float(info['episode_return'][0]),steps=int(info['episode_length'][0]),start=int(env.episode_start[0]),reason=int(info['reason'][0]),success=bool(info['success'][0])))
                if sample is not None:sample.save(int(info['reason'][0]),True)
                episode+=1;reset()
            counts['real_controls']+=1
        torch.cuda.synchronize();times['collect']+=time.time()-t0
        states=torch.stack(ss);nextstates=torch.stack(ns);controls=torch.stack(uu);phases=torch.stack(ph);lasts=torch.stack(la);actions=torch.cat(ac);rewards=torch.cat(rr);dones=torch.cat(dd)
        row=dict(update=update,steps=counts['real_controls'],method=args.method,mean_reward=float(rewards.mean()),position_mm=float(torch.stack(errors)[:,0].mean()*1000),rotation_deg=float(torch.stack(errors)[:,1].mean()*180/np.pi),action_clip_fraction=float(torch.stack(errors)[:,2].mean()))
        row.update({k:float(v) for k,v in zip(['reward_pos','reward_rot','reward_fingers','reward_wrist','reward_action','reward_change'],torch.stack(parts).mean(0))})
        # New-policy block diagnostics always precede fitting that block.
        if update==1 or counts['real_controls']%args.wm_interval==0:
            md=model_diagnostics(model,states,nextstates,controls,actions,phases,lasts,rewards,dones,env)
            counts['diagnostic_predictions']+=md.get('predictions',0)
            validation={**model_valid(model,replay,8),**model_valid(model,replay,32)};counts['diagnostic_predictions']+=sum(v for k,v in validation.items() if k.endswith('model_predictions'))
            diags.write(json.dumps(dict(step=counts['real_controls'],before_fit=True,**md,**validation))+'\n');diags.flush()
        if args.method=='S' and counts['real_controls']%args.wm_interval==0:
            t0=time.time();fit=model_fit(model,mopt,replay,args.wm_updates)
            counts['wm_updates']+=fit['wm_updates'];counts['wm_training_samples']+=fit['wm_training_samples']
            torch.cuda.synchronize();times['model_fit']+=time.time()-t0;row.update(fit)
            version=counts['real_controls'];path=out/'models'/f'version_{version:06d}.pt';path.parent.mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(),optimizer=mopt.state_dict(),normalization=norm,counts=counts),path)
            wm_versions.append(dict(version=version,new_episode_ids=[e['id'] for e in replay.episodes if e['split']=='train'],validation_episode_ids=[e['id'] for e in replay.episodes if e['split']=='validation'],old_train_only=True,fit=fit))
            write(out/'model_versions.json',wm_versions)
        if args.method=='P':
            t0=time.time();values=torch.cat(vv);boot=torch.cat(nv);adv=torch.zeros_like(rewards);gae=0.
            for t in reversed(range(128)):
                gae=rewards[t]+.99*boot[t]-values[t]+.99*.95*(~dones[t])*gae;adv[t]=gae
            ret=adv+values;an=(adv-adv.mean())/(adv.std()+1e-8);obsbuf=torch.cat(ob);oldlp=torch.cat(lp);metrics=[]
            for epoch in range(4):
                ix=torch.randperm(128,device='cuda',generator=update_rng);dist=policy.distribution(obsbuf[ix]);newlp=dist.log_prob(actions[ix]).sum(-1);logratio=newlp-oldlp[ix];ratio=logratio.exp()
                ploss=torch.maximum(-an[ix]*ratio,-an[ix]*ratio.clamp(.8,1.2)).mean()
                value=policy.value(obsbuf[ix]);vloss=.5*torch.maximum((value-ret[ix]).square(),(values[ix]+(value-values[ix]).clamp(-.2,.2)-ret[ix]).square()).mean()
                loss=ploss+.5*vloss-.001*dist.entropy().sum(-1).mean();popt.zero_grad(set_to_none=True);loss.backward();g=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
                assert torch.isfinite(g);popt.step();metrics.append([float(g),float(vloss.detach()),float(((ratio-1)-logratio).mean().detach())])
            counts['actor_updates']+=4;counts['critic_updates']+=4;row.update(grad_norm=float(np.mean(metrics,0)[0]),value_loss=float(np.mean(metrics,0)[1]),approx_kl=float(np.mean(metrics,0)[2]),advantage_std=float(adv.std()))
            torch.cuda.synchronize();times['actor']+=time.time()-t0
        else:
            t0=time.time();policy.critic.requires_grad_(False);model.requires_grad_(False);ix=torch.arange(0,128,4,device='cuda')
            aopt.zero_grad(set_to_none=True)
            loss,imagobs,targets,imdiag=imagine(policy,model,states[ix].detach(),phases[ix],lasts[ix].detach(),env,args.horizon)
            assert torch.isfinite(loss) and torch.isfinite(targets).all()
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.)
            assert torch.isfinite(gn);assert all(x.grad is None for x in model.parameters())
            for x in policy.actor.parameters():x.grad.clamp_(-.1,.1)
            aopt.step();counts['actor_updates']+=1;counts['imagined_transitions']+=imdiag['imagined_transitions'];counts['computed_model_transitions']+=imdiag['computed_model_transitions']
            row.update(imdiag,grad_norm=float(gn),target_mean=float(targets.mean()),target_std=float(targets.std()))
            torch.cuda.synchronize();times['actor']+=time.time()-t0
            t0=time.time();policy.critic.requires_grad_(True);vl=[]
            # Fixed detached imagined TD(lambda) targets, unnormalized original-return units.
            for epoch in range(5):
                order=torch.randperm(len(targets),device='cuda',generator=update_rng)
                for ids in order.chunk(4):
                    if not len(ids):continue
                    v=policy.value(imagobs[ids]);lossv=.5*(v-targets[ids]).square().mean();vopt.zero_grad(set_to_none=True);lossv.backward();vg=torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.)
                    assert torch.isfinite(vg);vopt.step();vl.append(float(lossv.detach()));counts['critic_updates']+=1
            vopt.zero_grad(set_to_none=True);row['value_loss']=float(np.mean(vl))
            row['critic_real_td_rmse']=float((policy.value(torch.cat(ob)).detach()-(rewards+.99*torch.cat(nv))).square().mean().sqrt())
            torch.cuda.synchronize();times['critic']+=time.time()-t0
        row['process_seconds']=time.time()-started
        if update==1 or update%8==0:training.write(json.dumps(row)+'\n');training.flush()
        if update%16==0:write(out/'progress.json',dict(**counts,process_seconds=time.time()-started,times=times,latest=row))
        if counts['real_controls']%8192==0:checkpoint('recovery_latest.pt')
        if counts['real_controls']%args.interval==0:
            if sample is not None:sample.save('evaluation_boundary_partial',False)
            evaluate(counts['real_controls']);replay.save(out);write(out/'episodes.json',episodes)
        if update%128==0:print('PROGRESS',args.name,counts['real_controls'],round(time.time()-started,1),flush=True)
    checkpoint('checkpoint_final.pt');replay.save(out);write(out/'episodes.json',episodes)
    if not stop[0] and curve[-1]['steps']!=counts['real_controls']:evaluate(counts['real_controls'])
    write(out/'summary.json',dict(**counts,seconds=time.time()-started,times=times,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,stop='signal_budget' if stop[0] else 'interaction_target',episode_count=len(episodes),model_frozen_actor=True,critic_frozen_actor=True,model_horizon=args.horizon,initial_policy=str(source)))
    training.close();diags.close();print('COMPLETE',args.name,json.dumps(counts),flush=True)
if __name__=='__main__':main()

