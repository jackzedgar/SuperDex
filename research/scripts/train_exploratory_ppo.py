"""Matched continuation from the original checkpoint: only reset sampling differs."""
import argparse,time,csv,copy,json,hashlib
import numpy as np,torch
from real_env import RealEnv
from train_real_ppo import Policy
from exploration_common import RUN,ROOT,DATA,GPU,write_json,read_json,activate,resource_check,Contacts,EpisodeLog,finish_timing


@torch.no_grad()
def evaluate(env,policy,out,contacts,step):
    activate(env);initial=np.load(ROOT/'runs/grasp_repair_20261026/fixed_initials.npz');reports=[]
    for i in range(8):
        env.reset(torch.tensor([0],device=env.device),False);env.qpos[0]=torch.tensor(initial['qpos'][i],device=env.device);env.qvel[0]=torch.tensor(initial['qvel'][i],device=env.device)
        log=EpisodeLog(out/f'eval_{step:06d}'/f'trial_{i:02d}.npz',env,dict(kind='fixed_evaluation',trial=i,additional_training_steps=step))
        first=None
        for k in range(51):
            a=policy.actor(env.observe());_,r,done,info=env.step(a);c=contacts.read();log.add(env,a,k,r,info,c)
            if first is None and (float(info['pos_error'][0])>.03 or float(info['rot_error'][0])>np.pi/6):first=(k+1)*.02
            if done[0]:break
        reason=int(info['reason'][0]);log.save(reason,True)
        reports.append(dict(trial=i,steps=k+1,return_=float(info['episode_return'][0]),success=bool(info['success'][0]),first_strict_failure_s=first,termination_s=(k+1)*.02 if reason else None,reason=reason,position_mean_m=float(info['mean_pos'][0]),rotation_mean_deg=float(np.rad2deg(float(info['mean_rot'][0])))))
    result=dict(additional_steps=step,successes=sum(r['success'] for r in reports),n=8,mean_return=float(np.mean([r['return_'] for r in reports])),mean_length=float(np.mean([r['steps'] for r in reports])),trials=reports)
    write_json(out/f'eval_{step:06d}'/'summary.json',result);print('EVAL',json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True);return result


def main():
    started=time.perf_counter();p=argparse.ArgumentParser();p.add_argument('--arm',choices=['uniform_rsi','start50'],required=True);p.add_argument('--steps',type=int,default=40960);p.add_argument('--seconds',type=float,default=1200);p.add_argument('--trend',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);out=RUN/args.arm;out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check())
    assert args.steps<=(81920 if args.trend else 40960) and args.steps%128==0 and args.seconds<=(1030 if args.trend else 1200)
    config=read_json(ROOT/'configs'/('learning_trend_budget.json' if args.trend else 'exploration_budget.json'));start_prob=config['ppo']['arms'][args.arm];seed=config['ppo']['seed'];base_steps=61440 if args.trend else 20480;interval=20480 if args.trend else 10240
    source=ROOT/config['source_run']/args.arm/'checkpoint_final.pt' if args.trend else ROOT/'runs/real_track_20261026/ppo/checkpoint.pt';ckpt=torch.load(source,map_location='cpu',weights_only=True)
    if args.trend:
        assert RUN.resolve()!=(ROOT/config['source_run']).resolve() and ckpt['additional_steps']==40960
        assert ckpt['config']['start_probability']==start_prob and ckpt['config']['seed']==seed
    env=RealEnv(1);ec=Contacts(env);ev=RealEnv(1);vc=Contacts(ev);assert env.qpos.data_ptr()!=ev.qpos.data_ptr();activate(env)
    policy=Policy(118).to('cuda:0');policy.load_state_dict(ckpt['policy']);optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5);optimizer.load_state_dict(ckpt['optimizer'])
    torch.manual_seed(seed);np.random.seed(seed)
    if args.trend:torch.cuda.set_rng_state(ckpt['cuda_rng'].cpu())
    steps_before=[float(v['step']) for v in optimizer.state.values()];assert steps_before and min(steps_before)==max(steps_before)
    write_json(out/'config.json',dict(arm=args.arm,start_probability=start_prob,seed=seed,additional_steps_cap=args.steps,process_seconds_cap=args.seconds,source_checkpoint=str(source),source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_updates=base_steps//128,source_training_steps=base_steps,checkpoint_interval=interval,optimizer_step_before=min(steps_before),cuda_rng_restored=args.trend,continuation_boundary='Fresh environment reset at checkpoint continuation; saved policy, optimizer and (trend mode) CUDA RNG restored.',changed_factor='reset mixture only',original_training_config=ckpt['config'],active_physics_worlds=1,evaluation='Separate single-world data and shared device, executed serially; exact old eight qpos/qvel tensors.'))
    evals=[evaluate(ev,policy,out,vc,0)];activate(env)
    episode_id=0
    def reset():
        phase_zero=bool(torch.rand((),device=env.device)<start_prob)
        env.reset(torch.tensor([0],device=env.device),random_phase=not phase_zero)
        return EpisodeLog(out/'episodes'/f'episode_{episode_id:06d}.npz',env,dict(kind='training',arm=args.arm,episode_id=episode_id,start_phase=int(env.phase[0]),start_branch=phase_zero))
    log=reset();obs=env.observe();nstep=0;writer=None;csvfile=(out/'training.csv').open('w',newline='');last_update_seconds=4.;stop='step_budget';eval_seconds=0.;completed=[]
    for update in range(1,args.steps//128+1):
        if time.perf_counter()-started+last_update_seconds+15>args.seconds:stop='wall_budget';break
        t0=time.perf_counter();ob=[];ac=[];lp=[];val=[];rewards=[];dones=[];nvs=[];diag=[];rparts=[];episode_rows=[]
        for t in range(128):
            with torch.no_grad():
                dist=policy.distribution(obs);a=dist.sample();logp=dist.log_prob(a).sum(-1);v=policy.value(obs);phase=int(env.phase[0]);last=env.last_action.clone()
                nextobs,r,done,info=env.step(a)
                if not bool(info['finite'].all()):raise RuntimeError('Nonfinite physical state invalidates this PPO run.')
                nv=policy.value(nextobs)*(~info['failure']);c=ec.read();log.add(env,a,phase,r,info,c)
                ref=env.ref[env.phase];ep=info['pos_error'];er=info['rot_error'];eq=(env.qpos[:,6:26]-ref[:,6:26]).square().mean(-1);ew=(env.qpos[:,:3]-ref[:,:3]).square().sum(-1);clipped=a.clamp(-1,1)
                parts=.02*torch.stack([5*torch.exp(-(ep/.04)**2),2*torch.exp(-(er/.5)**2),.5*torch.exp(-eq/.1),.5*torch.exp(-ew/.001),-.03*clipped.square().mean(-1),-.05*(clipped-last).square().mean(-1)],-1)
                assert torch.allclose(parts.sum(-1),r,atol=1e-6,rtol=1e-5)
            ob.append(obs);ac.append(a);lp.append(logp);val.append(v);rewards.append(r);dones.append(done);nvs.append(nv);rparts.append(parts[0].cpu().numpy());diag.append([float(ep[0]),float(er[0]),float(r[0]),float((a.abs()>1).float().mean()),c[0]>0])
            if done[0]:
                reason=int(info['reason'][0]);log.save(reason,True);episode_rows.append([float(info['episode_return'][0]),int(info['episode_length'][0]),reason]);episode_id+=1;log=reset()
            obs=env.observe();nstep+=1
        rr=torch.stack(rewards);vv=torch.stack(val);dd=torch.stack(dones);nn=torch.stack(nvs);adv=torch.zeros_like(rr);gae=torch.zeros(1,device=env.device)
        for t in reversed(range(128)):delta=rr[t]+.99*nn[t]-vv[t];gae=delta+.99*.95*(~dd[t])*gae;adv[t]=gae
        returns=adv+vv;obsbuf=torch.stack(ob).reshape(-1,118);actions=torch.stack(ac).reshape(-1,26);oldlp=torch.stack(lp).flatten();oldval=vv.flatten();rt=returns.flatten();raw_adv=adv.flatten();ad=(raw_adv-raw_adv.mean())/(raw_adv.std()+1e-8)
        diagnostics=[]
        for epoch in range(4):
            ix=torch.randperm(128,device=env.device);dist=policy.distribution(obsbuf[ix]);newlp=dist.log_prob(actions[ix]).sum(-1);lr=newlp-oldlp[ix];ratio=lr.exp();ploss=torch.maximum(-ad[ix]*ratio,-ad[ix]*ratio.clamp(.8,1.2)).mean()
            vnew=policy.value(obsbuf[ix]);vloss=.5*torch.maximum((vnew-rt[ix]).square(),(oldval[ix]+(vnew-oldval[ix]).clamp(-.2,.2)-rt[ix]).square()).mean();entropy=dist.entropy().sum(-1).mean();loss=ploss+.5*vloss-.001*entropy
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);assert torch.isfinite(grad);optimizer.step()
            diagnostics.append([float(ploss.detach()),float(vloss.detach()),float(entropy.detach()),float(((ratio-1)-lr).mean().detach()),float(grad),float(((ratio-1).abs()>.2).float().mean())])
        x=np.array(diag);d=np.mean(diagnostics,0);pr=np.mean(rparts,0);completed.extend(episode_rows)
        ev_before=float(1-(rt-oldval).var()/rt.var().clamp_min(1e-8));row=dict(update=update,additional_steps=nstep,total_steps_with_original=base_steps+nstep,process_seconds=time.perf_counter()-started,episode_return=float(np.mean([e[0] for e in episode_rows])) if episode_rows else None,episode_length=float(np.mean([e[1] for e in episode_rows])) if episode_rows else None,episodes=len(episode_rows),mean_reward=x[:,2].mean(),position_mean_m=x[:,0].mean(),rotation_mean_deg=np.rad2deg(x[:,1].mean()),contact_fraction=x[:,4].mean(),action_clip_fraction=x[:,3].mean(),policy_loss=d[0],value_loss=d[1],entropy=d[2],approx_kl=d[3],preclip_grad_norm=d[4],ppo_clip_fraction=d[5],explained_variance_before_update=ev_before,advantage_mean=float(raw_adv.mean()),advantage_std=float(raw_adv.std()))
        row.update({name:float(value) for name,value in zip(['reward_obj_pos','reward_obj_rot','reward_fingers','reward_wrist','reward_action','reward_action_change'],pr)})
        if writer is None:writer=csv.DictWriter(csvfile,fieldnames=row.keys());writer.writeheader()
        writer.writerow(row);csvfile.flush();last_update_seconds=time.perf_counter()-t0
        if update%20==0:print(json.dumps(row),flush=True)
        if nstep%interval==0:
            torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),additional_steps=nstep,config=read_json(out/'config.json'),cuda_rng=torch.cuda.get_rng_state()),out/f'checkpoint_{nstep:06d}.pt')
            et=time.perf_counter();evals.append(evaluate(ev,policy,out,vc,nstep));eval_seconds+=time.perf_counter()-et;activate(env)
    log.save('budget_end_partial_episode',False);csvfile.close()
    torch.save(dict(policy=policy.state_dict(),optimizer=optimizer.state_dict(),additional_steps=nstep,config=read_json(out/'config.json'),cuda_rng=torch.cuda.get_rng_state()),out/'checkpoint_final.pt')
    if evals[-1]['additional_steps']!=nstep:et=time.perf_counter();evals.append(evaluate(ev,policy,out,vc,nstep));eval_seconds+=time.perf_counter()-et
    seconds=finish_timing('ppo_'+args.arm,started,dict(additional_steps=nstep,steps_per_process_second=nstep/(time.perf_counter()-started),evaluation_seconds=eval_seconds,stop_reason=stop,seed=seed,peak_torch_memory_mb=torch.cuda.max_memory_allocated()/2**20))
    write_json(out/'summary.json',dict(arm=args.arm,additional_steps=nstep,total_steps_with_original=base_steps+nstep,process_seconds=seconds,stop_reason=stop,completed_training_episodes=episode_id,evaluations=evals,no_new_physics_or_reward_change=True));print('COMPLETE',args.arm,nstep,seconds,flush=True)


if __name__=='__main__':main()
