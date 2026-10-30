"""Bounded PPO smoke experiment on one real clip. Not SuperDex G1."""
from pathlib import Path
import argparse,json,time,csv,random
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from real_env import RealEnv,ROOT,DATA,GPU


class Policy(nn.Module):
    def __init__(self,obs_dim):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,26))
        self.critic=nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        for net in [self.actor,self.critic]:
            for layer in net:
                if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.actor[-1].weight);nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.logstd=nn.Parameter(torch.full((26,),-1.6))

    def distribution(self,obs):return Normal(self.actor(obs),self.logstd.exp())
    def value(self,obs):return self.critic(obs).squeeze(-1)


@torch.no_grad()
def _evaluate_once(env,policy,out,label,eval_seed=20261026):
    # Same deterministic perturbation seed before/after; all episodes start at t=0.
    torch.manual_seed(eval_seed)
    ids=torch.arange(env.n,device=env.device);env.reset(ids,False,noise=True)
    active=torch.ones(env.n,dtype=torch.bool,device=env.device)
    episodes=[];qs=[];vs=[];actions=[];controls=[];phases=[];rewards=[];errors=[];obslog=[]
    first_done=False
    for _ in range(env.length-1):
        obs=env.observe();action=policy.actor(obs);phase=env.phase.clone()
        before=env.qpos[0].clone();beforev=env.qvel[0].clone()
        _,reward,done,info=env.step(action)
        if not first_done:
            qs.append(before.cpu().numpy());vs.append(beforev.cpu().numpy());actions.append(action[0].cpu().numpy());controls.append(env.ctrl[0].cpu().numpy());phases.append(phase[0].item());rewards.append(reward[0].item());errors.append([info['pos_error'][0].item(),info['rot_error'][0].item()]);obslog.append(obs[0].cpu().numpy())
            final=env.qpos[0].cpu().numpy().copy()
            first_done=bool(done[0])
        ended=(done&active).nonzero().flatten()
        for i in ended.tolist():
            episodes.append(dict(env=i,return_=info['episode_return'][i].item(),length=info['episode_length'][i].item(),mean_position_m=info['mean_pos'][i].item(),mean_rotation_deg=float(np.rad2deg(info['mean_rot'][i].item())),final_position_m=info['pos_error'][i].item(),final_rotation_deg=float(np.rad2deg(info['rot_error'][i].item())),success=bool(info['success'][i]),reason=int(info['reason'][i])))
        active[ended]=False
        if not active.any():break
    assert len(episodes)==env.n
    report=dict(label=label,episodes=episodes,n=env.n,successes=sum(e['success'] for e in episodes),
                return_mean=float(np.mean([e['return_'] for e in episodes])),episode_length_mean=float(np.mean([e['length'] for e in episodes])),
                object_position_mean_m=float(np.mean([e['mean_position_m'] for e in episodes])),object_rotation_mean_deg=float(np.mean([e['mean_rotation_deg'] for e in episodes])),
                failure_counts={name:sum(e['reason']==i for e in episodes) for i,name in enumerate(['horizon','position_gt_15cm','rotation_gt_80deg','joint_limit','nonfinite'])},
                success_rule='Full 51 controls from source frame 118; EVERY control pos <= 3 cm and rotation <= 30 degrees; no failure. This is a pilot criterion, not final benchmark success.',
                caveat=f'Same clip and seed as development; {env.n} small initial perturbations, not independent tasks or training seeds. A horizon episode may still violate the success thresholds.')
    (out/f'evaluation_{label}.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/f'rollout_{label}.npz',qpos=np.array(qs),qvel=np.array(vs),raw_action=np.array(actions),ctrl=np.array(controls),phase=np.array(phases),reward=rewards,errors=errors,observation=np.array(obslog),final_qpos=final)
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
    return report


def evaluate(env,policy,out,label):
    reports=[_evaluate_once(env,policy,out,label if i==0 else f'{label}_{i:02d}',20261026+i) for i in range(8)]
    result=reports[0].copy();episodes=[dict(e,trial=i) for i,r in enumerate(reports) for e in r['episodes']]
    result.update(episodes=episodes,n=len(episodes),successes=sum(e['success'] for e in episodes),
                  return_mean=float(np.mean([e['return_'] for e in episodes])),episode_length_mean=float(np.mean([e['length'] for e in episodes])),
                  object_position_mean_m=float(np.mean([e['mean_position_m'] for e in episodes])),object_rotation_mean_deg=float(np.mean([e['mean_rotation_deg'] for e in episodes])),
                  failure_counts={name:sum(e['reason']==i for e in episodes) for i,name in enumerate(['horizon','position_gt_15cm','rotation_gt_80deg','joint_limit','nonfinite'])},
                  caveat='8 sequential episodes in ONE world, paired small perturbation seeds before/after. Same development clip; not independent tasks or training seeds.')
    (out/f'evaluation_{label}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=160);p.add_argument('--worlds',type=int,default=1);p.add_argument('--seconds',type=int,default=900);p.add_argument('--seed',type=int,default=7);p.add_argument('--output',type=Path,default=ROOT/'runs/real_track_20261026/ppo');args=p.parse_args()
    assert 1<=args.updates<=500 and 1<=args.worlds<=128 and 30<=args.seconds<=1800
    repair_gate=ROOT/'configs/grasp_repair_gate.json'
    if repair_gate.exists():
        gate=json.loads(repair_gate.read_text())
        assert gate['ready_for_bounded_ppo'],gate['reason']
    assert json.loads((ROOT/'runs/real_track_20261026/qualification_gpu.json').read_text())['ready_for_bounded_ppo']
    assert args.worlds==1,'Only the single-world pilot passed the complete consistency qualification.'
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    env=RealEnv(args.worlds);policy=Policy(env.obs_dim).to(env.device);optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5)
    config=dict(seed=args.seed,worlds=args.worlds,max_updates=args.updates,max_training_wall_seconds=args.seconds,rollout_steps=128,epochs=4,minibatch=128,learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip=.2,value_coef=.5,entropy_coef=.001,grad_norm=.5,
                obs_dim=env.obs_dim,act_dim=26,action='Unbounded diagonal Normal recorded in log-prob; environment clips sampled action to [-1,1]. PPO ratios use the original raw action.',
                action_scale=env.scale.cpu().tolist(),reset='reference-state initialization at random phases, at least 10 controls remaining; evaluation always phase 0',
                bootstrap='Time-limit truncations bootstrap final observation; true failures do not. GAE recursion stops at every reset.',
                reward='dt*(5 exp(-(position/0.04)^2)+2 exp(-(angle/0.5)^2)+0.5 exp(-finger_mse/0.1)+0.5 exp(-wrist_squared_error/0.001)-0.03 mean(action^2)-0.05 mean(delta_action^2))',
                gpu_uuid=GPU,source_trajectory=str(DATA/'trajectory.npz'),scope='single-clip PPO pilot; no world model or actor model-gradient update',physics_dt=.002,control_dt=.02,
                hardware_assumptions='40 g object; isotropic friction 1; virtual 6 DOF wrist; no hand self-collision or hand-floor collision; actuator force ranges from original URDF')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    before=evaluate(env,policy,out,'before')
    torch.manual_seed(args.seed)
    env.reset(torch.arange(env.n,device=env.device),True)
    obs=env.observe();start=time.perf_counter();global_steps=0;completed=[];last_loss={};saved_first=False
    csvfile=(out/'training.csv').open('w',newline='');writer=None
    for update in range(1,args.updates+1):
        if time.perf_counter()-start>args.seconds:break
        obsbuf=[];actbuf=[];logbuf=[];valbuf=[];rewbuf=[];donebuf=[];nextvalbuf=[];ended_records=[]
        diagnostic=[]
        for step in range(128):
            with torch.no_grad():
                dist=policy.distribution(obs);act=dist.sample();logp=dist.log_prob(act).sum(-1);value=policy.value(obs)
                nextobs,reward,done,info=env.step(act)
                if not info['finite'].all():raise RuntimeError('Nonfinite physical state; training aborted')
                nv=policy.value(nextobs)*(~info['failure'])
            obsbuf.append(obs);actbuf.append(act);logbuf.append(logp);valbuf.append(value);rewbuf.append(reward);donebuf.append(done);nextvalbuf.append(nv)
            diagnostic.append(torch.stack([info['pos_error'].mean(),info['rot_error'].mean(),reward.mean()]))
            ended=done.nonzero().flatten()
            if len(ended):
                rec=torch.stack([info['episode_return'][ended],info['episode_length'][ended],info['mean_pos'][ended],info['mean_rot'][ended],info['reason'][ended]],dim=-1).cpu().numpy()
                ended_records.extend(rec.tolist());env.reset(ended,True)
            obs=env.observe();global_steps+=env.n
        rewards=torch.stack(rewbuf);values=torch.stack(valbuf);dones=torch.stack(donebuf);nextvalues=torch.stack(nextvalbuf)
        advantages=torch.zeros_like(rewards);gae=torch.zeros(env.n,device=env.device)
        for t in reversed(range(128)):
            delta=rewards[t]+.99*nextvalues[t]-values[t]
            gae=delta+.99*.95*(~dones[t])*gae;advantages[t]=gae
        returns=advantages+values
        ob=torch.stack(obsbuf).reshape(-1,env.obs_dim);ac=torch.stack(actbuf).reshape(-1,26);oldlp=torch.stack(logbuf).flatten();oldval=values.flatten();rt=returns.flatten();ad=advantages.flatten();ad=(ad-ad.mean())/(ad.std()+1e-8)
        losses=[]
        for epoch in range(4):
            for ix in torch.randperm(len(ob),device=env.device).split(128):
                dist=policy.distribution(ob[ix]);newlp=dist.log_prob(ac[ix]).sum(-1);ratio=(newlp-oldlp[ix]).exp()
                ploss=torch.maximum(-ad[ix]*ratio,-ad[ix]*ratio.clamp(.8,1.2)).mean()
                value=policy.value(ob[ix]);vloss=.5*torch.maximum((value-rt[ix]).square(),(oldval[ix]+(value-oldval[ix]).clamp(-.2,.2)-rt[ix]).square()).mean()
                entropy=dist.entropy().sum(-1).mean();loss=ploss+.5*vloss-.001*entropy
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();grad=nn.utils.clip_grad_norm_(policy.parameters(),.5)
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite policy gradient')
                optimizer.step()
                losses.append([ploss.item(),vloss.item(),entropy.item(),((ratio-1)-(newlp-oldlp[ix])).mean().item(),grad.item()])
        diag=torch.stack(diagnostic).mean(0).cpu().numpy();avg=np.mean(losses,axis=0);ep=np.array(ended_records)
        row=dict(update=update,environment_steps=global_steps,seconds=time.perf_counter()-start,mean_step_reward=float(diag[2]),object_position_mean_m=float(diag[0]),object_rotation_mean_deg=float(np.rad2deg(diag[1])),episodes=len(ep),episode_return=float(ep[:,0].mean()) if len(ep) else None,episode_length=float(ep[:,1].mean()) if len(ep) else None,policy_loss=avg[0],value_loss=avg[1],entropy=avg[2],approx_kl=avg[3],gradient_norm=avg[4])
        if writer is None:writer=csv.DictWriter(csvfile,fieldnames=list(row));writer.writeheader()
        writer.writerow(row);csvfile.flush();completed.extend(ended_records)
        if update%10==0:print(json.dumps(row),flush=True)
        if update%40==0:torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'update':update,'config':config},out/'checkpoint.pt')
    csvfile.close();elapsed=time.perf_counter()-start
    torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'update':row['update'],'config':config},out/'checkpoint.pt')
    after=evaluate(env,policy,out,'after')
    np.save(out/'training_episodes.npy',np.array(completed))
    summary=dict(status='bounded_run_completed',updates=row['update'],environment_steps=global_steps,training_seconds=elapsed,stop_reason='update_budget' if row['update']==args.updates else 'wall_time_budget',
                 before={k:v for k,v in before.items() if k!='episodes'},after={k:v for k,v in after.items() if k!='episodes'},
                 peak_torch_memory_mb=torch.cuda.max_memory_allocated()/1e6,not_D0=True,not_G1=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
