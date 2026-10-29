"""PPO-Sim collector learning entry. Shared source/physics/reward; independent of W0/W1."""
import argparse,json,time,signal,math,os
from pathlib import Path
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
from a1_state import encode
from a1_policy import CollectorPolicy,policy_inputs
from a1_reset import ResetSampler
stop=False
def onstop(*_):
 global stop;stop=True
def atomic_save(value,path):
 temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)

class Runner:
 def __init__(self,env,policy,seed,reset_mode='adaptive_anchor'):
  assert reset_mode in {'adaptive_anchor','uniform_reference'}
  self.reset_mode=reset_mode
  self.env=env;self.policy=policy;self.sampler=ResetSampler(env.task,seed);self.rng=np.random.default_rng(seed)
  self.state_history=torch.zeros(1,3,287,device=env.device);self.action_history=torch.zeros(1,3,52,device=env.device)
  self.episodes=[];self.bootstrap_controls=0;self.real_controls=0;self.ep_rewards=[];self.ep_narrow=[];self.pending=[]
  self.reset()
  with torch.no_grad():
   self.inference()
   self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph,stream=env.torch_stream):self.inference()
 def reset(self):
  selected=self.sampler.choose() if self.reset_mode=='adaptive_anchor' else None
  if self.reset_mode=='uniform_reference':
   self.bootstrap=False;phase=int(self.rng.integers(self.env.task.length-1));self.bin=phase//25;snap=None
  elif selected is None:
   self.bootstrap=True;self.bin=int(self.rng.integers(self.sampler.nbins));phase=min(self.bin*25,self.env.task.length-2);snap=None
  else:self.bootstrap=False;self.bin,phase,snap=selected
  self.env.reset(phase,seed=int(self.rng.integers(2**31)),randomize=True,snapshot=snap)
  self.state_history.copy_(self.env.state()[:,None].expand(-1,3,-1));self.action_history.zero_()
  self.ep_rewards=[];self.ep_narrow=[];self.pending=[(0,self.env.snapshot())]
 def inference(self):
  e=self.env
  self.obs,self.goal=policy_inputs(e.task,self.state_history,self.action_history,e.phase)
  self.value_input=torch.cat([encode(self.state_history[:,-1],e.task.mask),self.goal,e.context,e.last_action],-1)
  self.action,self.logprob,self.value=self.policy.act(self.obs,self.goal,self.value_input,e.task.mask)
 def advance(self):
  e=self.env;s,r,done,info=e.step(self.action);self.real_controls+=1
  if self.bootstrap:self.bootstrap_controls+=1
  self.state_history.copy_(torch.cat([self.state_history[:,1:],s[:,None]],1));self.action_history.copy_(torch.cat([self.action_history[:,1:],e.last_action[:,None]],1))
  self.ep_rewards.append(float(r));self.ep_narrow.append(float(info['narrow_reward']))
  if len(self.ep_rewards)%25==0:self.pending.append((len(self.ep_rewards),e.snapshot()))
  if bool(done):
   for index,snap in self.pending:
    b=int(snap['phase'])//25
    if index+25<=len(self.ep_rewards) and b not in self.sampler.anchors:
     pool=self.sampler.fallback.setdefault(b,[])
     if len(pool)<8:pool.append(snap)
     else:pool[int(self.rng.integers(8))]=snap
   self.sampler.update(self.bin,self.ep_narrow)
   self.episodes.append(dict(task=e.task.meta['task_id'],controls=len(self.ep_rewards),start_phase=e.start_phase,bootstrap=self.bootstrap,return_sum=float(sum(self.ep_rewards)),failure_reason=int(info['reason']),regular_success=bool(info['regular_success']),strict_success=bool(info['strict_success']),mean_errors=info['mean_errors'].cpu().tolist()[0]))
   self.reset()
  return r.clone(),done.clone()
 @torch.no_grad()
 def rollout(self,n):
  obs=[];goals=[];critic=[];actions=[];logprob=[];values=[];rewards=[];dones=[]
  for _ in range(n):
   if stop:break
   self.graph.replay()
   obs.append(self.obs.clone());goals.append(self.goal.clone());critic.append(self.value_input.clone());actions.append(self.action.clone());logprob.append(self.logprob.clone());values.append(self.value.clone())
   r,d=self.advance();rewards.append(r);dones.append(d)
  self.graph.replay();last=self.value.clone()
  data={k:torch.cat(v) for k,v in [('obs',obs),('goal',goals),('value_input',critic),('action',actions),('old_logprob',logprob),('old_value',values),('reward',rewards),('done',dones)]}
  if not len(rewards):return None
  adv=torch.zeros_like(data['reward']);carry=torch.zeros_like(last);nextv=last
  for t in reversed(range(len(rewards))):
   alive=1-data['done'][t].float();delta=data['reward'][t]+.99*nextv*alive-data['old_value'][t]
   carry=delta+.99*.95*alive*carry;adv[t]=carry;nextv=data['old_value'][t]
  data['advantage']=adv;data['return']=adv+data['old_value'];data['mask']=self.env.task.mask.expand(len(adv),2).clone()
  return data

def ppo_update(policy,opt,data,generator,lr):
 for group in opt.param_groups:group['lr']=lr
 adv=data['advantage'];adv=(adv-adv.mean())/adv.std().clamp_min(1e-6);n=len(adv);rows=[]
 for epoch in range(5):
  order=torch.randperm(n,generator=generator,device=adv.device)
  for ids in order.chunk(4):
   dist=policy.distribution(data['obs'][ids],data['goal'][ids]);active=data['mask'][ids].repeat_interleave(26,-1)
   logp=(dist.log_prob(data['action'][ids])*active).sum(-1);logratio=logp-data['old_logprob'][ids];ratio=logratio.exp()
   loss_actor=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
   value=policy.value(data['value_input'][ids]);clipped=data['old_value'][ids]+(value-data['old_value'][ids]).clamp(-.2,.2)
   loss_value=.5*torch.maximum((value-data['return'][ids]).square(),(clipped-data['return'][ids]).square()).mean()
   entropy=(dist.entropy()*active).sum(-1).mean();loss=loss_actor+.5*loss_value-.001*entropy
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
   opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
   assert torch.isfinite(norm)
   torch.nn.utils.clip_grad_value_(policy.parameters(),.1);opt.step()
   rows.append([float(loss_actor),float(loss_value),float(entropy),float(norm),float(((ratio-1)-logratio).mean()),float((abs(ratio-1)>.2).float().mean())])
 return dict(zip(['actor_loss','value_loss','entropy','preclip_gradient_norm','approx_kl','clip_fraction'],np.array(rows).mean(0).tolist()))

@torch.no_grad()
def evaluate(env,policy,seeds):
 rows=[]
 for seed in seeds:
  errors=[];terms=[];contacts=[]
  env.reset(0,seed=int(seed),randomize=True);sh=env.state()[:,None].repeat(1,3,1);ah=torch.zeros(1,3,52,device=env.device);firstfail=None;rr=0.
  for step in range(env.task.length-1):
   obs,g=policy_inputs(env.task,sh,ah,env.phase);a=policy.distribution(obs,g).mean*env.task.mask.repeat_interleave(26,-1)
   s,r,done,info=env.step(a);rr+=float(r)
   errors.append([float(info['pos_error']),float(info['rot_error'])]);terms.append(info['terms'].cpu().tolist()[0]);contacts.append(env.contact_summary.cpu().tolist())
   if firstfail is None and (float(info['pos_error'])>.015 or float(info['rot_error'])>torch.pi/12):firstfail=(step+1)*.02
   sh=torch.cat([sh[:,1:],s[:,None]],1);ah=torch.cat([ah[:,1:],env.last_action[:,None]],1)
   if bool(done):break
  rows.append(dict(seed=int(seed),controls=step+1,regular_success=bool(info['regular_success']),strict_success=bool(info['strict_success']),failure_reason=int(info['reason']),first_strict_failure_s=firstfail,return_sum=rr,mean_errors=info['mean_errors'].cpu().tolist()[0],error_trace_m_rad=errors,reward_terms_mean=np.mean(terms,axis=0).tolist(),contact_summary_mean=np.mean(contacts,axis=0).tolist()))
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--steps',type=int,default=8192);p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=280043);a=p.parse_args()
 signal.signal(signal.SIGTERM,onstop);signal.signal(signal.SIGINT,onstop);torch.set_num_threads(2);torch.manual_seed(a.seed)
 out=RUN/a.run;out.mkdir(exist_ok=False);env=A1Env(ROOT/'data/a1_processed/tasks'/a.task);policy=CollectorPolicy(a.seed).to(env.device)
 opt=torch.optim.Adam(policy.parameters(),lr=2e-4,betas=(.9,.95));gen=torch.Generator(device=env.device).manual_seed(a.seed+1)
 runner=Runner(env,policy,a.seed);start=time.monotonic();history=[];update=0;seeds=[280046+i for i in range(4)]
 write(out/'configuration.json',dict(scope='Full-size PPO-Sim entry development; not frozenProbe40 collector or A1 data',task=a.task,steps_cap=a.steps,seed=a.seed,architecture='Enc/Dec2048,1024,512,256 latent32; critic256,128',optimizer='Adam2e-4 beta.9,.95 cosine0.03 floor at80percent',rollout=512,epochs=5,minibatches=4,gamma=.99,gae_lambda=.95,ppo_clip=.2,entropy_weight=.001,perception='Ideal simulation; no uncalibrated real-camera corruption',bootstrap='Only when no available anchors/replay states; uniform reference attempts explicitly tagged; replay admissible after25survivingcontrols',development_eval_seeds=seeds))
 initial=evaluate(env,policy,seeds);write(out/'evaluation_000000.json',initial);runner.reset()
 while runner.real_controls<a.steps and not stop:
  data=runner.rollout(min(512,a.steps-runner.real_controls))
  if data is None:break
  update+=1;fraction=min(1,runner.real_controls/(a.steps*.8));lr=2e-4*(.03+.97*.5*(1+math.cos(math.pi*fraction)))
  diag=ppo_update(policy,opt,data,gen,lr)
  row=dict(update=update,real_controls=runner.real_controls,elapsed_seconds=time.monotonic()-start,bootstrap_controls=runner.bootstrap_controls,mean_rollout_reward=float(data['reward'].mean()),action_saturation_fraction=float((data['action'].abs()>1).float().mean()),learning_rate=lr,**diag)
  history.append(row);print(json.dumps(row),flush=True);write(out/'learning_curve.json',history)
  if runner.real_controls%2048==0 or runner.real_controls>=a.steps:
   atomic_save(dict(policy=policy.state_dict(),optimizer=opt.state_dict(),controls=runner.real_controls,updates=update,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler=runner.sampler.state_dict()),out/f'checkpoint_{runner.real_controls:07d}.pt')
   evaluation=evaluate(env,policy,seeds);write(out/f'evaluation_{runner.real_controls:06d}.json',evaluation);runner.reset()
 atomic_save(dict(policy=policy.state_dict(),optimizer=opt.state_dict(),controls=runner.real_controls,updates=update,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),sampler=runner.sampler.state_dict()),out/'checkpoint_final.pt')
 write(out/'episodes.json',runner.episodes);write(out/'summary.json',dict(status='stopped' if stop else 'finished',real_training_controls=runner.real_controls,updates=update,bootstrap_controls=runner.bootstrap_controls,training_seconds=time.monotonic()-start,episodes=len(runner.episodes),peak_torch_bytes=torch.cuda.max_memory_allocated(),formal_A1=False))
if __name__=='__main__':main()
