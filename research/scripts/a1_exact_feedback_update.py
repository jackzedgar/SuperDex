"""Original PPO/evaluation functions, with explicit per-sample reference phase only."""
import numpy as np,torch
from a1_policy import policy_inputs
def ppo_update(policy,opt,data,generator,lr):
 for group in opt.param_groups:group['lr']=lr
 adv=data['advantage'];adv=(adv-adv.mean())/adv.std().clamp_min(1e-6);n=len(adv);rows=[]
 for epoch in range(5):
  order=torch.randperm(n,generator=generator,device=adv.device)
  for ids in order.chunk(4):
   dist=policy.distribution(data['obs'][ids],data['goal'][ids],data['phase'][ids]);active=data['mask'][ids].repeat_interleave(26,-1)
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
   obs,g=policy_inputs(env.task,sh,ah,env.phase);a=policy.distribution(obs,g,env.phase).mean*env.task.mask.repeat_interleave(26,-1)
   s,r,done,info=env.step(a);rr+=float(r)
   errors.append([float(info['pos_error']),float(info['rot_error'])]);terms.append(info['terms'].cpu().tolist()[0]);contacts.append(env.contact_summary.cpu().tolist())
   if firstfail is None and (float(info['pos_error'])>.015 or float(info['rot_error'])>torch.pi/12):firstfail=(step+1)*.02
   sh=torch.cat([sh[:,1:],s[:,None]],1);ah=torch.cat([ah[:,1:],env.last_action[:,None]],1)
   if bool(done):break
  rows.append(dict(seed=int(seed),controls=step+1,regular_success=bool(info['regular_success']),strict_success=bool(info['strict_success']),failure_reason=int(info['reason']),first_strict_failure_s=firstfail,return_sum=rr,mean_errors=info['mean_errors'].cpu().tolist()[0],error_trace_m_rad=errors,reward_terms_mean=np.mean(terms,axis=0).tolist(),contact_summary_mean=np.mean(contacts,axis=0).tolist()))
 return rows