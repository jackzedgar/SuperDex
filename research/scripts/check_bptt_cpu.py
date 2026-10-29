"""Focused CPU checks for newly introduced objective boundaries/Jacobians."""
import torch
from types import SimpleNamespace
from train_real_ppo import Policy
from bptt_common import imagine,survival
from world_model_mini import compose
class Toy(torch.nn.Module):
 def forward(self,s,u):
  d=torch.zeros(len(s),64);d[:,:26]=(u-s[:,:26])*.1
  return compose(s,d,u)
p=Policy(118);p.critic.requires_grad_(False);m=Toy()
q=torch.zeros(52,33);q[:,29]=1
env=SimpleNamespace(ref=q,target=torch.zeros(52,26),length=52,lo=torch.ones(26)*-2,hi=torch.ones(26)*2,scale=torch.ones(26)*.1)
s=torch.zeros(2,143);s[:,29]=1;s[:,:3]=.02;last=torch.zeros(2,26)
loss,o,t,diag=imagine(p,m,s,torch.tensor([0,50]),last,env,32)
loss.backward()
assert all(x.grad is None for x in p.critic.parameters())
assert all(torch.isfinite(x.grad).all() for x in p.actor.parameters())
assert sum(float(x.grad.abs().sum()) for x in p.actor.parameters())>0
assert diag['imagined_transitions']==33
with torch.no_grad():
 a=imagine(p,m,s[1:],torch.tensor([50]),last[1:],env,32)[0]
 p.critic[-1].bias.add_(100)
 b=imagine(p,m,s[1:],torch.tensor([50]),last[1:],env,32)[0]
 assert torch.allclose(a,b,atol=1e-7),'Reference end incorrectly bootstraps'
 c=imagine(p,m,s[:1],torch.tensor([0]),last[:1],env,8)[0]
 p.critic[-1].bias.add_(100)
 d=imagine(p,m,s[:1],torch.tensor([0]),last[:1],env,8)[0]
 assert (c-d).abs()>1,'Sampling cutoff incorrectly treated terminal'
print('PASS finite actor Jacobian; frozen critic; end reward retained; no terminal bootstrap; sampling-cutoff bootstrap; valid transition count')
