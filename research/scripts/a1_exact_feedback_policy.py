"""Twelfth authorized candidate: frozen teacher plus bounded residual actor."""
import numpy as np
import torch
from torch import nn
from a1_policy import CollectorPolicy
from a1_original_action_search_math import knot_times,torch_knot_bias

class ExactFeedbackPolicy(nn.Module):
 def __init__(self,parent_state,knots,arm,seed=280055):
  super().__init__();assert arm in ('candidate','direct');self.arm=arm
  self.base=CollectorPolicy(280043);self.base.load_state_dict(parent_state)
  for name,p in self.base.named_parameters():p.requires_grad_(name.startswith('critic.'))
  torch.manual_seed(seed)
  self.residual=nn.Sequential(nn.Linear(578,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,52))
  nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
  k=torch.as_tensor(np.asarray(knots,dtype=np.float32)).clone();assert k.shape==(4,52) and bool(torch.isfinite(k).all())
  if arm=='direct':k.zero_()
  self.register_buffer('teacher_knots',k);self.register_buffer('teacher_times',torch.tensor(knot_times(98),dtype=torch.float32));self.register_buffer('fixed_std',torch.full((52,),.03))
 def phase_bias(self,phase):
  assert phase.ndim==1
  return torch_knot_bias(self.teacher_knots[None].expand(len(phase),-1,-1),phase,self.teacher_times,torch.ones(len(phase),52,device=phase.device))
 def components(self,obs,goal,phase):
  z=torch.tanh(self.base.encoder(goal));x=torch.cat([z,obs],-1);base=torch.tanh(self.base.decoder(x));res=.1*torch.tanh(self.residual(x))
  return base,self.phase_bias(phase),res
 def distribution(self,obs,goal,phase):
  base,bias,res=self.components(obs,goal,phase);mean=(base+bias+res).clamp(-1,1)
  return torch.distributions.Normal(mean,self.fixed_std.expand_as(mean),validate_args=False)
 def value(self,x):return self.base.value(x)
 def act(self,obs,goal,value_input,mask,phase):
  d=self.distribution(obs,goal,phase);a=d.mean+d.stddev*torch.randn_like(d.mean);active=mask.repeat_interleave(26,-1)
  return a*active,(d.log_prob(a)*active).sum(-1),self.value(value_input)
 def frozen_actor_unchanged(self,parent):
  return all(torch.equal(v.detach().cpu(),parent[k]) for k,v in self.base.state_dict().items() if not k.startswith('critic.'))

def setup_gate(rows,max_action_error):
 return bool(len(rows)==3 and np.isfinite(max_action_error) and max_action_error<=1e-5 and all(r['seed']==280044 and r['complete'] and r['controls']==98 and r['failure_reason']==0 and r['regular_success'] for r in rows))
