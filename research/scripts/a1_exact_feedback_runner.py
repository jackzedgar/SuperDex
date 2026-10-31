"""Phase-aware inference only; original physical environments/reset/GAE inherited."""
import numpy as np
import torch
from a1_policy import policy_inputs
from a1_state import encode
from a1_formal_start_pool import FormalStartPoolCandidate
from a1_original_action_search_pool import WitnessPool
from a1_original_action_search_math import knot_bias

class ExactFormalRunner(FormalStartPoolCandidate):
 def inference(self):
  self.obs,self.goal=policy_inputs(self.task,self.state_history,self.action_history,self.phase)
  self.value_input=torch.cat([encode(self.state_history[:,-1],self.mask),self.goal,self.context,self.last_action],-1)
  self.action,self.logprob,self.value=self.policy.act(self.obs,self.goal,self.value_input,self.mask,self.phase)

class ExactWitnessPool(WitnessPool):
 @torch.no_grad()
 def inference(self):
  self.obs,self.goal=policy_inputs(self.task,self.state_history,self.action_history,self.phase)
  self.mean=self.policy.base.distribution(self.obs,self.goal).mean*self.mask
  self.bias=self.policy.phase_bias(self.phase)*self.mask
  self.action=self.policy.distribution(self.obs,self.goal,self.phase).mean*self.mask
 @torch.no_grad()
 def check_inputs(self,indices,knots):
  for i in indices:
   e=self.envs[i];o,g=policy_inputs(e.task,self.state_history[i:i+1],self.action_history[i:i+1],e.phase)
   inp=max(float((o-self.obs[i:i+1]).abs().max()),float((g-self.goal[i:i+1]).abs().max()));assert inp<=2e-5
   base=self.policy.base.distribution(o,g).mean*e.task.mask.repeat_interleave(26,-1)
   bias=torch.as_tensor(knot_bias(knots[i],int(e.phase),98,e.task.mask[0].cpu().numpy()),device='cuda',dtype=torch.float32)[None]
   teacher=(base+bias).clamp(-1,1);delta=float((teacher-self.action[i:i+1]).abs().max());assert delta<=1e-5
   direct=self.policy.distribution(o,g,e.phase).mean;assert float((direct-self.action[i:i+1]).abs().max())<=1e-5
   self.audit['max_teacher_action_error']=max(self.audit.get('max_teacher_action_error',0.),delta);self.audit['max_input_error']=max(self.audit['max_input_error'],inp);self.audit['input_checks']+=1
   assert torch.equal(self.state_history[i,-1],e.state()[0]) and torch.equal(self.action_history[i,-1],e.last_action[0])
 @torch.no_grad()
 def run(self,knots,seeds,should_stop,validate=False):
  assert len(seeds)==1 and seeds[0]==280044 and np.array_equal(np.asarray(knots[0],dtype=np.float32),self.policy.teacher_knots.cpu().numpy())
  self.checked_indices=[0];self.checked_knots=np.asarray(knots,dtype=np.float32)
  graph=self.graph;owner=self
  class CheckedGraph:
   def replay(self):
    graph.replay();owner.check_inputs([0],owner.checked_knots)
  self.graph=CheckedGraph()
  try:return super().run(knots,seeds,should_stop,validate=True)
  finally:self.graph=graph
