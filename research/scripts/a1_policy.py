"""Reference-conditioned PPO-Sim collector network and causal observations for full A1."""
import torch
from torch import nn
from a1_state import H,P,R,V,W,yaw_frame,local_vec,mat6
from a1_task_runtime import states_from_coordinates
from world_model_mini import qmatrix
OFFSETS=[1,2,3,4,6,8,12,16,24,32]
GOAL_DIM=1000;OBS_DIM=546;VALUE_DIM=337+GOAL_DIM+276+52

def policy_inputs(task,state_history,action_history,phase):
 s=state_history[:,-1]
 if not hasattr(task,'policy_time_constants'):
  task.policy_time_constants=s.new_tensor([.04,.02,0.,1.,1.,1.]);task.policy_offsets=torch.tensor(OFFSETS,device=s.device)
 mask=task.mask.expand(len(s),-1);frame,origin=yaw_frame(s,mask)
 hands=state_history[:,:,:274].reshape(len(s),3,2,H);parts=[]
 for t in range(3):
  for h in range(2):
   a=hands[:,t,h]
   value=torch.cat([a[:,:20],a[:,20:40]*.1,local_vec(a[:,P]-origin,frame)*5,mat6(frame.transpose(-1,-2)@qmatrix(a[:,R])),local_vec(a[:,V],frame),local_vec(a[:,W],frame)*.1,action_history[:,t,h*26:(h+1)*26]],-1)
   parts.append(value*mask[:,h,None])
 for t in range(3):
  o=state_history[:,t,274:]
  for h in range(2):
   wrist=hands[:,-1,h];rw=qmatrix(wrist[:,R])
   value=torch.cat([local_vec(o[:,:3]-wrist[:,P],rw)*10,mat6(rw.transpose(-1,-2)@qmatrix(o[:,3:7]))],-1)
   parts.append(value*mask[:,h,None])
 parts.append(task.policy_time_constants.expand(len(s),6))
 obs=torch.cat(parts,-1).clamp(-20,20);assert obs.shape[-1]==OBS_DIM
 # All desired references are known targets, never future physical outcomes.
 offsets=task.policy_offsets;index=phase[:,None]+offsets;valid=index<task.length
 ref=task.ref[index.clamp(max=task.length-1)];rhand=ref[:,:,:274].reshape(len(s),10,2,H);obj=ref[:,:,274:]
 ds=task.ref_sdf[index.clamp(max=task.length-1)];goals=[]
 for t in range(10):
  for h in range(2):
   a=rhand[:,t,h];rw=qmatrix(a[:,R])
   value=torch.cat([local_vec(a[:,P]-origin,frame)*5,mat6(frame.transpose(-1,-2)@rw),a[:,:20],local_vec(obj[:,t,:3]-a[:,P],rw)*10,mat6(rw.transpose(-1,-2)@qmatrix(obj[:,t,3:7])),ds[:,t,h].clamp(-.003,.05)*100],-1)
   goals.append(value*mask[:,h,None]*valid[:,t,None])
 goals.extend([valid.to(s.dtype),mask,task.bps.expand(len(s),-1)])
 goal=torch.cat(goals,-1).clamp(-20,20);assert goal.shape[-1]==GOAL_DIM
 return obs,goal

def mlp(widths,final_gain=None):
 layers=[]
 for i,(a,b) in enumerate(zip(widths[:-1],widths[1:])):
  layer=nn.Linear(a,b);nn.init.orthogonal_(layer.weight,gain=2**.5);nn.init.zeros_(layer.bias);layers.append(layer)
  if i<len(widths)-2:layers.append(nn.SiLU())
 if final_gain is not None:
  nn.init.orthogonal_(layers[-1].weight,gain=final_gain);nn.init.zeros_(layers[-1].bias)
 return nn.Sequential(*layers)

class CollectorPolicy(nn.Module):
 def __init__(self,seed=280043):
  super().__init__();torch.manual_seed(seed)
  self.encoder=mlp([GOAL_DIM,2048,1024,512,256,32],.01)
  self.decoder=mlp([32+OBS_DIM,2048,1024,512,256,52],0.)
  self.critic=mlp([VALUE_DIM,256,128,1],1.)
  self.logstd=nn.Parameter(torch.full((52,),-.9))
 def distribution(self,obs,goal):
  z=torch.tanh(self.encoder(goal));mean=torch.tanh(self.decoder(torch.cat([z,obs],-1)))
  return torch.distributions.Normal(mean,self.logstd.clamp(-3.,.5).exp(),validate_args=False)
 def value(self,value_input):return self.critic(value_input).squeeze(-1)
 def act(self,obs,goal,value_input,mask,deterministic=False):
  dist=self.distribution(obs,goal);a=dist.mean if deterministic else dist.mean+dist.stddev*torch.randn_like(dist.mean)
  active=mask.repeat_interleave(26,-1)
  return a*active,(dist.log_prob(a)*active).sum(-1),self.value(value_input)
