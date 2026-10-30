"""Shared full-A1 task tensors and exact documented reward; historical pilots untouched."""
import json
from pathlib import Path
import numpy as np,torch,mujoco
from a1_state import *
from world_model_mini import qmatrix,qmul,qconj,qlog
from a1_data_sources import ROOT
from a1_geometry import A1Geometry
from a1_portable_model import load_task_model

def states_from_coordinates(qp,qv,sides,base_position,base_quaternion,history=None,servo=None):
 n=qp.shape[0];s=empty_state(n,device=qp.device,dtype=qp.dtype);axes=torch.eye(3,device=qp.device,dtype=qp.dtype)
 for k,side in enumerate(sides):
  h=['right','left'].index(side);p=qp[:,k*26:k*26+26];v=qv[:,k*26:k*26+26]
  rb=qmatrix(base_quaternion[k]);rot=euler_quaternion(p[:,3:6]);q=qmul(base_quaternion[k].expand(n,4),rot)
  rx=qmatrix(qexp(p[:,3,None]*axes[0]));ry=qmatrix(qexp(p[:,4,None]*axes[1]))
  local_omega=v[:,3,None]*axes[0]+world_vec(v[:,4,None]*axes[1],rx)+world_vec(v[:,5,None]*axes[2],rx@ry)
  dyn=torch.cat([p[:,6:],v[:,6:],base_position[k]+world_vec(p[:,:3],rb),q,world_vec(v[:,:3],rb),world_vec(local_omega,rb)],-1)
  s[:,h*H:h*H+53]=dyn
  if history is not None:s[:,h*H+53:h*H+131]=history[:,h].flatten(1)
  if servo is not None:s[:,h*H+131:h*H+137]=servo[:,h]
 o=len(sides)*26
 s[:,274:281]=qp[:,o:o+7];s[:,281:284]=qv[:,o:o+3]
 s[:,284:287]=world_vec(qv[:,o+3:o+6],qmatrix(qp[:,o+3:o+7]))
 return s

def angle_error(a,b):
 return torch.linalg.vector_norm(qlog(qmul(qconj(b),a)),dim=-1)
def kernel(error_squared,sigma,narrow=False):
 z=error_squared/(sigma*sigma)
 return torch.exp(-z) if narrow else .7*torch.exp(-z)+.3*torch.exp(-z/25)
def reward_metrics(s,ref,action,last_action,mask,geom,ids,ref_keypoints,ref_sdf,hand_length,lower,upper,narrow=False):
 hs=s[:,:274].reshape(-1,2,H);rh=ref[:,:274].reshape(-1,2,H)
 op2=(s[:,274:277]-ref[:,274:277]).square().sum(-1);oa=angle_error(s[:,277:281],ref[:,277:281])
 ov2=(s[:,281:284]-ref[:,281:284]).square().sum(-1);ow2=(s[:,284:287]-ref[:,284:287]).square().sum(-1)
 wp2=(hs[...,P]-rh[...,P]).square().sum(-1);wa=angle_error(hs[...,R],rh[...,R])
 points=geom.points(s,keypoints=True)
 kpdist=torch.linalg.vector_norm(points-ref_keypoints,dim=-1);kp=kpdist.mean(-1);tip=kpdist[...,:5].mean(-1)
 def hands(x):return (x*mask).sum(-1)/mask.sum(-1).clamp_min(1)
 distances,_=geom.sdf(points[...,:5,:].flatten(1,2),ids);distances=distances.reshape(-1,2,5)
 gate=(ref_sdf<.01)*mask[...,None]
 contact=kernel((distances-ref_sdf.clamp_min(0)).square(),.01,narrow)
 nc=gate.sum((-1,-2));ct=torch.where(nc>0,(contact*gate).sum((-1,-2))/nc.clamp_min(1),torch.ones_like(nc,dtype=s.dtype))
 mid=(lower+upper)/2;half=(upper-lower)*.45
 limit=(torch.relu(mid-half-hs[...,:20])+torch.relu(hs[...,:20]-mid-half)).sum(-1)
 smooth=((action-last_action).reshape(-1,2,26).square().sum(-1)*mask).sum(-1)
 terms=torch.stack([2*kernel(op2,.02,narrow),kernel(oa.square(),.3,narrow),.15*kernel(ov2,.25,narrow),.15*kernel(ow2,3.14,narrow),.25*hands(kernel(wp2,.03,narrow)),.25*hands(kernel(wa.square(),.3,narrow)),hands(kernel(kp.square(),.1*hand_length,narrow)),ct,-.1*smooth,-2*hands(limit)],-1)*.02
 ncontact=kernel((distances-ref_sdf.clamp_min(0)).square(),.01,True)
 nct=torch.where(nc>0,(ncontact*gate).sum((-1,-2))/nc.clamp_min(1),torch.ones_like(nc,dtype=s.dtype))
 narrow_positive=2*kernel(op2,.02,True)+kernel(oa.square(),.3,True)+.15*kernel(ov2,.25,True)+.15*kernel(ow2,3.14,True)+.25*hands(kernel(wp2,.03,True))+.25*hands(kernel(wa.square(),.3,True))+hands(kernel(kp.square(),.1*hand_length,True))+nct
 narrow_reward=.02*(narrow_positive-.1*smooth-2*hands(limit))
 finite=torch.isfinite(s).all(-1)
 ep=op2.sqrt();ew=torch.where(mask>0,wp2.sqrt(),torch.zeros_like(wp2)).max(-1).values
 failure=(ep>.06)|(oa>.8)|(ew>.1)|~finite
 reason=torch.where(~finite,4,torch.where(ep>.06,1,torch.where(oa>.8,2,torch.where(ew>.1,3,0))))
 return terms.sum(-1),dict(terms=terms,narrow_reward=narrow_reward,pos_error=ep,rot_error=oa,wrist_error=ew,keypoint_error=kp,tip_error=tip,failure=failure,reason=reason,finite=finite)

class TaskTensors:
 def __init__(self,path,device='cuda'):
  self.path=Path(path);self.meta=json.loads((self.path/'qualification.json').read_text());self.sides=self.meta['hands']
  self.m=load_task_model(self.path);m=self.m
  self.device=device;self.nu=m.nu;self.n=len(self.sides);self.mask=torch.zeros(1,2,device=device)
  for side in self.sides:self.mask[0,['right','left'].index(side)]=1
  self.raw=dict(np.load(self.path/'reference.npz'))
  self.qpos=torch.as_tensor(self.raw['qpos'],dtype=torch.float32,device=device);self.qvel=torch.as_tensor(self.raw['qvel'],dtype=torch.float32,device=device)
  self.length=len(self.qpos)
  self.base_p=torch.tensor(np.array([m.body_pos[m.body(side+'_wrist').id] for side in self.sides]),dtype=torch.float32,device=device)
  self.base_q=torch.tensor(np.array([m.body_quat[m.body(side+'_wrist').id] for side in self.sides]),dtype=torch.float32,device=device)
  self.ref=states_from_coordinates(self.qpos,self.qvel,self.sides,self.base_p,self.base_q)
  self.ids=torch.zeros(1,dtype=torch.long,device=device)
  key=self.meta['object']['object_key']
  kin=[dict(np.load(ROOT/f'data/a1_processed/assets/hands/{side}/kinematics.npz')) for side in ['right','left']]
  self.geom=A1Geometry(kin,[dict(np.load(ROOT/f'data/a1_geometry/objects/{key}/sdf.npz'))]).to(device)
  with torch.no_grad():
   self.ref_keypoints=self.geom.points(self.ref,keypoints=True)
   ds,_=self.geom.sdf(self.ref_keypoints[...,:5,:].flatten(1,2),self.ids.expand(self.length));self.ref_sdf=ds.reshape(-1,2,5)
  self.lower=torch.zeros(1,2,20,device=device);self.upper=torch.zeros_like(self.lower);self.scale=torch.zeros(1,2,26,device=device)
  self.hand_length=torch.zeros(1,2,device=device)
  for k,side in enumerate(self.sides):
   h=['right','left'].index(side);sl=slice(k*26+6,k*26+26)
   self.lower[0,h]=torch.tensor(m.jnt_range[sl,0],device=device);self.upper[0,h]=torch.tensor(m.jnt_range[sl,1],device=device)
   self.scale[0,h,:3]=.02;self.scale[0,h,3:6]=.2
   self.scale[0,h,6:]=torch.tensor(.25*np.max(abs(m.actuator_forcerange[sl]),axis=1)/m.actuator_gainprm[sl,0],device=device)
   zero=torch.zeros(1,20,device=device);tips=self.geom.hands[h](zero)[:,:5];self.hand_length[0,h]=torch.linalg.vector_norm(tips,dim=-1).max()
  self.hand_length=self.hand_length.clamp_min(.01)
  self.base_command=self._reference_commands()
  bps=np.load(ROOT/'data/a1_geometry/bps_pca128.npz');i=list(bps['object_keys']).index(key);self.bps=torch.tensor(bps['object_codes'][i],device=device)
 def _reference_commands(self):
  # Nominal velocity feedforward plus causal wrist lag/delay inverse; no learned
  # outcomes are used. Full known reference allows advance by nominal delay1.
  m=self.m;n=self.length;out=torch.zeros(n,2,26,device=self.device)
  q=self.qpos[:,:self.nu];v=self.qvel[:,:self.nu]
  kv=torch.tensor(-m.actuator_biasprm[:,2]/m.actuator_gainprm[:,0],dtype=q.dtype,device=self.device)
  desired=q+kv*v
  lo=torch.tensor(m.actuator_ctrlrange[:,0],device=q.device,dtype=q.dtype);hi=torch.tensor(m.actuator_ctrlrange[:,1],device=q.device,dtype=q.dtype)
  desired=desired.clamp(lo,hi)
  for k,side in enumerate(self.sides):
   h=['right','left'].index(side);local=desired[:,k*26:k*26+26]
   worldp=self.base_p[k]+world_vec(local[:,:3],qmatrix(self.base_q[k]))
   worldq=qmul(self.base_q[k].expand(n,4),euler_quaternion(local[:,3:6]))
   one=(torch.arange(n,device=q.device)+1).clamp(max=n-1);two=(one+1).clamp(max=n-1)
   out[:,h,:3]=worldp[two]+(worldp[two]-worldp[one])
   advanced=qmul(worldq[one],qexp(2*qlog(qmul(qconj(worldq[one]),worldq[two]))))
   out[:,h,3:6]=matrix_euler(qmatrix(advanced));out[:,h,6:]=local[one,6:]
  return out.flatten(1)
