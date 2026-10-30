"""Full A1 single physical world, both hand slots, causal controls and episode DR."""
import os,time,json,copy
import numpy as np,torch,mujoco,mujoco_warp as mjw,warp as wp
from runtime_config import GPU
from a1_task_runtime import TaskTensors,states_from_coordinates,reward_metrics
from a1_state import H,analytic_control,empty_state,matrix_euler,euler_quaternion,world_vec,local_vec
from world_model_mini import qmatrix,qmul,qconj,qlog
from a1_world_model import CONTEXT,control_context
from a1_contacts import ContactDiagnostics

def near_euler(r,reference):
 a=matrix_euler(r);b=torch.stack([a[...,0]+torch.pi,torch.pi-a[...,1],a[...,2]+torch.pi],-1)
 a=a+2*torch.pi*torch.round((reference-a)/(2*torch.pi));b=b+2*torch.pi*torch.round((reference-b)/(2*torch.pi))
 return torch.where(((a-reference).square().sum(-1)<(b-reference).square().sum(-1))[...,None],a,b)

class A1Env:
 def __init__(self,task_path,capture=True,determinism='constraint',*,allow_training_velocity_override=False):
  self.allow_training_velocity_override=bool(allow_training_velocity_override)
  assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU and torch.cuda.device_count()==1
  assert str(torch.cuda.get_device_properties(0).uuid)==GPU[4:]
  self.device='cuda:0';self.n=1;self.physical_controls=0
  wp.init();wp.set_device(self.device)
  self.torch_stream=torch.cuda.Stream();torch.cuda.set_stream(self.torch_stream);self.stream=wp.stream_from_torch(self.torch_stream);wp.set_stream(self.stream)
  self.task=TaskTensors(task_path,self.device);t=self.task;self.m=t.m;m=self.m
  self.nominal={name:getattr(m,name).copy() for name in ['body_mass','body_inertia','body_ipos','body_iquat','geom_friction','actuator_gainprm','actuator_biasprm']}
  self.object_body=m.body('object').id;self.hand_geom=np.flatnonzero(m.geom_contype==1);self.floor_geom=m.geom('floor').id
  d=mujoco.MjData(m);d.qpos[:]=t.raw['qpos'][0];d.qvel[:]=t.raw['qvel'][0];mujoco.mj_forward(m,d)
  assert determinism in {'constraint','all'}
  self.wm=mjw.put_model(m);self.wm.opt.graph_conditional=False;self.wm.opt.deterministic=mjw.DeterminismType[determinism.upper()]
  self.determinism=determinism
  self.wd=mjw.put_data(m,d,nworld=1,nconmax=512,njmax=1536)
  self.qpos=wp.to_torch(self.wd.qpos);self.qvel=wp.to_torch(self.wd.qvel);self.ctrl=wp.to_torch(self.wd.ctrl)
  self.context=torch.zeros(1,CONTEXT,device=self.device)
  self.history=torch.zeros(1,2,3,26,device=self.device);self.servo=torch.zeros(1,2,6,device=self.device)
  self.phase=torch.zeros(1,dtype=torch.long,device=self.device);self.episode_start=self.phase.clone()
  self.hold_mode=torch.zeros(1,1,device=self.device,dtype=torch.bool);self.hold_command=torch.zeros(1,2,26,device=self.device)
  self.contacts=ContactDiagnostics(self)
  self.actions=torch.zeros(1,52,device=self.device);self.last_action=torch.zeros_like(self.actions);self.requested=torch.zeros_like(self.actions);self.applied=torch.zeros_like(self.actions)
  self.sums=torch.zeros(1,6,device=self.device);self.count=torch.zeros(1,device=self.device);self.all_strict=torch.ones(1,dtype=torch.bool,device=self.device)
  self.lo=torch.tensor(m.actuator_ctrlrange[:,0],device=self.device,dtype=torch.float32);self.hi=torch.tensor(m.actuator_ctrlrange[:,1],device=self.device,dtype=torch.float32)
  self.reset(0,seed=0,randomize=False);start=time.monotonic()
  self._advance();self.physical_controls+=1;wp.synchronize()
  if capture:
   self.graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph,stream=self.torch_stream,capture_error_mode='thread_local'):
    with wp.ScopedCapture(stream=self.stream,external=True,capture_mode=wp.CaptureMode.THREAD_LOCAL) as cap:
     self._advance()
   self.warp_capture=cap
  else:self.graph=None
  self.compile_seconds=time.monotonic()-start;self.reset(0,seed=0,randomize=False)
 def state(self):
  return states_from_coordinates(self.qpos,self.qvel,self.task.sides,self.task.base_p,self.task.base_q,self.history,self.servo)
 def _local(self,u):
  out=[];t=self.task
  for k,side in enumerate(t.sides):
   h=['right','left'].index(side);a=u[:,h]
   pos=local_vec(a[:,:3]-t.base_p[k],qmatrix(t.base_q[k]))
   rot=qmatrix(qmul(qconj(t.base_q[k]).expand(len(a),4),euler_quaternion(a[:,3:6])))
   euler=near_euler(rot,self.qpos[:,k*26+3:k*26+6])
   out.append(torch.cat([pos,euler,a[:,6:]],-1))
  return torch.cat(out,-1)
 def _world(self,u):
  out=torch.zeros(1,2,26,device=self.device);t=self.task
  for k,side in enumerate(t.sides):
   h=['right','left'].index(side);a=u[:,k*26:k*26+26]
   out[:,h,:3]=t.base_p[k]+world_vec(a[:,:3],qmatrix(t.base_q[k]))
   out[:,h,3:6]=matrix_euler(qmatrix(qmul(t.base_q[k].expand(len(a),4),euler_quaternion(a[:,3:6]))))
   out[:,h,6:]=a[:,6:]
  return out
 def _advance(self):
  t=self.task;previous=self.state();action=self.actions.clamp(-1,1)*t.mask.repeat_interleave(26,-1)
  raw=t.base_command[self.phase].reshape(1,2,26)+action.reshape(1,2,26)*t.scale
  raw=torch.where(self.hold_mode[:,:,None],self.hold_command,raw)
  limited=self._local(raw).clamp(self.lo,self.hi);u=self._world(limited).flatten(1);self.requested.copy_(u)
  delay,tau,offset=control_context(self.context)
  applied,servo=analytic_control(previous,u,t.mask,delay,tau,offset)
  self.ctrl.copy_(self._local(applied).clamp(self.lo,self.hi));self.applied.copy_(self._world(self.ctrl).flatten(1))
  for _ in range(10):mjw.step(self.wm,self.wd)
  # Contact diagnostics belong to the endpoint state. This extra forward is part
  # of the registered A1 simulator configuration, not a CPU contact reconstruction.
  mjw.forward(self.wm,self.wd)
  self.contact_summary=self.contacts.step();self.contact_details=self.contacts.details
  self.servo.copy_(servo);self.history.copy_(torch.cat([self.history[:,:,1:],u.reshape(1,2,1,26)],2))
  self.phase.copy_((self.phase+(~self.hold_mode[:,0]).long()).clamp(max=t.length-1));self.current_state=self.state()
  self.reward,self.info=reward_metrics(self.current_state,t.ref[self.phase],action,self.last_action,t.mask,t.geom,t.ids,t.ref_keypoints[self.phase],t.ref_sdf[self.phase],t.hand_length,t.lower,t.upper)
  self.count+=1;self.sums+=torch.cat([self.info['pos_error'][:,None],self.info['rot_error'][:,None],self.info['keypoint_error'],self.info['tip_error']],-1)
  self.all_strict&=(self.info['pos_error']<=.015)&(self.info['rot_error']<=torch.pi/12)&self.info['finite']
  self.timeout=self.phase==t.length-1;self.done=self.info['failure']|self.timeout;mean=self.sums/self.count[:,None]
  hand_good=(((mean[:,2:4]<=.08)&(mean[:,4:6]<=.06))|(t.mask==0)).all(-1)
  complete=self.timeout&~self.info['failure']&(self.episode_start==0)
  self.info['regular_success']=complete&(mean[:,0]<=.03)&(mean[:,1]<=torch.pi/6)&hand_good
  self.info['strict_success']=complete&(mean[:,0]<=.015)&(mean[:,1]<=torch.pi/12)&hand_good
  self.info['all_frames_strict']=complete&self.all_strict
  self.info['mean_errors']=mean;self.last_action.copy_(action)
 @torch.no_grad()
 def step(self,action):
  self.physical_controls+=1
  self.actions.copy_(action)
  if self.graph is None:self._advance()
  else:self.graph.replay()
  return self.current_state,self.reward,self.done,self.info
 def randomize(self,seed,enabled=True):
  rng=np.random.default_rng(seed);m=self.m;t=self.task
  for name,value in self.nominal.items():getattr(m,name)[:]=value
  handfric=float(rng.uniform(.4,1.2)) if enabled else .8;tablefric=float(rng.uniform(.3,.9)) if enabled else .6
  m.geom_friction[self.hand_geom,0]=handfric;m.geom_friction[self.floor_geom,0]=tablefric
  ob=self.object_body
  massmult=float(rng.uniform(.7,1.4)) if enabled else 1.
  m.body_mass[ob]*=massmult;m.body_inertia[ob]*=massmult
  if enabled:
   m.body_ipos[ob]+=rng.uniform(-.01,.01,3);m.body_inertia[ob]*=rng.uniform(.7,1.3,3)
  # Positive rigid-body inertia must satisfy triangle inequalities. Independent
  # diagonal scaling can violate them: reject this context draw, never repair
  # silently. Rejection is deterministic and recorded.
  rejects=0
  while m.body_inertia[ob].max()>=m.body_inertia[ob].sum()/2:
   rejects+=1;assert rejects<100000,'No physically admissible inertia draw within bounded rejection budget'
   m.body_inertia[ob]=self.nominal['body_inertia'][ob]*massmult*rng.uniform(.7,1.3,3)
  context=np.zeros(CONTEXT,dtype=np.float32)
  for k,side in enumerate(t.sides):
   h=['right','left'].index(side);sl=slice(k*26+6,k*26+26)
   scale=float(rng.uniform(.9,1.1)) if enabled else 1.
   m.actuator_gainprm[sl,0]*=scale;m.actuator_biasprm[sl,1:3]*=scale
   context[h*68:h*68+20]=rng.uniform(-.01,.01,20) if enabled else 0
   context[h*68+20:h*68+40]=m.actuator_gainprm[sl,0];context[h*68+40:h*68+60]=-m.actuator_biasprm[sl,2]
   extra=int(rng.integers(2)) if enabled else 0
   context[h*68+60:h*68+62]=[1+extra,extra];context[h*68+62:h*68+68]=.02
  context[136:139]=[m.body_mass[ob],handfric,tablefric];context[139:142]=m.body_ipos[ob];context[142:145]=m.body_inertia[ob]
  context[145:148]=qlog(torch.tensor(m.body_iquat[ob])).numpy();context[148:]=t.bps.cpu().numpy()
  d=mujoco.MjData(m);mujoco.mj_setConst(m,d)
  for name in [*self.nominal,'body_subtreemass','body_invweight0','dof_invweight0']:
   if not hasattr(self.wm,name):continue
   target=wp.to_torch(getattr(self.wm,name));v=torch.as_tensor(getattr(m,name),device=self.device,dtype=target.dtype)
   target.copy_(v.expand_as(target))
  self.context.copy_(torch.tensor(context,device=self.device)[None]);self.context_record=dict(seed=seed,randomized=enabled,inertia_draw_rejections=rejects,context=context.tolist(),assumptions='Simulated, not measured',mass_scale=massmult)
 @torch.no_grad()
 def reset(self,phase=0,seed=0,randomize=True,snapshot=None,*,training_hand_qvel=None):
  if training_hand_qvel is not None:
   assert self.allow_training_velocity_override and snapshot is None,'Training velocity override requires explicit opt-in and cannot restore a snapshot'
   training_hand_qvel=torch.as_tensor(training_hand_qvel,device=self.device,dtype=self.qvel.dtype)
   assert training_hand_qvel.shape==(self.task.nu,) and bool(torch.isfinite(training_hand_qvel).all())
  self.hold_mode.fill_(False)
  self.randomize(seed,randomize);t=self.task;self.phase.fill_(phase);self.episode_start.fill_(phase);self.start_phase=int(phase)
  self.qpos.copy_(t.qpos[phase:phase+1]);self.qvel.copy_(t.qvel[phase:phase+1])
  if training_hand_qvel is not None:self.qvel[:,:t.nu].copy_(training_hand_qvel.reshape(1,t.nu))
  qworld=self._world(self.qpos[:,:t.nu]);self.servo.copy_(qworld[:,:,:6]);self.history.copy_(qworld[:,:,None].expand(-1,-1,3,-1))
  self.ctrl.copy_(self.qpos[:,:t.nu]);self.hold_command.copy_(qworld)
  if snapshot is not None:
   for name in ['qpos','qvel','ctrl']:getattr(self,name).copy_(torch.as_tensor(snapshot[name],device=self.device))
   self.history.copy_(torch.as_tensor(snapshot['history'],device=self.device));self.servo.copy_(torch.as_tensor(snapshot['servo'],device=self.device))
  for name in ['qacc_warmstart','qacc','qfrc_applied','xfrc_applied']:
   if hasattr(self.wd,name):wp.to_torch(getattr(self.wd,name)).zero_()
  wp.to_torch(self.wd.time).zero_();self.last_action.zero_();self.actions.zero_();self.sums.zero_();self.count.zero_();self.all_strict.fill_(True)
  mjw.forward(self.wm,self.wd)
  if hasattr(self,'current_state'):self.current_state.copy_(self.state())
  else:self.current_state=self.state()
  return self.current_state
 def snapshot(self):
  return {**{name:getattr(self,name).detach().cpu().numpy().copy() for name in ['qpos','qvel','ctrl','history','servo']},'qacc_warmstart':wp.to_torch(self.wd.qacc_warmstart).cpu().numpy().copy(),'phase':int(self.phase.item()),'context':self.context_record}
