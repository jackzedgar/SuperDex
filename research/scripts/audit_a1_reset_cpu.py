"""Diagnostic CPU replay only. No training edits, no dataset or success claims."""
import json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import numpy as np,torch,mujoco
torch.set_num_threads(1)
from a1_task_runtime import TaskTensors,states_from_coordinates,angle_error
from a1_state import analytic_control,euler_quaternion,matrix_euler,world_vec,local_vec
from world_model_mini import qmatrix,qmul,qconj

def near_euler(r,ref):
 a=matrix_euler(r);b=torch.stack([a[...,0]+torch.pi,torch.pi-a[...,1],a[...,2]+torch.pi],-1)
 a+=2*torch.pi*torch.round((ref-a)/(2*torch.pi));b+=2*torch.pi*torch.round((ref-b)/(2*torch.pi))
 return torch.where(((a-ref).square().sum(-1)<(b-ref).square().sum(-1))[...,None],a,b)
def world(t,u):
 out=torch.zeros(1,2,26)
 for k,side in enumerate(t.sides):
  h=['right','left'].index(side);a=u[:,k*26:k*26+26]
  out[:,h,:3]=t.base_p[k]+world_vec(a[:,:3],qmatrix(t.base_q[k]));out[:,h,3:6]=matrix_euler(qmatrix(qmul(t.base_q[k].expand(1,4),euler_quaternion(a[:,3:6]))));out[:,h,6:]=a[:,6:]
 return out
def local(t,u,qp):
 out=[]
 for k,side in enumerate(t.sides):
  h=['right','left'].index(side);a=u[:,h]
  pos=local_vec(a[:,:3]-t.base_p[k],qmatrix(t.base_q[k]));rot=qmatrix(qmul(qconj(t.base_q[k]).expand(1,4),euler_quaternion(a[:,3:6])))
  out.append(torch.cat([pos,near_euler(rot,qp[:,k*26+3:k*26+6]),a[:,6:]],-1))
 return torch.cat(out,-1)
def replay(t,mode):
 m=t.m;d=mujoco.MjData(m);d.qpos[:]=t.raw['qpos'][0];d.qvel[:]=t.raw['qvel'][0]
 lo=torch.tensor(m.actuator_ctrlrange[:,0],dtype=torch.float32);hi=torch.tensor(m.actuator_ctrlrange[:,1],dtype=torch.float32)
 qp=torch.tensor(d.qpos[None],dtype=torch.float32);qv=torch.tensor(d.qvel[None],dtype=torch.float32)
 memory=qp[:,:t.nu].clone()
 if mode=='velocity_consistent':memory+=torch.tensor(-m.actuator_biasprm[:,2]/m.actuator_gainprm[:,0],dtype=torch.float32)*qv[:,:t.nu]
 memory=memory.clamp(lo,hi);uw=world(t,memory);hist=uw[:,:,None].repeat(1,1,3,1);servo=uw[:,:,:6].clone();d.ctrl[:]=memory.numpy()[0]
 delay=torch.tensor([[[1,0],[1,0]]]);tau=torch.ones(1,2,6)*.02;zero=torch.zeros(1,2,20)
 mujoco.mj_forward(m,d);rows=[]
 for phase in range(min(8,t.length-1)):
  s=states_from_coordinates(qp,qv,t.sides,t.base_p,t.base_q,hist,servo)
  raw=t.base_command[phase].reshape(1,2,26);u=world(t,local(t,raw,qp).clamp(lo,hi)).flatten(1)
  applied,servo=analytic_control(s,u,t.mask,delay,tau,zero);ctrl=local(t,applied,qp).clamp(lo,hi);d.ctrl[:]=ctrl.numpy()[0]
  for _ in range(10):mujoco.mj_step(m,d)
  mujoco.mj_forward(m,d);hist=torch.cat([hist[:,:,1:],u.reshape(1,2,1,26)],2)
  qp=torch.tensor(d.qpos[None],dtype=torch.float32);qv=torch.tensor(d.qvel[None],dtype=torch.float32)
  s=states_from_coordinates(qp,qv,t.sides,t.base_p,t.base_q,hist,servo);ref=t.ref[phase+1:phase+2]
  wp=(s[:,:274].reshape(1,2,137)[:,:,40:43]-ref[:,:274].reshape(1,2,137)[:,:,40:43]).norm(dim=-1)
  rows.append({'step':phase+1,'object_mm':float((s[:,274:277]-ref[:,274:277]).norm())*1000,'object_deg':float(angle_error(s[:,277:281],ref[:,277:281]))*180/np.pi,'wrist_mm':float((wp*t.mask).sum()/t.mask.sum())*1000,'ctrl_delta_firsthand_mm':((ctrl[0,:3]-t.qpos[phase,:3])*1000).tolist()})
 return rows
start=time.monotonic();manifest=json.loads(Path('configs/a1_probe40_v1.json').read_text());out=[]
for row in manifest['tasks']:
 t=TaskTensors(row['task_path'],device='cpu')
 out.append({'task':row['task_id'],'stratum':row['stratum'],'original':replay(t,'original'),'velocity_consistent':replay(t,'velocity_consistent')})
print(json.dumps({'scope':'CPU nominal zero-residual diagnostic. Not GPU validation; not trained policy success; no formal A1 data. Only reset actuator memory differs. Future control sequence, physics, physical initial qpos/qvel unchanged.','elapsed_seconds':time.monotonic()-start,'tasks':out},indent=2))
