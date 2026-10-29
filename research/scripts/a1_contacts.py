"""GPU endpoint contact summaries, force threshold and relative point velocity."""
import numpy as np,torch,warp as wp,mujoco_warp as mjw
class ContactDiagnostics:
 def __init__(self,env):
  self.e=env;self.cap=len(env.wd.contact.dist);device=env.device
  self.ids=wp.array(np.arange(self.cap,dtype=np.int32),device=device)
  self.force=wp.zeros(self.cap,dtype=wp.spatial_vector,device=device)
  m=env.m;self.types=torch.tensor(m.geom_contype,device=device);self.bodies=torch.tensor(m.geom_bodyid,device=device,dtype=torch.long)
  self.roots=torch.tensor(m.body_rootid,device=device,dtype=torch.long)
  hand=[]
  for g in range(m.ngeom):
   name=m.body(int(m.geom_bodyid[g])).name or ''
   hand.append(1 if name.startswith('right_') else 2 if name.startswith('left_') else 0)
  self.hands=torch.tensor(hand,device=device)
  self.range=torch.arange(self.cap,device=device)
 def step(self):
  e=self.e;c=e.wd.contact
  mjw.contact_force(e.wm,e.wd,self.ids,False,self.force)
  n=wp.to_torch(e.wd.nacon)[0];valid=self.range<n
  pair=wp.to_torch(c.geom).long().clamp(0,e.m.ngeom-1)
  types=self.types[pair];hand=(types==1).any(-1)&(types==2).any(-1)
  support=(types==4).any(-1)&(types==2).any(-1)
  force=wp.to_torch(self.force)
  dist=wp.to_torch(c.dist);active=valid&(force[:,0]>.02)&(dist<.002)
  hand=hand&active;support=support&active
  pos=wp.to_torch(c.pos);bid=self.bodies[pair]
  vel=wp.to_torch(e.wd.cvel)[0,bid];com=wp.to_torch(e.wd.subtree_com)[0,self.roots[bid]]
  pv=vel[...,3:]+torch.linalg.cross(vel[...,:3],pos[:,None,:]-com)
  relative=pv[:,1]-pv[:,0];normal=wp.to_torch(c.frame)[:,0,:]
  tangent=relative-(relative*normal).sum(-1,keepdim=True)*normal
  speed=torch.linalg.vector_norm(tangent,dim=-1)
  side=self.hands[pair].max(-1).values
  self.active=active;self.hand=hand;self.support=support;self.tangent_speed=speed;self.point_velocity=pv
  self.details=torch.cat([pair.float(),pos,force,tangent,dist[:,None],active.float()[:,None]],-1)
  self.summary=torch.stack([(hand&(side==1)).sum(),(hand&(side==2)).sum(),support.sum(),torch.where(hand,force[:,0],0).sum(),torch.where(support,force[:,0],0).sum(),torch.where(hand,speed,0).max(),(hand&(speed>.01)).sum(),torch.where(valid,(-dist).clamp_min(0),0).max(),n,(n>self.cap).float()]).float()
  return self.summary
