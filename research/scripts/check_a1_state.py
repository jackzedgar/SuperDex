"""Necessary CPU checks for the new A1 state algebra, not physics acceptance."""
import json,torch
from a1_state import *
from a1_data_sources import RUN,write
torch.set_num_threads(1);torch.manual_seed(280043)
s=empty_state(9,dtype=torch.float64);mask=torch.tensor([[1,1],[1,0],[0,1]]*3,dtype=s.dtype)
s[:,:]+=torch.randn_like(s)*.01
for h in range(2):s[:,h*H+43:h*H+47]=qnormalize(s[:,h*H+43:h*H+47])
s[:,277:281]=qnormalize(s[:,277:281])
target=s.clone()
for h in range(2):
 target[:,h*H:h*H+53]+=torch.randn(9,53,dtype=s.dtype)*.02
 target[:,h*H+43:h*H+47]=qnormalize(target[:,h*H+43:h*H+47])
target[:,274:]+=torch.randn(9,13,dtype=s.dtype)*.02;target[:,277:281]=qnormalize(target[:,277:281])
u=torch.randn(9,52,dtype=s.dtype)*.1;delay=torch.tensor([[[0,0],[1,1]],[[2,2],[3,3]],[[1,0],[0,1]]]*3)
tau=torch.ones(9,2,6,dtype=s.dtype)*.02;zero=torch.zeros(9,2,20,dtype=s.dtype)
d=delta_between(s,target,mask);recovered=compose(s,d,u,mask,delay,tau,zero)
errors=[]
for h in range(2):
 ok=mask[:,h]>0;errors.append((recovered[ok,h*H:h*H+53]-target[ok,h*H:h*H+53]).abs().max().item())
errors.append((recovered[:,274:]-target[:,274:]).abs().max().item())
assert max(errors)<1e-12,errors
stationary=s.clone();stationary[:,40:43]+=.17;stationary[:,43:47]=qmul(stationary[:,43:47],qexp(torch.tensor([.2,.3,.4],dtype=s.dtype).expand(9,3)))
assert delta_between(s,stationary,mask)[:,104:].abs().max()<1e-12
encoded=encode(s,mask);assert encoded.shape==(9,337) and torch.isfinite(encoded).all()
u.requires_grad_();out=compose(s,d,u,mask,delay,tau,zero);loss=out[:,53:137].square().sum()+out[:,190:274].square().sum();g=torch.autograd.grad(loss,u)[0]
assert torch.isfinite(g).all() and g.abs().sum()>0
# Explicit delay queue and first-order lag example.
z=empty_state(1,dtype=torch.float64);z[:,53:79]=1;z[:,79:105]=2;z[:,105:131]=3
uu=torch.ones(1,52,dtype=z.dtype)*4;mm=torch.tensor([[1.,0.]],dtype=z.dtype)
for lag in range(4):
 applied,sv=analytic_control(z,uu,mm,torch.full((1,2,2),lag),torch.zeros(1,2,6),torch.zeros(1,2,20))
 assert torch.allclose(applied[0,0,:3],torch.full((3,),4-lag,dtype=z.dtype))
 assert torch.allclose(applied[0,0,6:],torch.full((20,),4-lag,dtype=z.dtype))
 assert torch.allclose(euler_matrix(applied[0,0,3:6]),euler_matrix(torch.full((3,),4-lag,dtype=z.dtype)),atol=1e-12)
# Isotropic world-position / SO3 servo is equivariant under a common yaw.
yaw=qexp(torch.tensor([[0.,0.,.71]],dtype=s.dtype));rf=qmatrix(yaw)
a,sv=analytic_control(s,u,mask,delay,tau,zero)
ss=s.clone();uu=u.detach().clone().reshape(9,2,26)
for h in range(2):
 for start in [h*H+53,h*H+79,h*H+105,h*H+131]:
  ss[:,start:start+3]=world_vec(s[:,start:start+3],rf)
  ss[:,start+3:start+6]=matrix_euler(qmatrix(qmul(yaw,euler_quaternion(s[:,start+3:start+6]))))
 uu[:,h,:3]=world_vec(uu[:,h,:3],rf)
 uu[:,h,3:6]=matrix_euler(qmatrix(qmul(yaw,euler_quaternion(uu[:,h,3:6]))))
aa,svv=analytic_control(ss,uu.flatten(1),mask,delay,tau,zero)
assert torch.allclose(svv[...,:3],world_vec(sv[...,:3],rf[:,None]),atol=1e-12)
assert torch.allclose(euler_matrix(svv[...,3:6]),rf[:,None]@euler_matrix(sv[...,3:6]),atol=1e-12)
report={'checks':{'state_delta_compose_roundtrip':True,'stationary_object_zero_delta_under_wrist_motion':True,'single_left_right_and_bimanual_masks':True,'causal_delay0_to3':True,'finite_control_gradients':True,'servo_yaw_equivariance':True},'max_roundtrip_error':max(errors),'state_size':STATE,'network_state_encoding':ENCODED,'learned_delta_size':DELTA,'scope':'CPU state algebra only; no formal A1 data/learning/physics claim'}
write(RUN/'state_algebra_check.json',report);print(json.dumps(report,indent=2))
