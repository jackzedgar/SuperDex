"""Only newly affected policy/model chain, history and reference-end checks."""
from types import SimpleNamespace
import numpy as np,torch,mujoco
from exploration_common import RUN,ROOT,DATA,read_json,write_json
from core_policy_common import load_model,model_policy_objective
from train_real_ppo import Policy
torch.set_num_threads(1);cfg=read_json(RUN/'budget.json');ref=np.load(DATA/'trajectory.npz');ctrl=np.load(DATA/'controller_reference.npz')['ctrl'];m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
env=SimpleNamespace(ref=torch.tensor(ref['qpos'],dtype=torch.float64),target=torch.tensor(ctrl,dtype=torch.float64),length=len(ref['qpos']),lo=torch.tensor(m.actuator_ctrlrange[:,0]),hi=torch.tensor(m.actuator_ctrlrange[:,1]),scale=torch.tensor([.025]*3+[.18]*3+[.25]*20,dtype=torch.float64))
model=load_model(ROOT/cfg['model_source'],'cpu').double();policy=Policy(118).double();ck=torch.load(ROOT/cfg['policy_source'],map_location='cpu',weights_only=True);policy.load_state_dict(ck['policy'])
phase=torch.tensor([0,47,50,51]);state=torch.tensor(np.concatenate([ref['qpos'][phase],ref['qvel'][phase],np.tile(ctrl[phase],(1,3))],1),dtype=torch.float64);last=torch.zeros(4,26,dtype=torch.float64)
loss,returns,count=model_policy_objective(policy,model,state,phase,last,env);params=list(policy.actor.parameters());grads=torch.autograd.grad(loss,params);norm=torch.sqrt(sum(g.square().sum() for g in grads));assert torch.isfinite(norm) and norm>0
direction=[g/norm for g in grads];eps=1e-5
with torch.no_grad():
    for p,d in zip(params,direction):p.add_(eps*d)
    plus=model_policy_objective(policy,model,state,phase,last,env)[0]
    for p,d in zip(params,direction):p.add_(-2*eps*d)
    minus=model_policy_objective(policy,model,state,phase,last,env)[0]
    for p,d in zip(params,direction):p.add_(eps*d)
error=abs(float((plus-minus)/(2*eps)-norm));assert error<1e-5 and int(count)==9 and float(returns[-1])==0
u=torch.tensor(ctrl[1:5],dtype=torch.float64);out=model(state,u)
assert torch.equal(out[:,65:117],state[:,91:143]) and torch.equal(out[:,117:143],u)
assert all(p.grad is None for p in model.parameters())
write_json(RUN/'implementation_check.json',dict(policy_through_frozen_model_gradient=True,finite_difference_absolute_error=error,actor_gradient_norm=float(norm),valid_transitions=int(count),reference_end_return_zero=True,control_history_shift_correct=True,model_parameter_gradients_absent=True,scope='New differentiable policy chain/history/end masking only; old grasp/physics tests not repeated.'))
print('New policy-model-chain checks passed',error)
