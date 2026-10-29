"""Training-only instantaneous velocity-consistent actuator reset memory.

Identical to the completed preregistered DiagnosticEnv factor. Physical reset,
reference, randomization, reward, subsequent dynamics and evaluation are unchanged.
"""
import torch

@torch.no_grad()
def apply_velocity_reset_memory(env):
 m=env.m;dtype=env.qpos.dtype
 gain=torch.as_tensor(m.actuator_gainprm[:,0],device=env.device,dtype=dtype)
 bias=torch.as_tensor(m.actuator_biasprm[:,2],device=env.device,dtype=dtype)
 target=(env.qpos[:,:env.task.nu]+(-bias/gain)*env.qvel[:,:env.task.nu]).clamp(env.lo,env.hi)
 command=env._world(target)
 env.ctrl.copy_(target);env.history.copy_(command[:,:,None].expand(-1,-1,3,-1));env.servo.copy_(command[:,:,:6])
 env.current_state.copy_(env.state())
