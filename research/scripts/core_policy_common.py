"""Differentiable policy-through-frozen-model objective; no future real targets."""
from pathlib import Path
import numpy as np,torch
from exploration_common import RUN,ROOT,DATA,read_json,write_json
from world_model_mini import WorldModel,Geometry,encode,qlog,qmul,qconj
from probe_model_gradient import reward_from_state

def load_model(path,device='cuda'):
    c=torch.load(path,map_location='cpu',weights_only=False)
    model=WorldModel('W1',Geometry(ROOT/read_json(RUN/'budget.json')['model_geometry']),c['normalization'])
    model.load_state_dict(c['model'])
    return model.to(device).eval().requires_grad_(False)

def observe_state(s,phase,last,env):
    current=env.ref[phase];nxt=env.target[(phase+1).clamp(max=env.length-1)]
    q=s[:,29:33];r=current[:,29:33];q=q*torch.where((q*r).sum(-1,keepdim=True)<0,-1.,1.)
    return torch.cat([s[:,:26],s[:,33:59]*.1,nxt-s[:,:26],(s[:,26:29]-current[:,26:29])*10,q-r,s[:,59:65]*.1,phase[:,None]/(env.length-1),last],-1).clamp(-20,20)

def model_policy_objective(policy,model,state,phase,last,env,horizon=4):
    s=state;ph=phase;previous=last;returns=torch.zeros(len(s),device=s.device,dtype=s.dtype);valid_count=torch.zeros((),device=s.device,dtype=torch.long)
    for _ in range(horizon):
        valid=ph<env.length-1
        action=policy.actor(observe_state(s,ph,previous,env)).clamp(-1,1)
        nxt=(ph+1).clamp(max=env.length-1)
        ctrl=torch.maximum(env.lo,torch.minimum(env.hi,env.target[nxt]+action*env.scale))
        proposed=model(s,ctrl)
        reward,_=reward_from_state(proposed,env.ref[nxt],action,previous)
        returns=returns+reward*valid;valid_count=valid_count+valid.sum()
        s=torch.where(valid[:,None],proposed,s);previous=torch.where(valid[:,None],action,previous);ph=nxt
    return -returns.mean(),returns,valid_count

def physical_state(env,history):
    return torch.cat([env.qpos.clone(),env.qvel.clone(),history.flatten(1)],-1)

@torch.no_grad()
def model_diagnostics(model,states,nextstates,controls,actions,phases,lasts,rewards,dones,env,batch=16):
    n=len(states);valid=[]
    for t in range(n-3):
        if not bool(dones[t:t+3].any()) and int(phases[t])+4<env.length:valid.append(t)
    if not valid:return dict(windows=0,predictions=0)
    ix=torch.tensor(valid,device=states.device)
    # Deterministic evenly spaced diagnostics, independent of optimizer RNG.
    ix=ix[torch.linspace(0,len(ix)-1,min(batch,len(ix)),device=ix.device).long()]
    s=states[ix];initial=s;previous=lasts[ix];predr=torch.zeros(len(ix),device=s.device);realr=predr.clone();result={}
    for j in range(4):
        a=actions[ix+j].clamp(-1,1);s=model(s,controls[ix+j]);r,_=reward_from_state(s,env.ref[phases[ix]+j+1],a,previous);predr+=r;realr+=rewards[ix+j];previous=a
        if j in [0,3]:
            truth=nextstates[ix+j];result[f'position_{j+1}_mm']=float(torch.linalg.vector_norm(s[:,26:29]-truth[:,26:29],dim=-1).mean()*1000)
            result[f'rotation_{j+1}_deg']=float(torch.linalg.vector_norm(qlog(qmul(qconj(truth[:,29:33]),s[:,29:33])),dim=-1).mean()*180/np.pi)
    z=(encode(initial)-model.input_mean)/model.input_std
    result.update(windows=len(ix),predictions=len(ix)*4,predicted_reward_4=float(predr.mean()),real_reward_4=float(realr.mean()),reward_bias_4=float((predr-realr).mean()),reward_mae_4=float((predr-realr).abs().mean()),input_over_3std_fraction=float((z.abs()>3).float().mean()))
    return result
