"""Full-A1 W0/W1: matched 3x2048 SiLU dynamics, analytic control history."""
import math
import torch
from torch import nn
import a1_state as state

HAND_CONTEXT=68
CONTEXT=276
COMMON_INPUT=state.ENCODED+state.COMMAND_ENCODED+CONTEXT
SDF_INPUT=168

def control_context(context):
    hand=context[...,:136].reshape(*context.shape[:-1],2,HAND_CONTEXT)
    return hand[...,60:62],hand[...,62:68],hand[...,:20]

def common_features(s,u,mask,context):
    frame,origin=state.yaw_frame(s,mask)
    commands=u.reshape(*s.shape[:-1],2,26)
    parts=[state.encode(s,mask)]
    for h in range(2):
        parts.append(state.encode_command(commands[...,h,:],frame,origin)*mask[...,h,None])
    parts.append(context)
    return torch.cat(parts,-1)

class A1WorldModel(nn.Module):
    def __init__(self,kind,geometry=None,seed=23):
        super().__init__()
        assert kind in ('W0','W1')
        if kind=='W1' and geometry is None:raise ValueError('W1 requires actual shared task geometry')
        self.kind=kind;self.geometry=geometry
        size=COMMON_INPUT+(SDF_INPUT if kind=='W1' else 0)
        self.net=nn.Sequential(nn.Linear(size,2048),nn.SiLU(),nn.Linear(2048,2048),nn.SiLU(),nn.Linear(2048,2048),nn.SiLU(),nn.Linear(2048,state.DELTA))
        self.register_buffer('input_mean',torch.zeros(COMMON_INPUT))
        self.register_buffer('input_std',torch.ones(COMMON_INPUT))
        self.register_buffer('delta_std',torch.ones(state.DELTA))
        self.register_buffer('loss_std',torch.ones(state.DYNAMICS))
        self.register_buffer('geometry_scale',torch.tensor([.03,1,1,1]*42))
        self.reset_matched(seed)
    def reset_matched(self,seed):
        # Exactly the same initial common function: W1's extra columns start at
        # zero; all common columns and downstream weights use identical draws.
        rng=torch.Generator(device='cpu').manual_seed(seed)
        layers=[x for x in self.net if isinstance(x,nn.Linear)]
        with torch.no_grad():
            for i,layer in enumerate(layers):
                fan=COMMON_INPUT if i==0 else layer.in_features
                common=torch.empty(layer.out_features,fan).uniform_(-1/math.sqrt(fan),1/math.sqrt(fan),generator=rng)
                layer.weight.zero_();layer.weight[:,:fan].copy_(common)
                layer.bias.zero_()
            layers[-1].weight.zero_()
    def forward(self,s,u,mask,context,object_id):
        x=(common_features(s,u,mask,context)-self.input_mean)/self.input_std
        if self.kind=='W1':
            x=torch.cat([x,self.geometry.features(s,mask,object_id)/self.geometry_scale],-1)
        delta=self.net(x)*self.delta_std
        delay,tau,offset=control_context(context)
        return state.compose(s,delta,u,mask,delay,tau,offset)
    def rollout(self,start,commands,mask,context,object_id):
        outputs=[];s=start
        for h in range(commands.shape[1]):
            s=self(s,commands[:,h],mask,context,object_id);outputs.append(s)
        return torch.stack(outputs,1)
    def dynamics_loss(self,pred,target,mask):
        error=(state.encode_dynamics(pred)-state.encode_dynamics(target))/self.loss_std
        weight=torch.cat([mask[...,0,None].expand(*mask.shape[:-1],55),mask[...,1,None].expand(*mask.shape[:-1],55),mask.new_full((*mask.shape[:-1],15),4)],-1)
        while weight.ndim<error.ndim:weight=weight.unsqueeze(-2)
        # Sum over rollout time, normalized by present weighted components.
        return ((error.square()*weight).sum(-1)/weight.sum(-1)).sum(-1).mean()
    def config(self):
        return dict(kind=self.kind,hidden=[2048]*3,activation='SiLU',common_input=COMMON_INPUT,sdf_input=SDF_INPUT if self.kind=='W1' else 0,state=state.STATE,delta=state.DELTA,parameters=sum(p.numel() for p in self.net.parameters()),control='complete limited target before causal delay',initial_extra_columns='zero; identical common initial function',loss='8-step autoregression, normalized physical state, object weight4, absent-hand mask')

class StreamingMoments:
    """Float64 Welford sufficient statistics; fit only training episodes."""
    def __init__(self,width):self.n=0;self.mean=torch.zeros(width,dtype=torch.float64);self.m2=torch.zeros_like(self.mean)
    def add(self,values):
        x=values.detach().reshape(-1,values.shape[-1]).cpu().double()
        if not len(x):return
        n=len(x);mean=x.mean(0);m2=((x-mean)**2).sum(0);d=mean-self.mean
        total=self.n+n;self.m2+=m2+d*d*(self.n*n/total);self.mean+=d*(n/total);self.n=total
    def std(self,floor=1e-4):
        assert self.n>1
        return (self.m2/(self.n-1)).sqrt().clamp_min(floor).float()
