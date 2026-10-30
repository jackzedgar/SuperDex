"""Exploratory W0/W1: matched residual MLP; W1 adds only 21 SDF/gradient queries.

Single object, fixed physical context. Unlike formal W0/W1, this miniature uses
simulator generalized coordinates and 3x256 hidden layers, not yaw canonicalization
or 3x2048. No learned reward, termination or contact labels enter either predictor.
"""
import numpy as np,torch
from torch import nn


def qmul(a,b):
    aw,av=a[...,:1],a[...,1:];bw,bv=b[...,:1],b[...,1:]
    return torch.cat([aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.linalg.cross(av,bv)],-1)
def qconj(q):return torch.cat([q[...,:1],-q[...,1:]],-1)
def qnormalize(q):return q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-8)
def qexp(v):
    angle=torch.linalg.vector_norm(v,dim=-1,keepdim=True);half=angle*.5
    return torch.cat([half.cos(),.5*torch.sinc(half/torch.pi)*v],-1)
def qlog(q):
    q=qnormalize(q);q=q*torch.where(q[...,:1]<0,-1.,1.);v=q[...,1:];norm=torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    scale=torch.where(norm>1e-6,2*torch.atan2(norm,q[...,:1])/norm.clamp_min(1e-8),2*torch.ones_like(norm));return v*scale
def qmatrix(q):
    q=qnormalize(q);w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)
def rotate(q,p):return (qmatrix(q)@p[...,None]).squeeze(-1)
def encode(s):
    # qpos 33 followed by qvel 32, then three previous executed controls.
    r=qmatrix(s[...,29:33])[...,:,:2].reshape(*s.shape[:-1],6)
    return torch.cat([s[...,:29],r,s[...,33:]],-1)
def delta_state(a,b):return torch.cat([b[...,:29]-a[...,:29],qlog(qmul(qconj(a[...,29:33]),b[...,29:33])),b[...,33:65]-a[...,33:65]],-1)
def compose(s,d,u):
    q=qnormalize(qmul(s[...,29:33],qexp(d[...,29:32])))
    return torch.cat([s[...,:29]+d[...,:29],q,s[...,33:65]+d[...,32:],s[...,91:143],u],-1)


class Geometry(nn.Module):
    def __init__(self,path):
        super().__init__();a=np.load(path/'kinematics.npz');g=np.load(path/'sdf_grid.npz')
        for k in a.files:self.register_buffer(k,torch.tensor(a[k],dtype=torch.float32))
        for k in ['values','lo','hi','spacing']:self.register_buffer('grid_'+k,torch.tensor(g[k],dtype=torch.float32))
        self.shape=tuple(g['values'].shape)
    def points(self,q):
        b=len(q);r=torch.eye(3,device=q.device,dtype=q.dtype).expand(b,5,3,3);p=torch.zeros(b,5,3,device=q.device,dtype=q.dtype);links=[]
        base_r=qmatrix(self.body_quat);angles=q[:,6:26].reshape(b,5,4)-self.joint_qpos0
        for j in range(4):
            p=p+(r@self.body_pos[:,j,:,None]).squeeze(-1);r=r@base_r[:,j];anchor=p+(r@self.joint_pos[:,j,:,None]).squeeze(-1)
            r=r@qmatrix(qexp(self.joint_axis[:,j]*angles[:,:,j,None]));p=anchor-(r@self.joint_pos[:,j,:,None]).squeeze(-1)
            if j<3:links.append(p+(r@self.body_ipos[:,j,:,None]).squeeze(-1))
        tips=p+(r@self.tip_pos[...,None]).squeeze(-1);local=torch.cat([tips,*links,self.palm_point.expand(b,1,3)],1)
        e=torch.eye(3,device=q.device,dtype=q.dtype);base=qmatrix(self.wrist_base_quat);rw=base.expand(b,3,3)
        for j in range(3):rw=rw@qmatrix(qexp(q[:,3+j,None]*e[j]))
        wrist_position=self.wrist_base_pos+(base@q[:,:3,None]).squeeze(-1)
        world=(rw[:,None]@local[...,None]).squeeze(-1)+wrist_position[:,None]
        return (qmatrix(q[:,29:33]).transpose(-1,-2)[:,None]@(world-q[:,None,26:29])[...,None]).squeeze(-1)
    def sdf(self,points):
        p=points.clamp(self.grid_lo,self.grid_hi);x=(p-self.grid_lo)/self.grid_spacing
        imax=torch.tensor(self.shape,device=x.device)-2;i=torch.minimum(x.floor().long(),imax);i=i.clamp_min(0);t=x-i
        value=torch.zeros_like(x[...,0]);grad=torch.zeros_like(x)
        for a in [0,1]:
            for b in [0,1]:
                for c in [0,1]:
                    v=self.grid_values[i[...,0]+a,i[...,1]+b,i[...,2]+c];bits=[a,b,c];w=[t[...,j] if bits[j] else 1-t[...,j] for j in range(3)]
                    value=value+v*w[0]*w[1]*w[2]
                    grad=grad+torch.stack([v*(1 if bits[j] else -1)*w[(j+1)%3]*w[(j+2)%3]/self.grid_spacing[j] for j in range(3)],-1)
        off=points-p;distance=torch.linalg.vector_norm(off,dim=-1);inside_axis=(points>=self.grid_lo)&(points<=self.grid_hi)
        grad=grad*inside_axis+off/distance[...,None].clamp_min(1e-8)
        return value+distance,grad
    def forward(self,s):
        d,g=self.sdf(self.points(s[:,:33]));return torch.cat([d.clamp_min(-.003)[...,None]*100,g],-1).flatten(1)


class WorldModel(nn.Module):
    def __init__(self,variant,geometry,normalization):
        super().__init__();self.variant=variant;self.geometry=geometry if variant=='W1' else None
        for k,v in normalization.items():self.register_buffer(k,torch.tensor(v,dtype=torch.float32))
        dim=145+26+(84 if variant=='W1' else 0)
        self.net=nn.Sequential(nn.Linear(dim,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,64))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,s,u):
        features=[(encode(s)-self.input_mean)/self.input_std,(u-self.control_mean)/self.control_std]
        if self.geometry is not None:features.append(self.geometry(s))
        d=self.net(torch.cat(features,-1).clamp(-20,20))*self.delta_std+self.delta_mean
        return compose(s,d,u)
    def loss(self,pred,target):
        err=(encode(pred)[...,:67]-encode(target)[...,:67])/self.state_std
        return (err.square()*self.loss_weight).mean()


def simple_step(s,u,kind):
    d=torch.zeros(*s.shape[:-1],64,device=s.device,dtype=s.dtype)
    if kind!='persistence':
        d[...,:26]=s[...,33:59]*.02;d[...,26:29]=s[...,59:62]*.02;d[...,29:32]=s[...,62:65]*.02
        if kind=='ballistic':d[...,28]-=.5*9.81*.02*(.02+.002);d[...,60]-=9.81*.02
    return compose(s,d,u)
