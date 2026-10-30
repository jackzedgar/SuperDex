"""Same W1 MLP; object outputs are two half-interval velocity impulses.

All 64 outputs stay active. Linear impulses are in world coordinates, angular
impulses in the local body frame. Ten 2 ms semi-implicit substeps integrate the
SAME acceleration into velocity and pose. The prior is gravity, not contact or
full rigid-body dynamics; learned residual includes contact/gyroscopic effects.
"""
import numpy as np,torch
from world_model_mini import WorldModel,encode,delta_state,qmul,qexp,qnormalize

DT=.02
SUB=.002
HALF=.01

def integrate_object(s,d):
    p=s[...,26:29];q=s[...,29:33];v=s[...,59:62];w=s[...,62:65]
    gravity=s.new_tensor([0.,0.,-9.81])
    for lin,ang in [(d[...,26:29],d[...,29:32]),(d[...,58:61],d[...,61:64])]:
        dv=gravity*SUB+lin/5;dw=ang/5
        for _ in range(5):
            v=v+dv;w=w+dw;p=p+v*SUB;q=qnormalize(qmul(q,qexp(w*SUB)))
    return p,q,v,w

def residual_normalization(data,manifest,base):
    aa=[];bb=[]
    for e in manifest['episodes']:
        if e['split']=='train':
            k=e['states_start'];n=e['steps'];aa.append(data['states'][k:k+n]);bb.append(data['states'][k+1:k+n+1])
    a=torch.tensor(np.concatenate(aa));b=torch.tensor(np.concatenate(bb));delta=delta_state(a,b);g=a.new_tensor([0.,0.,-9.81])
    dv=b[:,59:62]-a[:,59:62]-DT*g
    dp=b[:,26:29]-a[:,26:29]-DT*a[:,59:62]-.5*DT*(DT+SUB)*g
    early=(dp-.5*(HALF+SUB)*dv)/HALF
    dw=b[:,62:65]-a[:,62:65];rotation=delta[:,29:32]-DT*a[:,62:65]
    early_w=(rotation-.5*(HALF+SUB)*dw)/HALF
    # Angular inversion is a small-angle normalization proxy only; training uses
    # the unchanged state-space loss through exact quaternion compositions.
    values=torch.cat([early,early_w,dv-early,dw-early_w],-1)
    result={k:v.copy() for k,v in base.items()};indices=list(range(26,32))+list(range(58,64))
    result['delta_mean'][indices]=values.mean(0).numpy()
    result['delta_std'][indices]=values.std(0,unbiased=False).clamp_min(.01).numpy()
    return result

class ResidualWorldModel(WorldModel):
    def __init__(self,geometry,normalization):super().__init__('W1',geometry,normalization)
    def forward(self,s,u):
        features=[(encode(s)-self.input_mean)/self.input_std,(u-self.control_mean)/self.control_std,self.geometry(s)]
        d=self.net(torch.cat(features,-1).clamp(-20,20))*self.delta_std+self.delta_mean
        p,q,v,w=integrate_object(s,d)
        return torch.cat([s[...,:26]+d[...,:26],p,q,s[...,33:59]+d[...,32:58],v,w,s[...,91:143],u],-1)
