"""Two-hand FK and packed object SDFs for full A1, with consistent derivatives."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from a1_state import H,R,P,Q
from world_model_mini import qmatrix,qexp

class HandKinematics(nn.Module):
    def __init__(self,arrays):
        super().__init__()
        for key in ['body_pos','body_quat','body_ipos','joint_axis','joint_pos','joint_qpos0','tip_pos','palm_point','root_pos','root_quat']:
            self.register_buffer(key,torch.as_tensor(arrays[key],dtype=torch.float32))
    def forward(self,q,keypoints=False):
        batch=q.shape[:-1]
        r=qmatrix(self.root_quat).expand(*batch,5,3,3)
        p=self.root_pos.expand(*batch,5,3)
        br=qmatrix(self.body_quat);angles=q.reshape(*batch,5,4)-self.joint_qpos0
        links=[]
        for j in range(4):
            p=p+(r@self.body_pos[:,j,:,None]).squeeze(-1);r=r@br[:,j]
            anchor=p+(r@self.joint_pos[:,j,:,None]).squeeze(-1)
            r=r@qmatrix(qexp(self.joint_axis[:,j]*angles[...,j,None]))
            p=anchor-(r@self.joint_pos[:,j,:,None]).squeeze(-1)
            if j<3:links.append(p+(r@self.body_ipos[:,j,:,None]).squeeze(-1))
            if j==2:mid=p
        tips=p+(r@self.tip_pos[...,None]).squeeze(-1)
        return torch.cat([tips,mid],-2) if keypoints else torch.cat([tips,*links,self.palm_point.expand(*batch,1,3)],-2)

class PackedSDF(nn.Module):
    def __init__(self,grids):
        super().__init__();offsets=[];sizes=[];los=[];his=[];spacings=[];tables=[];offset=0
        for grid in grids:
            values=np.asarray(grid['values'],dtype=np.float32)
            lo=np.asarray(grid['lo']);hi=np.asarray(grid['hi']);spacing=(hi-lo)/(np.array(values.shape)-1)
            assert len(values.shape)==3 and min(values.shape)>=2
            if 'gradient' in grid:grad=np.asarray(grid['gradient'])
            else:grad=np.stack(np.gradient(values,*spacing,edge_order=1),-1)
            table=np.concatenate([values[...,None],grad],-1).reshape(-1,4).astype(np.float16)
            assert np.isfinite(table).all()
            offsets.append(offset);sizes.append(values.shape);los.append(lo);his.append(hi);spacings.append(spacing);tables.append(table);offset+=len(table)
        for name,value,dtype in [('table',np.concatenate(tables),torch.float16),('offsets',offsets,torch.long),('sizes',sizes,torch.long),('lo',np.array(los),torch.float32),('hi',np.array(his),torch.float32),('spacing',np.array(spacings),torch.float32)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
    def forward(self,points,object_id):
        # points [B,N,3]. Interpolated precomputed gradients are retained in the
        # table for auditing, but the feature uses the exact derivative of the
        # trilinear scalar, avoiding the document's inconsistent gradient claim.
        lo=self.lo[object_id,None];hi=self.hi[object_id,None];spacing=self.spacing[object_id,None]
        sizes=self.sizes[object_id,None];offset=self.offsets[object_id,None]
        projected=torch.maximum(lo,torch.minimum(hi,points));x=(projected-lo)/spacing
        i=torch.minimum(x.floor().long().clamp_min(0),sizes-2);t=x-i
        value=torch.zeros_like(x[...,0]);grad=torch.zeros_like(x)
        for a in [0,1]:
            for b in [0,1]:
                for c in [0,1]:
                    bits=[a,b,c]
                    index=offset+((i[...,0]+a)*sizes[...,1]+i[...,1]+b)*sizes[...,2]+i[...,2]+c
                    v=self.table[index,0].to(points.dtype)
                    w=[t[...,j] if bits[j] else 1-t[...,j] for j in range(3)]
                    value=value+v*w[0]*w[1]*w[2]
                    grad=grad+torch.stack([v*(1 if bits[j] else -1)*w[(j+1)%3]*w[(j+2)%3]/spacing[...,j] for j in range(3)],-1)
        off=points-projected;outside=torch.linalg.vector_norm(off,dim=-1)
        axis_inside=((points>=lo)&(points<=hi)).to(points.dtype)
        grad=grad*axis_inside+off/outside[...,None].clamp_min(1e-12)
        return value+outside,grad

class A1Geometry(nn.Module):
    def __init__(self,kinematics,grids):
        super().__init__()
        self.hands=nn.ModuleList([HandKinematics(k) for k in kinematics])
        self.sdf=PackedSDF(grids)
    @classmethod
    def from_manifest(cls,path):
        path=Path(path);meta=json.loads(path.read_text())
        def load(p):
            with np.load(path.parent/p) as x:return {k:x[k] for k in x.files}
        return cls([load(p) for p in meta['hand_kinematics']],[load(p) for p in meta['object_sdf_paths']])
    def points(self,s,keypoints=False):
        hs=s[...,:274].reshape(*s.shape[:-1],2,H);o=s[...,274:]
        points=[]
        for h in range(2):
            local=self.hands[h](hs[...,h,Q],keypoints=keypoints)
            world=(qmatrix(hs[...,h,R])[...,None,:,:]@local[...,None]).squeeze(-1)+hs[...,h,None,P]
            points.append((qmatrix(o[...,3:7]).transpose(-1,-2)[...,None,:,:]@(world-o[...,:3].unsqueeze(-2))[...,None]).squeeze(-1))
        return torch.stack(points,-3)
    def features(self,s,mask,object_id):
        p=self.points(s);d,g=self.sdf(p.flatten(-3,-2),object_id)
        feat=torch.cat([d.clamp_min(-.003)[...,None],g],-1).reshape(*s.shape[:-1],2,21,4)
        return (feat*mask[...,None,None]).flatten(-3)
