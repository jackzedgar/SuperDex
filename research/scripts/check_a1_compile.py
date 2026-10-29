"""Bounded full-size graph compilation measurement; synthetic fixture only."""
import argparse,json,time,copy
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_geometry import A1Geometry
from a1_world_model import A1WorldModel,CONTEXT
from a1_state import empty_state
def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['W0','W1'],required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(280043)
 from torch._inductor import config as inductor_config
 inductor_config.compile_threads=4
 kin=[dict(np.load(ROOT/f'data/a1_processed/assets/hands/{side}/kinematics.npz')) for side in ['right','left']]
 geom=A1Geometry(kin,[dict(np.load(ROOT/'runs/explore_20261026/wm/sdf_grid.npz'))]).cuda()
 batch=512;s=empty_state(batch,device='cuda');s[:,42]=.15;s[:,179]=.15;s[:,276]=.1
 mask=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],device='cuda').repeat((batch+2)//3,1)[:batch]
 c=torch.zeros(batch,CONTEXT,device='cuda')
 for h in range(2):
  c[:,h*68+20:h*68+40]=1.97392088;c[:,h*68+40:h*68+60]=.1256637;c[:,h*68+60]=1;c[:,h*68+62:h*68+68]=.02
 c[:,136]=.04;c[:,137:139]=.8
 u=torch.zeros(batch,8,52,device='cuda');u[:,:,2]=.15;u[:,:,28]=.15
 ids=torch.zeros(batch,dtype=torch.long,device='cuda')
 target=s[:,None].repeat(1,8,1);target[:,:,274:277]+=torch.randn(batch,8,3,device='cuda')*.001
 m=A1WorldModel(a.kind,geom if a.kind=='W1' else None).cuda()
 with torch.no_grad():m.net[-1].weight.normal_(0,1e-5)
 expected=m(s,u[:,0],mask,c,ids).detach()
 original=m.forward;m.forward=torch.compile(original,fullgraph=True,dynamic=False)
 started=time.monotonic();actual=m(s,u[:,0],mask,c,ids);torch.cuda.synchronize()
 assert torch.allclose(actual,expected,atol=3e-6,rtol=3e-6),float((actual-expected).abs().max())
 print('forward compiled',time.monotonic()-started,flush=True)
 opt=torch.optim.Adam(m.parameters(),lr=2e-4);timings=[];losses=[]
 for i in range(6):
  torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
  pred=m.rollout(s,u,mask,c,ids);loss=m.dynamics_loss(pred,target,mask);loss.backward()
  gn=torch.nn.utils.clip_grad_norm_(m.net.parameters(),1.);assert torch.isfinite(gn)
  opt.step();torch.cuda.synchronize();timings.append(time.monotonic()-t);losses.append(float(loss))
  print('compiled update',i,timings[-1],losses[-1],flush=True)
 report=dict(kind=a.kind,scope='Synthetic performance/equivalence only; not A1 result',compile_plus_checks_seconds=time.monotonic()-started,update_times=timings,steady_microbatch_seconds=float(np.median(timings[-3:])),estimated_accumulated_adam_seconds=float(np.median(timings[-3:])*40),losses=losses,forward_max_difference=float((actual.detach()-expected).abs().max()),peak_bytes=torch.cuda.max_memory_allocated(),physics_steps=0,model_transitions=6*512*8)
 write(RUN/f'compiled_model_{a.kind}_check.json',report)
if __name__=='__main__':main()
