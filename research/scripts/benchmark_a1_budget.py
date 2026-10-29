"""Bounded throughput calibration only; no saved research model or success claim."""
import json,time,argparse
import numpy as np,torch
from torch import nn
from world_model_mini import WorldModel,Geometry,encode,compose
from bptt_common import ROOT,WMROOT
class BenchModel(WorldModel):
 def __init__(self,variant,norm):
  super().__init__(variant,Geometry(WMROOT),norm)
  self.net=nn.Sequential(nn.Linear(500+(168 if variant=='W1' else 0),2048),nn.SiLU(),nn.Linear(2048,2048),nn.SiLU(),nn.Linear(2048,2048),nn.SiLU(),nn.Linear(2048,64))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,s,u):
  f=[(encode(s)-self.input_mean)/self.input_std,(u-self.control_mean)/self.control_std,s.new_zeros(len(s),329)]
  if self.variant=='W1':
   g=self.geometry(s);f += [g,self.geometry(s)]
  d=self.net(torch.cat(f,-1).clamp(-20,20))*self.delta_std+self.delta_mean
  return compose(s,d,u)
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['W0','W1'],required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(1801);start=time.time()
 z=np.load(ROOT/'runs/explore_20261026/wm/dataset.npz');states=torch.tensor(z['states'],device='cuda');controls=torch.tensor(z['controls'],device='cuda');windows=z['windows_8_train']
 c=torch.load(WMROOT/'W1_final.pt',map_location='cpu',weights_only=False);m=BenchModel(a.variant,c['normalization']).cuda();opt=torch.optim.Adam(m.parameters(),lr=2e-4);rows=[]
 for batch in [256,512]:
  rng=np.random.default_rng(1801);ix=windows[rng.choice(len(windows),batch,replace=True)];si=torch.tensor(ix[:,0],device='cuda');ui=torch.tensor(ix[:,1],device='cuda')
  targets=states[si[:,None]+torch.arange(9,device='cuda')];u=controls[ui[:,None]+torch.arange(8,device='cuda')]
  for i in range(10):
   torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);s=targets[:,0];loss=0
   for h in range(8):s=m(s,u[:,h]);loss=loss+m.loss(s,targets[:,h+1])/8
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.cuda.synchronize()
   elapsed=time.perf_counter()-t
   if i>=2:rows.append(dict(batch=batch,seconds=elapsed,window_steps=8,model_transitions=batch*8,finite=bool(torch.isfinite(loss))))
  print(a.variant,batch,float(np.median([r['seconds'] for r in rows if r['batch']==batch])),flush=True)
 out=ROOT/'runs/a1_budget_20261028'/('benchmark_'+a.variant+'.json')
 result=dict(variant=a.variant,parameters=sum(x.numel() for x in m.parameters()),rows=rows,peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,scope='Timing proxy only: original one-hand state recurrence, representative500/668 input widths, two SDF queries for W1; zero extra context, no yaw/bimanual pipeline. Full formal throughput still requires calibration; not a predictive experiment.',precision='float32, no AMP',gpu=torch.cuda.get_device_name(0))
 out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
