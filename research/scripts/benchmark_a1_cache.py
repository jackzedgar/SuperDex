"""Frozen-policy throughput measurement on the complete uniform development task sampler."""
import argparse,json,time,gc,torch
from a1_data_sources import ROOT,RUN,write
from a1_policy import CollectorPolicy
from a1_multitask_runner import MultiRunner

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cache',type=int,required=True);p.add_argument('--determinism',default='constraint');a=p.parse_args()
 torch.set_num_threads(2);torch.manual_seed(280043);out=RUN/a.run;out.mkdir(exist_ok=False)
 ck=RUN/'collector_full_v3_lr_001/checkpoint_final.pt';c=torch.load(ck,map_location='cpu',weights_only=False)
 policy=CollectorPolicy().cuda();policy.load_state_dict(c['policy']);policy.eval()
 manifest=json.loads((ROOT/'configs/a1_probe40_v1.json').read_text())
 write(out/'registration.json',dict(cache_capacity=a.cache,determinism=a.determinism,controls=2048,seed=71929,scope='Implementation throughput only; frozen policy; all40tasks uniform; no final test, no quality admission. Cache-dependent constructor RNG draws make timings distribution-matched, not paired trajectories.'))
 start=time.monotonic();r=MultiRunner(manifest,policy,71929,cache_limit=a.cache,reset_mode='uniform_reference',determinism=a.determinism)
 records=[]
 for i in range(4):
  before=time.monotonic()
  with torch.no_grad():data=r.rollout(512)
  torch.cuda.synchronize();seconds=time.monotonic()-before
  records.append(dict(block=i,controls=512,seconds=seconds,controls_per_second=512/seconds,cache_size=len(r.cache),cache_creations=r.cache_creations,cache_creation_seconds=r.cache_creation_seconds,allocated_mib=torch.cuda.memory_allocated()/2**20,reserved_mib=torch.cuda.memory_reserved()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
  write(out/'progress.json',records);print(json.dumps(records[-1]),flush=True);del data
 write(out/'summary.json',dict(status='complete',seconds=time.monotonic()-start,records=records,task_controls=r.task_controls))
if __name__=='__main__':main()
