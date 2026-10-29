"""All predeclared checkpoints; fixed new conditions never used for fitting."""
import argparse,time,json
import numpy as np,torch
from real_env import RealEnv
from train_real_ppo import Policy
from train_core_policy import evaluate_policy
from exploration_common import Contacts
from bptt_common import ROOT,RUN,write
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);a=p.parse_args();start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 env=RealEnv(1);contact=Contacts(env);policy=Policy(118).cuda();folder=RUN/a.arm;rows=[];counts=0
 paths=[(p,int(p.stem.split('_')[1])) for p in sorted(folder.glob('step_*.pt'))]
 final_count=json.loads((folder/'summary.json').read_text())['real_controls']
 if final_count not in [step for _,step in paths]:paths.append((folder/'checkpoint_final.pt',final_count))
 assert paths
 for path,step in paths:
  if step%65536!=0 and step!=final_count:continue
  policy.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy'])
  for label,source in [('regression16',ROOT/'runs/core_iteration_20261027/heldout_initials.npz'),('new32',RUN/'new_initials.npz')]:
   out=folder/f'fixed_evaluation/{label}/step_{step:06d}'
   if (out/'summary.json').exists():r=json.loads((out/'summary.json').read_text())
   else:r=evaluate_policy(env,policy,out,dict(np.load(source)),contact);counts+=r['physical_controls']
   rows.append(dict(arm=a.arm,steps=step,set=label,**r))
   write(folder/'fixed_results.json',rows);print(a.arm,step,label,r['successes'],flush=True)
 write(folder/'fixed_timing.json',dict(seconds=time.time()-start,physical_controls=counts,checkpoints=[p.name for p,_ in paths],selection='All predeclared multiples of65536, including zero; no checkpoint selection'))
if __name__=='__main__':main()
