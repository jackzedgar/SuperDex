"""CPU critic calibration against stored deterministic evaluation returns."""
import json,time
from types import SimpleNamespace
import numpy as np,torch
from train_real_ppo import Policy
from core_policy_common import observe_state
from bptt_common import ROOT,RUN,write
def main():
 torch.set_num_threads(1);started=time.time();d=ROOT/'runtime/lambda/processed/clip000000_obj000004_right'
 ref=torch.tensor(np.load(d/'trajectory.npz')['qpos'],dtype=torch.float32);target=torch.tensor(np.load(d/'controller_reference.npz')['ctrl'],dtype=torch.float32)
 env=SimpleNamespace(ref=ref,target=target,length=len(ref));rows=[]
 for method in ['P','S']:
  for seed in [43,59]:
   arm=RUN/f'core_{method}_{seed}'
   for folder in sorted((arm/'fixed_evaluation/new32').glob('step_*')):
    step=int(folder.name.split('_')[1]);policy=Policy(118);checkpoint=arm/f'step_{step:06d}.pt';checkpoint=checkpoint if checkpoint.exists() else arm/'checkpoint_final.pt';policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy']);data=[];actual=[]
    for trial in range(32):
     x=np.load(folder/f'trial_{trial:02d}.npz');state=np.r_[x['qpos'][0],x['qvel'][0],np.tile(x['initial_ctrl'],3)];data.append(state);actual.append(float((x['reward']*.99**np.arange(len(x['reward']))).sum()))
    with torch.no_grad():
     s=torch.tensor(np.array(data),dtype=torch.float32);obs=observe_state(s,torch.zeros(32,dtype=torch.long),torch.zeros(32,26),env);pred=policy.value(obs).numpy()
    err=pred-np.array(actual);rows.append(dict(method=method,seed=seed,step=step,predicted_mean=float(pred.mean()),actual_discounted_mean=float(np.mean(actual)),bias=float(err.mean()),rmse=float(np.sqrt(np.mean(err**2))),predictions=pred.tolist(),actual_returns=actual))
 write(RUN/'analysis/critic_calibration.json',dict(rows=rows,cpu_seconds=time.time()-started,gpu_seconds=0,caveat='Deterministic frozen-policy evaluation; P critic was trained under stochastic collection, so this is an informative calibration diagnostic, not an unbiased on-policy value error comparison. No evaluation target enters fitting.'))
 print('critic calibration rows',len(rows))
if __name__=='__main__':main()

