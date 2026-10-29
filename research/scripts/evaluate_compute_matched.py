"""Same-GPU evaluation of outcome-blind compute-matched snapshots."""
import json,time
import numpy as np,torch
from bptt_common import ROOT,RUN,write
from real_env import RealEnv
from train_real_ppo import Policy
from exploration_common import Contacts
from train_core_policy import evaluate_policy
def main():
 start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1);env=RealEnv(1);contacts=Contacts(env);policy=Policy(118).cuda();initials=dict(np.load(RUN/'new_initials.npz'));rows=[];controls=0
 for seed in [43,59]:
  folder=RUN/f'compute_matched/seed_{seed}';selection=json.loads((folder/'selection.json').read_text())
  for method in ['P','S']:
   c=torch.load(folder/(method+'.pt'),map_location='cpu',weights_only=False);policy.load_state_dict(c['policy'])
   result=evaluate_policy(env,policy,folder/f'evaluation_{method}',initials,contacts);controls+=result['physical_controls']
   rows.append(dict(seed=seed,method=method,training_controls=c['counts']['real_controls'],common_compute_ceiling=selection['common_core_process_ceiling_seconds'],actual_checkpoint_compute_seconds=selection['policy_snapshots'][method]['gpu_process_seconds_at_save'],**result))
   write(RUN/'compute_matched/evaluation_summary.json',dict(rows=rows,physical_controls=controls,seconds=time.time()-start,scope='SameGPU0, fixednew32; snapshots selected by runtime only. Actual saved-time gap explicitly retained. Repeat of first finisher endpoint is separate, never overwrites original fixed evaluation.'));print(seed,method,c['counts']['real_controls'],result['successes'],flush=True)
if __name__=='__main__':main()
