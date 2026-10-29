"""New fixed conditions, all scheduled shared-policy checkpoints."""
import time,json
import numpy as np,torch
from train_bptt_multitask import TaskEnv,ROOT,RUN,TASK_META
from train_real_ppo import Policy
from train_core_policy import evaluate_policy
from exploration_common import Contacts,activate
from bptt_common import write
def main():
 start=time.time();torch.set_num_threads(1);torch.set_num_interop_threads(1);envs=[TaskEnv(i) for i in [0,1]];contacts=[Contacts(e) for e in envs];policy=Policy(127).cuda();out=RUN/'shared_two_objects';rows=[];count=0
 initials=[dict(np.load(ROOT/t['new_initials'])) for t in TASK_META]
 for path in sorted(out.glob('step_*.pt')):
  step=int(path.stem.split('_')[1]);policy.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['policy'])
  for i,env in enumerate(envs):
   activate(env);folder=out/f'new_fixed_evaluation/step_{step:06d}/task_{i}'
   r=evaluate_policy(env,policy,folder,initials[i],contacts[i]);count+=r['physical_controls']
   rows.append(dict(step=step,task=i,**r));write(out/'new_fixed_results.json',rows);print('SHARED_NEW',step,i,r['successes'],flush=True)
 write(out/'new_fixed_timing.json',dict(seconds=time.time()-start,physical_controls=count,selection='Every scheduled shared-policy checkpoint, new32 per task, no fitting or selection'))
if __name__=='__main__':main()
