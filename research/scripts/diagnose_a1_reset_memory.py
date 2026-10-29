"""Paired GPU development diagnostic; no fitting, task exclusion or final-test access."""
import argparse,gc,hashlib,json,signal,time
from pathlib import Path
import numpy as np,torch,warp as wp
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
from a1_policy import CollectorPolicy
from train_a1_collector import evaluate
stop=False
def halt(*_):
 global stop;stop=True

def velocity_memory(qpos,qvel,nu,gain,bias,lower,upper):
 """Instantaneous PD velocity feedforward; no future simulation state used."""
 target=qpos[:,:nu]+(-bias/gain)*qvel[:,:nu]
 return target.clamp(lower,upper)

class DiagnosticEnv(A1Env):
 reset_memory_mode='original'
 def reset(self,phase=0,seed=0,randomize=True,snapshot=None):
  result=super().reset(phase,seed,randomize,snapshot)
  if self.reset_memory_mode=='velocity_consistent':
   assert snapshot is None
   m=self.m;dtype=self.qpos.dtype
   gain=torch.as_tensor(m.actuator_gainprm[:,0],device=self.device,dtype=dtype)
   bias=torch.as_tensor(m.actuator_biasprm[:,2],device=self.device,dtype=dtype)
   target=velocity_memory(self.qpos,self.qvel,self.task.nu,gain,bias,self.lo,self.hi)
   command=self._world(target)
   self.ctrl.copy_(target);self.history.copy_(command[:,:,None].expand(-1,-1,3,-1));self.servo.copy_(command[:,:,:6])
   self.current_state.copy_(self.state());result=self.current_state
  return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args()
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(2)
 out=RUN/a.run;out.mkdir(exist_ok=False)
 c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 assert c['formal'] and c['configuration']['reset_mode']=='uniform_reference'
 manifest=json.loads((ROOT/'configs/a1_probe40_v1.json').read_text())
 assert hashlib.sha256((ROOT/'configs/a1_probe40_v1.json').read_bytes()).hexdigest()==c['manifest_sha256']
 policy=CollectorPolicy().to('cuda');policy.load_state_dict(c['policy']);policy.eval()
 seeds=c['configuration']['development_eval_seeds'];rows=[];start=time.monotonic()
 write(out/'registration.json',dict(checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),parent_controls=c['controls'],seeds=seeds,tasks=[r['task_id'] for r in manifest['tasks']],factor='Actuator delay/servo reset memory: static pose versus instantaneous q+kv/kp*qvel. Physical qpos/qvel, subsequent commands, policy, DR, task set and success thresholds unchanged.',scope='Paired development diagnostic only. No policy fitting and no A1/G1 success claim. CPU results motivated this check; candidate was not selected on locked test.'))
 for index,row in enumerate(manifest['tasks']):
  if stop:break
  env=DiagnosticEnv(ROOT/row['task_path']);modes=['original','velocity_consistent'] if index%2==0 else ['velocity_consistent','original']
  # Same seed must give exactly equal physical state/context in the two modes.
  start_states={}
  for mode in modes:
   env.reset_memory_mode=mode;env.reset(0,seed=seeds[0],randomize=True)
   start_states[mode]=(env.qpos.clone(),env.qvel.clone(),env.context.clone())
  assert all(torch.equal(x,y) for x,y in zip(start_states['original'],start_states['velocity_consistent']))
  for mode in modes:
   if stop:break
   env.reset_memory_mode=mode
   result=evaluate(env,policy,seeds)
   rows.extend(dict(task=row['task_id'],stratum=row['stratum'],mode=mode,**x) for x in result)
   write(out/'conditions.json',rows)
  env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
 counts={mode:dict(n=sum(r['mode']==mode for r in rows),regular=sum(r['regular_success'] for r in rows if r['mode']==mode),strict=sum(r['strict_success'] for r in rows if r['mode']==mode)) for mode in ['original','velocity_consistent']}
 complete=all(v['n']==40*len(seeds) for v in counts.values())
 write(out/'summary.json',dict(status='complete' if complete else 'budget_stop',seconds=time.monotonic()-start,counts=counts,formal_A1_complete=False,interpretation='Compare paired conditions and per-stratum errors before any training change. This script never selects a training revision or declares strong-positive A1.'))
 print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
