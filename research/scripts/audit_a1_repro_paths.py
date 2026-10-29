"""Bounded development-only audit of reset, cache and CUDA capture reproducibility."""
import argparse,gc,hashlib,json,time
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
from a1_policy import CollectorPolicy
from train_a1_collector import Runner,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 torch.set_num_threads(2);out=RUN/a.run;out.mkdir(exist_ok=False)
 ck=RUN/'collector_full_v3_lr_001/checkpoint_final.pt';c=torch.load(ck,map_location='cpu',weights_only=False)
 policy=CollectorPolicy().cuda();policy.load_state_dict(c['policy']);policy.eval()
 manifest=json.loads((ROOT/'configs/a1_probe40_v1.json').read_text());seeds=c['configuration']['development_eval_seeds']
 affected='in_hand_reorientation_or_regrasp_clip-000077_o016_f060-119_rv2'
 chosen=[next(r for r in manifest['tasks'] if r['task_id']==affected)]
 for st in ['stable_grasp_transport','bimanual_handover','light_object_tool_use_below100g']:chosen.append(next(r for r in manifest['tasks'] if r['stratum']==st))
 write(out/'registration.json',dict(checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),tasks=[r['task_id'] for r in chosen],seeds=seeds,scope='Implementation audit only. Chosen by exact reproducibility discrepancy plus first frozen-manifest task in other strata. No training, no task exclusion and no final test.'))
 start=time.monotonic();records=[]
 def measure(env,row,mode):
  result=evaluate(env,policy,seeds)
  records.extend(dict(task=row['task_id'],stratum=row['stratum'],mode=mode,**r) for r in result)
  write(out/'conditions.json',records)
  print(json.dumps(dict(task=row['task_id'],mode=mode,regular=sum(r['regular_success'] for r in result),strict=sum(r['strict_success'] for r in result))),flush=True)
 for row in chosen:
  env=A1Env(ROOT/row['task_path']);measure(env,row,'fresh_captured');measure(env,row,'same_env_repeat')
  runner=Runner(env,policy,71929,reset_mode='uniform_reference')
  with torch.no_grad():runner.rollout(64)
  measure(env,row,'after_stochastic_rollout');del runner
  env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
  env=A1Env(ROOT/row['task_path']);runner=Runner(env,policy,71929,reset_mode='uniform_reference');measure(env,row,'fresh_with_runner')
  del runner;env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
  if row['task_id']==affected:
   env=A1Env(ROOT/row['task_path'],capture=False);measure(env,row,'fresh_uncaptured');measure(env,row,'uncaptured_repeat')
   env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
 write(out/'summary.json',dict(status='complete',seconds=time.monotonic()-start,conditions=len(records),formal_A1_complete=False))
if __name__=='__main__':main()
