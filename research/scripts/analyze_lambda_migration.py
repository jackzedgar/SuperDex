"""CPU analysis of migration results; never updates a model or frozen evidence."""
import json,csv,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main():
 run=ROOT/'runs/lambda_migration_20261027';core=ROOT/'runs/core_iteration_20261027';new=read(run/'frozen_evaluation/results.json');old=read(core/'core_heldout/results.json');completion=read(core/'core_completion.json');table=[];paired=[]
 for row in new:
  if row['subset']=='heldout':baseline=next(x for x in old if x['arm']==row['arm'] and x['steps']==completion['steps_per_arm'] and x['repeat']==row['repeat']);folder=ROOT/baseline['folder']
  else:
   folder=(ROOT/completion['final_checkpoints'][row['arm']]).parent/'evaluation'/f"step_{completion['steps_per_arm']:06d}";baseline=read(folder/'summary.json')
  target=ROOT/row['folder'];trialrows=[]
  for i in range(row['n']):
   with np.load(folder/f'trial_{i:02d}.npz') as a,np.load(target/f'trial_{i:02d}.npz') as b:
    n=min(len(a['position_error']),len(b['position_error']))
    trialrows.append(dict(trial=i,common_steps=n,source_position_mm=float(np.mean(a['position_error'][:n])*1000),lambda_position_mm=float(np.mean(b['position_error'][:n])*1000),source_rotation_deg=float(np.rad2deg(np.mean(a['rotation_error'][:n]))),lambda_rotation_deg=float(np.rad2deg(np.mean(b['rotation_error'][:n]))),initial_qpos_max_delta=float(np.max(np.abs(a['qpos'][0]-b['qpos'][0]))),initial_qvel_max_delta=float(np.max(np.abs(a['qvel'][0]-b['qvel'][0]))),first_action_max_delta=float(np.max(np.abs(a['raw_action'][0]-b['raw_action'][0]))),first_control_max_delta=float(np.max(np.abs(a['ctrl'][0]-b['ctrl'][0])))))
  aggregate={k:float(np.mean([t[k] for t in trialrows])) for k in ['common_steps','source_position_mm','lambda_position_mm','source_rotation_deg','lambda_rotation_deg']}
  result=dict(arm=row['arm'],subset=row['subset'],repeat=row['repeat'],n=row['n'],source_successes=baseline['successes'],lambda_successes=row['successes'],source_return=baseline['mean_return'],lambda_return=row['mean_return'],source_mean_length=baseline['mean_length'],lambda_mean_length=row['mean_length'],source_median_first_failure_s=baseline['median_first_failure_s'],lambda_median_first_failure_s=row['median_first_failure_s'],source_failure_reasons=dict(collections.Counter(str(t['reason']) for t in baseline['trials'])),lambda_failure_reasons=dict(collections.Counter(str(t['reason']) for t in row['trials'])),**aggregate)
  table.append(result);paired.append(dict(**result,trials=trialrows,source_folder=str(folder.relative_to(ROOT)),lambda_folder=row['folder']))
 ledger=read(run/'gpu_ledger.json');gpu_seconds=sum(r['process_seconds'] for r in ledger);assert gpu_seconds<=1200
 smokes={}
 for arm in ['P','M']:
  summary=read(run/'smoke'/arm/'summary.json');rows=list(csv.DictReader((run/'smoke'/arm/'training.csv').open()));smokes[arm]=dict(summary=summary,updates=rows)
  assert summary['training_interactions']==128 and summary['ppo_optimizer_updates']==4 and summary['actor_parameter_l2_change']>0
  assert summary['optimizer_step_after']==summary['optimizer_step_before']+4 and summary['model_frozen']
  if arm=='M':assert summary['auxiliary_model_predictions']==64 and float(rows[0]['model_actor_norm'])>0
 wm=read(run/'wm_smoke/wm/config.json');assert wm['paired_updates']==2
 report={'endpoint_comparison':table,'paired_source_lambda_trials':paired,'gpu_process_seconds':gpu_seconds,'gpu_process_minutes':gpu_seconds/60,'gpu_ledger':ledger,'smoke':smokes,'world_model_smoke':wm,'source_hardware':'RTX A5000 / driver 580.178.04 / Warp 1.18.0 CUDA 13.4','lambda_hardware':'RTX 2080 Ti / driver 535.247.01 / Warp 1.18.0+cu12 CUDA 12.9','interpretation':'Hardware migration validation, not a new algorithm comparison. Repeated endpoint evaluations measure the same initial conditions; smoke updates are not research learning results.'}
 (run/'migration_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 with (run/'source_lambda_comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
 print(json.dumps({'table':table,'gpu_process_seconds':gpu_seconds,'smoke_steps':{a:smokes[a]['summary']['training_interactions'] for a in smokes},'wm_updates_each':wm['paired_updates']},indent=2))
if __name__=='__main__':main()
