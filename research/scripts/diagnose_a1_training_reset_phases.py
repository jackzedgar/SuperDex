"""Preregistered random-phase reset audit; no optimizer, policy admission or final test."""
import argparse,gc,hashlib,json,signal,time
from pathlib import Path
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_policy import CollectorPolicy,policy_inputs
from diagnose_a1_reset_memory import DiagnosticEnv
stop=False
def halt(*_):
 global stop;stop=True

@torch.no_grad()
def trace(env,policy,phase,seed,limit):
 env.reset(phase,seed=seed,randomize=True)
 sh=env.state()[:,None].repeat(1,3,1);ah=torch.zeros(1,3,52,device=env.device)
 physical=[x.detach().cpu().numpy().tobytes() for x in [env.qpos,env.qvel,env.context]]
 initial_hash=hashlib.sha256(b''.join(physical)).hexdigest();errors=[];rewards=[]
 requested=min(limit,env.task.length-1-phase)
 for _ in range(requested):
  obs,goal=policy_inputs(env.task,sh,ah,env.phase)
  action=policy.distribution(obs,goal).mean*env.task.mask.repeat_interleave(26,-1)
  state,reward,done,info=env.step(action)
  errors.append([float(info['pos_error']),float(info['rot_error'])]);rewards.append(float(reward))
  sh=torch.cat([sh[:,1:],state[:,None]],1);ah=torch.cat([ah[:,1:],env.last_action[:,None]],1)
  if bool(done):break
 return dict(initial_physical_context_sha256=initial_hash,start_phase=phase,requested_controls=requested,controls=len(errors),survived_requested_prefix=len(errors)==requested and not bool(info['failure']),full32_window=requested==limit,reached_reference_end=bool(env.timeout),failure_reason=int(info['reason']),finite=bool(info['finite']),error_trace_m_rad=errors,reward_sum=sum(rewards))

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args()
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(2)
 config=json.loads((ROOT/'configs/a1_training_reset_phase_audit_20261029.json').read_text())
 out=RUN/a.run;out.mkdir(exist_ok=False);start=time.monotonic()
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 assert checkpoint['formal'] and checkpoint['configuration']['reset_mode']=='uniform_reference'
 manifest_path=ROOT/'configs/a1_probe40_v1.json';manifest=json.loads(manifest_path.read_text())
 assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==checkpoint['manifest_sha256']
 policy=CollectorPolicy().cuda();policy.load_state_dict(checkpoint['policy']);policy.eval()
 rng=np.random.default_rng(config['phase_seed']);planned=[]
 for task in manifest['tasks']:
  with np.load(ROOT/task['task_path']/'reference.npz') as data:length=len(data['qpos'])
  phases=sorted(int(x) for x in rng.choice(np.arange(1,length-1),size=2,replace=False))
  planned.append(dict(task=task['task_id'],stratum=task['stratum'],path=task['task_path'],phases=phases))
 write(out/'registration.json',dict(configuration=config,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),parent_controls=checkpoint['controls'],planned_tasks=planned,planned_conditions=640,scope='640 short development trajectories:40tasks x2nonzero reference phases x2DRseeds x2memory modes x2repeats. No whole-trajectory success rate, no policy training or quality passport, no locked test.'))
 rows=[]
 for task_index,task in enumerate(planned):
  if stop:break
  env=DiagnosticEnv(ROOT/task['path'])
  for phase_index,phase in enumerate(task['phases']):
   for seed_index,seed in enumerate(config['dr_seeds']):
    if stop:break
    order=['original','velocity_consistent','velocity_consistent','original'] if (task_index+phase_index+seed_index)%2==0 else ['velocity_consistent','original','original','velocity_consistent']
    repeats={'original':0,'velocity_consistent':0};initial_hash=None
    for mode in order:
     if stop:break
     env.reset_memory_mode=mode;result=trace(env,policy,phase,int(seed),config['maximum_controls_per_trajectory'])
     if initial_hash is None:initial_hash=result['initial_physical_context_sha256']
     assert initial_hash==result['initial_physical_context_sha256'],'Physical state/context changed across paired memory modes'
     repeats[mode]+=1;rows.append(dict(task=task['task'],stratum=task['stratum'],seed=seed,mode=mode,repeat=repeats[mode],**result))
     write(out/'conditions.json',rows)
    write(out/'progress.json',dict(conditions=len(rows),tasks_started=task_index+1,seconds=time.monotonic()-start))
   if stop:break
  env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
  print(json.dumps(dict(task=task['task'],conditions=len(rows))),flush=True)
 counts={mode:dict(n=sum(r['mode']==mode for r in rows),survived_prefix=sum(r['mode']==mode and r['survived_requested_prefix'] for r in rows),full32_windows=sum(r['mode']==mode and r['full32_window'] for r in rows)) for mode in ['original','velocity_consistent']}
 write(out/'summary.json',dict(status='complete' if len(rows)==640 else 'budget_stop',seconds=time.monotonic()-start,conditions=len(rows),counts=counts,formal_A1_complete=False,interpretation='Assess paired same-phase effects, across-repeat variability and all four task strata before selecting the remaining revision. Short-prefix survival is not full-trajectory collector success or G1 prediction accuracy. No training is launched by this audit.'))
if __name__=='__main__':main()
