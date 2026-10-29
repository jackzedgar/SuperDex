"""Frozen-policy versus zero-residual paired development audit, not training."""
import argparse,gc,getpass,hashlib,json,os,signal,socket,time
from pathlib import Path
import numpy as np,torch

CONTROLLERS=('learned','zero_residual')
stop=False
def halt(*_):
 global stop;stop=True
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def order_for(task_index,seed_index):
 a,b=CONTROLLERS
 return (a,b,b,a) if (task_index+seed_index)%2==0 else (b,a,a,b)
def state_hash(values):
 h=hashlib.sha256()
 for value in values:
  a=value.detach().cpu().contiguous().numpy();h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
 return h.hexdigest()
def policy_hash(policy):return state_hash([v for _,v in sorted(policy.state_dict().items())])
def action_for(env,policy,sh,ah,controller,input_fn):
 if controller=='zero_residual':return torch.zeros_like(env.last_action)
 if controller!='learned':raise ValueError(controller)
 obs,goal=input_fn(env.task,sh,ah,env.phase)
 return policy.distribution(obs,goal).mean*env.task.mask.repeat_interleave(26,-1)

def endpoint_metrics(env,info):
 """Endpoint forces/limits are diagnostics, not substep maxima or new thresholds."""
 import warp as wp
 if not hasattr(env,'_audit_force'):
  env._audit_force=wp.to_torch(env.wd.actuator_force)
  env._audit_limits=torch.tensor(np.max(abs(env.m.actuator_forcerange),axis=1),device=env.device,dtype=env.qpos.dtype)
  env._audit_limited=torch.tensor(env.m.actuator_forcelimited.astype(bool),device=env.device)&(env._audit_limits>0)
  assert bool(env._audit_limited.any()),'No force-limited actuators for the registered occupancy diagnostic'
 reference=env.task.qpos[env.phase]
 finger=torch.cat([env.qpos[:,k*26+6:k*26+26]-reference[:,k*26+6:k*26+26] for k in range(len(env.task.sides))],-1)
 ratio=env._audit_force[:,env._audit_limited].abs()/env._audit_limits[env._audit_limited]
 at_control_limit=(env.ctrl<=env.lo+1e-6)|(env.ctrl>=env.hi-1e-6)
 active=env.task.mask.repeat_interleave(26,-1).bool()
 values=torch.stack([info['wrist_error'].reshape(()),finger.abs().mean(),ratio.max(),(ratio>=1-1e-5).float().mean(),at_control_limit.float().mean(),(env.last_action[active].abs()>=1-1e-5).float().mean()])
 return values.detach().cpu().tolist()

@torch.no_grad()
def trace(env,policy,seed,controller,input_fn,measure_fn=endpoint_metrics,should_stop=lambda:stop):
 env.reset(0,seed=int(seed),randomize=True)
 initial=state_hash([env.qpos,env.qvel,env.context,env.ctrl,env.history,env.servo,env.last_action])
 sh=env.state()[:,None].repeat(1,3,1);ah=torch.zeros(1,3,52,device=env.device)
 errors=[];measurements=[];contacts=[];total_reward=0.;info=None;complete=False
 for _ in range(env.task.length-1):
  if should_stop():break
  action=action_for(env,policy,sh,ah,controller,input_fn)
  state,reward,done,info=env.step(action)
  errors.append([float(info['pos_error']),float(info['rot_error'])]);measurements.append(measure_fn(env,info));contacts.append(env.contact_summary.detach().cpu().tolist());total_reward+=float(reward)
  sh=torch.cat([sh[:,1:],state[:,None]],1);ah=torch.cat([ah[:,1:],env.last_action[:,None]],1)
  if bool(done):complete=True;break
 finite=bool(info['finite']) if info is not None else None
 assert np.isfinite(errors).all() and np.isfinite(measurements).all(),'Nonfinite diagnostic trajectory'
 return dict(seed=int(seed),controller=controller,initial_state_context_memory_sha256=initial,start_phase=0,requested_controls=env.task.length-1,controls=len(errors),complete_episode=complete,regular_success=bool(info['regular_success']) if complete else None,strict_success=bool(info['strict_success']) if complete else None,reached_reference_end=bool(env.timeout) if info is not None else False,failure_reason=int(info['reason']) if info is not None else None,finite=finite,error_trace_m_rad=errors,endpoint_metrics=measurements,contact_trace=contacts,return_sum=total_reward,mean_errors=info['mean_errors'].detach().cpu().tolist()[0] if complete else None)

def summarize(rows,expected):
 counts={c:dict(recorded=sum(r['controller']==c for r in rows),complete=sum(r['controller']==c and r['complete_episode'] for r in rows),regular=sum(r['controller']==c and r['regular_success'] is True for r in rows),strict=sum(r['controller']==c and r['strict_success'] is True for r in rows)) for c in CONTROLLERS}
 return dict(status='complete' if len(rows)==expected and all(r['complete_episode'] for r in rows) else 'budget_stop_or_incomplete',conditions=len(rows),planned_conditions=expected,counts=counts,formal_A1_complete=False,quality_admission=False,interpretation='Paired development control diagnostic only. Incomplete coverage is inconclusive. Both controllers failing does not prove reference impossibility. No PPO updates, new quality passports or G1 claims.')

def validate_parent(root,config):
 run=root/'runs/a1_full_20261028';folder=run/config['parent_run'];summary=json.loads((folder/'summary.json').read_text())
 assert summary['status'] in ['bounded_stop','steps_complete','development_regression_stop'],'Unexpected parent termination'
 assert set(summary.get('bands',{}))!={'20','50','80'},'Quality prerequisites complete: this diagnostic must be skipped'
 assert not all((folder/f'quality_{q}_passport.json').exists() for q in ['20','50','80']),'All quality passports exist: review formal pipeline instead'
 assert not list(folder.glob('fatal_rank*.json')),'Unexpected parent failures require separate review'
 jobs=json.loads((run/'gpu_ledger.json').read_text())['jobs']
 own=[j for j in jobs if j['name']==config['run']];assert len(own)==1
 own=own[0];assert own['phase']=='implementation' and own['limit_seconds']==config['maximum_gpu_seconds'] and own['status']=='running' and own['child_pid']==os.getpid(),'Run only through the registered budget supervisor'
 parent=[j for j in jobs if config['parent_run'] in j.get('command',[]) and 'scripts/train_a1_shared_collector.py' in j.get('command',[])]
 assert len(parent)==4 and all(j['status']=='finished' for j in parent)
 assert all(j.get('returncode')==0 or (j.get('reason')=='process_budget' and summary['status']=='bounded_stop') for j in parent),'Unexpected worker return code'
 assert not any(j['status']=='running' and j['name']!=config['run'] for j in jobs),'Another A1 GPU reservation is active'
 for j in parent:
  supervisor=Path('/proc')/str(j['supervisor_pid'])/'stat'
  if supervisor.exists():
   stat=supervisor.read_text().split()
   assert stat[21]!=j['supervisor_start_ticks'] or stat[2] in ['Z','X'],'Parent budget supervisor still active'
  pid=j.get('child_pid');p=Path('/proc')/str(pid)
  if p.exists():
   command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   assert not ('train_a1_shared_collector.py' in command and config['parent_run'] in command and (p/'stat').read_text().split()[2] not in ['Z','X']),'Parent worker still active'
 return folder/'checkpoint_final.pt'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
 root=Path(__file__).resolve().parents[1]
 assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran'
 config=json.loads(args.config.read_text());assert config['status']=='PREREGISTERED_NOT_LAUNCHED'
 assert config['controllers']==list(CONTROLLERS) and config['repeats']==2 and config['maximum_gpu_seconds']==1800
 for name,digest in config['source_sha256'].items():assert sha(root/name)==digest,name
 checkpoint_path=validate_parent(root,config);checkpoint_sha=sha(checkpoint_path)
 manifest_path=root/'configs/a1_probe40_v1.json';manifest=json.loads(manifest_path.read_text())
 checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
 parent_summary=json.loads((checkpoint_path.parent/'summary.json').read_text())
 assert checkpoint['controls']==parent_summary['real_controls'] and checkpoint['bands']==parent_summary['bands'],'Final checkpoint/summary disagreement'
 assert checkpoint['formal'] and checkpoint['manifest_sha256']==sha(manifest_path) and checkpoint['configuration']['reset_mode']=='uniform_reference'
 seeds=checkpoint['configuration']['development_eval_seeds'];assert seeds==config['development_seeds'] and len(manifest['tasks'])==40
 assert [r['task_id'] for r in manifest['tasks']]==config['task_ids']
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(2)
 from a1_env import A1Env
 from a1_policy import CollectorPolicy,policy_inputs
 from a1_data_sources import write
 out=root/'runs/a1_full_20261028'/config['run'];out.mkdir(exist_ok=False)
 started=time.monotonic();policy=CollectorPolicy().cuda();policy.load_state_dict(checkpoint['policy']);policy.eval();before=policy_hash(policy)
 expected=len(manifest['tasks'])*len(seeds)*4
 write(out/'registration.json',dict(configuration=config,configuration_sha256=sha(args.config),checkpoint=str(checkpoint_path.relative_to(root)),checkpoint_sha256=checkpoint_sha,parent_controls=checkpoint['controls'],policy_sha256=before,planned_conditions=expected,endpoint_metric_columns=['wrist_position_max_m','finger_joint_absolute_error_mean_rad','actuator_force_over_limit_max','actuator_force_limit_occupancy_fraction','actuator_control_limit_occupancy_fraction','residual_clip_occupancy_fraction'],endpoint_metric_scope='At control endpoints only, not maxima across the ten physical substeps.',scope='Original phase0 initialization in both modes; deterministic policy mean versus zero residual, same reference and DR, no updates or final-test access.'))
 rows=[]
 for task_index,task in enumerate(manifest['tasks']):
  if stop:break
  env=A1Env(root/task['task_path'])
  try:
   for seed_index,seed in enumerate(seeds):
    repeats={c:0 for c in CONTROLLERS};initial=None
    for controller in order_for(task_index,seed_index):
     if stop:break
     result=trace(env,policy,seed,controller,policy_inputs)
     if initial is None:initial=result['initial_state_context_memory_sha256']
     assert initial==result['initial_state_context_memory_sha256'],'Paired initial state/context/command memory differ'
     repeats[controller]+=1;rows.append(dict(task=task['task_id'],stratum=task['stratum'],repeat=repeats[controller],**result));write(out/'conditions.json',rows)
    write(out/'progress.json',dict(conditions=len(rows),seconds=time.monotonic()-started,task=task['task_id']))
    if stop:break
  finally:
   env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
  print(json.dumps(dict(task=task['task_id'],conditions=len(rows))),flush=True)
 after=policy_hash(policy);assert before==after,'Policy weights changed during diagnostic'
 summary=summarize(rows,expected);summary.update(seconds=time.monotonic()-started,policy_unchanged=True,checkpoint_sha256=checkpoint_sha)
 write(out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
