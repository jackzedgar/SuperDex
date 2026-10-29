"""Four independent single-world workers, one synchronous shared PPO policy.
File IPC lives in this run only; consumed rollout temporaries are removed.
"""
import argparse,json,time,signal,math,hashlib,os
from pathlib import Path
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_policy import CollectorPolicy
from a1_development_monitor import regression_check
from a1_multitask_runner import MultiRunner
from train_a1_collector import atomic_save,ppo_update
stop=False
def halt(*_):
 global stop;stop=True
def wait(path,out,timeout=1800):
 start=time.monotonic()
 while not path.exists():
  if stop or list(out.glob('fatal_rank*.json')):raise InterruptedError('Stopped or peer failed; preserve last common checkpoint')
  if time.monotonic()-start>timeout:raise TimeoutError(str(path))
  time.sleep(.2)
 return path
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--run',required=True);p.add_argument('--manifest',default='configs/a1_probe40_v1.json');p.add_argument('--steps',type=int,default=2097152);p.add_argument('--rollout',type=int,default=512);p.add_argument('--eval-every',type=int,default=32768);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--config',default='configs/a1_collector_full_v1.json');a=p.parse_args()
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(2)
 out=RUN/a.run;out.mkdir(exist_ok=True);ipc=out/'ipc';ipc.mkdir(exist_ok=True)
 config=json.loads((ROOT/a.config).read_text());manifest=json.loads((ROOT/a.manifest).read_text());seed=config['seed']
 assert manifest['status']=='FROZEN_PROBE40_V1'
 torch.manual_seed(seed+a.rank*1000);policy=CollectorPolicy(seed).to('cuda');torch.manual_seed(seed+1000*a.rank+100)
 opt=torch.optim.Adam(policy.parameters(),lr=2e-4,betas=(.9,.95)) if a.rank==0 else None
 generator=torch.Generator(device='cuda').manual_seed(seed+1)
 initial=load(a.resume) if a.resume else None
 controls=initial['controls'] if initial else 0;iteration=initial['iteration'] if initial else 0
 if initial:
  assert initial.get('configuration',{}).get('reset_mode','adaptive_anchor')==config.get('reset_mode','adaptive_anchor'),'Use a fresh initialization for protocol correction'
  policy.load_state_dict(initial['policy'])
  if opt:opt.load_state_dict(initial['optimizer']);generator.set_state(initial['ppo_generator'])
 start=time.monotonic();runner=None;stats=None;last_evaluated=-1;bands={};quality_complete=False;development_stop=False;stage_start_controls=controls;regressions=[]
 def checkpoint(label):
  if a.rank==0:
   value=dict(policy=policy.state_dict(),optimizer=opt.state_dict(),controls=controls,iteration=iteration,global_sampler=stats,ppo_generator=generator.get_state(),configuration=config,manifest_sha256=hashlib.sha256((ROOT/a.manifest).read_bytes()).hexdigest(),bands=bands,formal=not a.smoke)
   atomic_save(value,out/label)
 def evaluation():
  nonlocal last_evaluated,bands,quality_complete,development_stop
  rows=runner.evaluation(a.rank,a.workers,config['development_eval_seeds'])
  write(out/f'evaluation_{controls:08d}_rank{a.rank}.json',rows)
  if a.rank==0:
   allrows=[]
   for rank in range(a.workers):
    file=wait(out/f'evaluation_{controls:08d}_rank{rank}.json',out);allrows+=json.loads(file.read_text())
   sr=sum(r['regular_success'] for r in allrows)/len(allrows);strict=sum(r['strict_success'] for r in allrows)/len(allrows)
   summary=dict(controls=controls,regular_success_rate=sr,strict_success_rate=strict,n=len(allrows),per_task={row['task_id']:dict(regular=sum(r['regular_success'] for r in allrows if r['task']==row['task_id']),strict=sum(r['strict_success'] for r in allrows if r['task']==row['task_id']),n=sum(r['task']==row['task_id'] for r in allrows)) for row in manifest['tasks']},all_conditions=allrows,scope='development only; full trajectories from phase0; no evaluation episode enters model data')
   baseline_path=out/f'evaluation_{stage_start_controls:08d}.json'
   if config.get('regression_stop_enabled') and controls>stage_start_controls and baseline_path.exists():
    check=regression_check(summary,json.loads(baseline_path.read_text()));regressions.append(check['joint_degradation']);summary['continuation_comparison']=check
    development_stop=len(regressions)>=3 and all(regressions[-3:])
   write(out/f'evaluation_{controls:08d}.json',summary)
   for name,interval in config['quality_bands'].items():
    if name not in bands and interval[0]<=sr<=interval[1]:
     bands[name]=dict(controls=controls,regular_success_rate=sr,strict_success_rate=strict);checkpoint(f'quality_{name}.pt')
     q=out/f'quality_{name}.pt'
     write(out/f'quality_{name}_passport.json',dict(status='DEVELOPMENT_BAND_REACHED',quality=name,checkpoint=str(q.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),probe40_frozen=True,task_ids=[r['task_id'] for r in manifest['tasks']],manifest_sha256=hashlib.sha256((ROOT/a.manifest).read_bytes()).hexdigest(),formal=not a.smoke,reset_mode=config.get('reset_mode','adaptive_anchor'),task_sampling=config.get('task_sampling','global_usable_bins'),**bands[name]))
   print('EVAL',controls,sr,strict,flush=True)
   write(ipc/f'evaluated_{controls:08d}.json',dict(done=True,quality_complete=set(bands)=={'20','50','80'} and not a.smoke,development_stop=development_stop))
  else:wait(ipc/f'evaluated_{controls:08d}.json',out)
  result=json.loads((ipc/f'evaluated_{controls:08d}.json').read_text());quality_complete=result['quality_complete'];development_stop=result.get('development_stop',False);last_evaluated=controls
 try:
  runner=MultiRunner(manifest,policy,seed+10000*a.rank,cache_limit=config.get('cache_capacity_per_worker',2),reset_mode=config.get('reset_mode','adaptive_anchor'),determinism=config.get('simulation_determinism','constraint'))
  if initial:runner.load_stats(initial['global_sampler'])
  if a.rank==0:
   stats=runner.stats();bands=initial.get('bands',{}) if initial else {}
   write(out/'run_configuration.json',dict(config,run=a.run,actual_steps_target=a.steps,workers=a.workers,rollout_per_worker=a.rollout,formal=not a.smoke,source_commit=os.popen('git rev-parse HEAD').read().strip(),resume=str(a.resume) if a.resume else None))
   checkpoint('checkpoint_initial.pt')
  if not a.smoke:evaluation()
  if initial:
   rngfile=a.resume.parent/f'rank{a.rank}_latest_rng.pt'
   if rngfile.exists():
    saved=load(rngfile);assert saved['iteration']<=iteration
    runner.rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    runner.reset()
    write(out/f'resume_rank{a.rank}.json',dict(parent_checkpoint=str(a.resume),parent_controls=controls,optimizer_restored=a.rank==0,worker_rng_restored=True,rng_source_iteration=saved['iteration'],simulation_restart='Fresh valid reference reset after initial evaluation; ongoing physics/solver history is not restored. Not bitwise physical continuation.'))
   else:raise RuntimeError('Resume requires saved worker RNG state; no silent RNG reseed')
  while controls<a.steps and not stop and not quality_complete and not development_stop:
   if a.rank==0:
    atomic_save(dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},stats=stats),ipc/'weights_latest.pt')
    write(ipc/f'command_{iteration:07d}.json',dict(iteration=iteration,controls=controls))
   else:
    wait(ipc/f'command_{iteration:07d}.json',out);packet=load(ipc/'weights_latest.pt');policy.load_state_dict(packet['policy']);runner.load_stats(packet['stats'])
   runner.events=[];runner.fallback_events=[];runner.episodes=[]
   n=min(a.rollout,(a.steps-controls)//a.workers)
   if n<=0:break
   before=time.monotonic();data=runner.rollout(n,lambda:stop)
   if data is None:break
   payload=dict(data={k:v.cpu() for k,v in data.items()},events=runner.events,fallback_events=runner.fallback_events,episodes=runner.episodes,controls=len(data['reward']),bootstrap_controls=runner.bootstrap_controls,task_controls=runner.task_controls,seconds=time.monotonic()-before,cache_size=len(runner.cache),warmup_controls=runner.warmup_controls,evaluation_controls=runner.evaluation_controls,peak_torch_bytes=torch.cuda.max_memory_allocated())
   atomic_save(payload,ipc/f'rollout_{iteration:07d}_rank{a.rank}.pt')
   atomic_save(dict(rng=runner.rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),controls=controls+n*a.workers,iteration=iteration),out/f'rank{a.rank}_latest_rng.pt')
   if a.rank==0:
    packets=[load(wait(ipc/f'rollout_{iteration:07d}_rank{rank}.pt',out)) for rank in range(a.workers)]
    joined={k:torch.cat([row['data'][k] for row in packets]).to('cuda') for k in data};added=len(joined['reward']);controls+=added
    fraction=min(1,controls/(a.steps*.8));lr=2e-4*(.03+.97*.5*(1+math.cos(math.pi*fraction)))
    if config.get('learning_rate_mode')=='constant_continuation':lr=float(config['continuation_learning_rate'])
    diag=ppo_update(policy,opt,joined,generator,lr)
    runner.merge(stats,[row['events'] for row in packets],[row['fallback_events'] for row in packets]);stats=runner.stats()
    row=dict(iteration=iteration+1,real_controls=controls,elapsed_seconds=time.monotonic()-start,mean_rollout_reward=float(joined['reward'].mean()),action_saturation_fraction=float((joined['action'].abs()>1).float().mean()),learning_rate=lr,worker_stats=[{k:v for k,v in r.items() if k not in ['data','events','fallback_events','episodes']} for r in packets],**diag)
    with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    with (out/'training_episodes.jsonl').open('a') as f:
     for packet in packets:
      for ep in packet['episodes']:f.write(json.dumps(dict(iteration=iteration+1,**ep))+'\n')
    print(json.dumps(row),flush=True)
    for rank in range(a.workers):(ipc/f'rollout_{iteration:07d}_rank{rank}.pt').unlink()
    atomic_save(dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},stats=stats,controls=controls),ipc/'updated_latest.pt')
    write(ipc/f'updated_{iteration:07d}.json',dict(controls=controls))
   else:
    wait(ipc/f'updated_{iteration:07d}.json',out);packet=load(ipc/'updated_latest.pt');policy.load_state_dict(packet['policy']);runner.load_stats(packet['stats']);controls=packet['controls']
   iteration+=1
   if a.rank==0:checkpoint('checkpoint_latest.pt')
   if not a.smoke and (controls-last_evaluated>=a.eval_every or controls>=a.steps):
    if a.rank==0:checkpoint(f'checkpoint_{controls:08d}.pt')
    evaluation()
   if a.smoke and controls>=a.steps:evaluation()
  if a.rank==0:checkpoint('checkpoint_final.pt');write(out/'summary.json',dict(status='bounded_stop' if stop else 'development_regression_stop' if development_stop else 'quality_complete' if quality_complete else 'steps_complete',real_controls=controls,iterations=iteration,seconds=time.monotonic()-start,bands=bands,formal=not a.smoke))
 except BaseException as error:
  write(out/f'fatal_rank{a.rank}.json',dict(error=repr(error),controls=controls,iteration=iteration))
  if a.rank==0 and runner is not None:checkpoint('checkpoint_interrupted.pt')
  raise
if __name__=='__main__':main()
