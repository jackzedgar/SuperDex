"""Bounded same-method expert followup with independent original-world concurrency."""
import argparse,datetime,getpass,hashlib,json,os,signal,socket,time,traceback
from pathlib import Path
import torch
import warp as wp
from a1_policy import CollectorPolicy
from a1_env import A1Env
from a1_original_pool_runner import OriginalPoolRunner
from train_a1_collector import ppo_update,evaluate
R=Path(__file__).resolve().parents[1];STOP=False
def signal_stop(*_):
 global STOP
 STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def save(p,x):
 t=p.with_suffix(p.suffix+'.tmp');torch.save(x,t);t.replace(p)
def same(a,b):
 if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:same(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
 assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran'
 c=json.loads(a.config.read_text());assert c['max_new_controls']==131072 and c['batch_controls']==2048 and c['nworld']==8 and c['lr']==3e-5
 assert c['scientific_revision']==3 and c['training_memory']=='velocity_consistent' and c['reset_mode']=='uniform_reference'
 for path,digest in c['source_hashes'].items():assert sha(R/path)==digest,path
 parent_path=R/c['parent_checkpoint'];assert sha(parent_path)==c['parent_sha256']
 out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
 signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop);torch.set_num_threads(1)
 start=time.monotonic();train_deadline=start+c['training_seconds'];deadline=start+c['job_limit_seconds']-100
 parent=torch.load(parent_path,map_location='cpu',weights_only=False);policy=CollectorPolicy(280043).cuda();policy.load_state_dict(parent['policy'])
 assert not parent.get('smoke',False) and parent['configuration']['task']==c['task_id']
 opt=torch.optim.Adam(policy.parameters(),lr=3e-5,betas=(.9,.95));opt.load_state_dict(parent['optimizer']);gen=torch.Generator(device='cuda');gen.set_state(parent['ppo_generator'].cpu())
 same(policy.state_dict(),parent['policy']);same(opt.state_dict(),parent['optimizer'])
 # Dedicated original evaluation environment never resets a training lane.
 eval_env=A1Env(R/c['task_path'],capture=True,determinism='constraint')
 runner=OriginalPoolRunner(R/c['task_path'],policy,parent['worker_rng'],validate=False)
 torch.set_rng_state(parent['torch_rng'].cpu());torch.cuda.set_rng_state(parent['cuda_rng'].cpu())
 prior_controls=parent['cumulative_task_controls'];prior_updates=parent['cumulative_task_updates'];controls=0;updates=0;uncommitted=0;evaluations=[];eval_controls=0;last_eval_controls=None
 write(out/'restore_gate.json',dict(status='PASS',policy_optimizer_exact=True,parent_sha256=c['parent_sha256'],parent_task_controls=prior_controls,parent_task_updates=prior_updates,smoke_weights_reused=False))
 write(out/'run_configuration.json',dict(c,started_beijing=now(),fresh_physical_resets=True,lane_rng_derivation=runner.lane_seeds,configuration_sha256=sha(a.config),formal_A1_complete=False))
 class Guard:
  def __getattr__(self,k):return getattr(eval_env,k)
  def reset(self,*args,**kwargs):
   if STOP or time.monotonic()>=deadline:raise InterruptedError('Evaluation bounded stop')
   x=eval_env.reset(*args,**kwargs);w=eval_env._world(eval_env.qpos[:,:eval_env.task.nu]);assert torch.equal(eval_env.ctrl,eval_env.qpos[:,:eval_env.task.nu]) and torch.equal(eval_env.history,w[:,:,None].expand(-1,-1,3,-1)) and torch.equal(eval_env.servo,w[:,:,:6]);return x
  def step(self,a):
   nonlocal eval_controls
   if STOP or time.monotonic()>=deadline:raise InterruptedError('Evaluation bounded stop')
   eval_controls+=1;return eval_env.step(a)
 guard=Guard()
 def checkpoint(name):
  save(out/name,dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},optimizer=opt.state_dict(),ppo_generator=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
   lane_rng=runner.rng_state(),parent_controls=parent['parent_controls'],parent_iteration=parent['parent_iteration'],prior_task_controls=prior_controls,prior_task_updates=prior_updates,new_controls=controls,new_updates=updates,cumulative_task_controls=prior_controls+controls,cumulative_task_updates=prior_updates+updates,
   physical_training_controls=runner.real_controls,ongoing_episode_controls=runner.ep_controls,configuration=c,smoke=False,formal_A1=False,quality_passports_allowed=False))
 def development():
  nonlocal last_eval_controls
  rows=[];torch.cuda.synchronize();torch.cuda.set_stream(eval_env.torch_stream);wp.set_stream(eval_env.stream)
  for seed in c['development_seeds']:
   try:rows += evaluate(guard,policy,[seed])
   except InterruptedError:break
   write(out/'evaluation_progress.json',dict(new_controls=controls,conditions=rows,complete=len(rows)==4))
  result=dict(scope='single_task_development_not_full160',task=c['task_id'],new_controls=controls,cumulative_task_controls=prior_controls+controls,complete=len(rows)==4,conditions=rows,n=len(rows),regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),original_initialization=True,quality_passport=False,time_beijing=now())
  write(out/f'evaluation_{controls:07d}.json',result)
  if result['complete']:write(out/'evaluation_latest_complete.json',result);last_eval_controls=controls
  evaluations.append(result);torch.cuda.set_stream(runner.stream);return result
 checkpoint('checkpoint_initial.pt');reason='control_target'
 try:
  while controls<c['max_new_controls'] and not STOP and time.monotonic()<train_deadline:
   before=time.monotonic();first_episode=len(runner.episodes);data=runner.rollout(2048,lambda:STOP or time.monotonic()>=train_deadline)
   if data is None:reason='training_time_cap';break
   n=len(data['reward'])
   if n!=2048:uncommitted=n;save(out/'pending_rollout.pt',{k:v.cpu() for k,v in data.items()});reason='training_time_cap';break
   assert all(bool(torch.isfinite(v).all()) for v in data.values())
   assert torch.equal(data['mask'],runner.mask[0].expand(2048,2))
   if updates==0:
    with torch.no_grad():
     ids=torch.arange(0,2048,31,device='cuda');dist=policy.distribution(data['obs'][ids],data['goal'][ids]);lp=(dist.log_prob(data['action'][ids])*data['mask'][ids].repeat_interleave(26,-1)).sum(-1);error=float((lp-data['old_logprob'][ids]).abs().max());assert error<=5e-4
    write(out/'preupdate_gate.json',dict(status='PASS',behavior_logprob_max_error=error,finite=True,total_controls=2048,layout='8x256',mask_exact=True))
   metrics=ppo_update(policy,opt,data,gen,3e-5);controls+=2048;updates+=1
   row=dict(time_beijing=now(),task=c['task_id'],new_controls=controls,cumulative_task_controls=prior_controls+controls,updates=updates,seconds=time.monotonic()-start,update_seconds=time.monotonic()-before,physical_training_controls=runner.real_controls,mean_rollout_reward=float(data['reward'].mean()),**metrics)
   with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   with (out/'training_episodes.jsonl').open('a') as f:
    for ep in runner.episodes[first_episode:]:f.write(json.dumps(dict(update=updates,**ep))+'\n')
   checkpoint('checkpoint_latest.pt');write(out/'progress.json',dict(status='TRAINING',**row));print('UPDATE',updates,controls,flush=True)
   if controls%32768==0:development()
  if STOP:reason='budget_or_signal'
  elif controls<c['max_new_controls']:reason='training_time_cap'
  checkpoint('checkpoint_final.pt')
  if last_eval_controls!=controls and not STOP and time.monotonic()<deadline:development()
  final=next((e for e in reversed(evaluations) if e['complete']),None)
  write(out/'summary.json',dict(status=reason,task=c['task_id'],new_controls=controls,updates=updates,cumulative_task_controls=prior_controls+controls,
   physical_training_controls=runner.real_controls,uncommitted_rollout_controls=uncommitted,physical_evaluation_controls=eval_controls,
   physical_total_env_controls=sum(e.physical_controls for e in runner.envs)+eval_env.physical_controls,
   latest_complete_development=final,final_checkpoint_evaluated=last_eval_controls==controls,checkpoint_sha256=sha(out/'checkpoint_final.pt'),seconds=time.monotonic()-start,
   full_A1_complete=False,quality_passports_created=False,time_beijing=now()))
 except BaseException as exc:
  checkpoint('checkpoint_failed_last_committed.pt');write(out/'fatal.json',dict(error=repr(exc),traceback=traceback.format_exc(),new_controls=controls,physical_training_controls=runner.real_controls,time_beijing=now()));raise

if __name__=='__main__':main()
