"""Approved eleventh matched PPO: unchanged original environments, fixed 64 updates."""
import argparse,datetime,getpass,hashlib,json,signal,socket,time,traceback
from pathlib import Path
import numpy as np,torch,warp as wp
from a1_policy import CollectorPolicy
from a1_env import A1Env
from a1_formal_start_pool import FormalStartPoolCandidate
from train_a1_collector import ppo_update,evaluate
from a1_teacher_coverage_helpers import factual_episode
R=Path(__file__).resolve().parents[1];STOP=False
def halt(*_):
 global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def save(p,v):
 q=p.with_suffix(p.suffix+'.tmp');torch.save(v,q);q.replace(p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args();c=json.loads(args.config.read_text())
 assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
 assert c['scientific_revision']==11 and c['authorization_reply']=='批准第十一项≤1.4 GPUh验证'
 assert c['job_limit_seconds']==2340 and c['max_new_controls']==131072 and c['updates']==64 and c['batch_controls']==2048
 for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
 assert sha(R/c['parent_checkpoint'])==c['parent_sha256']
 clone_summary=json.loads((R/c['clone_summary']).read_text());assert clone_summary['gate_passed'] and clone_summary['clone_updates']==2000 and clone_summary['cumulative_clone_updates']==4000 and len(clone_summary['per_source_rmse'])==9 and max(clone_summary['per_source_rmse'])<=.01
 assert sha(R/clone_summary['checkpoint'])==clone_summary['checkpoint_sha256']
 start=time.monotonic();train_deadline=start+2140;deadline=start+2280;out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
 parent=torch.load(R/c['parent_checkpoint'],map_location='cpu',weights_only=False);assert parent['configuration']['task_id']==c['task_id'] and not parent['smoke']
 initial=parent
 if c['arm']=='clone':
  initial=torch.load(R/clone_summary['checkpoint'],map_location='cpu',weights_only=False);assert initial['parent_sha256']==c['parent_sha256'] and initial['clone_updates']==2000 and initial['critic_logstd_unchanged']
 else:assert c['arm']=='direct'
 policy=CollectorPolicy(280043).cuda();policy.load_state_dict(initial['policy'])
 for k in parent['policy']:
  if k.startswith('critic.') or k=='logstd':assert torch.equal(policy.state_dict()[k].cpu(),parent['policy'][k])
 opt=torch.optim.Adam(policy.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8);assert not opt.state
 gen=torch.Generator(device='cuda');gen.manual_seed(c['ppo_seed'])
 eval_env=A1Env(R/c['task_path'],capture=True,determinism='constraint')
 root=np.random.default_rng(c['reset_root_seed']).bit_generator.state
 runner=FormalStartPoolCandidate(R/c['task_path'],policy,root,validate=True)
 torch.manual_seed(c['action_seed']);torch.cuda.manual_seed(c['action_seed'])
 prior_controls=parent['cumulative_task_controls'];prior_updates=parent['cumulative_task_updates'];controls=0;updates=0;uncommitted=0;development=None;eval_controls=0;reason='CONTROL_TARGET'
 write(out/'run_configuration.json',dict(c,started_beijing=now(),initial_checkpoint_sha256=clone_summary['checkpoint_sha256'] if c['arm']=='clone' else c['parent_sha256'],lane_seeds=runner.lane_seeds,optimizer_fresh=True,original_physics=True,formal_A1=False))
 def checkpoint(name):
  save(out/name,dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},optimizer=opt.state_dict(),ppo_generator=gen.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),lane_rng=runner.rng_state(),new_controls=controls,new_updates=updates,cumulative_task_controls=prior_controls+controls,cumulative_task_updates=prior_updates+updates,physical_training_controls=runner.real_controls,configuration=c,smoke=False,formal_A1=False,quality_passports_allowed=False))
 checkpoint('checkpoint_initial.pt')
 class Guard:
  def __getattr__(self,k):return getattr(eval_env,k)
  def reset(self,*a,**kw):
   if STOP or time.monotonic()>=deadline:raise InterruptedError('development deadline')
   result=eval_env.reset(*a,**kw);w=eval_env._world(eval_env.qpos[:,:eval_env.task.nu]);assert int(eval_env.phase)==0 and torch.equal(eval_env.ctrl,eval_env.qpos[:,:eval_env.task.nu]) and torch.equal(eval_env.history,w[:,:,None].expand(-1,-1,3,-1)) and torch.equal(eval_env.servo,w[:,:,:6]);return result
  def step(self,a):
   nonlocal eval_controls
   if STOP or time.monotonic()>=deadline:raise InterruptedError('development deadline')
   eval_controls+=1;return eval_env.step(a)
 try:
  while updates<64 and not STOP and time.monotonic()<train_deadline:
   before=time.monotonic();first_ep=len(runner.episodes);data=runner.rollout(2048,lambda:STOP or time.monotonic()>=train_deadline)
   if data is None:reason='TRAINING_TIME_CAP';break
   if len(data['reward'])!=2048:
    uncommitted=len(data['reward']);save(out/'uncommitted_rollout.pt',{k:v.cpu() for k,v in data.items()});reason='TRAINING_TIME_CAP';break
   assert all(bool(torch.isfinite(v).all()) for v in data.values());assert torch.equal(data['mask'],runner.mask[0].expand(2048,2))
   if updates==0:
    with torch.no_grad():
     ids=torch.arange(0,2048,31,device='cuda');dist=policy.distribution(data['obs'][ids],data['goal'][ids]);lp=(dist.log_prob(data['action'][ids])*data['mask'][ids].repeat_interleave(26,-1)).sum(-1);err=float((lp-data['old_logprob'][ids]).abs().max());assert err<=5e-4
    write(out/'implementation_gate.json',dict(status='PASS',behavior_logprob_max_error=err,runner_audit=runner.audit,optimizer_fresh=True,original_reset=True,initial_critic_logstd_identical=True))
    runner.validate=False
   metrics=ppo_update(policy,opt,data,gen,3e-5);updates+=1;controls+=2048
   row=dict(time_beijing=now(),arm=c['arm'],new_controls=controls,updates=updates,seconds=time.monotonic()-start,update_seconds=time.monotonic()-before,physical_training_controls=runner.real_controls,mean_rollout_reward=float(data['reward'].mean()),**metrics)
   with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   reset_index={(r['lane'],r['episode_id']):r for r in runner.reset_records}
   with (out/'training_episodes.jsonl').open('a') as f:
    for ep in runner.episodes[first_ep:]:f.write(json.dumps(dict(update=updates,**factual_episode(ep,reset_index)))+'\n')
   checkpoint('checkpoint_latest.pt');write(out/'progress.json',dict(status='TRAINING',**row));print('UPDATE',updates,controls,flush=True)
  if STOP:reason='BUDGET_OR_SIGNAL'
  elif updates<64:reason='TRAINING_TIME_CAP'
  checkpoint('checkpoint_final.pt')
  # Opening development is allowed only after the pre-registered matched update target.
  if updates==64 and controls==131072 and not STOP and time.monotonic()<deadline:
   torch.cuda.synchronize();torch.cuda.set_stream(eval_env.torch_stream);wp.set_stream(eval_env.stream);rows=[]
   for seed in c['development_seeds']:
    try:rows+=evaluate(Guard(),policy,[seed])
    except InterruptedError:break
    write(out/'development_progress.json',dict(conditions=rows,complete=len(rows)==4,new_controls=controls))
   development=dict(complete=len(rows)==4,conditions=rows,regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),n=len(rows),original_initialization=True,new_controls=controls,scope='single_task_development_not_full160')
   write(out/'development.json',development)
  write(out/'training_resets.json',runner.reset_records)
  reset_index={(r['lane'],r['episode_id']):r for r in runner.reset_records}
  write(out/'all_completed_training_episodes.json',dict(committed_controls=controls,physical_controls=runner.real_controls,episodes=[factual_episode(ep,reset_index) for ep in runner.episodes]))
  summary=dict(status=reason,time_beijing=now(),arm=c['arm'],new_controls=controls,updates=updates,physical_training_controls=runner.real_controls,uncommitted_rollout_controls=uncommitted,physical_evaluation_controls=eval_controls,physical_total_env_controls=sum(e.physical_controls for e in runner.envs)+eval_env.physical_controls,development=development,checkpoint_sha256=sha(out/'checkpoint_final.pt'),runner_audit=runner.audit,training_resets=len(runner.reset_records),training_reset_phases=sorted({r['phase'] for r in runner.reset_records}),training_reset_memory_modes=sorted({r['training_memory'] for r in runner.reset_records}),seconds=time.monotonic()-start,full_A1_complete=False,quality_passports_created=False)
  write(out/'summary.json',summary);print(json.dumps(summary),flush=True)
 except BaseException as ex:
  checkpoint('checkpoint_failed_last_committed.pt');write(out/'training_resets.json',runner.reset_records);write(out/'fatal.json',dict(error=repr(ex),traceback=traceback.format_exc(),new_controls=controls,physical_training_controls=runner.real_controls,time_beijing=now()));raise
if __name__=='__main__':main()
