"""Approved eleventh fixed actor data-coverage refinement and retention gate."""
import argparse,datetime,getpass,hashlib,json,signal,socket,time,traceback
from pathlib import Path
import numpy as np,torch,warp as wp
from a1_policy import CollectorPolicy
from a1_env import A1Env
from train_a1_collector import evaluate
from a1_teacher_warmstart_helpers import clone_parameters
from a1_teacher_coverage_helpers import validate_data,per_source_rmse,coverage_gate
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
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args();c=json.loads(args.config.read_text())
 assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
 assert c['scientific_revision']==11 and c['authorization_reply']=='批准第十一项≤1.4 GPUh验证' and c['job_limit_seconds']==360
 for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
 for k in ['parent','initial_clone']:assert sha(R/c[k+'_checkpoint'])==c[k+'_sha256']
 assert sha(R/c['data_path'])==c['data_sha256'] and sha(R/c['data_manifest'])==c['data_manifest_sha256']
 start=time.monotonic();deadline=start+300;out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
 write(out/'run_configuration.json',dict(c,started_beijing=now(),formal_A1=False))
 parent=torch.load(R/c['parent_checkpoint'],map_location='cpu',weights_only=False);initial=torch.load(R/c['initial_clone_checkpoint'],map_location='cpu',weights_only=False)
 assert initial['clone_updates']==2000 and initial['parent_sha256']==c['parent_sha256'] and not initial['smoke']
 policy=CollectorPolicy(280043);policy.load_state_dict(initial['policy']);policy.eval();updates=0;rmse=float('inf');errors=[];rows=[];eval_env=None;reason='NOT_STARTED'
 try:
  manifest=json.loads((R/c['data_manifest']).read_text());data=dict(np.load(R/c['data_path']));validate_data(data,manifest)
  assert manifest['task_id']==c['task_id'] and set(json.loads((R/c['task_path']/'qualification.json').read_text())['hands'])=={'right','left'}
  for s in manifest['sources']:assert sha(R/s['path'])==s['sha256']
  obs=torch.from_numpy(data['obs']).cuda();goal=torch.from_numpy(data['goal']).cuda();target=torch.from_numpy(data['teacher_target_action']).cuda();source_index=torch.from_numpy(data['source_index']).cuda()
  write(out/'teacher_provenance.json',dict(data_path=c['data_path'],data_sha256=c['data_sha256'],manifest=manifest,formal_collection=False,world_model_data=False,counterfactual_targets_not_executed=278))
  policy=policy.cuda();params=clone_parameters(policy);opt=torch.optim.Adam(params,lr=1e-4,betas=(.9,.95),eps=1e-8);torch.manual_seed(280053);torch.cuda.manual_seed(280053)
  for step in range(1,2001):
   if STOP or time.monotonic()>=deadline:reason='PREPARATION_TIME_CAP';break
   pred=policy.distribution(obs,goal).mean;loss=(pred-target).square().mean();assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step();updates=step
   if step==1 or step%100==0:
    row=dict(update=step,loss=float(loss),rmse_before_update=float(loss.sqrt()),preclip_gradient_norm=float(norm),seconds=time.monotonic()-start)
    with (out/'clone_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    write(out/'progress.json',dict(stage='FIXED_COVERAGE_REFINEMENT',**row));print('REFINE',step,float(loss),flush=True)
  with torch.no_grad():
   pred=policy.distribution(obs,goal).mean;rmse=float((pred-target).square().mean().sqrt());errors=per_source_rmse(pred,target,source_index)
  frozen=[k for k in parent['policy'] if k.startswith('critic.') or k=='logstd'];assert all(torch.equal(policy.state_dict()[k].cpu(),parent['policy'][k]) for k in frozen)
  cp=out/'checkpoint_coverage_fixed_final.pt';save(cp,dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},clone_optimizer=opt.state_dict(),clone_updates=updates,prior_clone_updates=2000,cumulative_clone_updates=2000+updates,clone_rmse=rmse,per_source_rmse=errors,parent_sha256=c['parent_sha256'],initial_clone_sha256=c['initial_clone_sha256'],data_sha256=c['data_sha256'],configuration=c,smoke=False,critic_logstd_unchanged=True,formal_A1=False))
  fit=updates==2000 and len(errors)==9 and all(0<=e<=.01 for e in errors)
  if fit and not STOP and time.monotonic()<deadline:
   eval_env=A1Env(R/c['task_path'],capture=True,determinism='constraint');torch.cuda.set_stream(eval_env.torch_stream);wp.set_stream(eval_env.stream)
   class Guard:
    def __getattr__(self,k):return getattr(eval_env,k)
    def reset(self,*a,**kw):
     if STOP or time.monotonic()>=deadline:raise InterruptedError('coverage replay deadline')
     result=eval_env.reset(*a,**kw);w=eval_env._world(eval_env.qpos[:,:eval_env.task.nu]);assert int(eval_env.phase)==0 and torch.equal(eval_env.ctrl,eval_env.qpos[:,:eval_env.task.nu]) and torch.equal(eval_env.history,w[:,:,None].expand(-1,-1,3,-1)) and torch.equal(eval_env.servo,w[:,:,:6]);return result
    def step(self,a):
     if STOP or time.monotonic()>=deadline:raise InterruptedError('coverage replay deadline')
     assert torch.isfinite(a).all() and float(a.abs().max())<=1.;return eval_env.step(a)
   for repeat in range(3):
    try:row=evaluate(Guard(),policy,[280044])[0]
    except InterruptedError:break
    rows.append(dict(row,repeat=repeat));write(out/'replay_progress.json',dict(conditions=rows,complete=len(rows)==3))
   reason='COVERAGE_REPLAY_COMPLETE' if len(rows)==3 else 'COVERAGE_REPLAY_INCOMPLETE'
  elif updates==2000:reason='SOURCE_FIT_GATE_FAILED' if not fit else 'PREPARATION_TIME_CAP'
  passed=coverage_gate(updates,errors,rows)
  result=dict(status='COVERAGE_GATE_PASS' if passed else 'COVERAGE_GATE_NOT_PASSED',reason=reason,time_beijing=now(),clone_updates=updates,prior_clone_updates=2000,cumulative_clone_updates=2000+updates,clone_rmse=rmse,per_source_rmse=errors,source_fit_gate=fit,replays=rows,regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),n=len(rows),gate_passed=passed,checkpoint=str(cp.relative_to(R)),checkpoint_sha256=sha(cp),data_sha256=c['data_sha256'],critic_logstd_unchanged=True,physical_evaluation_controls=eval_env.physical_controls if eval_env else 0,seconds=time.monotonic()-start,full_A1_complete=False,no_PPO_started_here=True)
  write(out/'summary.json',result);print(json.dumps(result),flush=True)
 except BaseException as ex:
  save(out/'checkpoint_failed.pt',dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},clone_updates=updates,configuration=c));write(out/'fatal.json',dict(error=repr(ex),traceback=traceback.format_exc(),clone_updates=updates,physical_evaluation_controls=eval_env.physical_controls if eval_env else 0,time_beijing=now()));raise
if __name__=='__main__':main()
