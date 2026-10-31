"""Approved ninth stage: fixed actor cloning and original full-episode retention gate."""
import argparse,datetime,getpass,hashlib,json,os,signal,socket,time,traceback
from pathlib import Path
import numpy as np,torch,warp as wp
from a1_policy import CollectorPolicy,policy_inputs
from a1_task_runtime import TaskTensors
from a1_env import A1Env
from train_a1_collector import evaluate
from a1_teacher_warmstart_helpers import causal_histories,clone_parameters,clone_gate
R=Path(__file__).resolve().parents[1];STOP=False
def halt(*_):
 global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def save(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');torch.save(v,tmp);tmp.replace(p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args();c=json.loads(args.config.read_text())
 assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
 assert c['scientific_revision']==9 and c['authorization_reply']=='批准第九项≤1.5 GPUh验证' and c['job_limit_seconds']==360
 for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
 assert sha(R/c['parent_checkpoint'])==c['parent_sha256']
 start=time.monotonic();deadline=start+300;out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
 write(out/'run_configuration.json',dict(c,started_beijing=now(),formal_A1=False))
 parent=torch.load(R/c['parent_checkpoint'],map_location='cpu',weights_only=False);assert not parent['smoke'] and parent['configuration']['task_id']==c['task_id']
 policy=CollectorPolicy(280043);policy.load_state_dict(parent['policy']);policy.eval();task=TaskTensors(R/c['task_path'],device='cpu')
 obs=[];goals=[];targets=[];audits=[];updates=0;rmse=float('inf');eval_env=None;rows=[];reason='NOT_STARTED'
 try:
  with torch.no_grad():
   for t in c['teaching_data']:
    p=R/t['path'];assert sha(p)==t['sha256'];z=np.load(p)
    state=torch.from_numpy(z['state']);action=torch.from_numpy(z['last_action']);assert state.shape==(99,287) and action.shape==(98,52)
    np.testing.assert_array_equal(z['phase'],np.arange(99));assert not any(z['done'][:-1]) and z['done'][-1]
    sh,ah=causal_histories(state,action);o,g=policy_inputs(task,sh,ah,torch.arange(98));mean=policy.distribution(o,g).mean*task.mask.repeat_interleave(26,-1)
    err=float((mean+torch.from_numpy(z['bias'])-torch.from_numpy(z['raw_action'])).abs().max());assert err<5e-5
    assert torch.equal(action,torch.from_numpy(z['raw_action']).clamp(-1,1)*task.mask.repeat_interleave(26,-1))
    obs.append(o);goals.append(g);targets.append(action);audits.append(dict(path=t['path'],sha256=t['sha256'],raw_reconstruction_max_error=err,controls=98))
  obs=torch.cat(obs).cuda();goal=torch.cat(goals).cuda();target=torch.cat(targets).cuda();mask=task.mask.repeat_interleave(26,-1).expand(294,52).cuda()
  assert len(target)==294 and bool(torch.isfinite(obs).all()) and bool(torch.isfinite(goal).all())
  np.savez_compressed(out/'teacher_inputs.npz',obs=obs.cpu().numpy(),goal=goal.cpu().numpy(),executed_action=target.cpu().numpy(),mask=mask.cpu().numpy())
  write(out/'teacher_provenance.json',dict(traces=audits,controls=294,initialization_families=1,seed=280044,formal_collection=False,development_used=False,inputs_sha256=sha(out/'teacher_inputs.npz')))
  policy=policy.cuda();params=clone_parameters(policy);opt=torch.optim.Adam(params,lr=1e-4,betas=(.9,.95),eps=1e-8)
  torch.manual_seed(280053);torch.cuda.manual_seed(280053)
  for step in range(1,2001):
   if STOP or time.monotonic()>=deadline:reason='PREPARATION_TIME_CAP';break
   mean=policy.distribution(obs,goal).mean;loss=((mean-target).square()*mask).sum()/mask.sum();assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm);opt.step();updates=step
   if step==1 or step%100==0:
    row=dict(update=step,loss=float(loss),rmse_before_update=float(loss.sqrt()),preclip_gradient_norm=float(norm),seconds=time.monotonic()-start)
    with (out/'clone_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    write(out/'progress.json',dict(stage='FIXED_ACTOR_CLONING',**row));print('CLONE',step,float(loss),flush=True)
  with torch.no_grad():rmse=float((((policy.distribution(obs,goal).mean-target).square()*mask).sum()/mask.sum()).sqrt())
  frozen=[k for k in parent['policy'] if k.startswith('critic.') or k=='logstd'];assert all(torch.equal(policy.state_dict()[k].cpu(),parent['policy'][k]) for k in frozen)
  cp=out/'checkpoint_clone_fixed_final.pt';save(cp,dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},clone_optimizer=opt.state_dict(),clone_updates=updates,clone_rmse=rmse,parent_sha256=c['parent_sha256'],configuration=c,smoke=False,critic_logstd_unchanged=True,formal_A1=False))
  if updates==2000 and rmse<=.02 and not STOP and time.monotonic()<deadline:
   eval_env=A1Env(R/c['task_path'],capture=True,determinism='constraint');torch.cuda.set_stream(eval_env.torch_stream);wp.set_stream(eval_env.stream)
   class Guard:
    def __getattr__(self,k):return getattr(eval_env,k)
    def reset(self,*a,**kw):
     if STOP or time.monotonic()>=deadline:raise InterruptedError('clone replay deadline')
     result=eval_env.reset(*a,**kw);w=eval_env._world(eval_env.qpos[:,:eval_env.task.nu])
     assert int(eval_env.phase)==0 and torch.equal(eval_env.ctrl,eval_env.qpos[:,:eval_env.task.nu]) and torch.equal(eval_env.history,w[:,:,None].expand(-1,-1,3,-1)) and torch.equal(eval_env.servo,w[:,:,:6]);return result
    def step(self,a):
     if STOP or time.monotonic()>=deadline:raise InterruptedError('clone replay deadline')
     assert torch.isfinite(a).all() and float(a.abs().max())<=1.;return eval_env.step(a)
   for repeat in range(3):
    try:row=evaluate(Guard(),policy,[280044])[0]
    except InterruptedError:break
    rows.append(dict(row,repeat=repeat));write(out/'replay_progress.json',dict(conditions=rows,complete=len(rows)==3))
   reason='CLONE_REPLAY_COMPLETE' if len(rows)==3 else 'CLONE_REPLAY_INCOMPLETE'
  elif updates==2000:reason='FIT_GATE_FAILED' if rmse>.02 else 'PREPARATION_TIME_CAP'
  passed=clone_gate(updates,rmse,rows)
  result=dict(status='CLONE_GATE_PASS' if passed else 'CLONE_GATE_NOT_PASSED',reason=reason,time_beijing=now(),clone_updates=updates,clone_rmse=rmse,replays=rows,regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),n=len(rows),gate_passed=passed,checkpoint=str(cp.relative_to(R)),checkpoint_sha256=sha(cp),critic_logstd_unchanged=True,physical_evaluation_controls=eval_env.physical_controls if eval_env else 0,seconds=time.monotonic()-start,full_A1_complete=False,no_PPO_started_here=True)
  write(out/'summary.json',result);print(json.dumps(result),flush=True)
 except BaseException as ex:
  save(out/'checkpoint_failed.pt',dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},clone_updates=updates,configuration=c))
  write(out/'fatal.json',dict(error=repr(ex),traceback=traceback.format_exc(),clone_updates=updates,physical_evaluation_controls=eval_env.physical_controls if eval_env else 0,time_beijing=now()));raise
if __name__=='__main__':main()
