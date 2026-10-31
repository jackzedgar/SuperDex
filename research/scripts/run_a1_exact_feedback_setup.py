"""Authorized twelfth unchanged-environment setup and3/3 retention gate; no updates."""
import argparse,datetime,getpass,hashlib,json,os,signal,socket,time,traceback
from pathlib import Path
import numpy as np,torch
from a1_exact_feedback_policy import ExactFeedbackPolicy,setup_gate
from a1_exact_feedback_runner import ExactWitnessPool
R=Path(__file__).resolve().parents[1];STOP=False
def halt(*_):
 global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();c=json.loads(a.config.read_text())
 assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
 assert c['scientific_revision']==12 and c['authorization_reply']=='批准第十二项≤1 GPUh' and c['job_limit_seconds']==180
 for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
 assert sha(R/c['parent_checkpoint'])==c['parent_sha256'] and sha(R/c['selected_controller'])==c['selected_controller_sha256']
 start=time.monotonic();deadline=start+115;out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False);signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
 write(out/'run_configuration.json',dict(c,started_beijing=now(),network_updates=0,formal_A1=False));pool=None;rows=[]
 try:
  parent=torch.load(R/c['parent_checkpoint'],map_location='cpu',weights_only=False);knots=np.asarray(json.loads((R/c['selected_controller']).read_text())['knots'],np.float32)
  policy=ExactFeedbackPolicy(parent['policy'],knots,'candidate').cuda().eval()
  for p in policy.parameters():p.requires_grad_(False)
  torch.cuda.synchronize();pool=ExactWitnessPool(R/c['task_path'],policy);assert pool.horizon==98
  initial={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()};cp=out/'checkpoint_initial_exact_teacher.pt';torch.save(dict(policy=initial,configuration=c,network_updates=0,smoke=False,formal_A1=False),cp)
  for repeat in range(3):
   if STOP or time.monotonic()>=deadline:break
   result=pool.run(knots[None],[280044],lambda:STOP or time.monotonic()>=deadline,validate=True)
   if not result:break
   row=result[0];trace=row.pop('trace');path=out/f'trace_{repeat:02d}_exact_teacher.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in trace.items()});row.update(repeat=repeat,trace_file=path.name,trace_sha256=sha(path),time_beijing=now())
   with (out/'conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   rows.append(row);write(out/'progress.json',dict(stage='EXACT_TEACHER_RETENTION',conditions=len(rows),last_regular=row['regular_success'],physical_controls=pool.controls))
  assert all(torch.equal(v.cpu(),initial[k]) for k,v in policy.state_dict().items()) and policy.frozen_actor_unchanged(parent['policy'])
  delta=pool.audit.get('max_teacher_action_error',float('inf'));passed=setup_gate(rows,delta)
  result=dict(status='EXACT_TEACHER_GATE_PASS' if passed else 'EXACT_TEACHER_GATE_NOT_PASSED',time_beijing=now(),gate_passed=passed,n=len(rows),regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),controls=[r['controls'] for r in rows],conditions=rows,network_updates=0,all_weights_unchanged=True,max_teacher_action_error=delta,physical_controls=pool.controls,warmup_controls=pool.warmup_controls,audit=pool.audit,checkpoint=str(cp.relative_to(R)),checkpoint_sha256=sha(cp),seconds=time.monotonic()-start,no_PPO_started_here=True,full_A1_complete=False)
  write(out/'summary.json',result);print(json.dumps(result),flush=True)
 except BaseException as ex:
  write(out/'fatal.json',dict(error=repr(ex),traceback=traceback.format_exc(),conditions=len(rows),physical_controls=pool.controls if pool else 0,time_beijing=now()));raise
if __name__=='__main__':main()
