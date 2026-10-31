"""Authorized tenth fixed comparison, unchanged physics, no network optimization."""
import argparse,datetime,getpass,hashlib,json,os,signal,socket,time,traceback
from pathlib import Path
import numpy as np,torch
from a1_policy import CollectorPolicy
from a1_original_action_search_pool import WitnessPool
from a1_matched_frozen_control import replace_frozen,interpretation,ORDER
R=Path(__file__).resolve().parents[1];STOP=False
def halt(*_):
 global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args();c=json.loads(args.config.read_text())
 assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
 assert c['scientific_revision']==10 and c['authorization_reply']=='批准第十项≤0.05 GPUh验证' and c['job_limit_seconds']==180
 for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
 for role in ['teacher','clone']:assert sha(R/c[role+'_checkpoint'])==c[role+'_sha256']
 assert sha(R/c['selected_controller'])==c['selected_controller_sha256']
 start=time.monotonic();deadline=start+115;out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
 write(out/'run_configuration.json',dict(c,started_beijing=now(),network_updates=0,formal_A1=False))
 pool=None;rows=[];swap_checks=[]
 try:
  states={role:torch.load(R/c[role+'_checkpoint'],map_location='cpu',weights_only=False)['policy'] for role in ['teacher','clone']}
  policy=CollectorPolicy(280043).cuda();policy.eval();replace_frozen(policy,states['teacher'])
  chosen=json.loads((R/c['selected_controller']).read_text());knots=np.asarray(chosen['knots'],np.float32);assert knots.shape==(4,52)
  pool=WitnessPool(R/c['task_path'],policy);assert pool.horizon==98
  def guard():return STOP or time.monotonic()>=deadline
  for index,role in enumerate(ORDER):
   if guard():break
   torch.cuda.synchronize();torch.cuda.set_stream(pool.stream);pointers=replace_frozen(policy,states[role]);pool.stream.synchronize()
   p=knots if role=='teacher' else np.zeros_like(knots)
   result=pool.run(p[None],[280044],guard,validate=True)
   assert len(result)==1
   row=result[0];trace=row.pop('trace');path=out/f'trace_{index:02d}_{role}.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in trace.items()})
   row.update(controller=role,index=index,trace_file=path.name,trace_sha256=sha(path),time_beijing=now())
   with (out/'conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   rows.append(row);swap_checks.append(dict(index=index,controller=role,pointers_preserved=True,weights_exact=True,parameter_pointer_map=pointers))
   assert all(torch.equal(v.cpu(),states[role][k]) for k,v in policy.state_dict().items())
   write(out/'progress.json',dict(stage='MATCHED_FROZEN_REPLAY',conditions=len(rows),controls=pool.controls,time_beijing=now(),last=dict(controller=role,regular=row['regular_success'],strict=row['strict_success'],controls=row['controls'])))
  for role in ['teacher','clone']:assert sha(R/c[role+'_checkpoint'])==c[role+'_sha256']
  assert sha(R/c['selected_controller'])==c['selected_controller_sha256']
  assert sum(r['controls'] for r in rows)==pool.controls and sum(e.physical_controls for e in pool.envs)==pool.controls+pool.warmup_controls
  result=dict(status='COMPLETE' if len(rows)==6 and all(r['complete'] for r in rows) else 'BOUNDED_STOP_INCOMPLETE',time_beijing=now(),interpretation=interpretation(rows),conditions=len(rows),grouped={role:dict(n=sum(r['controller']==role for r in rows),regular=sum(r['regular_success'] for r in rows if r['controller']==role),strict=sum(r['strict_success'] for r in rows if r['controller']==role),controls=[r['controls'] for r in rows if r['controller']==role]) for role in ['teacher','clone']},physical_controls=pool.controls,warmup_controls=pool.warmup_controls,audit=pool.audit,swap_checks=swap_checks,optimizer_steps=0,weights_unchanged=True,seconds=time.monotonic()-start,no_auto_PPO=True,full_A1_complete=False)
  write(out/'summary.json',result);print(json.dumps(result),flush=True)
 except BaseException as ex:
  write(out/'fatal.json',dict(error=repr(ex),traceback=traceback.format_exc(),conditions=len(rows),controls=pool.controls if pool else 0,time_beijing=now()));raise
if __name__=='__main__':main()
