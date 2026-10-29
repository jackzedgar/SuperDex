"""Single bounded experiment continuation on Lambda; no notifications or chat automation."""
import fcntl,getpass,hashlib,json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/a1_full_20261028'
PARENT='collector_full_v3_lr_001';NAME='reset_memory_development_20261029_001'
def write_status(stage,**kw):
 p=RUN/(NAME+'_queue.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,time_unix=time.time(),**kw),indent=2)+'\n');tmp.replace(p)
def main():
 assert socket.gethostname().lower()=='lambda' and getpass.getuser()=='youran';os.chdir(ROOT)
 with (RUN/(NAME+'_queue.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (RUN/NAME/'summary.json').exists():return
  registration=json.loads((RUN/(NAME+'_queue_registration.json')).read_text())
  for f,h in registration['source_sha256'].items():
   assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,('Source changed; inspect before diagnostic',f)
  started=time.time();write_status('WAITING_EXISTING_COLLECTOR',pid=os.getpid(),parent=PARENT,maximum_gpu_seconds=1800)
  while time.time()-started<12*3600:
   d=json.loads((RUN/'gpu_ledger.json').read_text());jobs=[j for j in d['jobs'] if PARENT in j.get('command',[])]
   if not jobs:raise RuntimeError('No registered parent jobs')
   if any(j['status']=='running' for j in jobs):time.sleep(45);continue
   p=RUN/PARENT/'summary.json'
   if not p.exists():raise RuntimeError('Parent has no committed summary; no diagnostic launched')
   parent=json.loads(p.read_text())
   if set(parent.get('bands',{}))=={'20','50','80'}:
    write_status('SKIPPED_QUALITY_COMPLETE',reason='Existing formal pipeline has priority.');return
   if parent['status'] not in ['bounded_stop','steps_complete','development_regression_stop']:raise RuntimeError('Unexpected parent outcome')
   ckpt=RUN/PARENT/'checkpoint_final.pt'
   if not ckpt.exists():raise RuntimeError('No final checkpoint')
   if any(j['name']==NAME for j in d['jobs']):raise RuntimeError('Diagnostic attempt already registered; do not duplicate')
   used=sum(j['seconds'] if j['status']=='finished' else j['limit_seconds'] for j in d['jobs'] if j['phase']=='revision1')
   if used+1800>d['phase_caps']['revision1']:raise RuntimeError('No revision1 budget for diagnostic')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
   gpuids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
   busy={j['gpu'] for j in d['jobs'] if j['status']=='running'};free=[]
   for line in gpuids.splitlines():
    i,u=[x.strip() for x in line.split(',')]
    if int(i) in range(4) and int(i) not in busy and u not in apps:free.append(int(i))
   if not free:time.sleep(45);continue
   for f,h in registration['source_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,('Source changed while waiting',f)
   cmd=[str(ROOT/'.venv-mjw/bin/python'),'scripts/a1_budget.py','--name',NAME,'--gpu',str(free[0]),'--seconds','1800','--phase','revision1','--','python','-u','scripts/diagnose_a1_reset_memory.py','--checkpoint',str(ckpt),'--run',NAME]
   write_status('RUNNING_BOUNDED_DIAGNOSTIC',gpu=free[0],command=cmd)
   with (RUN/(NAME+'_supervisor.log')).open('x') as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
   write_status('DIAGNOSTIC_FINISHED_REVIEW_REQUIRED',returncode=rc,result=str(RUN/NAME),formal_A1_complete=False);return
  write_status('WAIT_DEADLINE_REACHED',reason='No new experiment launched after12hours.')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write_status('REVIEW_REQUIRED',error=repr(e));raise
