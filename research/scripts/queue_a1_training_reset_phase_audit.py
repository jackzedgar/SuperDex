"""One bounded implementation diagnostic after the current collector; never starts training."""
import fcntl,getpass,hashlib,json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/a1_full_20261028'
PARENT='collector_full_v4_continue_001';NAME='training_reset_phase_audit_20261029_001'
def status(stage,**kw):
 p=RUN/(NAME+'_queue.json');temp=p.with_suffix('.tmp')
 temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],time_unix=time.time(),parent=PARENT,**kw),indent=2)+'\n');temp.replace(p)
def live_job(j):
 for key in ['supervisor_pid','child_pid']:
  pid=j.get(key)
  if pid is None:continue
  try:
   cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
   if PARENT in cmd and ('train_a1_shared_collector.py' in cmd or 'a1_budget.py' in cmd):return True
  except FileNotFoundError:pass
 return False
def main():
 assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran';os.chdir(ROOT)
 with (RUN/(NAME+'_queue.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  registration=json.loads((RUN/(NAME+'_queue_registration.json')).read_text())
  def verify():
   for f,h in registration['source_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,('Source changed; no audit launched',f)
  verify();started=time.time();status('WAITING_EXISTING_COLLECTOR',maximum_gpu_seconds=1800,budget_phase='implementation')
  while time.time()-started<6*3600:
   ledger=json.loads((RUN/'gpu_ledger.json').read_text());jobs=[j for j in ledger['jobs'] if PARENT in j.get('command',[])]
   if len(jobs)!=4:raise RuntimeError('Expected exactly four registered parent workers')
   if any(j['name']==NAME for j in ledger['jobs']):raise RuntimeError('Audit attempt already registered; never duplicate')
   if any(j['status']=='running' for j in jobs):
    if any(j['status']=='running' and not live_job(j) for j in jobs):raise RuntimeError('Parent ledger/process disagreement; inspect before launching')
    time.sleep(45);continue
   if any(live_job(j) for j in jobs):time.sleep(45);continue
   summary_path=RUN/PARENT/'summary.json'
   if not summary_path.exists():raise RuntimeError('Parent ended without committed summary')
   summary=json.loads(summary_path.read_text())
   if set(summary.get('bands',{}))=={'20','50','80'}:status('SKIPPED_QUALITY_COMPLETE',reason='Formal pipeline has priority');return
   if summary['status'] not in ['bounded_stop','steps_complete','development_regression_stop']:raise RuntimeError('Unexpected parent status')
   failures=[j for j in jobs if j.get('returncode')!=0 and j.get('reason')!='process_budget']
   if failures:
    errors=[json.loads(p.read_text()).get('error','') for p in (RUN/PARENT).glob('fatal_rank*.json')]
    expected_budget_peer_interrupt=summary['status']=='bounded_stop' and any(j.get('reason')=='process_budget' for j in jobs) and errors and all(e.startswith("InterruptedError('Stopped or peer failed;") for e in errors)
    if not expected_budget_peer_interrupt:raise RuntimeError('Unexpected parent failure')
   checkpoint=RUN/PARENT/'checkpoint_final.pt'
   if not checkpoint.exists():raise RuntimeError('Missing parent final checkpoint')
   used=sum(j['seconds'] if j['status']=='finished' else j['limit_seconds'] for j in ledger['jobs'] if j['phase']=='implementation')
   if used+1800>ledger['phase_caps']['implementation']:raise RuntimeError('Implementation budget unavailable')
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
   ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);busy={j['gpu'] for j in ledger['jobs'] if j['status']=='running'};free=[]
   for line in ids.splitlines():
    i,u=[x.strip() for x in line.split(',')]
    if int(i) in range(4) and int(i) not in busy and u not in apps:free.append(int(i))
   if not free:status('WAITING_IDLE_GPU');time.sleep(45);continue
   verify();cmd=[str(ROOT/'.venv-mjw/bin/python'),'scripts/a1_budget.py','--name',NAME,'--gpu',str(free[0]),'--seconds','1800','--phase','implementation','--','python','-u','scripts/diagnose_a1_training_reset_phases.py','--checkpoint',str(checkpoint),'--run',NAME]
   status('RUNNING_BOUNDED_PHASE_AUDIT',gpu=free[0],command=cmd)
   with (RUN/(NAME+'_supervisor.log')).open('x') as log:rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
   status('PHASE_AUDIT_FINISHED_REVIEW_REQUIRED',returncode=rc,result=str(RUN/NAME),formal_A1_complete=False);return
  status('WAIT_DEADLINE_REACHED',reason='No diagnostic launched after6hours')
if __name__=='__main__':
 try:main()
 except BaseException as e:status('REVIEW_REQUIRED',error=repr(e));raise
