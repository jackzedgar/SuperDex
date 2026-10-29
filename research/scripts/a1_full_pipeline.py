"""Lambda-only CPU coordinator for the already-authorized full A1 stages.
Every GPU child is launched through the hard aggregate process-time supervisor.
Never fabricates missing policy-quality levels or opens test for tuning.
"""
import os,time,json,subprocess,socket,getpass,hashlib,fcntl,argparse
from pathlib import Path
from a1_data_sources import ROOT,RUN,write
RUNNAME='collector_full_v1_001'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ledger():return json.loads((RUN/'gpu_ledger.json').read_text())
def status(stage,**kwargs):write(RUN/'pipeline_status.json',dict(stage=stage,time_unix=time.time(),**kwargs))
def available(phase):
 d=ledger();caps=d['phase_caps'];reserved=lambda j:j['seconds'] if j['status']=='finished' else j['limit_seconds']
 return min(caps[phase]-sum(reserved(j) for j in d['jobs'] if j['phase']==phase),240*3600-sum(reserved(j) for j in d['jobs'] if not j['phase'].startswith('revision')))
def launch(name,gpu,seconds,phase,cmd):
 seconds=int(min(seconds,available(phase)))
 if seconds<60:raise RuntimeError('No stage budget remains: '+phase)
 f=(RUN/(name+'.coordinator.log')).open('x')
 process=subprocess.Popen([str(ROOT/'.venv-mjw/bin/python'),'scripts/a1_budget.py','--name',name,'--gpu',str(gpu),'--seconds',str(seconds),'--phase',phase,'--',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 f.close();return process
def idle_gpus():
 d=ledger();busy={j['gpu'] for j in d['jobs'] if j['status']=='running'}
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
 for row in ids.splitlines():
  idx,uid=[v.strip() for v in row.split(',')]
  if uid in text:busy.add(int(idx))
 return [i for i in range(4) if i not in busy]
def cpu(cmd,name):
 status('CPU_'+name)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 with (RUN/(name+'.cpu.log')).open('x') as log:subprocess.run([str(ROOT/'.venv-mjw/bin/python'),*cmd],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def collect():
 manifest=json.loads((ROOT/'configs/a1_probe40_v1.json').read_text());queue=[]
 for quality in ['20','50','80']:
  passport=RUN/RUNNAME/f'quality_{quality}_passport.json';meta=json.loads(passport.read_text());assert meta['formal'] and meta['status']=='DEVELOPMENT_BAND_REACHED' and meta.get('reset_mode')=='uniform_reference';checkpoint=ROOT/meta['checkpoint'];assert sha(checkpoint)==meta['checkpoint_sha256']
  for i,task in enumerate(manifest['tasks']):
   path=ROOT/'data/a1_dataset/v1'/quality/task['task_id'];queue.append(dict(quality=quality,task=task['task_id'],path=path,checkpoint=checkpoint,passport=passport,seed=1800043000+i*100000,attempt=1))
 running={};complete=[];started=time.monotonic()
 while queue or running:
  for gpu,(proc,job) in list(running.items()):
   if proc.poll() is None:continue
   summary=job['path']/'summary.json';d=json.loads(summary.read_text()) if summary.exists() else {}
   if d.get('real_dataset_controls',0)>=100000 and d.get('status')=='quota_complete':complete.append(dict(task=job['task'],quality=job['quality'],**d))
   elif d.get('status')=='budget_stop' and job['attempt']<3 and available('collection')>=60:
    job['attempt']+=1;queue.insert(0,job)
   else:raise RuntimeError(f"Collection incomplete: {job['task']} quality{job['quality']} rc{proc.returncode}; preserve shard")
   del running[gpu]
  for gpu in idle_gpus():
   if gpu in running or not queue:continue
   job=queue.pop(0);key=hashlib.sha256(job['task'].encode()).hexdigest()[:10];name=f"data_q{job['quality']}_{key}_{job['attempt']:03d}"
   cmd=['python','scripts/collect_a1_episodes.py','--task',job['task'],'--quality',job['quality'],'--checkpoint',str(job['checkpoint']),'--passport',str(job['passport']),'--controls','100000','--seed',str(job['seed']),'--output',str(job['path'])]
   running[gpu]=(launch(name,gpu,9000,'collection',cmd),job)
   time.sleep(2)
  status('COLLECTING_FORMAL_DATA',completed_shards=len(complete),queued_shards=len(queue),running={str(g):dict(task=j['task'],quality=j['quality']) for g,(p,j) in running.items()},completed_controls=sum(r['real_dataset_controls'] for r in complete),wall_seconds=time.monotonic()-started)
  time.sleep(30)
 write(RUN/'collection_complete.json',complete)
def models():
 queue=[(kind,seed) for seed in [23,47,89] for kind in ['W0','W1']];running={};finished=[]
 while queue or running:
  for gpu,(proc,kind,seed,run) in list(running.items()):
   if proc.poll() is None:continue
   p=RUN/run/'summary.json';d=json.loads(p.read_text()) if p.exists() else {}
   if proc.returncode or d.get('updates')!=2000:raise RuntimeError(f'Full model incomplete: {run}, preserve checkpoint and budget')
   finished.append(run);del running[gpu]
  for gpu in idle_gpus():
   if gpu in running or not queue:continue
   kind,seed=queue[0];cap=5*3600 if kind=='W0' else 7*3600
   if available('models_initial')<cap and running:continue
   queue.pop(0);run=f'model_full_v1_{kind}_s{seed}'
   cmd=['python','scripts/train_a1_world_model.py','--manifest','data/a1_dataset/v1/frozen_manifest.json','--normalization','data/a1_dataset/v1/normalization.pt','--kind',kind,'--seed',str(seed),'--run',run]
   running[gpu]=(launch(run,gpu,cap,'models_initial',cmd),kind,seed,run);time.sleep(2)
  status('MATCHED_FULL_MODELS',completed=finished,queued=queue,running={str(g):r for g,(p,k,s,r) in running.items()})
  time.sleep(30)
 return finished
def final_if_development_passes(runs):
 criteria=[]
 for run in runs:
  if '_W1_' in run:
   p=RUN/run/'validation_2000/criteria.json';d=json.loads(p.read_text());criteria.append(dict(run=run,development_criteria=d,sha256=sha(p)))
 if not all(all(x['development_criteria']['all_present_criteria_pass'].values()) for x in criteria):
  status('INITIAL_COMPLETE_DEVELOPMENT_REVIEW_REQUIRED',reason='At least one W1 seed misses a registered development threshold or lacks a required phase. Locked test remains unopened. At most two one-factor revisions are authorized; inspect evidence before selecting.',criteria=criteria)
  return
 checkpoints=[RUN/run/'checkpoint_final.pt' for run in runs]
 decision=ROOT/'configs/a1_model_selection_v1.json';assert not decision.exists()
 write(decision,dict(status='FROZEN_BEFORE_LOCKED_TEST',selection_rule='Predeclared initial configuration accepted only if all three W1 seeds pass all original criteria on both task-balanced and natural development windows. No locked test used. Evaluate all six matched final checkpoints including W0.',configuration='initial_full_v1',development_evidence=criteria,checkpoint_sha256s=[sha(p) for p in checkpoints],checkpoints=[str(p.relative_to(ROOT)) for p in checkpoints]))
 queue=list(zip(runs,checkpoints));running={};done=[]
 while queue or running:
  for gpu,(proc,name) in list(running.items()):
   if proc.poll() is None:continue
   if proc.returncode:raise RuntimeError('Final evaluation failed '+name)
   done.append(name);del running[gpu]
  for gpu in idle_gpus():
   if gpu in running or not queue:continue
   model,checkpoint=queue.pop(0);run=model+'_locked_test'
   cmd=['python','scripts/evaluate_a1_final.py','--checkpoint',str(checkpoint),'--manifest','data/a1_dataset/v1/frozen_manifest.json','--decision',str(decision),'--run',run]
   running[gpu]=(launch(run,gpu,3600,'evaluation',cmd),run);time.sleep(2)
  status('LOCKED_FINAL_EVALUATION',completed=done,queued=len(queue));time.sleep(30)
 status('INITIAL_FULL_A1_EXECUTION_COMPLETE_ANALYSIS_PENDING',evaluations=done,formal_G1_claim=False,reason='All scheduled runs complete; aggregate uncertainty, per-object failures, costs and final scientific judgment still required. No automatic positive conclusion.')
def main():
 global RUNNAME
 parser=argparse.ArgumentParser();parser.add_argument('--collector-run',default=RUNNAME);args=parser.parse_args();RUNNAME=args.collector_run
 assert '/' not in RUNNAME and '..' not in RUNNAME
 assert socket.gethostname().lower()=='lambda' and getpass.getuser()=='youran';os.chdir(ROOT)
 with (RUN/'pipeline_controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  status('WAITING_FORMAL_COLLECTOR',run=RUNNAME)
  reservation_wait_start=time.monotonic()
  while True:
   folder=RUN/RUNNAME
   summary=folder/'summary.json'
   jobs=[j for j in ledger()['jobs'] if 'scripts/train_a1_shared_collector.py' in j.get('command',[]) and RUNNAME in j['command']]
   active=[j for j in jobs if j['status']=='running']
   errors=[json.loads(p.read_text()).get('error','') for p in folder.glob('fatal_rank*.json')]
   expected_interrupt=errors and all(e.startswith("InterruptedError('Stopped or peer failed;") for e in errors)
   budget_evidence=jobs and all(j.get('reason')=='process_budget' or (j['status']=='running' and time.time()-j['started']>=j['limit_seconds']-60) for j in jobs)
   if errors and not (expected_interrupt and budget_evidence):raise RuntimeError('Formal collector peer error; preserve all checkpoints')
   if summary.exists() and not active:
    if errors:
     d=json.loads(summary.read_text());assert d['status']=='bounded_stop'
     write(folder/'verified_budget_stop.json',dict(reason='All supervisors reached registered process budgets; expected peer interrupts, committed final checkpoint retained.',jobs=[j['name'] for j in jobs],errors=errors))
    break
   if not active and not summary.exists():
    if not jobs and time.monotonic()-reservation_wait_start<60:
     time.sleep(1);continue  # Budget supervisors may still be recording initial reservations.
    raise RuntimeError('Collector ended without committed summary')
   time.sleep(30)
  d=json.loads(summary.read_text())
  if set(d.get('bands',{}))!={'20','50','80'}:
   status('COLLECTOR_BOUNDED_END_QUALITY_LEVELS_MISSING',collector_summary=d,reason='Full original A1 prerequisites incomplete; no quality relabeling or12mformal collection with mismatched policies. Development evidence requires review within the remaining authorized budget.')
   return
  collect()
  cpu(['scripts/freeze_a1_dataset.py','--data-root','data/a1_dataset/v1','--output','data/a1_dataset/v1/frozen_manifest.json'],'freeze_formal_dataset')
  cpu(['scripts/fit_a1_normalization.py','--manifest','data/a1_dataset/v1/frozen_manifest.json','--output','data/a1_dataset/v1/normalization.pt'],'fit_formal_normalization')
  final_if_development_passes(models())
if __name__=='__main__':
 try:main()
 except BaseException as error:
  status('COORDINATOR_REVIEW_REQUIRED',error=repr(error),action='No new GPU jobs launched. Already launched bounded jobs retain their own independent supervisors; do not kill unrelated processes.')
  raise
