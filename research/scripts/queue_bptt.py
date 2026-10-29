"""Queue only this iteration's jobs on their already-authorized Lambda GPUs."""
from pathlib import Path
import json,time,subprocess,os
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/bptt_iteration_20261027'
def read(p):return json.loads(p.read_text())
def launch(name,gpu,seconds,phase,script,args):
    log=open(RUN/(name+'.supervisor.log'),'w')
    subprocess.Popen(['python3','scripts/bptt_budget.py',name,str(gpu),str(seconds),phase,'.venv-mjw/bin/python',script,*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def main():
    started=time.time();launched=set()
    while time.time()-started<22000:
        jobs={j['name']:j for j in read(RUN/'ledger.json')['jobs']}
        for gpu,method,seed in [(0,'P',43),(1,'S',43),(2,'P',59),(3,'S',59)]:
            arm=f'core_{method}_{seed}';name='fixed_'+arm
            if arm in jobs and jobs[arm]['status']=='finished' and name not in jobs and name not in launched:
                if (RUN/arm/'summary.json').exists():
                    launch(name,gpu,1400,'final_evaluation','scripts/evaluate_bptt.py',['--arm',arm]);launched.add(name)
                else:
                    (RUN/(arm+'_needs_attention.txt')).write_text('Core process ended without summary; no automatic continuation.');launched.add(name)
        dep=jobs.get('fixed_core_P_59')
        if dep and dep['status']=='finished' and dep.get('returncode')==0 and 'shared_two_objects' not in jobs and 'shared_two_objects' not in launched:
            launch('shared_two_objects',2,6500,'multitask','scripts/train_bptt_multitask.py',['--steps','131072']);launched.add('shared_two_objects')
        if all(jobs.get('fixed_core_'+m+'_'+str(s),{}).get('status')=='finished' for m in ['P','S'] for s in [43,59]) and jobs.get('shared_two_objects',{}).get('status')=='finished':break
        time.sleep(30)
    (RUN/'queue_finished.json').write_text(json.dumps(dict(started=started,ended=time.time(),launched=sorted(launched)),indent=2)+'\n')
if __name__=='__main__':main()

