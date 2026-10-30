"""Hard per-category and total process-wall limits; never auto-add experiments."""
import argparse,time,subprocess,sys,os,signal
from exploration_common import RUN,ROOT,read_json,write_json

p=argparse.ArgumentParser();p.add_argument('job',choices=['structure','gradient']);p.add_argument('--attempt',default='1');a=p.parse_args();cfg=read_json(RUN/'budget.json');path=RUN/'process_ledger.json';ledger=read_json(path) if path.exists() else dict(processes=[])
name=a.job+'_'+a.attempt;assert not any(r['name']==name for r in ledger['processes'])
spent=sum(r['wall_seconds'] for r in ledger['processes']);category=sum(r['wall_seconds'] for r in ledger['processes'] if r['category']==a.job);remaining=min(cfg['allocation_seconds'][a.job]-category,3600-spent)
assert remaining>30
command=[sys.executable,'-u','scripts/train_structure_residual.py' if a.job=='structure' else 'scripts/probe_model_gradient.py','--seconds',str(remaining-25)]
start=time.perf_counter();timeout=False
with (RUN/(name+'.log')).open('x') as f:
    child=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=child.wait(timeout=remaining-2)
    except subprocess.TimeoutExpired:
        timeout=True;os.killpg(child.pid,signal.SIGTERM)
        try:code=child.wait(timeout=1)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
ledger['processes'].append(dict(name=name,category=a.job,pid=child.pid,command=command,wall_seconds=time.perf_counter()-start,hard_cap_seconds=remaining-2,returncode=code,hard_timeout=timeout));write_json(path,ledger);print(ledger['processes'][-1],flush=True)
sys.exit(code)
