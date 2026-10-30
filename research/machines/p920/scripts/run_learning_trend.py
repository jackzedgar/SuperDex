"""Launch only the three requested learning jobs, serially, with hard wall caps."""
import argparse,time,subprocess,sys,os,signal
from exploration_common import RUN,ROOT,read_json,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--only',choices=['training','evaluation'],default='training');args=p.parse_args()
    budget=read_json(RUN/'budget.json');ledger_path=RUN/'process_ledger.json';ledger=read_json(ledger_path) if ledger_path.exists() else dict(processes=[],definition='Observed child-process wall time, including imports, CPU/I/O and evaluation inside each job. Only owned child process groups are signaled at hard timeout. CPU-only artifact work is excluded. Not kernel-only GPU time.')
    if args.only=='training':jobs=[('ppo_uniform_rsi','ppo',1050,['scripts/train_exploratory_ppo.py','--trend','--arm','uniform_rsi','--steps','81920','--seconds','1025']),('ppo_start50','ppo',1050,['scripts/train_exploratory_ppo.py','--trend','--arm','start50','--steps','81920','--seconds','1025']),('world_models','world_models',900,['scripts/train_world_models_mini.py','--trend','--updates','2000','--seconds','875'])]
    else:jobs=[('canonical_evaluation','evaluation_and_anomalies',300,['scripts/evaluate_learning_trend.py'])]
    for name,category,cap,cmd in jobs:
        assert not any(x['name']==name for x in ledger['processes']),f'{name} already attempted; do not silently rerun or spend extra budget.'
        spent=sum(x['wall_seconds'] for x in ledger['processes']);category_spent=sum(x['wall_seconds'] for x in ledger['processes'] if x['category']==category)
        effective=min(cap,budget['total_gpu_process_seconds_cap']-spent-2,budget['allocation_seconds'][category]-category_spent-2)
        assert effective>5
        print('START',name,'hard_cap_seconds',effective,flush=True);started=time.perf_counter();timeout=False
        with (RUN/(name+'.log')).open('x') as f:
            proc=subprocess.Popen([sys.executable,'-u',*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=effective)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(proc.pid,signal.SIGTERM)
                try:code=proc.wait(timeout=1)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
        duration=time.perf_counter()-started;ledger['processes'].append(dict(name=name,category=category,pid=proc.pid,wall_seconds=duration,hard_cap_seconds=effective,returncode=code,hard_timeout=timeout,command=[sys.executable,*cmd]));ledger['total_seconds']=sum(x['wall_seconds'] for x in ledger['processes']);write_json(ledger_path,ledger)
        print('FINISH',name,'returncode',code,'wall_seconds',duration,flush=True)
    print('Requested',args.only,'jobs ended; no automatic extra experiments.',flush=True)


if __name__=='__main__':main()
