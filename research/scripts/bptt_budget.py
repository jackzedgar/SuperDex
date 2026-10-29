"""Cumulative GPU-process accounting. Own child processes only."""
import os,sys,json,time,fcntl,subprocess,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/bptt_iteration_20261027'
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def locked(fn):
    with (RUN/'ledger.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);p=RUN/'ledger.json'
        x=json.loads(p.read_text()) if p.exists() else {'cap_seconds':86400,'jobs':[]}
        ret=fn(x);write(p,x);return ret
def used(x):
    now=time.time()
    return sum(j.get('seconds',now-j['started']) for j in x['jobs'])
def main():
    # name GPU seconds phase [command ...]
    name,gpu,limit,phase=sys.argv[1:5];limit=float(limit);cmd=sys.argv[5:]
    import socket,getpass
    assert socket.gethostname().lower()=='lambda' and getpass.getuser()=='youran'
    # CPU-only admission wait. Never kill or change another task.
    wait_started=time.time()
    while True:
        uid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
        occupants=[r for r in apps.splitlines() if uid in r]
        if not occupants and free>=2200:break
        if time.time()-wait_started>300:
            def skip(x):
                x['jobs'].append(dict(name=name,gpu=int(gpu),limit=limit,phase=phase,started=time.time(),ended=time.time(),seconds=0,status='finished',returncode=2,reason='CPU admission resource skip',occupants=occupants,memory_free_mib=free,cpu_wait_seconds=time.time()-wait_started))
            locked(skip);return
        time.sleep(30)
    def reserve(x):
        assert not any(j['name']==name for j in x['jobs'])
        active=sum(max(0,j['limit']-(time.time()-j['started'])) for j in x['jobs'] if j['status']=='running')
        reserve=0 if phase=='final_evaluation' else 3600
        assert used(x)+active+limit<=86400-reserve,(used(x),active,limit)
        x['jobs'].append(dict(name=name,gpu=int(gpu),limit=limit,phase=phase,started=time.time(),status='running',pid=os.getpid(),command=cmd,cpu_admission_seconds=time.time()-wait_started,gpu_uuid=uid))
    locked(reserve);start=time.time();reason='completed';child=None
    try:
        env=os.environ.copy();env['SUPERDEX_GPU_INDEX']=gpu
        # Activation resolves an actual Lambda UUID; no inherited old UUID.
        env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('SUPERDEX_GPU_UUID',None)
        import shlex
        script='source ./activate_mjw.sh; exec '+shlex.join(cmd)
        with (RUN/(name+'.stdout.log')).open('w') as log:
            child=subprocess.Popen(['bash','-lc',script],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(2)
                if int(time.time()-start)%120<2:
                    size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
                    if size>2_000_000_000:reason='iteration_output_cap'
                if time.time()-start>limit-35:reason='process_time_limit'
                if shutil.disk_usage(ROOT).free<1_300_000_000:reason='disk_reserve'
                if reason!='completed':
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
            code=child.returncode
    except BaseException as e:
        reason=repr(e);code=-1
        if child and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait()
    finally:
        def finish(x):
            j=next(j for j in x['jobs'] if j['name']==name)
            j.update(status='finished',ended=time.time(),seconds=time.time()-start,returncode=code,reason=reason)
            x['used_seconds']=used(x)
        locked(finish)
if __name__=='__main__':main()

