"""Lambda-only A1 GPU-process budget supervisor. Conservative whole-process wall accounting."""
import argparse,fcntl,json,os,signal,subprocess,time,shutil,socket,getpass,shlex,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/a1_full_20261028'
CAP=330*3600;PHASES={k:int(v*3600) for k,v in json.loads((ROOT/'configs/a1_execution_allocation_v2.json').read_text())['limits_hours'].items()}
assert sum(PHASES.values())==CAP and sum(v for k,v in PHASES.items() if not k.startswith('revision'))==240*3600
def atomic(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def elapsed(j,now):
    if j.get('status')=='finished':return j['seconds']
    if j.get('status')=='remote_reserved':return 0
    return max(0,now-j['started'])
def mutate(fn):
    RUN.mkdir(exist_ok=True)
    with (RUN/'gpu_ledger.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);p=RUN/'gpu_ledger.json'
        x=json.loads(p.read_text()) if p.exists() else {'cap_seconds':CAP,'authorized_by':'configs/a1_full_execution_20261028.json','phase_caps':PHASES,'jobs':[]}
        out=fn(x);now=time.time();x['last_updated']=now;x['used_seconds']=sum(elapsed(j,now) for j in x['jobs']);atomic(p,x);return out
def existing():
    p=RUN/'gpu_ledger.json'
    return json.loads(p.read_text()) if p.exists() else {'jobs':[],'used_seconds':0,'cap_seconds':CAP}
def main():
    p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--gpu',type=int,choices=range(4));p.add_argument('--seconds',type=float);p.add_argument('--phase',choices=PHASES);p.add_argument('--status',action='store_true');p.add_argument('--remote-reserve',type=Path);p.add_argument('--remote-reconcile',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.status:print(json.dumps(existing(),indent=2));return
    if a.remote_reserve or a.remote_reconcile:
        assert not (a.remote_reserve and a.remote_reconcile)
        from a1_remote_budget import reserve,reconcile
        result=reserve(json.loads(a.remote_reserve.read_text()),sys.modules[__name__]) if a.remote_reserve else reconcile(json.loads(a.remote_reconcile.read_text()),sys.modules[__name__])
        print(json.dumps(result,indent=2));return
    assert socket.gethostname().lower()=='lambda' and getpass.getuser()=='youran'
    assert a.name and a.phase and a.seconds>=60 and a.command
    assert '/' not in a.name and '..' not in a.name
    auth=json.loads((ROOT/'configs/a1_full_execution_20261028.json').read_text());assert auth['new_gpu_seconds_cap']==CAP
    uid=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    if any(uid in line for line in apps.splitlines()):raise RuntimeError('GPU occupied; no task modified. Retry after resource review.')
    assert shutil.disk_usage(ROOT).free>=5_000_000_000
    stop={'reason':None};child=None
    def signal_handler(sig,_):stop['reason']='supervisor_signal_'+str(sig)
    signal.signal(signal.SIGTERM,signal_handler);signal.signal(signal.SIGINT,signal_handler)
    start=time.time()
    def reserve(x):
        now=time.time();jobs=x['jobs']
        assert not any(j['name']==a.name for j in jobs),'Use a unique attempt name; preserve failures.'
        assert not any(j.get('gpu_uuid')==uid and j['status']!='finished' for j in jobs),'Own active reservation on GPU'
        # Active jobs reserve their whole authorized limit, not just elapsed time.
        def total(rows):return sum(j['seconds'] if j['status']=='finished' else max(j['limit_seconds'],elapsed(j,now)) for j in rows)
        overall=total(jobs);phase=total([j for j in jobs if j['phase']==a.phase])
        assert phase+a.seconds<=PHASES[a.phase],(a.phase,phase,a.seconds,PHASES[a.phase])
        assert overall+a.seconds<=CAP-(0 if a.phase=='evaluation' else 3600),(overall,a.seconds,CAP)
        if not a.phase.startswith('revision'):
            initial=total([j for j in jobs if not j['phase'].startswith('revision')]);assert initial+a.seconds<=240*3600
        jobs.append({'name':a.name,'gpu':a.gpu,'gpu_uuid':uid,'phase':a.phase,'limit_seconds':a.seconds,'started':start,'status':'running','supervisor_pid':os.getpid(),'supervisor_start_ticks':Path('/proc/self/stat').read_text().split()[21],'command':a.command})
    mutate(reserve)
    code=-1;reason='completed';peak=None;storage_bytes=None
    try:
        env=os.environ.copy();env['SUPERDEX_GPU_INDEX']=str(a.gpu);env.pop('CUDA_VISIBLE_DEVICES',None);env.pop('SUPERDEX_GPU_UUID',None)
        cmd=a.command[1:] if a.command[0]=='--' else a.command
        with (RUN/(a.name+'.stdout.log')).open('x') as log:
            child=subprocess.Popen(['bash','-lc','set -e; source ./activate_mjw.sh; exec '+shlex.join(cmd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            mutate(lambda x:next(j for j in x['jobs'] if j['name']==a.name).update(child_pid=child.pid))
            last_check=0
            while child.poll() is None:
                time.sleep(1);now=time.time()
                if now-start>=a.seconds-35:stop['reason']='process_budget'
                if shutil.disk_usage(ROOT).free<5_000_000_000:stop['reason']='filesystem_free_floor_5GB'
                if now-last_check>=30:
                    last_check=now
                    try:
                        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
                        for row in rows.splitlines():
                            fields=row.split(',')
                            if fields[0].strip()==str(child.pid):peak=max(peak or 0,int(fields[1].strip()))
                    except Exception:pass
                    if now-start>60:
                        paths=[RUN,ROOT/'data/a1_sources',ROOT/'data/a1_processed',ROOT/'data/a1_geometry',ROOT/'data/a1_dataset',ROOT/'data/raw/oakink2']
                        storage_bytes=sum(p.stat().st_size for base in paths if base.exists() for p in base.rglob('*') if p.is_file())
                        if storage_bytes>60_000_000_000:stop['reason']='new_output_capacity_60GB'
                if stop['reason']:
                    reason=stop['reason'];os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
            code=child.returncode
    except BaseException as e:
        reason=repr(e)
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    finally:
        end=time.time()
        def finish(x):
            next(j for j in x['jobs'] if j['name']==a.name).update(status='finished',ended=end,seconds=end-start,returncode=code,reason=reason,peak_process_gpu_memory_mib=peak,observed_output_bytes=storage_bytes)
        mutate(finish)
    raise SystemExit(code if code is not None and code>=0 else 1)
if __name__=='__main__':main()
