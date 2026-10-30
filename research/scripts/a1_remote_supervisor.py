"""One-use, bounded remote GPU grant; only its own child process group may be stopped."""
import argparse,getpass,hashlib,json,os,signal,socket,subprocess,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def process_ticks(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text()
        return text[text.rfind(')')+2:].split()[19]
    except FileNotFoundError:return None


def stop_owned_group(child,start_ticks,sig):
    if child.poll() is not None:return False
    assert process_ticks(child.pid)==start_ticks,'Refuse to signal a different process identity'
    assert os.getpgid(child.pid)==child.pid,'Expected isolated own process group'
    os.killpg(child.pid,sig);return True


def gpu_query(uuid):
    p=subprocess.run(['nvidia-smi','--id='+uuid,'--query-gpu=index,uuid,memory.total,memory.free',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
    rows=[r.strip().split(', ') for r in p.stdout.splitlines() if r.strip()];assert len(rows)==1
    index,actual,total,free=rows[0];assert actual==uuid
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],
                      capture_output=True,text=True,check=True,timeout=3)
    processes=[]
    for line in p.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)==3 and parts[1]==uuid:
            processes.append(dict(pid=int(parts[0]),memory_mib=int(parts[2]) if parts[2].isdigit() else None))
    return dict(index=int(index),uuid=actual,total_mib=int(total),free_mib=int(free),processes=processes)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('grant',type=Path);args=parser.parse_args()
    grant=json.loads(args.grant.read_text());sha=digest(grant)
    assert socket.gethostname().lower()==grant['hostname'].lower() and getpass.getuser()==grant['account']
    assert ROOT==Path(grant['working_directory']).resolve()
    assert grant['global_cap_seconds']==330*3600 and 60<=grant['limit_seconds']<=2700
    assert 1024<=grant['max_process_gpu_mib']<=4096 and grant['free_gpu_floor_mib']>=8192
    assert '/' not in grant['name'] and '..' not in grant['name']
    claims=ROOT/'runs/a1_remote_grant_claims';claims.mkdir(parents=True,exist_ok=True)
    with (claims/(grant['nonce']+'.json')).open('x') as claim:json.dump(dict(grant_sha256=sha,supervisor_pid=os.getpid()),claim)
    output=ROOT/'runs/a1_remote_jobs'/grant['name'];output.mkdir(parents=True,exist_ok=False)
    atomic(output/'grant.json',grant)
    receipt=dict(name=grant['name'],grant_sha256=sha,hostname=socket.gethostname(),account=getpass.getuser(),
                 gpu_uuid=grant['gpu_uuid'],supervisor_pid=os.getpid(),supervisor_start_ticks=process_ticks(os.getpid()),
                 status='preflight',seconds=0,child_pid=None,child_start_ticks=None,identity_checked_alive=False,
                 peak_process_gpu_mib=0,gpu_child_observed=False,reason='preflight',returncode=None)
    atomic(output/'receipt.json',receipt)
    child=None;started_mono=None;started_ticks=None;stop_reason=None;term_at=None
    external_stop={'reason':None}
    def stopped(sig,_):external_stop['reason']='supervisor_signal_'+str(sig)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        for relative,expected in grant['source_hashes'].items():
            path=(ROOT/relative).resolve();assert path.is_relative_to(ROOT) and path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
        assert grant['command'][0] in grant['source_hashes']
        gpu=gpu_query(grant['gpu_uuid']);assert gpu['index']==grant['gpu']
        assert gpu['free_mib']>=grant['free_gpu_floor_mib']+grant['max_process_gpu_mib'],'Insufficient free memory for bounded sharing'
        if not grant['allow_shared_gpu']:assert not gpu['processes'],'GPU already used'
        atomic(output/'prelaunch_gpu.json',gpu)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':grant['gpu_uuid'],'A1_REMOTE_MAX_GPU_MIB':str(grant['max_process_gpu_mib']),
             'A1_REMOTE_JOB_NAME':grant['name'],'A1_REMOTE_GRANT_SHA256':sha,'PYTHONNOUSERSITE':'1',
             'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2',
             'TRITON_CACHE_DIR':str(output/'triton_cache'),'TORCHINDUCTOR_CACHE_DIR':str(output/'inductor_cache'),
             'XDG_CACHE_HOME':str(output/'cache'),'WARP_CACHE_PATH':str(output/'warp_cache'),'PYTHONUNBUFFERED':'1',
             'CUDA_CACHE_PATH':str(output/'cuda_cache'),'TORCH_EXTENSIONS_DIR':str(output/'torch_extensions'),
             'SUPERDEX_ROOT':str(ROOT),'SUPERDEX_GPU_UUID':grant['gpu_uuid'],
             'A1_PORTABLE_MODEL_MANIFEST':'runs/a1_full_20261028/portable_models_20261030_001/manifest.json',
             'A1_REMOTE_PARENT_PID':str(os.getpid()),'A1_REMOTE_CHILD_WALL_SECONDS':str(grant['limit_seconds']-8)}
        env.pop('SUPERDEX_DATA_DIR',None)
        command=[grant['python'],'scripts/a1_remote_gpu_entry.py']+grant['command']
        assert external_stop['reason'] is None,'Supervisor was stopped before child launch'
        with (output/'child.log').open('x') as log:
            started_mono=time.monotonic();receipt['started']=time.time()
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started_ticks=process_ticks(child.pid);assert started_ticks
            receipt.update(status='running',child_pid=child.pid,child_start_ticks=started_ticks,identity_checked_alive=True,reason='running',command=command)
            atomic(output/'receipt.json',receipt)
            last_sample=0
            while child.poll() is None:
                elapsed=time.monotonic()-started_mono
                if external_stop['reason'] and stop_reason is None:stop_reason=external_stop['reason']
                if elapsed>=grant['limit_seconds']-15 and stop_reason is None:stop_reason='registered_process_budget'
                if elapsed-last_sample>=5 and stop_reason is None:
                    try:
                        gpu=gpu_query(grant['gpu_uuid']);last_sample=elapsed
                        own=[p for p in gpu['processes'] if p['pid']==child.pid]
                        if own:
                            receipt['gpu_child_observed']=True;memory=own[0]['memory_mib']
                            if memory is None:stop_reason='gpu_memory_measurement_unavailable'
                            else:
                                receipt['peak_process_gpu_mib']=max(receipt['peak_process_gpu_mib'],memory)
                                if memory>grant['max_process_gpu_mib']:stop_reason='own_gpu_memory_limit'
                        if gpu['free_mib']<grant['free_gpu_floor_mib']:stop_reason='shared_gpu_free_memory_floor'
                        with (output/'gpu_observations.jsonl').open('a') as f:f.write(json.dumps(dict(seconds=elapsed,gpu=gpu))+'\n')
                    except Exception:
                        receipt['gpu_query_error']=traceback.format_exc();stop_reason='gpu_observation_failed'
                if stop_reason and term_at is None:
                    stop_owned_group(child,started_ticks,signal.SIGTERM);term_at=time.monotonic()
                if term_at and (time.monotonic()-term_at>=5 or elapsed>=grant['limit_seconds']-5):
                    stop_owned_group(child,started_ticks,signal.SIGKILL)
                receipt.update(seconds=time.monotonic()-started_mono,observed_at=time.time(),reason=stop_reason or 'running',identity_checked_alive=child.poll() is None)
                atomic(output/'receipt.json',receipt);time.sleep(1)
            receipt['returncode']=child.wait()
        receipt['reason']=stop_reason or ('completed' if receipt['returncode']==0 else 'child_failed')
    except BaseException:
        receipt['error']=traceback.format_exc();receipt['reason']='supervisor_or_preflight_failed'
        if child is not None and child.poll() is None:
            stop_owned_group(child,started_ticks,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:stop_owned_group(child,started_ticks,signal.SIGKILL);child.wait(timeout=5)
        receipt['returncode']=child.returncode if child is not None else None
    finally:
        assert child is None or child.poll() is not None,'Do not issue a terminal receipt for a live child'
        receipt.update(status='finished',ended=time.time(),seconds=time.monotonic()-started_mono if started_mono is not None else 0,
                       identity_checked_alive=False,budget_violation=started_mono is not None and time.monotonic()-started_mono>grant['limit_seconds'])
        atomic(output/'receipt.json',receipt)


if __name__=='__main__':main()
