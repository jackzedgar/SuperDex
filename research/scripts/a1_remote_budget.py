"""Remote grants share the original Lambda ledger and its locked phase limits."""
import hashlib,json,time,uuid
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def reserved_seconds(jobs,now):
    return sum(float(j['seconds']) if j['status']=='finished' else
               float(j['limit_seconds']) if j['status']=='remote_reserved' else
               max(float(j['limit_seconds']),max(0,now-float(j.get('started',now))))
               for j in jobs)


def make_remote_grant(request,ledger,phases,registry,now,nonce):
    assert ledger['cap_seconds']==330*3600 and ledger['phase_caps']==phases
    assert registry['global_cap_seconds']==ledger['cap_seconds']
    name=request['name'];assert name and '/' not in name and '..' not in name
    node=registry['nodes'][request['node']];phase=request['phase'];seconds=float(request['limit_seconds'])
    assert phase in phases and 60<=seconds<=registry['maximum_single_validation_seconds']
    gpu=int(request['gpu']);assert str(gpu) in node['gpus']
    assert request['gpu_uuid']==node['gpus'][str(gpu)]
    assert 0<int(request['max_process_gpu_mib'])<=node['max_process_gpu_mib']
    assert int(request['free_gpu_floor_mib'])>=node['free_gpu_floor_mib']
    assert request['command'] and request['command'][0] in registry['allowed_scripts']
    assert all(isinstance(s,str) for s in request['command'])
    assert request['source_hashes'] and request['command'][0] in request['source_hashes']
    for relative,sha in request['source_hashes'].items():
        p=Path(relative);assert not p.is_absolute() and '..' not in p.parts and len(sha)==64
    jobs=ledger['jobs'];assert not any(j['name']==name for j in jobs),'Unique attempt name required'
    assert not any(j.get('gpu_uuid')==request['gpu_uuid'] and j['status']!='finished' for j in jobs),'Own active grant on GPU'
    assert reserved_seconds([j for j in jobs if j['phase']==phase],now)+seconds<=phases[phase]
    assert reserved_seconds(jobs,now)+seconds<=ledger['cap_seconds']-(0 if phase=='evaluation' else 3600)
    if not phase.startswith('revision'):
        assert reserved_seconds([j for j in jobs if not j['phase'].startswith('revision')],now)+seconds<=240*3600
    grant=dict(version=1,name=name,nonce=nonce,node=request['node'],hostname=node['hostname'],account=node['account'],
        working_directory=node['working_directory'],python=node['python'],gpu=gpu,gpu_uuid=request['gpu_uuid'],
        phase=phase,limit_seconds=seconds,max_process_gpu_mib=int(request['max_process_gpu_mib']),
        free_gpu_floor_mib=int(request['free_gpu_floor_mib']),allow_shared_gpu=node['allow_shared_gpu'],
        command=request['command'],source_hashes=request['source_hashes'],job_registration_sha256=request['job_registration_sha256'],
        global_cap_seconds=ledger['cap_seconds'],reserved_at_unix=now,authority=registry['authorization_files'],
        accounting_rule='One GPU child process; actual whole-process wall seconds. Full grant remains reserved until SSH-verified receipt reconciliation.')
    digest=canonical_hash(grant)
    job=dict(name=name,node=request['node'],gpu=gpu,gpu_uuid=request['gpu_uuid'],phase=phase,limit_seconds=seconds,
             started=now,reserved_at_unix=now,status='remote_reserved',grant_sha256=digest,grant=grant,
             command=request['command'],remote_observation_required=True)
    jobs.append(job)
    return grant


def reconcile_remote_receipt(ledger,receipt,now):
    job=next(j for j in ledger['jobs'] if j['name']==receipt['name'])
    assert job.get('grant') and receipt['grant_sha256']==job['grant_sha256']
    g=job['grant']
    for field in ['hostname','account','gpu_uuid']:
        assert receipt[field]==g[field],field
    assert receipt['status'] in ['running','finished']
    seconds=float(receipt['seconds']);assert 0<=seconds<365*24*3600
    if job['status']=='finished':
        assert receipt['status']=='finished' and canonical_hash(receipt)==job['final_receipt_sha256'],'Terminal receipt is immutable'
        return dict(status='already_reconciled',name=job['name'])
    assert job['status'] in ['remote_reserved','remote_running']
    if receipt['status']=='running':
        assert receipt['child_pid'] and receipt['child_start_ticks'] and receipt['identity_checked_alive']
        job.update(status='remote_running',started=float(receipt['started']),
                   child_pid=receipt['child_pid'],child_start_ticks=str(receipt['child_start_ticks']),
                   last_remote_observed_seconds=seconds,last_remote_observed_at=float(receipt['observed_at']),
                   remote_observation_required=True)
        return dict(status='running_reconciled',name=job['name'])
    assert receipt['identity_checked_alive'] is False
    job.update(status='finished',started=float(receipt.get('started',job['reserved_at_unix'])),
               ended=float(receipt['ended']),seconds=seconds,returncode=receipt['returncode'],reason=receipt['reason'],
               child_pid=receipt.get('child_pid'),child_start_ticks=receipt.get('child_start_ticks'),
               peak_process_gpu_memory_mib=receipt.get('peak_process_gpu_mib'),
               budget_violation=seconds>float(g['limit_seconds']),remote_observation_required=False,
               final_receipt_sha256=canonical_hash(receipt),final_receipt=receipt)
    return dict(status='finished_reconciled',name=job['name'],seconds=seconds,budget_violation=job['budget_violation'])


def reserve(request,budget):
    assert budget.socket.gethostname()=='Lambda' and budget.getpass.getuser()=='youran'
    registry=json.loads((budget.ROOT/'configs/a1_remote_nodes_20261030.json').read_text())
    registration=(budget.ROOT/request['job_registration_path']).resolve()
    assert registration.is_relative_to(budget.ROOT/'reports')
    assert hashlib.sha256(registration.read_bytes()).hexdigest()==request['job_registration_sha256']
    registered=json.loads(registration.read_text())['jobs'][request['name']]
    for field in ['node','gpu','gpu_uuid','phase','limit_seconds','max_process_gpu_mib','free_gpu_floor_mib','command']:
        assert registered[field]==request[field],field
    for relative,expected in request['source_hashes'].items():
        path=(budget.ROOT/relative).resolve();assert path.is_relative_to(budget.ROOT)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
    grant=budget.mutate(lambda ledger:make_remote_grant(request,ledger,budget.PHASES,registry,time.time(),uuid.uuid4().hex))
    directory=budget.RUN/'remote_grants';directory.mkdir(exist_ok=True)
    path=directory/(grant['name']+'.json');assert not path.exists()
    budget.atomic(path,grant)
    return dict(grant=grant,grant_sha256=canonical_hash(grant),path=str(path.relative_to(budget.ROOT)))


def reconcile(receipt,budget):
    assert budget.socket.gethostname()=='Lambda' and budget.getpass.getuser()=='youran'
    return budget.mutate(lambda ledger:reconcile_remote_receipt(ledger,receipt,time.time()))
