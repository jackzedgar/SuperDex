"""Bounded original-world collection correctness/cost probe. Never formal data."""
import argparse,datetime,gc,getpass,hashlib,json,math,os,signal,socket,time,traceback
from pathlib import Path
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_policy import CollectorPolicy
from a1_collection_original_pool import CollectionPool
STOP=False
def halt(*_):
    global STOP
    STOP=True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
    c=json.loads(args.config.read_text());assert c['formal'] is False and c['phase']=='implementation' and c['worlds']==8
    for path,digest in c['source_hashes'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/c['checkpoint'])==c['checkpoint_sha256']
    assert c['mode'] in ['correctness_gate','full40_timing'] and c['controls_per_task']==512
    if c['mode']=='full40_timing':
        gate=json.loads((ROOT/c['gate_summary']).read_text());assert gate['status']=='complete' and gate['implementation_gate_passed'] and len(gate['rows'])==4
        assert gate['execution_code_hashes']==c['execution_code_hashes']
    out=RUN/c['run'];out.mkdir(exist_ok=False);write(out/'registration.json',c)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
    started=time.monotonic();deadline=started+c['internal_seconds'];pool=None;rows=[]
    stopped=lambda:STOP or time.monotonic()>=deadline
    try:
        parent=torch.load(ROOT/c['checkpoint'],map_location='cpu',weights_only=False);assert parent['configuration']['reset_mode']=='uniform_reference'
        policy=CollectorPolicy().cuda();policy.load_state_dict(parent['policy']);policy.logstd.fill_(math.log(.1));policy.eval()
        for p in policy.parameters():p.requires_grad_(False)
        definitions=json.loads((ROOT/'data/raw/hot3d/clip_definitions.json').read_text())
        for index,t in enumerate(c['tasks']):
            if stopped():break
            task_path=ROOT/t['task_path'];source=json.loads((task_path/'source_selection.json').read_text())
            family=definitions[str(int(source['clip'].split('-')[1]))]['sequence_id'] if source['source']=='hot3d' else source['sequence']
            write(out/'progress.json',dict(stage='BUILD',task=t['task_id'],tasks_complete=len(rows),time_beijing=now(),seconds=time.monotonic()-started))
            before=time.monotonic()
            pool=CollectionPool(task_path,policy,out/t['task_id'],t['seed'],dict(path=c['checkpoint'],sha256=c['checkpoint_sha256']),family,parent['configuration']['reset_mode'],validate=c['mode']=='correctness_gate',should_stop=stopped)
            def progress(p):write(out/'progress.json',dict(stage='COLLECT',task=t['task_id'],tasks_complete=len(rows),time_beijing=now(),**p))
            summary=pool.collect(c['controls_per_task'],stopped,progress)
            summary.update(task=t['task_id'],stratum=t['stratum'],seed=t['seed'],policy_checkpoint_sha256=c['checkpoint_sha256'],wall_seconds_including_environment_build=time.monotonic()-before)
            write(out/t['task_id']/'summary.json',summary);rows.append(summary)
            write(out/'partial_summary.json',dict(rows=rows,time_beijing=now(),formal=False))
            torch.cuda.synchronize();torch.cuda.set_stream(torch.cuda.default_stream());pool=None;gc.collect();torch.cuda.empty_cache()
            if summary['status']!='quota_complete':break
        assert all(torch.equal(v.detach().cpu(),parent['policy'][k]) for k,v in policy.state_dict().items() if k!='logstd')
        assert torch.equal(policy.logstd.detach().cpu(),torch.full((52,),math.log(.1)))
        complete=len(rows)==len(c['tasks']) and all(r['status']=='quota_complete' for r in rows)
        gate=complete and all(r['audit']['input_checks']>=8 and r['audit']['packed_checks']>=8 and r['audit']['reset_checks']>=8 for r in rows)
        if c['mode']=='correctness_gate':gate=gate and all(r['audit']['reset_isolation_checks']==r['audit']['reset_checks'] for r in rows)
        result=dict(status='complete' if complete else 'bounded_stop',time_beijing=now(),mode=c['mode'],formal=False,tasks_expected=len(c['tasks']),tasks_complete=sum(r['status']=='quota_complete' for r in rows),rows=rows,implementation_gate_passed=gate,execution_code_hashes=c['execution_code_hashes'],seconds_including_build_io=time.monotonic()-started,physical_controls=sum(r['all_physical_controls_this_process'] for r in rows),real_controls=sum(r['new_controls_this_process'] for r in rows),policy_weights_unchanged_except_registered_collection_sigma=True,optimizer_updates=0,quality_passports_created=False,full_A1_complete=False)
        write(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    except BaseException as exc:
        write(out/'fatal.json',dict(error=repr(exc),traceback=traceback.format_exc(),time_beijing=now(),rows_complete=len(rows),physical_controls_live=pool.real_controls if pool else None))
        if pool is not None:
            for lane,buffer in pool.buffers.items():np.savez_compressed(out/f'failed_uncommitted_lane{lane}.npz',**{k:np.asarray(v) for k,v in buffer.arrays.items()})
        raise
if __name__=='__main__':main()
