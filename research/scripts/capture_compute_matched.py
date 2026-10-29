"""CPU-only outcome-blind near-equal-compute checkpoint capture."""
import pathlib,json,time,shutil,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];RUN=ROOT/'runs/bptt_iteration_20261027'
def main():
    out=RUN/'compute_matched';out.mkdir(exist_ok=True);captured=set()
    for _ in range(900):
        jobs={j['name']:j for j in json.loads((RUN/'ledger.json').read_text())['jobs']}
        for seed in [43,59]:
            if seed in captured or (out/f'seed_{seed}/selection.json').exists():captured.add(seed);continue
            pair={m:jobs.get(f'core_{m}_{seed}') for m in ['P','S']}
            finished=[(m,j) for m,j in pair.items() if j and j['status']=='finished' and (RUN/f'core_{m}_{seed}/checkpoint_final.pt').exists()]
            if not finished:continue
            first,j=min(finished,key=lambda z:z[1]['ended']);ceiling=j['seconds'];record=dict(seed=seed,first_finished=first,common_core_process_ceiling_seconds=ceiling,selection_uses_success=False,policy_snapshots={})
            dest=out/f'seed_{seed}';dest.mkdir(exist_ok=True)
            for method in ['P','S']:
                job=pair[method];d=RUN/f'core_{method}_{seed}'
                if job['status']=='finished' and job.get('seconds',1e20)<=ceiling+2:src=d/'checkpoint_final.pt';elapsed=job['seconds']
                else:
                    candidates=[p for p in [d/'recovery_latest.pt',*d.glob('step_*.pt')] if p.exists() and p.stat().st_mtime-job['started']<=ceiling]
                    src=max(candidates,key=lambda p:p.stat().st_mtime);elapsed=src.stat().st_mtime-job['started']
                    # Recovery is atomically replaced by trainer; source opened once.
                target=dest/(method+'.pt')
                with src.open('rb') as f:
                    stamp=__import__('os').fstat(f.fileno()).st_mtime
                    if src.name!='checkpoint_final.pt':elapsed=stamp-job['started']
                    if elapsed>ceiling+2:raise RuntimeError('Recovery replaced across time ceiling; retry with immutable scheduled checkpoint')
                    with target.open('wb') as g:shutil.copyfileobj(f,g)
                record['policy_snapshots'][method]=dict(source=str(src.relative_to(ROOT)),gpu_process_seconds_at_save=elapsed,seconds_under_common_ceiling=ceiling-elapsed,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            (dest/'selection.json').write_text(json.dumps(record,indent=2)+'\n');captured.add(seed)
        if len(captured)==2:break
        time.sleep(5)
if __name__=='__main__':main()

