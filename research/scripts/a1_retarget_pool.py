"""Bounded CPU preprocessing of a preregistered source-only candidate pool."""
import argparse,concurrent.futures,json,os,subprocess,time
from pathlib import Path
from a1_data_sources import ROOT,RUN,write
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v1');a=parser.parse_args()
    manifest=RUN/f'public_catalog/task_candidates_{a.version}.json';rows=json.loads(manifest.read_text())['candidates']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def one(pair):
        i,row=pair;task=row['task_id'];start=time.monotonic()
        if row['source']=='oakink2' and not (ROOT/'data/raw/oakink2/anno_preview'/f"{row['sequence']}.pkl").exists():
            return dict(task_id=task,status='WAITING_SOURCE',seconds=0)
        path=RUN/'retarget_logs'/f'{task}.log';path.parent.mkdir(exist_ok=True)
        with path.open('a') as log:
            result=subprocess.run([str(ROOT/'.venv-mjw/bin/python'),'-u','scripts/a1_prepare_task.py','--manifest',str(manifest),'--index',str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        q=ROOT/'data/a1_processed/tasks'/task/'qualification.json'
        report=json.loads(q.read_text()) if q.exists() else {}
        return dict(task_id=task,status='KINEMATIC_CANDIDATE' if report.get('kinematic_candidate') else 'REJECTED_OR_ERROR',returncode=result.returncode,seconds=time.monotonic()-start,qualification=str(q.relative_to(ROOT)) if q.exists() else None,gates=report.get('gates'))
    start=time.monotonic();results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(one,enumerate(rows)):
            results.append(r);write(RUN/f'retarget_pool_{a.version}_progress.json',dict(completed=len(results),total=len(rows),cpu_wall_seconds=time.monotonic()-start,results=results))
            print(json.dumps(r),flush=True)
    print('CPU_POOL_FINISHED',len(results),flush=True)
if __name__=='__main__':main()
