"""CPU geometry for all source-qualified rv2 candidates, without policy selection."""
import concurrent.futures,json,subprocess,os,time
from pathlib import Path
from a1_data_sources import ROOT,RUN,write
def main():
 keys=sorted({json.loads(p.read_text())['object']['object_key'] for p in (ROOT/'data/a1_processed/tasks').glob('*_rv2/qualification.json') if json.loads(p.read_text())['kinematic_candidate']})
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def one(key):
  log=RUN/'geometry_logs'/f'{key}.log';log.parent.mkdir(exist_ok=True)
  with log.open('a') as f:r=subprocess.run([str(ROOT/'.venv-mjw/bin/python'),'-u','scripts/a1_prepare_geometry.py','--object',key],env=env,stdout=f,stderr=subprocess.STDOUT)
  return dict(object_key=key,returncode=r.returncode,ready=(ROOT/f'data/a1_geometry/objects/{key}/grid.json').exists())
 start=time.monotonic();rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for r in pool.map(one,keys):
   rows.append(r);write(RUN/'geometry_pool_progress.json',dict(completed=len(rows),total=len(keys),cpu_wall_seconds=time.monotonic()-start,records=rows))
if __name__=='__main__':main()
