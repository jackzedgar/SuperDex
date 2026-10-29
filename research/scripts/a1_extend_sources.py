"""Bounded second source pool, preregistered before any policy/model outcomes."""
import concurrent.futures,json,time
from pathlib import Path
import numpy as np
from a1_data_sources import ROOT,RUN,fetch,write
from a1_hot3d_catalog import collect

def main():
 start=time.monotonic()
 catalog=json.loads((RUN/'public_catalog/hot3d_train_quest3.json').read_text())['entries']
 old=json.loads((RUN/'public_catalog/hot3d_screen_selection.json').read_text())['sources']
 oldpaths={e['path'] for e in old}
 remaining=sorted([e for e in catalog if e['path'].endswith('.tar') and e['path'] not in oldpaths],key=lambda e:e['path'])
 order=np.random.default_rng(280044).permutation(len(remaining))
 extra=[remaining[int(i)] for i in order[:96]]
 annotation=json.loads((RUN/'public_catalog/oakink2_anno_preview.json').read_text())['entries']
 amap={Path(e['path']).stem:e for e in annotation}
 tool=[]
 for p in sorted((ROOT/'data/a1_sources/oakink2/program/program_info').glob('*.json')):
  desc=json.loads((p.parent.parent/'desc_info'/p.name).read_text())
  for interval,row in json.loads(p.read_text()).items():
   if row['primitive']=='write_on_paper' and len(row.get('obj_list_rh') or [])==1 and row.get('primitive_rh')=='write_on_paper':
    tool.append(dict(sequence=p.stem,interval=interval,program=row,description=desc.get(interval),annotation=amap[p.stem]))
 unique={x['annotation']['path']:x['annotation'] for x in tool}
 prereg={'hot3d_additional_count':len(extra),'hot3d_seed':280044,'hot3d_rule':'Uniform without replacement from clips outside first64; expansion because handover had only6 unique clips, before any model outcomes.','hot3d_sources':extra,'oakink_tool_segments':tool,'oakink_annotation_bytes':sum(x['size'] for x in unique.values()),'tool_selection_rule':'All source-annotated right-hand write_on_paper segments with one manipulated object; semantic candidacy only, mass/physical/support checks still required.','frozen_probe40':False}
 write(RUN/'public_catalog/source_pool_v2_preregister.json',prereg)
 print('PREREG',len(extra),len(tool),len(unique),prereg['oakink_annotation_bytes'],flush=True)
 def oak(e):
  try:return {'path':str(fetch('oakink2',e)),'ok':True}
  except Exception as err:return {'path':e['path'],'error':repr(err)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(oak,e) for e in unique.values()]+[pool.submit(collect,e) for e in extra]
  results=[]
  for f in concurrent.futures.as_completed(futures):
   results.append(f.result());write(RUN/'public_catalog/source_pool_v2_progress.json',{'completed':len(results),'total':len(futures),'records':results,'cpu_wall_seconds':time.monotonic()-start})
 print('FINISHED',len(results),sum('error' in r for r in results),flush=True)
if __name__=='__main__':main()
