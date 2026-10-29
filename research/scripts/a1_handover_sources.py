"""One bounded source expansion based only on annotated handover neighborhoods."""
import concurrent.futures,json,time,hashlib
from pathlib import Path
from a1_data_sources import ROOT,RUN,write
from a1_hot3d_catalog import collect
def main():
 cat=json.loads((RUN/'public_catalog/hot3d_train_quest3.json').read_text())['entries']
 old=json.loads((RUN/'public_catalog/hot3d_screen_selection.json').read_text())['sources']
 extra=json.loads((RUN/'public_catalog/source_pool_v2_preregister.json').read_text())['hot3d_sources']
 existing={e['path'] for e in old+extra}
 screen=json.loads((RUN/'public_catalog/hot3d_reference_screen_v2_partial.json').read_text())
 rows=[r for r in screen['rows'] if 'bimanual_handover_candidate' in r['candidate_strata']]
 definitions=json.loads((ROOT/'data/raw/hot3d/clip_definitions.json').read_text())
 target_sequences={definitions[str(int(r['clip'].split('-')[1]))]['sequence_id'] for r in rows}
 candidates=[e for e in cat if e['path'].endswith('.tar') and e['path'] not in existing and definitions.get(str(int(Path(e['path']).stem.split('-')[1])),{}).get('sequence_id') in target_sequences]
 candidates.sort(key=lambda e:hashlib.sha256(('280045:'+e['path']).encode()).hexdigest());selected=candidates[:64]
 prereg=RUN/'public_catalog/source_pool_v3_preregister.json'
 assert not prereg.exists(),'Do not overwrite registered source choice'
 write(prereg,dict(selection='At most64 source clips from sequences with source-only handover proximity evidence; deterministic hash order seed280045; no policy/model outcomes',source_sequences=sorted(target_sequences),pool=len(candidates),selected=selected))
 start=time.monotonic();results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for r in pool.map(collect,selected):
   results.append(r);write(RUN/'public_catalog/source_pool_v3_progress.json',dict(completed=len(results),total=len(selected),cpu_wall_seconds=time.monotonic()-start,records=results))
 print('DONE',len(results),flush=True)
if __name__=='__main__':main()
