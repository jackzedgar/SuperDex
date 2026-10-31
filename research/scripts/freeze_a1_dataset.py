"""Freeze complete A1 episode shards, exact versions, family split and geometry."""
import argparse,json,hashlib,collections
from pathlib import Path
from a1_data_sources import ROOT,RUN,write
from a1_episode_contract import is_complete_episode
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 root=a.data_root.resolve();assert root.is_relative_to(ROOT);assert not a.output.exists()
 if a.smoke:assert root.is_relative_to(RUN)
 else:assert root.is_relative_to(ROOT/'data/a1_dataset')
 rows=[];seen=set();families={};complete_families=set();splits=collections.Counter();tasks=set();quality=set()
 for schema in sorted(root.rglob('schema.json')):
  path=schema.parent;meta=json.loads(schema.read_text());entries=[json.loads(x) for x in (path/'episodes.jsonl').read_text().splitlines() if x]
  assert entries
  for e in entries:
   assert e['episode_sha256'] not in seen,'Duplicate complete episode';seen.add(e['episode_sha256'])
   if e['family'] in families:assert families[e['family']]==e['split'],'Family leakage'
   families[e['family']]=e['split']
   if not a.smoke:assert e['metadata']['formal'] and e['metadata']['policy_noise_sigma']==.1
  recorded_entries=entries
  entries=[e for e in recorded_entries if is_complete_episode(e)]
  assert entries,'No complete episodes in '+str(path)
  for e in entries:splits[e['split']]+=e['controls'];complete_families.add(e['family'])
  total=sum(e['controls'] for e in entries);tasks.add(meta['task_id']);quality.add(meta['quality'])
  if not a.smoke:assert total>=100000,(path,total)
  files={}
  for name in ['schema.json','episodes.jsonl']+[k+'.bin' for k in meta['columns']]:
   f=path/name;files[name]=dict(bytes=f.stat().st_size,sha256=sha(f))
  rows.append(dict(path=str(path.relative_to(ROOT)),task_id=meta['task_id'],object_key=meta['object_key'],quality=meta['quality'],controls=total,episodes=len(entries),recorded_episodes=len(recorded_entries),excluded_incomplete_episode_ids=[e['id'] for e in recorded_entries if not is_complete_episode(e)],excluded_incomplete_controls=sum(e['controls'] for e in recorded_entries if not is_complete_episode(e)),files=files))
 probe=ROOT/'configs/a1_probe40_v1.json';pm=json.loads(probe.read_text())
 if not a.smoke:
  assert tasks=={r['task_id'] for r in pm['tasks']} and quality=={'20','50','80'} and len(rows)==120
  assert len({(r['task_id'],r['quality']) for r in rows})==120
 objects=sorted({r['object_key'] for r in rows})
 hands={}
 for side in ['right','left']:
  f=ROOT/f'data/a1_processed/assets/hands/{side}/kinematics.npz'
  if f.exists():hands[side]=dict(bytes=f.stat().st_size,sha256=sha(f))
 if not a.smoke:assert set(hands)=={'right','left'},'Freeze both hand kinematics before formal model training'
 write(a.output,dict(status='FROZEN_SMOKE_DATA' if a.smoke else 'FROZEN_FULL_A1_DATA_V1',formal=not a.smoke,frozen_integrity_version=1,hand_kinematics=hands,shards=rows,object_keys=objects,geometry={k:sha(ROOT/f'data/a1_geometry/objects/{k}/sdf.npz') for k in objects},probe40_sha256=sha(probe),controls=sum(r['controls'] for r in rows),episodes=sum(r['episodes'] for r in rows),recorded_episodes=len(seen),excluded_incomplete_controls=sum(r['excluded_incomplete_controls'] for r in rows),episode_eligibility='physical_failure_or_reference_end_only',split_controls=dict(splits),families=len(complete_families),recorded_families=len(families),split_protocol='Complete physical episodes; identical source-sequence/reset-seed family remains together across qualities. No overlapping windows across episodes; hash duplicates rejected. Source task and object may appear across splits, as required for A1, not A3.',training_sampling='Uniform objects, then uniform eligible windows within object. Distinct windows per full Adam update.',thresholds=json.loads((ROOT/'configs/a1_full_execution_20261028.json').read_text())['plan']['original_G1_thresholds']))
 print(a.output,dict(splits),flush=True)
if __name__=='__main__':main()
