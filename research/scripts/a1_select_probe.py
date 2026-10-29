"""Deterministic source/kinematic selection; never reads learned outcome files."""
import hashlib,json,argparse
import numpy as np,trimesh
from pathlib import Path
from PIL import Image,ImageDraw
from a1_data_sources import ROOT,RUN,write
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--version',default='v4');p.add_argument('--suffix',default='');a=p.parse_args()
 rows=json.loads((RUN/f'public_catalog/task_candidates_{a.version}.json').read_text())['candidates']
 order=['bimanual_handover','in_hand_reorientation_or_regrasp','stable_grasp_transport','light_object_tool_use_below100g'];used=set();selected=[];excluded=[]
 definitions=json.loads((ROOT/'data/raw/hot3d/clip_definitions.json').read_text())
 for st in order:
  taken=0
  for row in rows:
   if row['stratum']!=st:continue
   path=ROOT/'data/a1_processed/tasks'/row['task_id'];p=path/'qualification.json'
   quality=json.loads(p.read_text()) if p.exists() else {}
   key=row.get('clip',row.get('sequence'))
   if not quality.get('kinematic_candidate'):
    excluded.append(dict(task_id=row['task_id'],reason='Kinematic/source gate',gates=quality.get('gates'),error_file=str(path/'error.json')));continue
   asset=ROOT/'data/a1_processed/assets/objects'/row['object_key']
   meshes=[trimesh.load(p,force='mesh',process=False) for p in asset.glob('collision_*.obj')]
   verts=np.concatenate([m.vertices for m in meshes]);shape=np.maximum(64,np.ceil((np.ptp(verts,axis=0)+.1)/.003).astype(int)+1)
   if shape.max()>128:
    excluded.append(dict(task_id=row['task_id'],reason='Original SDF geometry limits: padding5cm spacing<=3mm shape<=128',required_shape=shape.tolist()));continue
   if key in used:
    excluded.append(dict(task_id=row['task_id'],reason='Source clip already represented in another stratum'));continue
   if taken>=10:
    excluded.append(dict(task_id=row['task_id'],reason='Beyond fixed first10 source-hash order'));continue
   used.add(key);taken+=1
   source_family=definitions[str(int(row['clip'].split('-')[1]))]['sequence_id'] if row['source']=='hot3d' else row['sequence']
   receipt=ROOT/'data/a1_sources/hot3d'/row['clip']/'source_receipt.json' if row['source']=='hot3d' else None
   selected.append(dict(**row,task_path=str(path.relative_to(ROOT)),source_family=source_family,reference_sha256=sha(path/'reference.npz'),scene_sha256=sha(path/'scene.xml'),qualification=quality,receipt=json.loads(receipt.read_text()) if receipt else None))
  assert taken==10,(st,taken)
 manifest=dict(status='SOURCE_AND_KINEMATIC_SELECTION_PENDING_VISUAL_REVIEW',source_candidate_manifest=f'runs/a1_full_20261028/public_catalog/task_candidates_{a.version}.json',selection='Predeclared source hash order, first10 qualifying per stratum; prioritize rare handovers then reorientation then transport, no repeated source clip. No policy/model outcomes or anchor coverage used.',task_count=40,object_count=len({r['object_key'] for r in selected}),strata={st:10 for st in order},source_review='HOT3D source RGB proximity audit and OakInk official write_on_paper annotation; physics screening is not learning outcome selection',tasks=selected,exclusions=excluded,old_pilot_results_unchanged=True,formal_policy_initialization='Fresh common seeded network, not development-trained checkpoint',physical_assumptions='Unmeasured simulated nominal masses/inertias/servo parameters are explicit in per-task qualification; not D0 or robot identification')
 out=RUN/f'probe40_proposed_manifest_{a.version}{a.suffix}.json';assert not out.exists();write(out,manifest)
 review=RUN/f'source_review_{a.version}{a.suffix}';review.mkdir(exist_ok=True)
 for st in order[:3]:
  tasks=[r for r in selected if r['stratum']==st]
  for page in range(2):
   canvas=Image.new('RGB',(1280,5*270),'white');draw=ImageDraw.Draw(canvas)
   for i,row in enumerate(tasks[page*5:page*5+5]):
    top=i*270;draw.text((6,top+3),f"{row['clip']} object{row['object_id']} {row['object_name']} frames {row['start_frame']}..{row['end_frame']}",fill='black')
    for j,frame in enumerate([0,50,100,149]):
     p=ROOT/'data/a1_sources/hot3d'/row['clip']/f'{frame:06d}.image_1201-1.jpg'
     im=Image.open(p).convert('RGB');im.thumbnail((320,235));canvas.paste(im,(320*j,top+30))
     draw.text((320*j+4,top+16),f'source frame{frame}',fill='black')
   canvas.save(review/f'{st}_{page}.jpg',quality=90)
 print('SELECTED',40,manifest['object_count'],{s:sum(t['stratum']==s for t in selected) for s in order},flush=True)
if __name__=='__main__':main()
