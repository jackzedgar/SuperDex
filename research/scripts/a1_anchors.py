"""Precompute documented nominal 20-control-step settling anchors, before policy training."""
import argparse,json,time,signal,hashlib
from pathlib import Path
import numpy as np,torch,warp as wp
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
stop=False
def onstop(*_):
 global stop;stop=True
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4);p.add_argument('--controls-cap',type=int,default=30000);p.add_argument('--tag',required=True);p.add_argument('--manifest',type=Path);a=p.parse_args()
 signal.signal(signal.SIGTERM,onstop);signal.signal(signal.SIGINT,onstop);torch.set_num_threads(2)
 rows=[]
 for q in sorted((ROOT/'data/a1_processed/tasks').glob('*_rv2/qualification.json')):
  meta=json.loads(q.read_text())
  if meta['kinematic_candidate']:rows.append((q.parent,meta))
 rows=[r for r in rows if int.from_bytes(hashlib.sha256(r[0].name.encode()).digest()[:4],'little')%a.shards==a.shard]
 if a.manifest:
  allowed={r['task_id'] for r in json.loads(a.manifest.read_text())['tasks']};rows=[r for r in rows if r[0].name in allowed]
 total=0;start=time.monotonic();summaries=[]
 for task,meta in rows:
  if stop or total>=a.controls_cap:break
  done=task/'anchors_v1.json'
  if done.exists() and json.loads(done.read_text()).get('complete'):continue
  if not (ROOT/f"data/a1_geometry/objects/{meta['object']['object_key']}/grid.json").exists():
   summaries.append(dict(task=task.name,status='WAITING_GEOMETRY'));continue
  env=A1Env(task);nbins=(env.task.length-1+24)//25
  anchors=[];binrows=[];attempts=[];controls0=total
  for b in range(nbins):
   if stop or total>=a.controls_cap:break
   found=None
   for frame in range(b*25,min((b+1)*25,env.task.length-1)):
    if stop or total>=a.controls_cap:break
    env.reset(frame,seed=280043,randomize=False);env.hold_mode.fill_(True)
    initial=env.qpos[0,env.task.nu:env.task.nu+3].clone();maxdrift=0.;maxpen=0.;finite=True;executed=0
    for _ in range(20):
     s,r,doneflag,info=env.step(torch.zeros(1,52,device=env.device));total+=1;executed+=1
     drift=float(torch.linalg.vector_norm(env.qpos[0,env.task.nu:env.task.nu+3]-initial))
     maxdrift=max(maxdrift,drift);maxpen=max(maxpen,float(env.contact_summary[7]));finite=bool(info['finite'])
     if maxdrift>.01 or maxpen>.002 or not finite or bool(env.contact_summary[9]) or stop:break
    ok=executed==20 and maxdrift<=.01 and maxpen<=.002 and finite
    attempt=dict(bin=b,source_frame=frame,controls=executed,max_drift_m=maxdrift,max_penetration_m=maxpen,finite=finite,accepted=ok)
    attempts.append(attempt)
    if ok:
     found=len(anchors);snap=env.snapshot();snap['bin']=b;anchors.append(snap);break
   complete_bin=not stop and total<a.controls_cap
   binrows.append(dict(bin=b,anchor_index=found,status='ANCHOR' if found is not None else 'NO_ANCHOR' if complete_bin else 'INCOMPLETE'))
  arrays={}
  if anchors:
   for key in ['qpos','qvel','ctrl','history','servo','qacc_warmstart']:arrays[key]=np.stack([x[key] for x in anchors])
   arrays['phase']=np.array([x['phase'] for x in anchors]);arrays['bin']=np.array([x['bin'] for x in anchors])
   np.savez_compressed(task/'anchors_v1.npz',**arrays)
  report=dict(task=task.name,complete=len(binrows)==nbins and all(x['status']!='INCOMPLETE' for x in binrows),bins=nbins,anchors=len(anchors),actual_controls=total-controls0,bin_results=binrows,attempts=attempts,protocol='Bin25steps; scan successive reference frames;20PD hold controls, object free. Max over holding interval penetration<=2mm and displacement<=1cm. Nominal context. No policy/model outcomes used. Coverage is reported, not a grasp success admission gate.',missing_bin_rule='Only real replay states subsequently surviving25controls may seed unanchored bins; no fabricated anchors.')
  write(task/'anchors_v1.json',report);summaries.append({k:report[k] for k in ['task','complete','bins','anchors','actual_controls']})
  write(RUN/f'{a.tag}.json',dict(status='RUNNING',gpu_process_seconds=time.monotonic()-start,actual_controls=total,tasks=summaries))
  print(json.dumps(summaries[-1]),flush=True);del env;torch.cuda.empty_cache()
 write(RUN/f'{a.tag}.json',dict(status='BOUNDED_STOP' if stop or total>=a.controls_cap else 'FINISHED_AVAILABLE_TASKS',gpu_process_seconds=time.monotonic()-start,actual_controls=total,tasks=summaries))
if __name__=='__main__':main()
