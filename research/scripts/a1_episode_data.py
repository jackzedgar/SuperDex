"""Append-only whole-episode A1 data with causal reconstruction and family-level splits."""
import hashlib,json,os
from pathlib import Path
import numpy as np,torch
from a1_state import H,STATE,empty_state

def pack_state(state):
 hs=state[...,:274].reshape(*state.shape[:-1],2,H)
 return torch.cat([hs[...,:,:53].flatten(-2),state[...,274:],hs[...,131:137].flatten(-2)],-1)
def unpack_state(physical,history):
 shape=physical.shape[:-1];s=torch.zeros(*shape,STATE,device=physical.device,dtype=physical.dtype)
 hs=s[...,:274].reshape(*shape,2,H)
 hs[...,:,:53]=physical[...,:106].reshape(*shape,2,53)
 hs[...,53:131]=history.reshape(*shape,2,78)
 hs[...,131:137]=physical[...,119:].reshape(*shape,2,6)
 s[...,274:]=physical[...,106:119];return s

def split_family(family):
 u=int.from_bytes(hashlib.sha256(('A1-split-280047:'+family).encode()).digest()[:8],'big')/2**64
 return 'train' if u<.7 else 'development' if u<.85 else 'test'

class EpisodeWriter:
 def __init__(self,path,task,quality,policy_checkpoint,source_family):
  self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.task=task;self.quality=quality;self.source_family=source_family
  self.widths={'state':131,'request':52,'action':52,'applied':52,'generalized':task.m.nq+task.m.nv,'contact':10,'reward_terms':10,'context':276,'initial_history':156,'contact_details':16}
  self.spec={k:dict(width=v,dtype='<f4') for k,v in self.widths.items()};self.spec.update(phase=dict(width=1,dtype='<i4'),terminated=dict(width=1,dtype='u1'),contact_ranges=dict(width=2,dtype='<i8'))
  meta=self.path/'schema.json'
  description=dict(version=2,contact_detail_columns=['geom0','geom1','world_x','world_y','world_z','contact_force_n','contact_force_t1','contact_force_t2','contact_torque_n','contact_torque_t1','contact_torque_t2','relative_tangent_world_x','relative_tangent_world_y','relative_tangent_world_z','distance_m','active_force_and_distance_gate'],task_id=task.meta['task_id'],object_key=task.meta['object']['object_key'],stratum=task.meta['stratum'],hands=task.sides,quality=quality,policy_checkpoint=policy_checkpoint,source_family=source_family,columns=self.spec,protocol='World physical state stored once,3step command history reconstructed within episode only. FP32 lossless. Whole initial-condition family70/15/15 split before windows.',mask=task.mask.cpu().tolist()[0])
  if meta.exists():assert json.loads(meta.read_text())==description,'Never repurpose a data shard'
  else:meta.write_text(json.dumps(description,indent=2))
  self.entries=[]
  index=self.path/'episodes.jsonl'
  if index.exists():
   for line in index.read_text().splitlines():
    if line:self.entries.append(json.loads(line))
  self.counts={k:0 for k in self.spec}
  for entry in self.entries:
   for k in self.spec:self.counts[k]=entry['offsets'][k]+entry['lengths'][k]
  # Only incomplete append tails after the last committed whole-episode index
  # are recoverable temporaries. Never truncate committed data.
  for k,spec in self.spec.items():
   file=self.path/(k+'.bin');expected=self.counts[k]*spec['width']*np.dtype(spec['dtype']).itemsize
   if file.exists():
    assert file.stat().st_size>=expected
    if file.stat().st_size>expected:
     with file.open('r+b') as f:f.truncate(expected)
  self.index=index
 def append(self,arrays,seed,phase_start,termination,metadata=None):
  assert set(arrays)==set(self.spec),(set(arrays)^set(self.spec))
  length=len(arrays['request']);assert length>0
  mask=self.task.mask.detach().cpu().numpy().reshape(1,2,1)
  assert np.max(abs(np.asarray(arrays['request']).reshape(-1,2,26)*(1-mask)))<1e-7,'Inactive-hand request must be zero'
  assert all(len(arrays[k])==length+1 for k in ['state','generalized','contact','phase','contact_ranges'])
  assert all(len(arrays[k])==length for k in ['action','applied','reward_terms','terminated'])
  assert len(arrays['context'])==len(arrays['initial_history'])==1
  digest=hashlib.sha256();converted={}
  for k,spec in self.spec.items():
   arr=np.asarray(arrays[k],dtype=spec['dtype']).reshape(-1,spec['width'])
   assert np.isfinite(arr).all(),k
   converted[k]=np.ascontiguousarray(arr);digest.update(k.encode());digest.update(converted[k].tobytes())
  checksum=digest.hexdigest()
  if any(e['episode_sha256']==checksum for e in self.entries):raise RuntimeError('Duplicate episode; keep one and audit family assignment')
  family=f'{self.source_family}:{seed}';entry=dict(id=len(self.entries),task=self.task.meta['task_id'],quality=self.quality,seed=int(seed),family=family,split=split_family(family),phase_start=int(phase_start),controls=length,termination=termination,episode_sha256=checksum,offsets=self.counts.copy(),lengths={k:len(v) for k,v in converted.items()},metadata=metadata or {})
  for k,arr in converted.items():
   with (self.path/(k+'.bin')).open('ab') as f:
    f.write(arr.tobytes());f.flush();os.fsync(f.fileno())
  with self.index.open('a') as f:f.write(json.dumps(entry,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
  self.entries.append(entry)
  for k in self.counts:self.counts[k]+=entry['lengths'][k]
  return entry

class EpisodeShard:
 def __init__(self,path):
  self.path=Path(path);self.meta=json.loads((self.path/'schema.json').read_text());self.episodes=[]
  # Keep immutable full provenance on disk. Window readers do not duplicate
  # large per-episode context/warmstart metadata in every training process.
  with (self.path/'episodes.jsonl').open() as index:
   for line in index:
    if not line.strip():continue
    e=json.loads(line);small={key:e[key] for key in ['id','seed','family','split','phase_start','controls','episode_sha256','termination']}
    for key in ['offsets','lengths']:small[key]={column:e[key][column] for column in self.meta['columns']}
    self.episodes.append(small)
  self.arrays={}
  for k,spec in self.meta['columns'].items():
   file=self.path/(k+'.bin');self.arrays[k]=np.memmap(file,dtype=spec['dtype'],mode='r').reshape(-1,spec['width'])
 def window(self,episode_ids,times,horizon):
  ids=np.asarray(episode_ids);times=np.asarray(times);es=[self.episodes[int(i)] for i in ids]
  assert all(0<=t and t+horizon<=e['controls'] for e,t in zip(es,times))
  sstart=np.array([e['offsets']['state'] for e in es]);ustart=np.array([e['offsets']['request'] for e in es])
  phys=np.asarray(self.arrays['state'][sstart[:,None]+times[:,None]+np.arange(horizon+1)],dtype=np.float32)
  commands=np.asarray(self.arrays['request'][ustart[:,None]+times[:,None]+np.arange(horizon)],dtype=np.float32)
  htimes=times[:,None]+np.arange(-3,0);hist=np.asarray(self.arrays['request'][ustart[:,None]+htimes.clip(min=0)],dtype=np.float32).reshape(-1,3,2,26).transpose(0,2,1,3).copy()
  initial=np.asarray(self.arrays['initial_history'][[e['offsets']['initial_history'] for e in es]]).reshape(-1,2,3,26)
  for j in range(3):
   invalid=htimes[:,j]<0
   if invalid.any():hist[invalid,:,j]=initial[invalid,:,htimes[invalid,j]+3]
  context=np.asarray(self.arrays['context'][[e['offsets']['context'] for e in es]],dtype=np.float32)
  # Targets need physical coordinates only; history is irrelevant to physical
  # loss/endpoint metrics and is explicitly zero rather than using future truth.
  start=unpack_state(torch.from_numpy(phys[:,0].copy()),torch.from_numpy(hist.copy()))
  target=unpack_state(torch.from_numpy(phys[:,1:].copy()),torch.zeros(len(ids),horizon,2,3,26))
  return dict(start=start,target=target,commands=torch.from_numpy(commands.copy()),context=torch.from_numpy(context.copy()),mask=torch.tensor(self.meta['mask']).expand(len(ids),2).clone())
