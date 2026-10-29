"""Whole-episode window sampling for frozen full A1 datasets.
Training samples objects uniformly, then windows uniformly within that object.
"""
import json,hashlib,fcntl,resource
from pathlib import Path
import numpy as np,torch
from a1_episode_data import EpisodeShard
from a1_episode_contract import is_complete_episode
from a1_geometry import A1Geometry
from a1_data_sources import ROOT
class WindowPool:
 def __init__(self,manifest,split,horizon=8):
  self.path=Path(manifest);self.meta=json.loads(self.path.read_text());self.split=split;self.horizon=horizon
  soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);needed=64+40*len(self.meta['shards'])
  if soft<needed:
   assert hard>=needed,'Per-process file descriptor hard limit insufficient for frozen shards'
   resource.setrlimit(resource.RLIMIT_NOFILE,(needed,hard))
  self.shards=[EpisodeShard(ROOT/r['path']) for r in self.meta['shards']];self.object_keys=self.meta['object_keys'];self.object_id={k:i for i,k in enumerate(self.object_keys)}
  self.groups={};self.task_groups={}
  for sid,shard in enumerate(self.shards):
   for eid,e in enumerate(shard.episodes):
    if not is_complete_episode(e):continue
    n=e['controls']-horizon+1
    if e['split']!=split or n<=0:continue
    key=shard.meta['object_key'];row=(sid,eid,n)
    self.groups.setdefault(key,[]).append(row);self.task_groups.setdefault(shard.meta['task_id'],[]).append(row)
  self.cdf={key:np.cumsum([r[2] for r in rows]) for key,rows in self.groups.items()}
  self.keys=sorted(self.groups);assert self.keys
 def sample_ids(self,n,rng,seen=None):
  available=sum(int(cdf[-1]) for cdf in self.cdf.values())
  if seen is not None and n>available-len(seen):raise ValueError(f'Requested {n} distinct windows, only {available-len(seen)} remain')
  result=[]
  while len(result)<n:
   key=self.keys[int(rng.integers(len(self.keys)))];cdf=self.cdf[key];i=int(rng.integers(cdf[-1]));j=int(np.searchsorted(cdf,i,side='right'));sid,eid,_=self.groups[key][j];t=i-(int(cdf[j-1]) if j else 0);identity=(sid,eid,t)
   if seen is not None:
    if identity in seen:continue
    seen.add(identity)
   result.append(identity)
  return np.asarray(result,dtype=np.int64)
 def batch(self,ids):
  ids=np.asarray(ids);result={};order=[]
  for sid in np.unique(ids[:,0]):
   positions=np.flatnonzero(ids[:,0]==sid);sub=ids[positions];shard=self.shards[int(sid)]
   data=shard.window(sub[:,1],sub[:,2],self.horizon);data['object_id']=torch.full((len(sub),),self.object_id[shard.meta['object_key']],dtype=torch.long)
   for key,value in data.items():result.setdefault(key,[]).append(value)
   order.extend(positions.tolist())
  inverse=np.argsort(order)
  return {key:torch.cat(value)[inverse] for key,value in result.items()}
 def contacts(self,ids,horizon=None):
  horizon=horizon or self.horizon;out=[]
  for sid,eid,t in np.asarray(ids):
   shard=self.shards[int(sid)];e=shard.episodes[int(eid)];offset=e['offsets']['contact']+int(t)
   out.append(np.array(shard.arrays['contact'][offset:offset+horizon+1]))
  return np.stack(out)
 def geometry(self,device='cuda'):
  kin=[dict(np.load(ROOT/f'data/a1_processed/assets/hands/{side}/kinematics.npz')) for side in ['right','left']]
  grids=[dict(np.load(ROOT/f'data/a1_geometry/objects/{key}/sdf.npz')) for key in self.object_keys]
  return A1Geometry(kin,grids).to(device)
 def fixed_windows(self,path,seed=280050,per_task=64,natural=4096):
  with Path(str(path)+'.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   return self._fixed_windows(path,seed,per_task,natural)
 def _fixed_windows(self,path,seed=280050,per_task=64,natural=4096):
  path=Path(path)
  manifest_sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
  if path.exists():
   d=np.load(path)
   assert 'manifest_sha256' in d.files and str(d['manifest_sha256'])==manifest_sha,'Fixed windows belong to a different or unverified manifest'
   assert str(d['split'])==self.split and int(d['horizon'])==self.horizon,'Fixed window split/horizon mismatch'
   return {k:d[k] for k in d.files}
  rng=np.random.default_rng(seed);balanced=[]
  for task,rows in sorted(self.task_groups.items()):
   cdf=np.cumsum([r[2] for r in rows]);n=min(per_task,int(cdf[-1]));picked=rng.choice(cdf[-1],size=n,replace=False)
   for i in picked:
    j=int(np.searchsorted(cdf,i,side='right'));sid,eid,_=rows[j];balanced.append((sid,eid,int(i)-(int(cdf[j-1]) if j else 0)))
  allrows=[r for rows in self.groups.values() for r in rows];cdf=np.cumsum([r[2] for r in allrows]);nat=[]
  for i in rng.choice(cdf[-1],size=min(natural,int(cdf[-1])),replace=False):
   j=int(np.searchsorted(cdf,i,side='right'));sid,eid,_=allrows[j];nat.append((sid,eid,int(i)-(int(cdf[j-1]) if j else 0)))
  values=dict(balanced=np.asarray(balanced,dtype=np.int64),natural=np.asarray(nat,dtype=np.int64),seed=np.array(seed),horizon=np.array(self.horizon),split=np.array(self.split),manifest_sha256=np.array(manifest_sha))
  np.savez_compressed(path,**values);return values
def save_model(model,path,**extra):
 # Geometry is immutable/shared and can be hundreds of MB. Never duplicate it
 # into every checkpoint. Record and validate the dataset geometry manifest.
 state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('geometry.')}
 tmp=Path(path).with_suffix('.tmp');torch.save(dict(model=state,model_configuration=model.config(),**extra),tmp);tmp.replace(path)
def load_model_state(model,state):
 missing,unexpected=model.load_state_dict(state,strict=False)
 assert not unexpected and all(k.startswith('geometry.') for k in missing),(missing,unexpected)
