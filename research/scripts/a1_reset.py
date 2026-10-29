"""Capped original bin-allocation mixture and immutable anchor lookup."""
import json
import numpy as np
class ResetSampler:
 def __init__(self,task,seed=280043):
  self.task=task;self.rng=np.random.default_rng(seed);self.nbins=(task.length-1+24)//25
  self.g=np.zeros(self.nbins);self.visits=np.zeros(self.nbins,dtype=int);self.memory=np.zeros(self.nbins)
  self.anchors={};self.fallback={};self.files=None
  meta=task.path/'anchors_v1.json';data=task.path/'anchors_v1.npz'
  if meta.exists() and data.exists():
   self.files=np.load(data)
   for i,b in enumerate(self.files['bin']):self.anchors[int(b)]=i
  active=list(self.anchors)
  if active:self.memory[active]=1/len(active)
 def probabilities(self):
  usable=sorted(set(self.anchors)|set(self.fallback));anchored=sorted(self.anchors)
  if not usable:return None
  p=np.zeros(self.nbins);p[usable]=.5/len(usable)
  if anchored:
   logits=-25*self.g[anchored]/5.8;weights=np.exp(logits-logits.max());weights/=weights.sum()
   memory=self.memory[anchored];memory=memory/memory.sum() if memory.sum() else np.ones(len(anchored))/len(anchored)
   p[anchored]+=.4*weights+.1*memory
  else:p[usable]=1/len(usable)
  cap=5/len(usable);lo,hi=1.,max(2.,1/max(p[p>0].min(),1e-30))
  for _ in range(60):
   mid=(lo+hi)/2
   if np.minimum(cap,mid*p).sum()<1:lo=mid
   else:hi=mid
  p=np.minimum(cap,hi*p);p/=p.sum();return p
 def choose(self):
  p=self.probabilities()
  if p is None:return None
  b=int(self.rng.choice(self.nbins,p=p))
  if b in self.anchors:
   i=self.anchors[b];snap={k:self.files[k][i] for k in ['qpos','qvel','ctrl','history','servo','qacc_warmstart']}
   phase=int(self.files['phase'][i])
  else:
   choices=self.fallback[b];snap=choices[int(self.rng.integers(len(choices)))];phase=snap['phase']
  return b,phase,snap
 def update(self,b,narrow_rewards):
  if b not in self.anchors:return
  reward=np.asarray(narrow_rewards[:100],float)/.02
  if not len(reward):return
  gamma=.99**np.arange(len(reward));value=float(np.sum(gamma*reward)/gamma.sum())
  self.g[b]=value if not self.visits[b] else .9*self.g[b]+.1*value;self.visits[b]+=1
  active=list(self.anchors);logits=-25*self.g[active]/5.8;q=np.exp(logits-logits.max());q/=q.sum()
  self.memory*=.999;self.memory[active]+=.001*q
 def state_dict(self):return dict(g=self.g,visits=self.visits,memory=self.memory,rng=self.rng.bit_generator.state,fallback=self.fallback)
 def load_state_dict(self,state):
  self.g=state['g'];self.visits=state['visits'];self.memory=state['memory'];self.rng.bit_generator.state=state['rng'];self.fallback=state.get('fallback',{})

def uniform_reference_state(task_ids,lengths,rng):
 """B1 reset: equal task probability, then every nonterminal frame equally."""
 tid=task_ids[int(rng.integers(len(task_ids)))]
 assert lengths[tid]>=2
 return tid,int(rng.integers(lengths[tid]-1))
