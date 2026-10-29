"""Shared-policy serial-world runner. No batched contact worlds or specialist actors."""
import json,math,gc,copy
from types import SimpleNamespace
import numpy as np,torch,warp as wp
from a1_data_sources import ROOT
from a1_env import A1Env
from a1_reset import ResetSampler,uniform_reference_state
from train_a1_collector import Runner,evaluate
class MultiRunner:
 def __init__(self,manifest,policy,seed,cache_limit=2,reset_mode='adaptive_anchor'):
  assert reset_mode in {'adaptive_anchor','uniform_reference'}
  self.reset_mode=reset_mode
  self.rows=manifest['tasks'];self.policy=policy;self.rng=np.random.default_rng(seed);self.samplers={};self.cache={};self.cache_limit=cache_limit;self.cache_order=[]
  for i,row in enumerate(self.rows):
   path=ROOT/row['task_path']
   with np.load(path/'reference.npz') as z:length=len(z['qpos'])
   self.samplers[row['task_id']]=ResetSampler(SimpleNamespace(path=path,length=length),seed+i)
  self.real_controls=0;self.warmup_controls=0;self.evaluation_controls=0;self.bootstrap_controls=0;self.episodes=[];self.events=[];self.fallback_events=[];self.task_controls={r['task_id']:0 for r in self.rows}
  self.active=None;self.reset()
 def get(self,tid):
  if tid not in self.cache:
   if len(self.cache)>=self.cache_limit:
    torch.cuda.synchronize();victim=self.cache_order.pop(0);old=self.cache.pop(victim)
    if self.active is old:self.active=None
    old.env.contacts.e=None;del old;gc.collect();torch.cuda.empty_cache()
   row=next(r for r in self.rows if r['task_id']==tid)
   env=A1Env(ROOT/row['task_path']);self.warmup_controls+=env.physical_controls;r=Runner(env,self.policy,int(self.rng.integers(2**31)),reset_mode=self.reset_mode);r.sampler=self.samplers[tid];self.cache[tid]=r
  if tid in self.cache_order:self.cache_order.remove(tid)
  self.cache_order.append(tid);r=self.cache[tid]
  if getattr(self,'stream_task',None)!=tid:
   torch.cuda.synchronize();torch.cuda.set_stream(r.env.torch_stream);wp.set_stream(r.env.stream);self.stream_task=tid
  return r
 def choose(self):
  if self.reset_mode=='uniform_reference':
   # B1 PPO-Sim: task-uniform, then uniform over every nonterminal reference frame.
   # No anchor/replay admission, learned return weighting, or phase0 oversampling.
   tid,phase=uniform_reference_state([r['task_id'] for r in self.rows],{tid:s.task.length for tid,s in self.samplers.items()},self.rng)
   return tid,phase//25,phase,None,False
  entries=[];anchored=[];gs=[];memory=[]
  for row in self.rows:
   tid=row['task_id'];s=self.samplers[tid];usable=sorted(set(s.anchors)|set(s.fallback))
   # The original rule assumes at least one admissible state. Until one exists
   # for a source-qualified task, explicit cold-start reference attempts supply
   # real 25-step-surviving replay states. They are never called settled anchors.
   if not usable:
    entries.append((tid,None));anchored.append(False);gs.append(0);memory.append(0)
   for b in usable:
    entries.append((tid,b));anchored.append(b in s.anchors);gs.append(s.g[b]);memory.append(s.memory[b])
  a=np.asarray(anchored);p=np.full(len(entries),.5/len(entries));g=np.asarray(gs);m=np.asarray(memory)
  if a.any():
   w=np.exp(-25*g[a]/5.8-(-25*g[a]/5.8).max());w/=w.sum();mm=m[a];mm=mm/mm.sum() if mm.sum() else np.ones(a.sum())/a.sum();p[a]+=.4*w+.1*mm
  else:p[:]=1/len(p)
  cap=5/len(p);lo,hi=1.,max(2.,1/p.min())
  for _ in range(50):
   mid=(lo+hi)/2
   if np.minimum(cap,mid*p).sum()<1:lo=mid
   else:hi=mid
  p=np.minimum(cap,hi*p);p/=p.sum()
  tid,b=entries[int(self.rng.choice(len(entries),p=p))];s=self.samplers[tid]
  if b is None:
   b=int(self.rng.integers(s.nbins));return tid,b,min(25*b,s.task.length-2),None,True
  if b in s.anchors:
   i=s.anchors[b];snap={k:s.files[k][i] for k in ['qpos','qvel','ctrl','history','servo','qacc_warmstart']};phase=int(s.files['phase'][i])
  else:snap=s.fallback[b][int(self.rng.integers(len(s.fallback[b])))];phase=int(snap['phase'])
  return tid,b,phase,snap,False
 def reset(self):
  tid,b,phase,snap,cold=self.choose();r=self.get(tid);self.active=r;self.tid=tid;self.bin=b;self.bootstrap=cold
  e=r.env;e.reset(phase,seed=int(self.rng.integers(2**31)),randomize=True,snapshot=snap)
  r.state_history.copy_(e.state()[:,None].expand(-1,3,-1));r.action_history.zero_();self.rewards=[];self.narrow=[];self.pending=[(0,e.snapshot())] if self.reset_mode=='adaptive_anchor' else []
 @torch.no_grad()
 def rollout(self,n,should_stop=lambda:False):
  cols={k:[] for k in ['obs','goal','value_input','action','old_logprob','old_value','reward','done','mask']}
  for _ in range(n):
   if should_stop():break
   r=self.active;e=r.env;r.graph.replay()
   for k,v in [('obs',r.obs),('goal',r.goal),('value_input',r.value_input),('action',r.action),('old_logprob',r.logprob),('old_value',r.value),('mask',e.task.mask)]:cols[k].append(v.clone())
   s,reward,done,info=e.step(r.action);cols['reward'].append(reward.clone());cols['done'].append(done.clone())
   self.real_controls+=1;self.task_controls[self.tid]+=1;self.bootstrap_controls+=int(self.bootstrap)
   r.state_history.copy_(torch.cat([r.state_history[:,1:],s[:,None]],1));r.action_history.copy_(torch.cat([r.action_history[:,1:],e.last_action[:,None]],1))
   self.rewards.append(float(reward));self.narrow.append(float(info['narrow_reward']))
   if self.reset_mode=='adaptive_anchor' and len(self.rewards)%25==0:self.pending.append((len(self.rewards),e.snapshot()))
   if bool(done):
    for t,snap in self.pending:
     b=int(snap['phase'])//25
     if t+25<=len(self.rewards) and b not in r.sampler.anchors:
      pool=r.sampler.fallback.setdefault(b,[])
      if len(pool)<8:pool.append(snap)
      else:pool[int(self.rng.integers(8))]=snap
      self.fallback_events.append((self.tid,b,snap))
    if self.reset_mode=='adaptive_anchor':
     self.events.append((self.tid,self.bin,self.narrow[:100]));r.sampler.update(self.bin,self.narrow)
    self.episodes.append(dict(task=self.tid,reset_mode=self.reset_mode,controls=len(self.rewards),start_phase=e.start_phase,bootstrap=self.bootstrap,return_sum=sum(self.rewards),failure_reason=int(info['reason']),regular_success=bool(info['regular_success']),strict_success=bool(info['strict_success']),mean_errors=info['mean_errors'].cpu().tolist()[0]))
    self.reset()
  if not cols['reward']:return None
  self.active.graph.replay();nextv=self.active.value.clone();carry=torch.zeros_like(nextv);d={k:torch.cat(v) for k,v in cols.items()};adv=torch.zeros_like(d['reward'])
  for t in reversed(range(len(adv))):
   alive=1-d['done'][t].float();delta=d['reward'][t]+.99*nextv*alive-d['old_value'][t];carry=delta+.99*.95*alive*carry;adv[t]=carry;nextv=d['old_value'][t]
  d['advantage']=adv;d['return']=adv+d['old_value'];return d
 def stats(self):return {tid:copy.deepcopy(s.state_dict()) for tid,s in self.samplers.items()}
 def load_stats(self,stats):
  for tid,state in stats.items():
   # Shared allocation statistics, distinct per-worker sampling RNG streams.
   local=self.samplers[tid].rng.bit_generator.state
   self.samplers[tid].load_state_dict(copy.deepcopy(state));self.samplers[tid].rng.bit_generator.state=local
 def merge(self,stats,event_sets,fallback_sets):
  self.load_stats(stats)
  for events in event_sets:
   for tid,b,rewards in events:self.samplers[tid].update(b,rewards)
  for events in fallback_sets:
   for tid,b,snap in events:
    pool=self.samplers[tid].fallback.setdefault(b,[])
    if len(pool)<8:pool.append(snap)
    else:pool[int(self.rng.integers(8))]=snap
 @torch.no_grad()
 def evaluation(self,rank,world_size,seeds):
  rows=[]
  for i,row in enumerate(self.rows):
   if i%world_size!=rank:continue
   r=self.get(row['task_id'])
   rows += [dict(task=row['task_id'],**x) for x in evaluate(r.env,self.policy,seeds)]
  self.evaluation_controls+=sum(row['controls'] for row in rows)
  self.reset();return rows
