"""Frozen PPO-Sim noisy physical collection; formal writes require a quality passport."""
import argparse,json,math,time,hashlib,signal
from pathlib import Path
import numpy as np,torch,warp as wp
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
from a1_policy import CollectorPolicy
from train_a1_collector import Runner
from a1_episode_data import EpisodeWriter,pack_state
from a1_episode_contract import is_complete_episode,complete_controls
stop=False
def onstop(*_):
 global stop;stop=True
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--quality',required=True);p.add_argument('--controls',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=1800043000);p.add_argument('--smoke',action='store_true');p.add_argument('--passport',type=Path);a=p.parse_args()
 signal.signal(signal.SIGTERM,onstop);signal.signal(signal.SIGINT,onstop);torch.set_num_threads(2);torch.manual_seed(a.seed)
 assert a.checkpoint.resolve().is_relative_to(ROOT)
 checkpoint_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
 if not a.smoke:
  assert a.passport is not None,'Formal collection needs frozenProbe40 and evaluated quality checkpoint'
  passport=json.loads(a.passport.read_text());assert passport['checkpoint_sha256']==checkpoint_sha and passport['quality']==a.quality and passport['status']=='DEVELOPMENT_BAND_REACHED'
  assert passport['probe40_frozen'] and passport.get('formal') is True and a.task in passport['task_ids']
  assert passport.get('reset_mode')=='uniform_reference','Original A1 PPO-Sim requires uniform reference initialization, not adaptive SuperDex resets'
 else:assert a.output.resolve().is_relative_to(RUN),'Smoke episodes cannot be mixed into formal dataset'
 env=A1Env(ROOT/'data/a1_processed/tasks'/a.task);policy=CollectorPolicy().to(env.device)
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);policy.load_state_dict(checkpoint['policy'])
 for param in policy.parameters():param.requires_grad_(False)
 policy.logstd.fill_(math.log(.1));policy.eval()
 reset_mode=checkpoint.get('configuration',{}).get('reset_mode','adaptive_anchor')
 if not a.smoke:assert reset_mode=='uniform_reference'
 runner=Runner(env,policy,a.seed,reset_mode=reset_mode)
 if 'sampler' in checkpoint:runner.sampler.load_state_dict(checkpoint['sampler'])
 if 'samplers' in checkpoint:runner.sampler.load_state_dict(checkpoint['samplers'][a.task])
 if 'global_sampler' in checkpoint:runner.sampler.load_state_dict(checkpoint['global_sampler'][a.task])
 # Collection RNG is independent of policy training; identical task seed across
 # quality checkpoints keeps repeated reset families in the same split.
 runner.sampler.rng=np.random.default_rng(a.seed+1);torch.manual_seed(a.seed)
 row=json.loads((env.task.path/'source_selection.json').read_text())
 if row['source']=='hot3d':
  definitions=json.loads((ROOT/'data/raw/hot3d/clip_definitions.json').read_text());family=definitions[str(int(row['clip'].split('-')[1]))]['sequence_id']
 else:family=row['sequence']
 writer=EpisodeWriter(a.output,env.task,a.quality,dict(path=str(a.checkpoint.resolve().relative_to(ROOT)),sha256=checkpoint_sha),family)
 total=complete_controls(writer.entries);initial_total=total;initial_stored=sum(x['controls'] for x in writer.entries);start=time.monotonic();episode=len(writer.entries)
 # Resume uses a fresh disjoint RNG suffix; no duplicate noise replay.
 runner.rng=np.random.default_rng(a.seed+episode*1000003);runner.sampler.rng=np.random.default_rng(a.seed+1+episode*1000003);torch.manual_seed(a.seed+episode*1000003)
 while total<a.controls and not stop:
  runner.reset();seed=int(env.context_record['seed']);start_phase=int(env.phase);history=env.history.cpu().numpy().reshape(1,156)
  arrays={key:[] for key in ['state','request','action','applied','generalized','contact','phase','reward_terms','terminated','contact_details','contact_ranges']}
  arrays['context']=[env.context.cpu().numpy()[0]];arrays['initial_history']=[history[0]]
  initial_warm=wp.to_torch(env.wd.qacc_warmstart).cpu().numpy()[0]
  def details(summary,values):
   n=int(summary[8]);start=len(arrays['contact_details']);arrays['contact_details'].extend(values[:n]);arrays['contact_ranges'].append([start,n])
  # Initial state is independently forwarded at reset; subsequent states use
  # the captured endpoint buffer, avoiding a second kinematics computation.
  initial_contact=env.contacts.step().cpu().numpy()
  arrays['state'].append(pack_state(env.current_state).cpu().numpy()[0].copy());arrays['generalized'].append(torch.cat([env.qpos,env.qvel],-1).cpu().numpy()[0].copy());arrays['contact'].append(initial_contact);arrays['phase'].append(start_phase)
  details(initial_contact,env.contacts.details.cpu().numpy())
  while True:
   runner.graph.replay();s,r,done,info=env.step(runner.action)
   # One state/action/status transfer instead of many synchronizing reads.
   fields=[pack_state(s),torch.cat([env.qpos,env.qvel],-1),env.requested,env.actions,env.applied,info['terms'],env.contact_summary[None],env.phase.float()[:,None],done.float()[:,None],info['finite'].float()[:,None]]
   lengths=[x.shape[-1] for x in fields];host=torch.cat(fields,-1).cpu().numpy()[0];parts=np.split(host,np.cumsum(lengths)[:-1])
   for key,value in zip(['state','generalized','request','action','applied','reward_terms','contact'],parts[:7]):arrays[key].append(value.copy())
   arrays['phase'].append(int(parts[7][0]));arrays['terminated'].append(int(parts[8][0]))
   details(parts[6],env.contact_details.cpu().numpy())
   runner.state_history.copy_(torch.cat([runner.state_history[:,1:],s[:,None]],1));runner.action_history.copy_(torch.cat([runner.action_history[:,1:],env.last_action[:,None]],1))
   assert bool(parts[9][0]) and not bool(parts[6][9]),'Nonfinite state or contact buffer overflow; affected collection stops'
   if bool(parts[8][0]) or stop:break
  if bool(info['failure']):reason='physics_failure_'+str(int(info['reason']))
  elif bool(env.timeout):reason='reference_end'
  else:reason='budget_truncation'
  entry=writer.append({k:np.asarray(v) for k,v in arrays.items()},seed,start_phase,reason,metadata=dict(formal=not a.smoke,reset_mode=reset_mode,initial_solver_warmstart=initial_warm.tolist(),context=env.context_record,policy_noise_sigma=.1,full_reference_start=start_phase==0,regular_success=bool(info['regular_success']),strict_success=bool(info['strict_success']),bootstrap=runner.bootstrap))
  if is_complete_episode(entry):total+=entry['controls']
  episode+=1
  if episode%10==0:
   write(a.output/'progress.json',dict(controls=total,episodes=episode,seconds=time.monotonic()-start,formal=not a.smoke));print('COLLECT',total,episode,time.monotonic()-start,flush=True)
 summary=dict(status='budget_stop' if stop else 'quota_complete',real_dataset_controls=total,episodes=sum(is_complete_episode(e) for e in writer.entries),stored_episodes=episode,stored_controls=sum(e['controls'] for e in writer.entries),excluded_incomplete_controls=sum(e['controls'] for e in writer.entries if not is_complete_episode(e)),seconds=time.monotonic()-start,new_controls_this_process=total-initial_total,new_stored_controls_this_process=sum(e['controls'] for e in writer.entries)-initial_stored,controls_per_second=(total-initial_total)/(time.monotonic()-start),all_physical_controls_this_process=env.physical_controls,formal=not a.smoke,quality=a.quality,policy_checkpoint_sha256=checkpoint_sha,split_counts={split:sum(e['controls'] for e in writer.entries if e['split']==split and is_complete_episode(e)) for split in ['train','development','test']})
 write(a.output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
