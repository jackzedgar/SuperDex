"""Bounded real single-world A1 path validation; preserves all failures and logs."""
import argparse,json,time
import numpy as np,torch,warp as wp
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
from a1_state import delta_between,compose
from a1_world_model import control_context
from grasp_batch_audit import gpu_contacts

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--steps',type=int,default=150);p.add_argument('--tag',required=True);p.add_argument('--no-capture',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);out=RUN/a.tag;out.mkdir(exist_ok=False);env=A1Env(ROOT/'data/a1_processed/tasks'/a.task,capture=not a.no_capture)
 rows=[];snapshots=[];results=[];physical=0;start=time.monotonic()
 for repeat,(seed,dr) in enumerate([(280043,False),(280043,False),(280044,True)]):
  env.reset(0,seed=seed,randomize=dr);episode=[];before=env.state().clone()
  for step in range(min(a.steps,env.task.length-1)):
   action=torch.zeros(1,52,device=env.device)
   if repeat==2:action[:,6:26]=.05*torch.sin(torch.tensor(step*.1,device=env.device))
   s,r,done,info=env.step(action);physical+=1
   assert bool(info['finite'].all()),(repeat,step)
   known=compose(before,delta_between(before,s,env.task.mask),env.requested,env.task.mask,*control_context(env.context))
   causal_error=float((known-s).abs().max())
   if causal_error>=3e-5:
    diff=(known-s).abs()[0];indices=torch.topk(diff,12).indices
    evidence=dict(repeat=repeat,step=step,indices=indices.tolist(),known=known[0,indices].tolist(),actual=s[0,indices].tolist(),before=before[0,indices].tolist(),requested=env.requested.tolist(),context=env.context.tolist(),phase=env.phase.tolist(),time=wp.to_torch(env.wd.time).tolist())
    write(out/'causal_failure.json',evidence);print(json.dumps(evidence),flush=True)
    np.savez_compressed(out/'causal_failure.npz',before=before.cpu().numpy(),after=s.cpu().numpy(),known=known.cpu().numpy(),requested=env.requested.cpu().numpy(),context=env.context.cpu().numpy())
    raise AssertionError(causal_error)
   row=dict(repeat=repeat,step=step+1,physics_time=float(wp.to_torch(env.wd.time)[0]),reward=float(r),failure=bool(info['failure']),reason=int(info['reason']),pos_m=float(info['pos_error']),rot_rad=float(info['rot_error']),causal_state_error=causal_error,terms=info['terms'].cpu().tolist())
   if step in [0,5,20,50] or bool(done):
    contacts,n=gpu_contacts(env);row['contacts']=contacts;row['gpu_total_contacts']=n
   rows.append(row);episode.append(s.cpu().numpy()[0]);snapshots.append(np.r_[env.qpos.cpu().numpy()[0],env.qvel.cpu().numpy()[0],env.requested.cpu().numpy()[0],env.ctrl.cpu().numpy()[0]])
   before=s.clone()
   if bool(done):break
  wp.synchronize();results.append(dict(seed=seed,randomization=dr,controls=len(episode),states=np.array(episode),context=env.context_record,final_info={k:v.cpu().tolist() for k,v in info.items()}))
 # Identical nominal replays: report amplification, not bitwise physics requirement.
 n=min(len(results[0]['states']),len(results[1]['states']))
 delta=np.linalg.norm(results[0]['states'][:n,274:277]-results[1]['states'][:n,274:277],axis=-1)
 for row in results:row.pop('states')
 report=dict(scope='New full-A1 runtime only; not qualification of policy success or G1',task=a.task,hands=env.task.sides,compile_seconds=env.compile_seconds,measured_seconds=time.monotonic()-start,actual_controls=physical,repeat_common_controls=n,repeat_max_object_position_difference_m=float(delta.max()),runs=results,peak_torch_bytes=torch.cuda.max_memory_allocated(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),graph_captures_control_physics_reward=not a.no_capture,contact_state_alignment='Endpoint forward at exact control-step time')
 write(out/'summary.json',report);write(out/'step_log.json',rows);np.savez_compressed(out/'state_action_log.npz',values=np.array(snapshots))
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
