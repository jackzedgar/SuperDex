"""Frozen endpoint migration evaluation only; no fitting or selection."""
import json,os,time,hashlib,importlib.metadata,argparse
from pathlib import Path
import numpy as np
import torch,warp as wp,mujoco,mujoco_warp as mjw
from real_env import RealEnv,ROOT,GPU
from exploration_common import Contacts,write_json,resource_check
from train_real_ppo import Policy
from train_core_policy import evaluate_policy
from world_model_mini import WorldModel,Geometry

def main():
 start=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
 parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/lambda_migration_20261027/frozen_evaluation');args=parser.parse_args()
 out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
 original=ROOT/'runs/core_iteration_20261027';completion=json.loads((original/'core_completion.json').read_text());jobs=[]
 for subset,repeats in [('development',1),('heldout',3)]:
  for repeat in range(repeats):
   for arm in (['P','M'] if repeat%2==0 else ['M','P']):
    jobs.append(dict(arm=arm,subset=subset,repeat=repeat,checkpoint=completion['final_checkpoints'][arm]))
 write_json(out/'protocol.json',dict(jobs=jobs,strict_success_unchanged=True,original_evaluate_policy=True,hardware_only=True,future_real_states_used=False,original_initials={s:hashlib.sha256((original/(s+'_initials.npz')).read_bytes()).hexdigest() for s in ['development','heldout']}))
 (out/'resources_start.txt').write_text(resource_check(2400))
 env=RealEnv(1);contacts=Contacts(env)
 assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid)==GPU[4:]
 policy=Policy(118).to('cuda');optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4,eps=1e-5);loads=[]
 for arm in ['P','M']:
  path=ROOT/completion['final_checkpoints'][arm];c=torch.load(path,map_location='cpu',weights_only=True);policy.load_state_dict(c['policy']);optimizer.load_state_dict(c['optimizer'])
  assert all(torch.isfinite(v).all() for v in policy.state_dict().values())
  assert all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
  loads.append(dict(arm=arm,checkpoint=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),optimizer_states=len(optimizer.state),optimizer_step=float(next(iter(optimizer.state.values()))['step'])))
 wmroot=ROOT/'runs/learning_trend_20261027/wm'
 for variant in ['W0','W1']:
  c=torch.load(wmroot/(variant+'_final.pt'),map_location='cpu',weights_only=False);model=WorldModel(variant,Geometry(wmroot),c['normalization']).to('cuda');model.load_state_dict(c['model']);opt=torch.optim.Adam(model.parameters());opt.load_state_dict(c['optimizer'])
  env.reset(torch.tensor([0],device='cuda'),False);history=env.ctrl[:,None].expand(1,3,26).clone();s=torch.cat([env.qpos,env.qvel,history.flatten(1)],-1)
  with torch.no_grad():
   for i in range(32):s=model(s,env.target[min(i+1,env.length-1)][None]);assert torch.isfinite(s).all()
  assert all(torch.isfinite(v).all() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
  loads.append(dict(model=variant,update=c['update'],optimizer_states=len(opt.state),finite_prediction_steps=32))
 write_json(out/'dependency_checkpoint_checks.json',dict(packages={n:importlib.metadata.version(n) for n in ['torch','mujoco','mujoco-warp','warp-lang','numpy','scipy']},gpu=GPU,gpu_name=torch.cuda.get_device_name(),torch_cuda=torch.version.cuda,warp_cuda=wp.get_cuda_toolkit_version(),warp_driver=wp.get_cuda_driver_version(),compile_seconds=env.compile_seconds,checkpoints=loads))
 results=[]
 for job in jobs:
  c=torch.load(ROOT/job['checkpoint'],map_location='cpu',weights_only=True);policy.load_state_dict(c['policy']);policy.eval()
  initials=dict(np.load(original/(job['subset']+'_initials.npz')))
  target=out/f"{job['subset']}_{job['arm']}_repeat_{job['repeat']}";r=evaluate_policy(env,policy,target,initials,contacts)
  results.append(dict(**job,folder=str(target.relative_to(ROOT)),**r));write_json(out/'results.json',results)
  print(json.dumps(dict(arm=job['arm'],subset=job['subset'],repeat=job['repeat'],successes=r['successes'],n=r['n'],elapsed=time.monotonic()-start)),flush=True)
 wp.synchronize();torch.cuda.synchronize()
 write_json(out/'summary.json',dict(complete=len(results)==len(jobs),planned_jobs=len(jobs),completed_jobs=len(results),physical_controls=sum(r['physical_controls'] for r in results),process_seconds=time.monotonic()-start,worlds=1,checkpoint_training=False,peak_torch_memory_mb=torch.cuda.max_memory_allocated()/2**20))
if __name__=='__main__':main()
