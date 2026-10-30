"""Nonzero, distinct state/history sentinels for first/last-world reset tests."""
import argparse,numpy as np,torch,warp as wp
from real_env import RealEnv
from grasp_lab import RUN,write_json

p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,choices=[2,8],required=True);a=p.parse_args();env=RealEnv(a.worlds);assert torch.cuda.mem_get_info()[0]>64*2**20
arrays=dict(qpos=env.qpos,qvel=env.qvel,ctrl=env.ctrl,time=wp.to_torch(env.wd.time),warmstart=wp.to_torch(env.wd.qacc_warmstart),phase=env.phase,steps=env.steps,last_action=env.last_action,returns=env.returns,possum=env.possum,rotsum=env.rotsum,all_good=env.all_good,episode_start=env.episode_start)
for name in ['qacc','qfrc_applied','xfrc_applied']:arrays[name]=wp.to_torch(getattr(env.wd,name))
results=[]
for selected in [0,a.worlds-1]:
    env.reset(torch.arange(a.worlds,device=env.device),False)
    for name,x in arrays.items():
        for world in range(a.worlds):
            if name=='qpos':x[world,0]+=.001*(world+1)
            elif name=='ctrl':x[world,0]+=.002*(world+1)
            elif x.dtype==torch.bool:x[world]=False
            elif x.dtype in [torch.int64,torch.int32]:x[world]=world+3
            else:x[world]+=(world+1)*.01
    before={name:x.clone() for name,x in arrays.items()};others=torch.arange(a.worlds,device=env.device)!=selected
    env.reset(torch.tensor([selected],device=env.device),False)
    diffs={name:float((x[others].double()-before[name][others].double()).abs().max()) for name,x in arrays.items()}
    target_ok=torch.equal(env.qpos[selected],env.ref[0]) and torch.equal(env.qvel[selected],env.refvel[0]) and torch.equal(env.ctrl[selected],env.target[0])
    target_ok=target_ok and all(bool((x[selected]==(1 if name=='all_good' else 0)).all()) for name,x in arrays.items() if name not in ['qpos','qvel','ctrl'])
    results.append(dict(selected_world=selected,other_world_max_changes=diffs,reset_target_correct=bool(target_ok),passed=all(v==0 for v in diffs.values()) and bool(target_ok)))
report=dict(nworld=a.worlds,tests=results,ctrl_shape=list(env.ctrl.shape),ctrl_stride=list(env.ctrl.stride()),qpos_stride=list(env.qpos.stride()),zero_copy_pointers=env.qpos.data_ptr()==env.wd.qpos.ptr and env.ctrl.data_ptr()==env.wd.ctrl.ptr,torch_stream_handle=env.torch_stream.cuda_stream,warp_stream_handle=env.stream.cuda_stream)
write_json(RUN/f'reset_markers_{a.worlds}.json',report);print(report)
