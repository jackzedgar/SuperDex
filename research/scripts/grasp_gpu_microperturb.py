"""One GPU world repeated with controlled representable microperturbations."""
import numpy as np,torch,warp as wp
from real_env import RealEnv
from grasp_lab import RUN,write_json

env=RealEnv(1);assert torch.cuda.mem_get_info()[0]>64*2**20,'Insufficient resource margin'
cases=[('repeat0',None,0),('repeat1',None,0)]+[(f'obj_{j}_{s:+d}',26+j,s*1e-7) for j in range(3) for s in [-1,1]]
traces={};actualeps={}
for name,j,eps in cases:
    env.reset(torch.tensor([0],device=env.device),False)
    if j is not None:
        before=env.qpos[0,j].item();env.qpos[0,j]+=eps;actualeps[name]=env.qpos[0,j].item()-before
    else:actualeps[name]=0
    qs=[]
    for k in range(1,env.length):
        env.ctrl[:]=env.target[k];wp.capture_launch(env.graph);qs.append(env.qpos[0].cpu().numpy().copy())
    traces[name]=np.asarray(qs)
base=traces['repeat0'];rows=[]
for name,j,eps in cases:
    pos=np.linalg.norm(traces[name][:,26:29]-base[:,26:29],axis=-1);first={}
    for limit in [1e-6,1e-4,.001,.005]:
        ids=np.flatnonzero(pos>limit);first[str(limit)]=float((ids[0]+1)*.02) if len(ids) else None
    rows.append(dict(case=name,actual_initial_epsilon_m=actualeps[name],max_position_difference_m=float(pos.max()),final_position_difference_m=float(pos[-1]),first_difference_times_s=first))
np.savez_compressed(RUN/'gpu_microperturb.npz',**traces)
write_json(RUN/'gpu_microperturb.json',dict(cases=rows,scope='ONE world; repeat plus 0.1 micrometre perturbations. Isolates sensitivity within GPU contact/solver arithmetic from cross-world indexing; does not prove physical correctness.'));print(rows)
