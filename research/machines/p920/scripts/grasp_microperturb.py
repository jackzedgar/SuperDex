"""Controlled CPU perturbations separate contact sensitivity from world indexing."""
import numpy as np,mujoco,json
from grasp_lab import DATA,RUN,model_for,write_json,cpu_contacts

m=model_for('baseline');ref=np.load(DATA/'trajectory.npz');ctrl=np.load(DATA/'controller_reference.npz')['ctrl'].astype(np.float32)
cases=[('identical_0',None,0),('identical_1',None,0)]+[(f'obj_{axis}_{sign:+d}_1e-7',26+axis,sign*1e-7) for axis in range(3) for sign in [-1,1]]
traces={};contacts={}
for name,j,epsilon in cases:
    d=mujoco.MjData(m);d.qpos[:]=ref['qpos'][0].astype(np.float32);d.qvel[:]=ref['qvel'][0].astype(np.float32)
    if j is not None:d.qpos[j]+=epsilon
    states=[];cs=[]
    for step in range(510):
        d.ctrl[:]=ctrl[step//10+1];mujoco.mj_step(m,d);states.append(d.qpos.copy())
        cs.append(cpu_contacts(m,d))
    traces[name]=np.asarray(states);contacts[name]=cs
base=traces['identical_0'];summary=[];evidence=[]
for name,j,epsilon in cases:
    pos=np.linalg.norm(traces[name][:,26:29]-base[:,26:29],axis=-1);first={}
    for threshold in [1e-6,1e-4,.001,.005]:
        ids=np.flatnonzero(pos>threshold);i=int(ids[0]) if len(ids) else None;first[str(threshold)]=None if i is None else (i+1)*.002
        if i is not None and threshold==.001:evidence.append(dict(case=name,time_s=(i+1)*.002,baseline_state=base[i].tolist(),perturbed_state=traces[name][i].tolist(),ctrl=ctrl[i//10+1].tolist(),baseline_contacts=contacts['identical_0'][i],perturbed_contacts=contacts[name][i]))
    summary.append(dict(case=name,initial_epsilon_m=epsilon,position_max_m=float(pos.max()),position_final_m=float(pos[-1]),first_difference_times=first))
np.savez_compressed(RUN/'cpu_microperturb.npz',**traces)
write_json(RUN/'cpu_microperturb.json',dict(cases=summary,evidence_at_first_1mm=evidence,interpretation='One CPU world at a time; identical-copy control plus deliberately specified 0.1 micrometre object perturbations. This is sensitivity evidence, not proof that every GPU implementation detail is correct.'))
print(json.dumps(summary,indent=2))
