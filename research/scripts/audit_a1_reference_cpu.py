import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('scripts').resolve()))
import numpy as np,torch,mujoco
torch.set_num_threads(1)
from a1_task_runtime import TaskTensors
start=time.monotonic();manifest=json.loads(Path('configs/a1_probe40_v1.json').read_text());rows=[]
for row in manifest['tasks']:
 t=TaskTensors(row['task_path'],device='cpu');m=t.m;d=mujoco.MjData(m)
 ob=m.body('object').id;og=set(np.flatnonzero(m.geom_bodyid==ob));frames=sorted(set([0,1,2,4,8,*np.linspace(0,t.length-2,16,dtype=int)]));frames=[f for f in frames if f<t.length-1]
 qv=t.raw['qvel'];qa=np.gradient(qv,.02,axis=0);records=[]
 for f in frames:
  d.qpos[:]=t.raw['qpos'][f];d.qvel[:]=qv[f];d.qacc[:]=qa[f];mujoco.mj_inverse(m,d)
  handcontacts=[];tablecontacts=[]
  for c in d.contact:
   g1,g2=int(c.geom1),int(c.geom2)
   if (g1 in og)==(g2 in og):continue
   g=g2 if g1 in og else g1
   if m.geom_contype[g]==1:handcontacts.append(float(c.dist))
   elif m.geom_contype[g]==4:tablecontacts.append(float(c.dist))
  tips=t.ref_sdf[f].numpy();mask=t.mask.numpy()[0].astype(bool);tip=(tips[mask]<.01).sum()
  force=d.qfrc_inverse.copy();mass=m.body_mass[ob]
  maxforce=np.max(abs(m.actuator_forcerange),axis=1)
  ratios=abs(force[:t.nu])/np.maximum(maxforce,1e-9)
  records.append({'frame':int(f),'robot_tips_within1cm':int(tip),'hand_contacts':len(handcontacts),'hand_contacts_within2mm':sum(x<=.002 for x in handcontacts),'table_contacts':len(tablecontacts),'unactuated_inverse_force_N':float(np.linalg.norm(force[t.nu:t.nu+3])),'unactuated_inverse_force_over_weight':float(np.linalg.norm(force[t.nu:t.nu+3])/(9.81*mass)),'unactuated_inverse_torque_Nm':float(np.linalg.norm(force[t.nu+3:t.nu+6])),'max_actuated_torque_ratio':float(ratios.max()),'actuated_dofs_over_limit':int((ratios>1).sum())})
 rows.append({'task':row['task_id'],'stratum':row['stratum'],'frames_total':t.length,'sampled':records})
print(json.dumps({'scope':'CPU reference geometry and nominal inverse-dynamics diagnostics; sampled frames, finite-difference acceleration, no task exclusion or policy-success inference. Unactuated inverse force is force needed for exact reference under the nominal soft-contact inverse model, not a proof of impossibility within tracking tolerance.','elapsed_seconds':time.monotonic()-start,'tasks':rows},indent=2))
