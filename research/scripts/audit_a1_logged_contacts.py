"""CPU kinematic audit of logged GPU contact tangent velocities (no resimulation)."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
from a1_data_sources import ROOT,RUN,write
from a1_episode_data import EpisodeShard
def main():
 reports=[]
 for gpu in [2,3]:
  shard=EpisodeShard(RUN/f'collection_throughput_gpu{gpu}_002');m=mujoco.MjModel.from_xml_path(str(ROOT/'data/a1_processed/tasks'/shard.meta['task_id']/'scene.xml'));d=mujoco.MjData(m)
  ob=m.body('object').id;orth=[];excess=[];badgate=0;count=0
  for e in shard.episodes[:10]:
   ctx=shard.arrays['context'][e['offsets']['context']];m.body_ipos[ob]=ctx[139:142]
   for t in range(0,e['controls']+1,3):
    v=shard.arrays['generalized'][e['offsets']['generalized']+t];d.qpos[:]=v[:m.nq];d.qvel[:]=v[m.nq:];mujoco.mj_forward(m,d)
    start,n=shard.arrays['contact_ranges'][e['offsets']['contact_ranges']+t];start+=e['offsets']['contact_details']
    for row in shard.arrays['contact_details'][start:start+n]:
     g0,g1=map(int,row[:2]);pos=row[2:5].astype(np.float64);vel=[]
     for g in [g0,g1]:
      jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,pos,int(m.geom_bodyid[g]));vel.append(jp@d.qvel)
     relative=vel[1]-vel[0];tangent=row[11:14];norm=np.linalg.norm(tangent)
     if norm>1e-5:orth.append(abs(np.dot(relative-tangent,tangent))/norm)
     excess.append(max(0,norm-np.linalg.norm(relative)));badgate+=int(bool(row[15]) != bool(row[5]>.02 and row[14]<.002));count+=1
  report=dict(task=shard.meta['task_id'],contacts_checked=count,tangent_orthogonality_velocity_error_max_m_s=float(max(orth,default=0)),tangent_speed_excess_over_relative_speed_max_m_s=float(max(excess,default=0)),active_gate_mismatches=badgate,scope='CPU point Jacobians at logged GPU qpos/qvel/contact points; verifies tangential projection is orthogonal to removed normal component and cannot exceed actual relative speed. This does not reconstruct GPU contact forces or assert bitwise CPU/GPU contact trajectories.')
  assert count>0 and badgate==0 and max(orth,default=0)<2e-4 and max(excess,default=0)<2e-4,report
  reports.append(report)
 write(RUN/'logged_contact_kinematic_audit.json',reports);print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
