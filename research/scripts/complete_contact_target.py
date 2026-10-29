"""Single-factor diagnostic: recruit index contact via servo targets only.

Initial states, physical parameters and geometric references remain unchanged.
The target is disabled when the ORIGINAL reference has fewer than two contacting
finger geometries. This gate is a model-derived heuristic, not annotated contact.
"""
import numpy as np,mujoco
from scipy.optimize import least_squares
from grasp_lab import DATA,RUN,model_for,write_json

m=model_for('baseline');d=mujoco.MjData(m);ref=np.load(DATA/'trajectory.npz');controls=np.load(DATA/'controller_reference.npz')['ctrl'].copy();original=controls.copy();index=m.geom('hand_geom_10').id;objgeoms=np.flatnonzero(m.geom_contype==2);rows=[]

def distance():
    candidates=[]
    for g in objgeoms:
        seg=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,index,int(g),.1,seg);candidates.append((dist,seg.copy()))
    return min(candidates,key=lambda v:v[0])

for k,q in enumerate(ref['qpos']):
    d.qpos[:]=q;mujoco.mj_forward(m,d)
    touching=set()
    for c in d.contact:
        if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}=={1,2}:touching.add(c.geom1 if m.geom_contype[c.geom1]==1 else c.geom2)
    initial_distance=distance()[0];gate=len(touching)>=2
    if gate:
        q0=controls[k,10:14].copy();lo=np.maximum(m.jnt_range[10:14,0]+1e-6,q0-.3);hi=np.minimum(m.jnt_range[10:14,1]-1e-6,q0+.3)
        def residual(x):
            d.qpos[10:14]=x;mujoco.mj_forward(m,d)
            return np.r_[distance()[0]+.001,.0003*(x-q0)]
        sol=least_squares(residual,np.clip(q0,lo,hi),bounds=(lo,hi),max_nfev=60,diff_step=1e-4,gtol=1e-9)
        controls[k,10:14]=sol.x;final_distance=distance()[0]
    else:final_distance=initial_distance
    rows.append(dict(frame=k,gate=gate,original_touching_geoms=len(touching),index_initial_gap_m=initial_distance,index_command_gap_m=final_distance,index_joint_delta=(controls[k,10:14]-original[k,10:14]).tolist()))
np.savez_compressed(RUN/'contact_completion_targets.npz',ctrl=controls,reference_gate=[r['gate'] for r in rows])
write_json(RUN/'contact_completion.json',dict(change='Index-finger servo target only; up to 0.3 rad; virtual command overlap 1mm; actual initial state unchanged',assumption='Geometric heuristic, not calibrated hardware or ground-truth contact',frames=rows))
print(rows[0]);print('gated frames',sum(r['gate'] for r in rows))
