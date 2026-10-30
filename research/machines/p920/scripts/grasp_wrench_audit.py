"""Linearized grasp wrench feasibility, not a nonlinear stability certificate."""
import json
import numpy as np,mujoco
from scipy.optimize import linprog
from grasp_lab import DATA,RUN,model_for,fixed_initials,write_json,cpu_contacts

m=model_for('baseline');d=mujoco.MjData(m);ref=np.load(DATA/'trajectory.npz');target=np.load(DATA/'controller_reference.npz')['ctrl'];initial=fixed_initials();body=m.body('object').id
results=[]
for trial in [-1]+list(range(8)):
    d.qpos[:]=ref['qpos'][0] if trial<0 else initial['qpos'][trial];d.qvel[:]=0;d.ctrl[:]=target[0];mujoco.mj_forward(m,d)
    W=[];J=[];group=[];details=[]
    for i,c in enumerate(d.contact):
        if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}!={1,2}:continue
        hg=c.geom1 if m.geom_contype[c.geom1]==1 else c.geom2;sign=1 if hg==c.geom1 else -1
        frame=c.frame.reshape(3,3);jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[hg]));center=d.xipos[body]
        for tangent in [[.5,0],[-.5,0],[0,.5],[0,-.5]]:
            # Half the actual sliding coefficient reserves friction capacity.
            f=sign*frame.T@np.r_[1,tangent]
            W.append(np.r_[f,np.cross(c.pos-center,f)]);J.append(jp[:,:26].T@f);group.append(int(hg))
        details.append(dict(geom=int(hg),point=c.pos.tolist(),normal=(sign*frame[0]).tolist()))
    W=np.array(W).T;J=np.array(J).T;groups=sorted(set(group));lower=[]
    for g in groups:lower.append(-np.array(group).__eq__(g).astype(float))
    # Each contacting finger's patch supplies >= 0.5 N, with existing torque bounds.
    A=np.vstack([J,-J,lower]);b=np.r_[m.actuator_forcerange[:,1],-m.actuator_forcerange[:,0],[-.5]*len(groups)]
    rhs=np.r_[-m.body_mass[body]*m.opt.gravity,np.zeros(3)]
    sol=linprog(np.ones(W.shape[1]),A_ub=A,b_ub=b,A_eq=W,b_eq=rhs,bounds=(0,None),method='highs')
    record=dict(trial=trial,contacts=len(details),distinct_hand_geoms=len(groups),object_com_world=d.xipos[body].tolist(),initial_unbalanced_linear_acceleration=d.qacc[26:29].tolist(),initial_unbalanced_angular_acceleration=d.qacc[29:].tolist(),linearized_static_wrench_feasible=bool(sol.success),message=sol.message,contacts_detail=details)
    wn=W.reshape(6,-1,4).mean(2).repeat(4,axis=1);jn=J.reshape(26,-1,4).mean(2).repeat(4,axis=1)
    wa=wn+2*(W-wn);ja=jn+2*(J-jn)
    actual=linprog(np.ones(wa.shape[1]),A_ub=np.vstack([ja,-ja]),b_ub=np.r_[m.actuator_forcerange[:,1],-m.actuator_forcerange[:,0]],A_eq=wa,b_eq=rhs,bounds=(0,None),method='highs')
    unlimited=linprog(np.ones(wa.shape[1]),A_eq=wa,b_eq=rhs,bounds=(0,None),method='highs')
    record.update(actual_mu1_no_reserve_feasible=bool(actual.success),actual_mu1_unlimited_actuator_feasible=bool(unlimited.success),actual_mu1_min_normal_force_n=float(sum(actual.x)) if actual.success else None)
    if sol.success:
        tau=J@sol.x;record.update(normal_force_sum_n=float(sum(sol.x)),equilibrium_residual=float(np.max(abs(W@sol.x-rhs))),actuator_torque_nm=tau.tolist(),ctrl=(d.qpos[:26]+tau/m.actuator_gainprm[:,0]).tolist())
    results.append(record)
write_json(RUN/'wrench_audit.json',dict(assumptions='Point contacts; linearized geometry; friction coefficient 0.5 used as conservative reserve vs model 1.0; >=0.5N per existing finger patch. No measurement-based material claim.',results=results))
print(json.dumps([{k:v for k,v in r.items() if k in ['trial','contacts','distinct_hand_geoms','initial_unbalanced_linear_acceleration','initial_unbalanced_angular_acceleration','linearized_static_wrench_feasible','normal_force_sum_n']} for r in results],indent=2))
