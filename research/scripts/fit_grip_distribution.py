"""Diagnose force distribution by bounded CPU servo-target optimization.

Only thumb/middle/ring preload multipliers change; no material/model changes.
Fit uses the unperturbed nominal state, then saved eight perturbations are tested.
"""
import json
import numpy as np,mujoco
from scipy.optimize import minimize
from grasp_lab import ROOT,DATA,RUN,write_json,quat_angle,model_for

m=model_for('baseline');ref=np.load(DATA/'trajectory.npz');target=np.load(DATA/'controller_reference.npz')['ctrl'];q0=ref['qpos'][0];delta=target[0]-q0[:26];d=mujoco.MjData(m)
history=[]


def objective(x):
    mujoco.mj_resetData(m,d);d.qpos[:]=q0
    bias=delta.copy()
    for factor,finger in zip(x,[0,2,3]):bias[6+4*finger:10+4*finger]*=factor
    d.ctrl[:]=np.clip(q0[:26]+bias,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);values=[]
    for k in range(200):
        mujoco.mj_step(m,d)
        if k%10==9:
            ep=np.linalg.norm(d.qpos[26:29]-q0[26:29]);er=quat_angle(d.qpos[29:],q0[29:]);values.append((ep/.01)**2+(er/.2)**2)
    score=float(np.mean(values)+.002*np.sum((x-1)**2));history.append(dict(x=x.tolist(),objective=score))
    return score


initial=objective(np.ones(3));fit=minimize(objective,np.ones(3),method='Powell',bounds=[(.1,4)]*3,options={'maxiter':15,'maxfev':400,'xtol':.01,'ftol':.001})
write_json(RUN/'grip_distribution_fit.json',dict(initial_objective=initial,final_objective=float(fit.fun),multipliers=fit.x.tolist(),fingers=['thumb','middle','ring'],evaluations=len(history),history=history,success=bool(fit.success),message=str(fit.message),scope='Nominal 0.4 s CPU static grip fit; a controller diagnosis, not learning or independent performance evidence. Validate 2 s and paired states separately.'))
print(json.dumps(dict(initial=initial,final=float(fit.fun),multipliers=fit.x.tolist(),evaluations=len(history))),flush=True)
