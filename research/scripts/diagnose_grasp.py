"""Small controller diagnosis: contact-normal preload, fixed mass/friction.

Only servo targets change. Original retargeted qpos and object path are immutable.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from qualify_real import DATA,ROOT,angle


def main():
    ref=np.load(DATA/'trajectory.npz');m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));d=mujoco.MjData(m)
    directions=[]
    for q in ref['qpos']:
        d.qpos[:]=q;mujoco.mj_forward(m,d);rows=[]
        for c in d.contact:
            if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}!={1,2}:continue
            hg=c.geom1 if m.geom_contype[c.geom1]==1 else c.geom2
            normal=c.frame[:3]*(1 if hg==c.geom1 else -1)
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[hg]))
            rows.append(normal@jp[:,6:26])
        if rows:
            J=np.asarray(rows);bias=np.linalg.lstsq(np.vstack([J,.003*np.eye(20)]),np.r_[np.ones(len(J))*.001,np.zeros(20)],rcond=None)[0]
        else:bias=np.zeros(20)
        directions.append(bias)
    directions=np.array(directions);trials=[];best=None
    for strength in [0,1,2,4,8]:
        ctrl=ref['ctrl'].copy();ctrl[:,6:]+=np.clip(strength*directions,-.25,.25)
        ctrl=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        result={'preload_mm':strength}
        for mode in ['hold','tracking']:
            mujoco.mj_resetData(m,d);d.qpos[:]=ref['qpos'][0]
            if mode=='tracking':d.qvel[:]=ref['qvel'][0]
            es=[];contact_steps=0
            for t in range(100 if mode=='hold' else len(ctrl)-1):
                k=0 if mode=='hold' else t+1;d.ctrl[:]=ctrl[k]
                for _ in range(10):mujoco.mj_step(m,d)
                mujoco.mj_forward(m,d)
                es.append([np.linalg.norm(d.qpos[26:29]-ref['qpos'][k,26:29]),angle(d.qpos[29:],ref['qpos'][k,29:])])
                contact_steps+=any({int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}=={1,2} for c in d.contact)
            result[mode]=dict(position_mean_m=float(np.mean(es,axis=0)[0]),position_final_m=float(es[-1][0]),rotation_mean_deg=float(np.rad2deg(np.mean(es,axis=0)[1])),contact_steps=contact_steps)
        trials.append(result)
        # Diagnostic selection, not an unbiased benchmark or separate learned policy.
        score=result['hold']['position_mean_m']+result['tracking']['position_mean_m']
        if best is None or score<best[0]:best=(score,strength,ctrl)
    np.savez(DATA/'controller_reference.npz',ctrl=best[2],preload_direction=directions)
    report=dict(trials=trials,selected_preload_mm=best[1],selection='minimum hold + tracking position mean error on this same pilot trajectory',
                baseline_qpos_unchanged=True,object_mass_and_friction_unchanged=True,
                caution='Controller targets contain up to 0.25 rad geometric preload; target is not a physically occupied reference pose. No force actuator or kinematic overwrite is applied to the object.')
    (ROOT/'runs/real_track_20261026/grasp_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
