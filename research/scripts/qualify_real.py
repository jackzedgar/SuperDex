"""CPU physical checks; reference visualization is never a policy result."""
from pathlib import Path
import json,argparse
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/clip000000_obj000004_right'


def angle(q,r):
    return 2*np.arccos(np.clip(abs(np.dot(q,r)),0,1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/real_track_20261026');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ref=np.load(DATA/'trajectory.npz');processing=json.loads((DATA/'processing_report.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));d=mujoco.MjData(m)
    assert (m.nq,m.nv,m.nu)==(33,32,26)
    contacts=[];hand_pen=[];floor_pen=[]
    for q in ref['qpos']:
        d.qpos[:]=q;mujoco.mj_forward(m,d)
        h=[];f=[]
        for c in d.contact:
            types={int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}
            if types=={1,2}:h.append(c.dist)
            if types=={2,4}:f.append(c.dist)
        contacts.append(len(h));hand_pen.append(max([max(0,-x) for x in h]+[0]));floor_pen.append(max([max(0,-x) for x in f]+[0]))
    trials={}
    for mode in ['stationary_hold','reference_control']:
        mujoco.mj_resetData(m,d);d.qpos[:]=ref['qpos'][0]
        if mode=='reference_control':d.qvel[:]=ref['qvel'][0]
        rows=[];warnings0=d.warning.number.copy();hand_steps=0
        n=100 if mode=='stationary_hold' else len(ref['qpos'])-1
        for t in range(n):
            k=0 if mode=='stationary_hold' else t+1
            d.ctrl[:]=ref['ctrl'][k]
            for _ in range(10):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            hand_steps+=any({int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}=={1,2} for c in d.contact)
            q=ref['qpos'][k]
            rows.append((np.linalg.norm(d.qpos[26:29]-q[26:29]),angle(d.qpos[29:],q[29:]),np.linalg.norm(d.qpos[:3]-q[:3]),np.sqrt(np.mean((d.qpos[6:26]-q[6:26])**2))))
            if not np.isfinite(d.qpos).all():raise RuntimeError('Nonfinite CPU state')
        rows=np.asarray(rows)
        trials[mode]=dict(steps=n,object_position_mean_m=float(rows[:,0].mean()),object_position_final_m=float(rows[-1,0]),object_position_max_m=float(rows[:,0].max()),object_rotation_mean_deg=float(np.rad2deg(rows[:,1].mean())),object_rotation_final_deg=float(np.rad2deg(rows[-1,1])),wrist_position_mean_m=float(rows[:,2].mean()),finger_joint_rmse_rad=float(rows[:,3].mean()),hand_contact_steps=hand_steps,warnings=(d.warning.number-warnings0).tolist(),finite=True)
        np.savez(a.output/f'cpu_{mode}.npz',errors=rows)
    checks=dict(no_missing_frames=True,source_rotation_valid=bool(np.allclose(np.linalg.det(ref['object_pose'][:,:3,:3]),1)),joint_limits=processing['joint_limit_violations']==0,
                tip_p95_under_15mm=processing['fingertip_error_p95_m']<.015,initial_hand_penetration_under_1mm=hand_pen[0]<.001,
                all_reference_penetration_under_2mm=max(hand_pen+floor_pen)<.002,other_hand_over_50mm=processing['other_hand_min_vertex_distance_m']>.05,
                contact_at_initial_grasp=contacts[0]>0,cpu_finite_no_warnings=all(t['finite'] and not any(t['warnings']) for t in trials.values()),
                wrist_servo_mean_under_30mm=trials['reference_control']['wrist_position_mean_m']<.03)
    result=dict(checks=checks,ready_for_bounded_ppo=all(checks.values()),cpu_trials=trials,reference_hand_penetration_max_m=max(hand_pen),reference_floor_penetration_max_m=max(floor_pen),initial_contact_count=contacts[0],min_reference_contact_count=min(contacts),
                note='Numerical/control admission checks do not certify dynamic feasibility or task success. Object free joint is never overwritten during either physical trial. Hand self-collision and hand-floor collision disabled in this pilot.')
    # Audit masks omitted from the minimal dynamics without changing that model.
    audit=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));audit.geom_conaffinity[audit.geom_contype==1]=7;audit.geom_conaffinity[audit.geom_contype==4]=3;ad=mujoco.MjData(audit);rows=[]
    for i,q in enumerate(ref['qpos']):
        ad.qpos[:]=q;mujoco.mj_forward(audit,ad)
        for c in ad.contact:
            types={int(audit.geom_contype[c.geom1]),int(audit.geom_contype[c.geom2])}
            if (types=={1} or types=={1,4}) and c.dist<-.0001:
                rows.append(dict(frame=i,penetration_m=-float(c.dist),pair=[audit.geom(c.geom1).name,audit.geom(c.geom2).name],kind='self' if types=={1} else 'hand_floor'))
    audit_result=dict(geometry_only=True,contact_masks_temporarily_enabled=True,counts=len(rows),maximum_penetration_m=max([r['penetration_m'] for r in rows]+[0]),worst=sorted(rows,key=lambda x:-x['penetration_m'])[:10])
    (a.output/'disabled_collision_audit.json').write_text(json.dumps(audit_result,indent=2)+'\n')
    (a.output/'qualification_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
