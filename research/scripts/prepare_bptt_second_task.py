"""Reproduce the fixed second-task control recipe on CPU; never tune on success."""
import argparse,json,time
import numpy as np,mujoco
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/bptt_iteration_20261027'
def main():
    p=argparse.ArgumentParser();p.add_argument('--task-id',type=int,default=1);p.add_argument('--check-only',action='store_true');a=p.parse_args();start=time.time()
    meta=next(t for t in json.loads((RUN/'task_manifest.json').read_text())['tasks'] if t['task_id']==a.task_id)
    assert a.task_id!=0,'Historical spoon controls remain immutable.'
    dpath=ROOT/meta['path'];ref=np.load(dpath/'trajectory.npz');m=mujoco.MjModel.from_xml_path(str(dpath/'scene.xml'));d=mujoco.MjData(m);directions=[];counts=[]
    for q in ref['qpos']:
        d.qpos[:]=q;mujoco.mj_forward(m,d);rows=[]
        for c in d.contact:
            if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}!={1,2}:continue
            hand=c.geom1 if m.geom_contype[c.geom1]==1 else c.geom2
            normal=c.frame[:3]*(1 if hand==c.geom1 else -1)
            jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[hand]));rows.append(normal@jp[:,6:26])
        bias=np.linalg.lstsq(np.vstack([rows,.003*np.eye(20)]),np.r_[np.full(len(rows),.001),np.zeros(20)],rcond=None)[0] if rows else np.zeros(20)
        directions.append(bias);counts.append(len(rows))
    ctrl=ref['ctrl'].copy();ctrl[:,6:]+=np.clip(np.array(directions),-.25,.25);ctrl=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);assert np.isfinite(ctrl).all()
    dest=dpath/'controller_reference.npz';difference=None
    if dest.exists():
        old=np.load(dest);difference=float(np.max(abs(ctrl-old['ctrl'])));assert difference<1e-10,'Existing reference differs; do not overwrite it.'
    elif not a.check_only:np.savez(dest,ctrl=ctrl,preload_direction=directions)
    else:raise FileNotFoundError(dest)
    print(json.dumps(dict(task_id=a.task_id,controls=len(ctrl)-1,max_existing_difference=difference,reference_contact_counts=counts,cpu_seconds=time.time()-start,selection='Fixed geometric1mm target, regularizer.003, joint bias clip.25rad; no policy or physical-success fitting')))
if __name__=='__main__':main()

