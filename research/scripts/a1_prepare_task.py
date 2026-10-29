"""Prepare a source-selected full-A1 physical reference, without policy-outcome selection."""
import argparse,copy,fcntl,hashlib,json,time,xml.etree.ElementTree as E
from pathlib import Path
import numpy as np,mujoco,trimesh,coacd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from a1_data_sources import ROOT,RUN,write
from a1_hand_assets import ASSETS,combined
from a1_source_reference import resample
def object_assets(row,raw):
    key=row['object_key'];out=ASSETS/'objects'/key;out.mkdir(parents=True,exist_ok=True)
    with (out/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done=out/'asset.json'
        if done.exists():return out,json.loads(done.read_text())
        mesh=raw['mesh'];sha=hashlib.sha256(raw['mesh_path'].read_bytes()).hexdigest()
        if mesh.extents.max()>.284:raise ValueError('Object too large for <=3mm and <=128 grid dimensions including 5cm margins')
        mesh.export(out/'visual.obj')
        coacd.set_log_level('warn')
        parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),threshold=.05,max_convex_hull=16,resolution=1000,mcts_nodes=10,mcts_iterations=50,mcts_max_depth=3,seed=0)
        for j,(v,f) in enumerate(parts):trimesh.Trimesh(v,f,process=False).export(out/f'collision_{j}.obj')
        solid=mesh if mesh.is_watertight and mesh.volume>1e-10 else mesh.convex_hull
        mass=.012 if row['stratum']=='light_object_tool_use_below100g' else float(np.clip(solid.volume*500,.02,.3))
        inertia=np.asarray(solid.moment_inertia)*mass/solid.volume
        values=np.linalg.eigvalsh(inertia)
        assert values.min()>0 and values.max()<values.sum()/2+1e-8
        meta=dict(object_key=key,source_mesh=str(raw['mesh_path'].relative_to(ROOT)),source_mesh_sha256=sha,parts=len(parts),mass_kg=mass,center_of_mass_m=solid.center_mass.tolist(),inertia_kgm2=inertia.tolist(),physical_parameters_status='SIMULATED nominal assumptions, not measured. Pen12g; other objects uniform density500kg/m3 clipped20-300g. Inertia from uniform source solid or convex-hull fallback.',inertia_geometry='source watertight solid' if solid is mesh else 'convex hull fallback',bounds_m=mesh.bounds.tolist(),extents_m=mesh.extents.tolist())
        write(done,meta);return out,meta

def build(row):
    started=time.monotonic();out=ROOT/'data/a1_processed/tasks'/row['task_id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'qualification.json').exists():return json.loads((out/'qualification.json').read_text())
    write(out/'source_selection.json',row)
    raw=resample(row);asset,meta=object_assets(row,raw)
    sides=row['hands'];root=combined(sides);world=root.find('worldbody');assets=root.find('asset')
    E.SubElement(assets,'mesh',name='object_visual_mesh',file=str(asset/'visual.obj'))
    obj=E.SubElement(world,'body',name='object');E.SubElement(obj,'freejoint',name='object_free')
    I=np.array(meta['inertia_kgm2']);full=[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]
    E.SubElement(obj,'inertial',pos=' '.join(map(str,meta['center_of_mass_m'])),mass=str(meta['mass_kg']),fullinertia=' '.join(map(str,full)))
    E.SubElement(obj,'geom',name='object_visual',type='mesh',mesh='object_visual_mesh',contype='0',conaffinity='0',density='0',group='1',rgba='.6 .35 .2 1')
    for j in range(meta['parts']):
        E.SubElement(assets,'mesh',name=f'object_mesh_{j}',file=str(asset/f'collision_{j}.obj'))
        E.SubElement(obj,'geom',name=f'object_collision_{j}',type='mesh',mesh=f'object_mesh_{j}',contype='2',conaffinity='5',density='0',friction='0 .005 .0001')
    scene=out/'scene.xml';E.ElementTree(root).write(scene)
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);nu=m.nu
    d.qpos[:nu]=np.clip(0,m.jnt_range[:nu,0],m.jnt_range[:nu,1]);mujoco.mj_forward(m,d)
    registrations={}
    for h,side in enumerate(sides):
        slot=['right','left'].index(side)
        ids=[m.site(f'{side}_mcp_{i}').id for i in range(2,6)]
        robot=np.vstack([d.site_xpos[ids],[0,0,0]])
        target=np.vstack([raw['keypoints_50'][0,slot,11:15],raw['wrist_50'][0,slot,:3,3]])
        u,_,vt=np.linalg.svd((robot-robot.mean(0)).T@(target-target.mean(0)))
        rot=vt.T@u.T
        if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
        pos=target.mean(0)-rot@robot.mean(0);T=np.eye(4);T[:3,:3]=rot;T[:3,3]=pos
        registrations[side]=T
        mount=next(b for b in root.iter('body') if b.get('name')==f'{side}_wrist')
        mount.set('pos',' '.join(map(str,pos)));mount.set('quat',' '.join(map(str,Rotation.from_matrix(rot).as_quat(scalar_first=True))))
    E.ElementTree(root).write(scene);m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    sites=np.array([m.site(f'{side}_{label}_{i}').id for side in sides for label in ['tip','mid'] for i in range(1,6)])
    object_geoms=set(np.flatnonzero(m.geom_contype==2).tolist());hand_geoms=np.flatnonzero(m.geom_contype==1);lookup={g:i for i,g in enumerate(hand_geoms)}
    lo,hi=m.jnt_range[:nu].T;qprev=np.clip(np.zeros(nu),lo+1e-5,hi-1e-5)
    weights=np.tile(np.repeat([1.]*5+[.4]*5,3),len(sides))
    wrists=[];points=[];references=[];errors=[];penetrations=[];nfevs=[]
    time_s=raw['resampled_time'];wrist_goals=[]
    for side in sides:
        slot=['right','left'].index(side);base=registrations[side]
        desired=raw['wrist_50'][:,slot]@np.linalg.inv(raw['wrist_50'][0,slot])@base
        local=np.linalg.inv(base)[None]@desired
        euler=np.unwrap(Rotation.from_matrix(local[:,:3,:3]).as_euler('XYZ'),axis=0)
        wrist_goals.append(np.concatenate([local[:,:3,3],euler],-1))
    for frame in range(len(time_s)):
        pose=raw['object'][frame];d.qpos[nu:]=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)]
        target=np.concatenate([raw['keypoints_50'][frame,['right','left'].index(side),:10] for side in sides])
        previous=qprev.copy();cache={}
        def calc(q):
            if 'q' in cache and np.array_equal(q,cache['q']):return cache['r'],cache['j']
            d.qpos[:nu]=q;mujoco.mj_forward(m,d)
            residual=((d.site_xpos[sites]-target).ravel()*weights)
            jac=[]
            for i,sid in enumerate(sites):
                jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,int(sid));jac.append(jp[:,:nu]*weights[i*3:(i+1)*3,None])
            jac=np.concatenate(jac)
            pens=np.zeros(len(hand_geoms));pj=np.zeros((len(hand_geoms),nu))
            for c in d.contact:
                g1,g2=int(c.geom1),int(c.geom2)
                hand=g1 if g2 in object_geoms else g2 if g1 in object_geoms else -1
                if hand not in lookup:continue
                k=lookup[hand];depth=min(float(c.dist)+.0005,0.)
                if depth<pens[k]:
                    pens[k]=depth;jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[hand]))
                    pj[k]=(1 if hand==g2 else -1)*c.frame[:3]@jp[:,:nu]
            r=[residual,.002*(q-previous),5*pens];js=[jac,.002*np.eye(nu),5*pj]
            for h,side in enumerate(sides):
                factor=np.array([.2]*3+[.01]*3);j=np.zeros((6,nu));j[:,h*26:h*26+6]=np.diag(factor)
                r.append(factor*(q[h*26:h*26+6]-wrist_goals[h][frame]));js.append(j)
            result=(np.concatenate(r),np.concatenate(js));cache.update(q=q.copy(),r=result[0],j=result[1]);return result
        lower,upper=lo+1e-6,hi-1e-6
        if row.get('retarget_version',1)>=2 and frame>0:
            lower=lower.copy();upper=upper.copy()
            for h in range(len(sides)):
                sl=slice(h*26+6,h*26+26)
                lower[sl]=np.maximum(lower[sl],qprev[sl]-.4)
                upper[sl]=np.minimum(upper[sl],qprev[sl]+.4)
        sol=least_squares(lambda q:calc(q)[0],qprev,jac=lambda q:calc(q)[1],bounds=(lower,upper),max_nfev=100 if frame==0 else 30,ftol=1e-6,xtol=1e-6,gtol=1e-6)
        qprev=sol.x;calc(qprev);references.append(d.qpos.copy());points.append(d.site_xpos[sites].copy());errors.append(np.linalg.norm(d.site_xpos[sites]-target,axis=-1));nfevs.append(sol.nfev)
        wrists.append(np.array([np.r_[d.xpos[m.body(f'{side}_wrist').id],d.xquat[m.body(f'{side}_wrist').id]] for side in sides]))
        penetrations.append(max([max(0.,-float(c.dist)) for c in d.contact if int(c.geom1) in object_geoms or int(c.geom2) in object_geoms]+[0.]))
        if frame%100==0:print(row['task_id'],frame,'/',len(time_s),'tip mean mm',np.mean(np.array(errors[-1]).reshape(-1,10)[:,:5])*1000,flush=True)
    qpos=np.array(references);qvel=np.zeros((len(qpos),m.nv))
    for j in range(len(qpos)-1):mujoco.mj_differentiatePos(m,qvel[j],.02,qpos[j],qpos[j+1])
    qvel[-1]=qvel[-2]
    np.savez_compressed(out/'reference.npz',time_s=time_s,source_time=raw['source_time'],qpos=qpos,qvel=qvel,ctrl=qpos[:,:nu],wrist_pose=np.array(wrists),object_pose=raw['object'],robot_points=np.array(points),human_keypoints=raw['keypoints_50'],errors_m=np.array(errors),penetration_m=np.array(penetrations),source_to_sim=raw['source_to_sim'])
    e=np.array(errors).reshape(len(time_s),len(sides),10)[:,:,:5]
    gates=dict(finite=bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),tip_p95_le20mm=bool(np.quantile(e,.95)<=.02),initial_penetration_le3mm=bool(penetrations[0]<=.003),max_penetration_le5mm=bool(max(penetrations)<=.005),joint_speed_le25rad_s=bool(max(np.max(abs(qvel[:,h*26+6:h*26+26])) for h in range(len(sides)))<=25),duration_ge0p8s=bool(time_s[-1]>=.8))
    report=dict(retarget_version=row.get('retarget_version',1),velocity_constraint_rad_s=20 if row.get('retarget_version',1)>=2 else None,task_id=row['task_id'],stratum=row['stratum'],source=row['source'],hands=sides,object=meta,frames=len(time_s),duration_s=float(time_s[-1]),tip_mean_mm=float(e.mean()*1000),tip_p95_mm=float(np.quantile(e,.95)*1000),max_penetration_mm=float(max(penetrations)*1000),initial_penetration_mm=float(penetrations[0]*1000),gates=gates,kinematic_candidate=all(gates.values()),source_support=raw['support_evidence'],source_to_sim=raw['source_to_sim'].tolist(),source_end_trim_s=raw['trim_at_end_s'],cpu_wall_seconds=time.monotonic()-started,physical_qualification='PENDING; no policy success or static-hold gate',source_file=str(raw['source_file'].relative_to(ROOT)))
    write(out/'qualification.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
    row=json.loads(a.manifest.read_text())['candidates'][a.index]
    try:report=build(row);print(json.dumps(report),flush=True)
    except Exception as err:
        write(ROOT/'data/a1_processed/tasks'/row['task_id']/'error.json',dict(task=row,error=repr(err)));raise
if __name__=='__main__':main()
