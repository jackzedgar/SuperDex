"""HOT3D -> 50 Hz, object-preserving Wuji landmark retargeting.

Raw observations are retained. This is a newly defined pilot task, not Probe-40.
"""
import argparse
from pathlib import Path
import json, sys, tarfile, hashlib
import xml.etree.ElementTree as E
import numpy as np
import torch
import mujoco
import trimesh
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'references/hand_tracking_toolkit'))
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json, UmeTrackHandPose, forward_kinematics


def transform(pose):
    T=np.eye(4)
    T[:3,:3]=R.from_quat(pose['quaternion_wxyz'],scalar_first=True).as_matrix()
    T[:3,3]=pose['translation_xyz']
    return T


def interpolate_pose(t, matrices, out_t):
    out=np.tile(np.eye(4),(len(out_t),1,1))
    out[:,:3,:3]=Slerp(t,R.from_matrix(matrices[:,:3,:3]))(out_t).as_matrix()
    for j in range(3): out[:,j,3]=np.interp(out_t,t,matrices[:,j,3])
    return out


def hand_xml(out):
    source=ROOT.parent/'assets/wuji-training/urdf/right.urdf'
    tree=E.parse(source); meshdir=(source.parent/'../meshes/right').resolve()
    tree.getroot().find('mujoco/compiler').set('meshdir',str(meshdir))
    for mesh in tree.getroot().iter('mesh'): mesh.set('filename',Path(mesh.get('filename')).name)
    tree.write(out/'hand_path_fixed.urdf',encoding='utf-8')
    m=mujoco.MjModel.from_xml_path(str(out/'hand_path_fixed.urdf'))
    mujoco.mj_saveLastXML(str(out/'hand_imported.xml'),m)
    root=E.parse(out/'hand_imported.xml').getroot()
    root.find('compiler').set('meshdir',str(meshdir))
    for i in range(1,6):
        parent=next(x for x in root.iter('body') if x.get('name')==f'right_finger{i}_link4')
        frame=next(x for x in parent.findall('frame') if any('tip_link' in g.get('mesh','') for g in x.findall('geom')))
        mesh=trimesh.load(meshdir/f'right_finger{i}_tip_link.STL',force='mesh')
        p=mesh.bounds.mean(0);p[2]=mesh.bounds[1,2]
        E.SubElement(frame,'site',name=f'tip_{i}',pos=' '.join(map(str,p)),size='.002',rgba='0 0.4 0.8 1')
        mid=next(x for x in root.iter('body') if x.get('name')==f'right_finger{i}_link3')
        E.SubElement(mid,'site',name=f'mid_{i}',pos='0 0 0',size='.001')
        mcp=next(x for x in root.iter('body') if x.get('name')==f'right_finger{i}_link2')
        E.SubElement(mcp,'site',name=f'mcp_{i}',pos='0 0 0',size='.001')
    world=root.find('worldbody')
    mount=E.Element('body',name='wrist',pos='0 0 0')
    E.SubElement(mount,'inertial',pos='0 0 0.04',mass='.25',diaginertia='.001 .001 .001')
    for i,axis in enumerate(['1 0 0','0 1 0','0 0 1']):
        E.SubElement(mount,'joint',name=f'wrist_p{i}',type='slide',axis=axis,range='-1 1',damping='5',armature='.01')
    for i,axis in enumerate(['1 0 0','0 1 0','0 0 1']):
        E.SubElement(mount,'joint',name=f'wrist_r{i}',type='hinge',axis=axis,range='-2.5 2.5',damping='.3',armature='.01')
    for child in list(world): world.remove(child);mount.append(child)
    world.append(mount)
    for b in mount.iter('body'): b.set('gravcomp','1')
    for g in mount.iter('geom'):
        visual=g.get('group')=='1'
        g.set('contype','0' if visual else '1');g.set('conaffinity','0' if visual else '2')
        g.set('friction','1.0 0.005 0.0001')
        if not visual: g.set('name',f'hand_geom_{len([x for x in mount.iter("geom") if x.get("name")])}')
    for j in mount.iter('joint'):
        if not j.get('name').startswith('wrist'):
            j.set('armature','0.0005');j.set('damping','.04')
    actuators=E.SubElement(root,'actuator')
    for i in range(3): E.SubElement(actuators,'position',name=f'wrist_p{i}',joint=f'wrist_p{i}',kp='3000',kv='100',forcerange='-100 100',ctrlrange='-1 1')
    for i in range(3): E.SubElement(actuators,'position',name=f'wrist_r{i}',joint=f'wrist_r{i}',kp='100',kv='5',forcerange='-15 15',ctrlrange='-2.5 2.5')
    for i in range(1,6):
        for j in range(1,5):
            name=f'right_finger{i}_joint{j}'
            joint=next(x for x in mount.iter('joint') if x.get('name')==name)
            E.SubElement(actuators,'position',name=name,joint=name,kp='3',kv='.08',
                         ctrlrange=joint.get('range'),forcerange=joint.get('actuatorfrcrange','-.4 .4'))
    option=root.find('option')
    if option is None: option=E.SubElement(root,'option')
    option.attrib.update(timestep='.002',integrator='implicitfast',solver='Newton',iterations='30',ls_iterations='50',gravity='0 0 -9.81')
    E.SubElement(world,'geom',name='floor',type='plane',size='2 2 .05',contype='4',conaffinity='2',rgba='.85 .85 .85 1')
    E.SubElement(world,'light',pos='0 -1 2',dir='0 0 -1',diffuse='.8 .8 .8')
    return root,mount


def main():
    p=argparse.ArgumentParser();p.add_argument('--clip',type=int,default=0);p.add_argument('--object',type=int,default=4)
    p.add_argument('--start',type=int,default=118);p.add_argument('--end',type=int,default=149)
    p.add_argument('--output',type=Path,help='Optional fresh task directory; existing baseline files are not overwritten.')
    args=p.parse_args()
    out=args.output.resolve() if args.output else ROOT/'data/processed'/f'clip{args.clip:06d}_obj{args.object:06d}_right'
    if args.output:assert not out.exists(), 'New task output must be a fresh directory.'
    out.mkdir(parents=True,exist_ok=True)
    rawroot=ROOT/'data/raw/hot3d';tarpath=rawroot/'train_quest3'/f'clip-{args.clip:06d}.tar'
    split=json.loads((rawroot/'clip_splits.json').read_text())
    # Preserve the upstream split file; directory is also explicitly train_quest3.
    infos=[];wp=[];op=[];kp=[];left=[];angles=[]
    with tarfile.open(tarpath) as tar:
        shape=from_json(json.load(tar.extractfile('__hand_shapes.json__'))['umetrack'])
        for i in range(150):
            info=json.load(tar.extractfile(f'{i:06d}.info.json'));infos.append(info)
            hands=json.load(tar.extractfile(f'{i:06d}.hands.json'))
            obj=json.load(tar.extractfile(f'{i:06d}.objects.json'))[str(args.object)][0]
            object_name=obj['object_name'];object_uid=obj['object_uid']
            op.append(transform(obj['T_world_from_object']))
            for side,dest in [('right',kp),('left',left)]:
                h=hands[side]['umetrack_pose'];w=transform(h['T_world_from_wrist'])
                pose=UmeTrackHandPose(torch.tensor(int(side=='right')),torch.tensor(h['joint_angles'],dtype=torch.float32),torch.tensor(w,dtype=torch.float32))
                with torch.no_grad(): points,_,_=forward_kinematics(pose,shape,requires_mesh=False)
                dest.append(points.numpy())
                if side=='right': wp.append(w);angles.append(h['joint_angles'])
        for i in [args.start,(args.start+args.end)//2,args.end]:
            (out/f'source_frame_{i:03d}.jpg').write_bytes(tar.extractfile(f'{i:06d}.image_1201-1.jpg').read())
    timestamps=np.array([x['ref_timestamp_ns'] for x in infos],dtype=np.int64)
    t=(timestamps-timestamps[0])*1e-9
    assert np.all(np.diff(t)>0) and np.max(np.abs(np.diff(t)-1/30))<1e-5
    wp,op,kp,left=map(np.asarray,(wp,op,kp,left))
    np.savez_compressed(out/'raw_reference.npz',timestamp_ns=timestamps,time_s=t,wrist_pose=wp,object_pose=op,
                        right_landmarks=kp,left_landmarks=left,human_joint_angles=np.array(angles))
    meshpath=rawroot/'object_models_eval'/f'obj_{args.object:06d}.glb'
    scene=trimesh.load(meshpath)
    mesh=scene.to_geometry() if isinstance(scene,trimesh.Scene) else scene
    assert 0.02<mesh.extents.max()<.6
    mesh.export(out/'object_visual.obj')
    # True source mesh is the visual. CoACD convex parts supply contact geometry.
    import coacd
    coacd.set_log_level('warn')
    parts=coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices),np.asarray(mesh.faces)),
                         threshold=.05,max_convex_hull=16,resolution=1000,mcts_nodes=10,mcts_iterations=50,mcts_max_depth=3,seed=0)
    for j,(v,f) in enumerate(parts): trimesh.Trimesh(v,f,process=False).export(out/f'object_collision_{j}.obj')
    # Quest source axes: +Y is vertical. Validate table level using the stationary spoon prefix.
    floor_samples=np.array([(mesh.vertices@op[i,:3,:3].T+op[i,:3,3])[:,1].min() for i in range(20)])
    prefix_floor=float(np.median(floor_samples))
    # Table height is not annotated. Fit a conservative lower envelope of source
    # mesh vertices; this preserves ALL object poses, and records the 3 mm fit shift.
    floor=float(min((mesh.vertices@pose[:3,:3].T+pose[:3,3])[:,1].min() for pose in op))-.0002
    A=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
    origin=np.array([op[args.start,0,3],floor,op[args.start,2,3]])
    T=np.eye(4);T[:3,:3]=A;T[:3,3]=-A@origin
    op2=T[None]@op;wp2=T[None]@wp
    kp2=kp@A.T-A@origin;left2=left@A.T-A@origin
    target_t=np.arange(t[args.start],t[args.end]+1e-9,.02)
    object_T=interpolate_pose(t,op2,target_t);wrist_T=interpolate_pose(t,wp2,target_t)
    human=np.stack([np.interp(target_t,t,kp2[:,j,c]) for j in range(20) for c in range(3)],axis=1).reshape(-1,20,3)
    other=np.stack([np.interp(target_t,t,left2[:,j,c]) for j in range(20) for c in range(3)],axis=1).reshape(-1,20,3)
    root,mount=hand_xml(out)
    root.find('asset').append(E.Element('mesh',name='real_visual',file=str(out/'object_visual.obj')))
    for j in range(len(parts)): E.SubElement(root.find('asset'),'mesh',name=f'obj_c{j}',file=str(out/f'object_collision_{j}.obj'))
    obj=E.SubElement(root.find('worldbody'),'body',name='object',pos='0 0 .3')
    E.SubElement(obj,'freejoint',name='object_free')
    # Estimated pilot mass/inertia; not supplied by the dataset.
    E.SubElement(obj,'inertial',pos=' '.join(map(str,mesh.center_mass)),mass='.04',diaginertia='.00032 .00032 .000015')
    E.SubElement(obj,'geom',name='object_visual',type='mesh',mesh='real_visual',contype='0',conaffinity='0',group='1',rgba='.65 .38 .15 1',density='0')
    for j in range(len(parts)): E.SubElement(obj,'geom',name=f'object_contact_{j}',type='mesh',mesh=f'obj_c{j}',contype='2',conaffinity='5',friction='1.0 .005 .0001',rgba='.65 .38 .15 1',density='0')
    scene_path=out/'scene.xml'
    E.ElementTree(root).write(scene_path)
    m=mujoco.MjModel.from_xml_path(str(scene_path));d=mujoco.MjData(m)
    # Register the robot palm to the observed proximal hand landmarks.
    d.qpos[6:26]=np.clip(np.zeros(20),m.jnt_range[6:26,0],m.jnt_range[6:26,1])
    mujoco.mj_forward(m,d)
    robot=np.array([d.site_xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'mcp_{i}')] for i in range(2,6)])
    human_mcp=human[0,[8,11,14,17]]
    # Include wrist to avoid the near-collinear MCP row rotation ambiguity.
    robot=np.vstack([robot,[0,0,0]])
    target=np.vstack([human_mcp,wrist_T[0,:3,3]])
    U,_,Vt=np.linalg.svd((robot-robot.mean(0)).T@(target-target.mean(0)))
    reg=Vt.T@U.T
    if np.linalg.det(reg)<0: Vt[-1]*=-1;reg=Vt.T@U.T
    reg_p=target.mean(0)-reg@robot.mean(0)
    mount.set('pos',' '.join(map(str,reg_p)))
    mount.set('quat',' '.join(map(str,R.from_matrix(reg).as_quat(scalar_first=True))))
    E.ElementTree(root).write(scene_path)
    m=mujoco.MjModel.from_xml_path(str(scene_path));d=mujoco.MjData(m)
    sites=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,f'{kind}_{i}') for kind in ['tip','mid'] for i in range(1,6)]
    hidx=[0,1,2,3,4,6,9,12,15,18]
    handgeoms=np.flatnonzero(m.geom_contype==1)
    objectgeoms=set(np.flatnonzero(m.geom_contype==2).tolist())
    qprev=np.zeros(26);qprev[6:]=np.clip(qprev[6:],m.jnt_range[6:26,0]+1e-5,m.jnt_range[6:26,1]-1e-5)
    lo,hi=m.jnt_range[:26].T
    references=[];errors=[];penetrations=[];robot_points=[];evals=[]
    weights=np.repeat(np.array([1]*5+[.4]*5),3)
    for frame in range(len(target_t)):
        pos=object_T[frame,:3,3];quat=R.from_matrix(object_T[frame,:3,:3]).as_quat(scalar_first=True)
        d.qpos[26:]=np.r_[pos,quat]
        q0=qprev.copy()
        def residual(q):
            d.qpos[:26]=q;mujoco.mj_forward(m,d)
            error=(d.site_xpos[sites]-human[frame,hidx]).ravel()*weights
            pens=np.zeros(len(handgeoms))
            for c in d.contact:
                g1,g2=int(c.geom1),int(c.geom2)
                h=g1 if g2 in objectgeoms else g2 if g1 in objectgeoms else -1
                if h in handgeoms:
                    ix=np.searchsorted(handgeoms,h);pens[ix]=min(pens[ix],c.dist+.0005)
            return np.r_[error,0.002*(q-q0),5*pens]
        sol=least_squares(residual,q0,bounds=(lo+1e-6,hi-1e-6),max_nfev=150 if frame==0 else 50,
                          ftol=1e-5,xtol=1e-5,gtol=1e-5,diff_step=1e-4)
        qprev=sol.x;residual(qprev)
        references.append(d.qpos.copy());robot_points.append(d.site_xpos[sites].copy())
        errors.append(np.linalg.norm(d.site_xpos[sites]-human[frame,hidx],axis=1));evals.append(sol.nfev)
        penetrations.append(max([max(0,-c.dist) for c in d.contact if int(c.geom1) in objectgeoms or int(c.geom2) in objectgeoms]+[0]))
        if frame%10==0: print('retarget',frame,'tip mm',np.mean(errors[-1][:5])*1000,'penetration mm',penetrations[-1]*1000,flush=True)
    qpos=np.array(references);qvel=np.zeros((len(qpos),m.nv))
    for i in range(len(qpos)-1): mujoco.mj_differentiatePos(m,qvel[i],.02,qpos[i],qpos[i+1])
    qvel[-1]=qvel[-2]
    np.savez_compressed(out/'trajectory.npz',time_s=target_t-target_t[0],source_time_s=target_t,
                        qpos=qpos,qvel=qvel,ctrl=qpos[:,:26],object_pose=object_T,human_landmarks=human,
                        other_hand_landmarks=other,robot_keypoints=np.array(robot_points),keypoint_error_m=np.array(errors),penetration_m=penetrations,
                        source_to_sim=T,reference_wrist_pose=wrist_T)
    # Approximate hand proximity is a diagnostic, not a contact label.
    from scipy.spatial import cKDTree
    tree=cKDTree(mesh.vertices)
    distances={}
    for name,points in [('right',human),('left',other)]:
        local=np.einsum('tji,tkj->tki',object_T[:,:3,:3],points[:,:5]-object_T[:,None,:3,3])
        distances[name]=tree.query(local.reshape(-1,3))[0].reshape(-1,5).min(1)
    np.savez(out/'hand_object_distances.npz',**distances)
    report=dict(source='HOT3D-Clips/train_quest3',clip_id=args.clip,object_id=args.object,object_name=object_name,object_uid=object_uid,
                sequence_id=infos[0]['sequence_id'],source_frame_range=[args.start,args.end],original_frames=150,
                frames_50hz=len(target_t),duration_s=float(target_t[-1]-target_t[0]),source_hz=30,target_hz=50,
                units='metres, radians, seconds; source timestamps preserved as integer ns',
                rotations='source wxyz active T_world_from_local; Slerp; MuJoCo wxyz',
                source_world_up='Y inferred from Quest/tabletop geometry, not a gravity measurement',floor_y_m=floor,floor_y_std_m=float(floor_samples.std()),
                floor_fit='lower envelope of original object vertices, minus 0.2 mm; table is not annotated',
                floor_offset_from_prefix_m=floor-prefix_floor,
                transformation=T.tolist(),mesh_extent_m=mesh.extents.tolist(),collision_parts=len(parts),
                object_mass_kg=.04,mass_source='pilot assumption, not measured',
                fingertip_error_median_m=float(np.median(np.array(errors)[:,:5])),fingertip_error_p95_m=float(np.quantile(np.array(errors)[:,:5],.95)),
                fingertip_error_max_m=float(np.max(np.array(errors)[:,:5])),
                frames_with_tip_error_over_15mm=int(np.any(np.array(errors)[:,:5]>.015,axis=1).sum()),
                unreachable_rule='Any fingertip error over 15 mm marks a geometrically infeasible frame for this pilot; not a certified reachability proof.',
                resampled_last_source_frame=float(target_t[-1]*30),
                penetration_max_m=float(np.max(penetrations)),penetration_p95_m=float(np.quantile(penetrations,.95)),
                joint_limit_violations=int(np.sum((qpos[:,:26]<lo)|(qpos[:,:26]>hi))),
                other_hand_min_vertex_distance_m=float(distances['left'].min()),
                selected_hand_median_min_vertex_distance_m=float(np.median(distances['right'])),
                object_translation_extent_m=np.ptp(object_T[:,:3,3],axis=0).tolist(),
                hand_toolkit_commit='c521930d966c871977aaa4d15f33bfd4b3781f09',
                exclusions='frames outside the selected single-hand subclip retained in raw_reference.npz; no missing frames interpolated',
                status='retargeted; physical qualification is separate')
    (out/'processing_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
