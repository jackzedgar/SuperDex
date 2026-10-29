"""Decode immutable source segments, preserving poses and explicit support assumptions."""
import ast,gzip,json,sys
from pathlib import Path
import numpy as np,torch,trimesh
from scipy.spatial.transform import Rotation,Slerp
from a1_data_sources import ROOT,RUN,safe_pickle
from a1_mano import ManoDecoder
from a1_screen_hot3d import transform,from_json,UmeTrackHandPose,forward_kinematics

A_Y_UP_TO_Z_UP=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
def mesh_for(source,oid):
    if source=='hot3d':path=ROOT/f'data/raw/hot3d/object_models_eval/obj_{int(oid):06d}.glb'
    else:path=ROOT/f'data/a1_sources/oakink2/object_repair/align_ds/{oid}/model.obj'
    mesh=trimesh.load(path,process=False)
    if isinstance(mesh,trimesh.Scene):mesh=mesh.to_geometry()
    if not .002<mesh.extents.max()<1.5:raise ValueError(('Source units or object size',oid,mesh.extents))
    return mesh,path

def support_height(all_poses,source,target_mesh,target_poses):
    candidates=[]
    for oid,poses in all_poses.items():
        valid=poses[np.isfinite(poses).all((1,2))]
        if len(valid)<3:continue
        samples=valid[np.linspace(0,len(valid)-1,5,dtype=int)]
        motion=np.linalg.norm(np.ptp(samples[:,:3,3],axis=0))
        angle=np.max((Rotation.from_matrix(samples[0,:3,:3]).inv()*Rotation.from_matrix(samples[:,:3,:3])).magnitude())
        if motion>.01 or angle>.05:continue
        try:mesh,_=mesh_for(source,oid)
        except (FileNotFoundError,ValueError):continue
        y=[float((mesh.vertices@p[:3,:3].T+p[:3,3])[:,1].min()) for p in samples]
        if np.ptp(y)<.008:candidates.append(dict(object_id=str(oid),height_m=float(np.median(y)),motion_m=float(motion)))
    if candidates:
        heights=np.array([c['height_m'] for c in candidates])
        counts=np.array([np.sum(np.abs(heights-y)<.01) for y in heights])
        center=heights[np.argmax(counts)];chosen=heights[np.abs(heights-center)<.01]
        return float(np.median(chosen))-.0002,dict(method='Dominant stationary source-object lower-surface cluster; inferred, not annotated table',candidates=candidates,cluster_count=int(len(chosen)))
    sample=target_poses[np.linspace(0,len(target_poses)-1,min(100,len(target_poses)),dtype=int)]
    floor=min(float((target_mesh.vertices@p[:3,:3].T+p[:3,3])[:,1].min()) for p in sample)-.0002
    return floor,dict(method='Target source lower envelope; unverified support height',candidates=[],requires_manual_or_physical_review=True)

def hot3d(row):
    packet=ROOT/f"data/a1_sources/hot3d/{row['clip']}/annotations.json.gz"
    d=json.loads(gzip.decompress(packet.read_bytes()));shape=from_json(d['__hand_shapes.json__']['umetrack'])
    indices=sorted(int(k[:6]) for k in d if k.endswith('.info.json'))
    ns=np.array([d[f'{i:06d}.info.json']['ref_timestamp_ns'] for i in indices],dtype=np.int64)
    t=(ns-ns[0])*1e-9;obj={};wr=np.full((len(t),2,4,4),np.nan);kp=np.full((len(t),2,15,3),np.nan)
    for n,i in enumerate(indices):
        for oid,rows in d[f'{i:06d}.objects.json'].items():
            if len(rows)!=1:continue
            obj.setdefault(oid,np.full((len(t),4,4),np.nan))[n]=transform(rows[0]['T_world_from_object'])
        for h,side in enumerate(['right','left']):
            hands=d[f'{i:06d}.hands.json'];info=(hands.get(side) or {}).get('umetrack_pose')
            if info is None:continue
            wr[n,h]=transform(info['T_world_from_wrist'])
            pose=UmeTrackHandPose(torch.tensor(int(side=='right')),torch.tensor(info['joint_angles'],dtype=torch.float32),torch.tensor(wr[n,h],dtype=torch.float32))
            with torch.no_grad():pts,_,_=forward_kinematics(pose,shape,requires_mesh=False)
            kp[n,h]=pts.numpy()[[0,1,2,3,4,6,9,12,15,18,5,8,11,14,17]]
    keep=(np.array(indices)>=row['start_frame'])&(np.array(indices)<=row['end_frame'])
    mesh,path=mesh_for('hot3d',row['object_id']);object_pose=obj[str(row['object_id'])]
    floor,proof=support_height(obj,'hot3d',mesh,object_pose)
    return dict(time=t[keep],wrist=wr[keep],landmarks=kp[keep],object_pose=object_pose[keep],mesh=mesh,mesh_path=path,source_floor=floor,support_evidence=proof,source_file=packet,source_frame_ids=np.array(indices)[keep])

def oakink(row):
    source=ROOT/'data/raw/oakink2/anno_preview'/f"{row['sequence']}.pkl"
    raw=safe_pickle(source)
    intervals=ast.literal_eval(row['interval'])
    chosen=intervals[1];assert chosen is not None
    oid=row['program']['obj_list_rh'][0]
    frames=np.array([i for i in sorted(raw['raw_mano']) if chosen[0]<=i<chosen[1] and i in raw['obj_transf'][oid]])
    if len(frames)<40:raise ValueError('Source interval too short or incomplete')
    assert np.max(np.diff(frames))<=3, 'Unexplained source mocap gaps'
    wr=np.zeros((len(frames),2,4,4));kp=np.zeros((len(frames),2,15,3))
    for h,(side,prefix) in enumerate([('right','rh'),('left','lh')]):
        layer=ManoDecoder(side)
        q=torch.cat([raw['raw_mano'][int(i)][prefix+'__pose_coeffs'] for i in frames]).float()
        beta=torch.cat([raw['raw_mano'][int(i)][prefix+'__betas'] for i in frames]).float()
        tsl=torch.cat([raw['raw_mano'][int(i)][prefix+'__tsl'] for i in frames]).float()
        joints=[]
        for start in range(0,len(q),64):
            with torch.no_grad():out=layer(q[start:start+64],beta[start:start+64],tsl[start:start+64])
            joints.append(out['joints'].numpy())
        kp[:,h]=np.concatenate(joints)[:,[4,8,12,16,20,2,6,10,14,18,1,5,9,13,17]]
        wr[:,h,:3,:3]=Rotation.from_quat(q[:,0].numpy(),scalar_first=True).as_matrix();wr[:,h,:3,3]=tsl.numpy();wr[:,h,3,3]=1
    poses=np.array([raw['obj_transf'][oid][int(i)] for i in frames])
    mesh,path=mesh_for('oakink2',oid)
    all_poses={o:np.array([p for _,p in sorted(rows.items())]) for o,rows in raw['obj_transf'].items()}
    floor,proof=support_height(all_poses,'oakink2',mesh,poses)
    # Coordinate inference is cross-checked using the SMPL-X body-up direction.
    qs=np.concatenate([v['world_rot'].numpy().reshape(1,4) for v in raw['raw_smplx'].values()])
    upright=Rotation.from_quat(qs,scalar_first=True).as_matrix()[:,:,1]
    proof['median_body_local_y_dot_world_y']=float(np.median(upright[:,1]))
    if proof['median_body_local_y_dot_world_y']<.7:raise ValueError('Source upright direction requires review')
    return dict(time=frames/120.,wrist=wr,landmarks=kp,object_pose=poses,mesh=mesh,mesh_path=path,source_floor=floor,support_evidence=proof,source_file=source,source_frame_ids=frames)

def resample(row):
    d=hot3d(row) if row['source']=='hot3d' else oakink(row)
    t=d['time'];assert len(t)>1 and np.all(np.diff(t)>0)
    query=t[0]+np.arange(int(np.floor((t[-1]-t[0])/.02))+1)*.02
    origin=d['object_pose'][0,:3,3].copy();origin[1]=d['source_floor']
    T=np.eye(4);T[:3,:3]=A_Y_UP_TO_Z_UP;T[:3,3]=-A_Y_UP_TO_Z_UP@origin
    def pose_interp(p):
        out=np.tile(np.eye(4),(len(query),1,1))
        out[:,:3,:3]=Slerp(t,Rotation.from_matrix(p[:,:3,:3]))(query).as_matrix()
        for c in range(3):out[:,c,3]=np.interp(query,t,p[:,c,3])
        return T[None]@out
    sides=row['hands'];wr=np.tile(np.eye(4),(len(query),2,1,1));kp=np.zeros((len(query),2,15,3))
    for h,side in enumerate(['right','left']):
        if side not in sides:continue
        assert np.isfinite(d['wrist'][:,h]).all() and np.isfinite(d['landmarks'][:,h]).all(), 'Missing active hand reference'
        wr[:,h]=pose_interp(d['wrist'][:,h])
        points=np.stack([np.interp(query,t,d['landmarks'][:,h,j,c]) for j in range(15) for c in range(3)],-1).reshape(-1,15,3)
        kp[:,h]=points@T[:3,:3].T+T[:3,3]
    assert np.isfinite(d['object_pose']).all()
    d.update(resampled_time=query-query[0],source_time=query,object=pose_interp(d['object_pose']),wrist_50=wr,keypoints_50=kp,source_to_sim=T,trim_at_end_s=float(t[-1]-query[-1]))
    return d
