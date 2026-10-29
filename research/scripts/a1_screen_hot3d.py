"""CPU source/reference screening only. No learned policy outcomes enter selection."""
import argparse,gzip,json,sys,time
from pathlib import Path
import numpy as np,torch,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from a1_data_sources import ROOT,RUN,write
sys.path.insert(0,str(ROOT/'references/hand_tracking_toolkit'))
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json,UmeTrackHandPose,forward_kinematics
torch.set_num_threads(1)
def transform(p):
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(p['quaternion_wxyz'],scalar_first=True).as_matrix();t[:3,3]=p['translation_xyz'];return t
def one(path):
    d=json.loads(gzip.decompress(path.read_bytes()));shape=from_json(d['__hand_shapes.json__']['umetrack'])
    indices=sorted(int(k[:6]) for k in d if k.endswith('.info.json'))
    kp=np.full((len(indices),2,20,3),np.nan);wr=np.full((len(indices),2,4,4),np.nan)
    times=np.array([d[f'{i:06d}.info.json']['ref_timestamp_ns'] for i in indices],dtype=np.int64)
    objects={}
    for k,i in enumerate(indices):
        hs=d[f'{i:06d}.hands.json']
        for side_id,side in enumerate(['right','left']):
            if not hs.get(side) or not hs[side].get('umetrack_pose'):continue
            h=hs[side]['umetrack_pose'];w=transform(h['T_world_from_wrist'])
            pose=UmeTrackHandPose(torch.tensor(int(side=='right')),torch.tensor(h['joint_angles'],dtype=torch.float32),torch.tensor(w,dtype=torch.float32))
            with torch.no_grad():pts,_,_=forward_kinematics(pose,shape,requires_mesh=False)
            kp[k,side_id]=pts.numpy();wr[k,side_id]=w
        for obj_id,obs in d[f'{i:06d}.objects.json'].items():
            if len(obs)!=1:continue
            ob=obs[0]
            row=objects.setdefault(obj_id,{'name':ob['object_name'],'uid':ob['object_uid'],'poses':np.full((len(indices),4,4),np.nan)})
            row['poses'][k]=transform(ob['T_world_from_object'])
    rows=[]
    for oid,ob in objects.items():
        meshpath=ROOT/f'data/raw/hot3d/object_models_eval/obj_{int(oid):06d}.glb'
        sc=trimesh.load(meshpath);mesh=sc.to_geometry() if isinstance(sc,trimesh.Scene) else sc;query=cKDTree(mesh.vertices)
        poses=ob['poses'];valid=np.isfinite(poses).all((1,2))
        dist=np.full((len(indices),2,5),np.inf);rel=np.full((len(indices),2,3,3),np.nan)
        for h in range(2):
            ok=valid & np.isfinite(kp[:,h]).all((1,2))
            pts=np.einsum('nij,nkj->nki',poses[ok,:3,:3].transpose(0,2,1),kp[ok,h,:5]-poses[ok,None,:3,3])
            dist[ok,h]=query.query(pts)[0]
            rel[ok,h]=wr[ok,h,:3,:3].transpose(0,2,1)@poses[ok,:3,:3]
        near=np.sort(dist,axis=-1)[:,:,1]<.025 # two fingertips near the visual surface; screening proxy only
        for start,end in [(0,len(indices)-1)]+[(s,s+59) for s in range(0,len(indices)-59,30)]:
            sl=slice(start,end+1);duration=(times[end]-times[start])*1e-9
            if not valid[sl].all():continue
            p=poses[sl,:3,3];extent=float(np.linalg.norm(np.ptp(p,axis=0)))
            angle=float(np.max((Rotation.from_matrix(poses[start,:3,:3]).inv()*Rotation.from_matrix(poses[sl,:3,:3])).magnitude()))
            both=float(near[sl].all(1).mean());freq=near[sl].mean(0);dominant=int(freq.argmax())
            relative=np.nan
            if np.isfinite(rel[sl,dominant]).all():relative=float(np.max((Rotation.from_matrix(rel[start,dominant]).inv()*Rotation.from_matrix(rel[sl,dominant])).magnitude()))
            n=end-start+1;early=near[start:start+max(5,n//5)].mean(0);late=near[end-max(5,n//5)+1:end+1].mean(0)
            transfer=bool((early[0]>.7 and early[1]<.3 and late[1]>.7 and late[0]<.3 or early[1]>.7 and early[0]<.3 and late[0]>.7 and late[1]<.3) and both>.02)
            kinds=[]
            if extent>.03 and freq[dominant]>.85 and freq[1-dominant]<.25 and relative<np.deg2rad(20):kinds.append('stable_grasp_transport_candidate')
            if freq[dominant]>.7 and freq[1-dominant]<.3 and relative>np.deg2rad(25):kinds.append('in_hand_reorientation_candidate')
            if transfer:kinds.append('bimanual_handover_candidate')
            rows.append({'clip':path.parent.name,'object_id':int(oid),'object_uid':ob['uid'],'object_name':ob['name'],'start_frame':indices[start],'end_frame':indices[end],'duration_s':duration,'translation_extent_mm':extent*1000,'object_rotation_extent_deg':np.rad2deg(angle),'wrist_relative_rotation_extent_deg':None if not np.isfinite(relative) else np.rad2deg(relative),'right_two_tip_proximity_fraction':float(freq[0]),'left_two_tip_proximity_fraction':float(freq[1]),'bimanual_proximity_fraction':both,'early_hand_proximity':early.tolist(),'late_hand_proximity':late.tolist(),'candidate_strata':kinds,'tool_use_status':'Requires source activity evidence/visual annotation; proximity alone is insufficient','mass_status':'Not measured or provided by source metadata','source_only_no_physics_or_policy_claim':True})
    return rows
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='hot3d_reference_screen.json');a=parser.parse_args()
    started=time.monotonic();rows=[];errors=[]
    for p in sorted((ROOT/'data/a1_sources/hot3d').glob('*/annotations.json.gz')):
        try:r=one(p);rows.extend(r);print(p.parent.name,'windows',len(r),'candidates',sum(bool(x['candidate_strata']) for x in r),flush=True)
        except Exception as e:errors.append({'clip':p.parent.name,'error':repr(e)});print(p,repr(e),flush=True)
    summary={'cpu_wall_seconds':time.monotonic()-started,'clips':len(set(r['clip'] for r in rows)),'objects':len(set(r['object_id'] for r in rows)),'candidate_windows':sum(bool(r['candidate_strata']) for r in rows),'candidate_counts':{k:sum(k in r['candidate_strata'] for r in rows) for k in ['stable_grasp_transport_candidate','in_hand_reorientation_candidate','bimanual_handover_candidate']},'scope':'Unfrozen candidate screening only; surface vertex-distance proxy is not physical contact; tool use not inferred from object name.','errors':errors,'rows':rows}
    write(RUN/'public_catalog'/a.output,summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
