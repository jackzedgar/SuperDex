"""One additional source clip, selected on reference geometry before policy trials."""
import sys,json,tarfile
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from exploration_common import ROOT,RUN,write_json
sys.path.insert(0,str(ROOT/'references/hand_tracking_toolkit'))
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json,UmeTrackHandPose,forward_kinematics
torch.set_num_threads(1)
def transform(p):
    out=np.eye(4);out[:3,:3]=Rotation.from_quat(p['quaternion_wxyz'],scalar_first=True).as_matrix();out[:3,3]=p['translation_xyz'];return out
path=ROOT/'data/raw/hot3d/train_quest3/clip-000001.tar';objects={};hands={'right':[],'left':[]};names={}
with tarfile.open(path) as archive:
    shape=from_json(json.load(archive.extractfile('__hand_shapes.json__'))['umetrack'])
    for i in range(150):
        obs=json.load(archive.extractfile(f'{i:06d}.objects.json'));hs=json.load(archive.extractfile(f'{i:06d}.hands.json'))
        for key,value in obs.items():
            names[key]=value[0]['object_name'];objects.setdefault(key,[]).append(transform(value[0]['T_world_from_object']))
        for side in ['right','left']:
            h=hs[side]['umetrack_pose'];pose=UmeTrackHandPose(torch.tensor(int(side=='right')),torch.tensor(h['joint_angles'],dtype=torch.float32),torch.tensor(transform(h['T_world_from_wrist']),dtype=torch.float32))
            with torch.no_grad():kp,_,_=forward_kinematics(pose,shape,requires_mesh=False)
            hands[side].append(kp.numpy())
    for k in [20,75,130]:(RUN/f'backup_source_{k:03d}.jpg').write_bytes(archive.extractfile(f'{k:06d}.image_1201-1.jpg').read())
hands={k:np.array(v) for k,v in hands.items()};rows=[];windows=[]
for key,poses in objects.items():
    p=np.array(poses)[:,:3,3];right=np.linalg.norm(hands['right'][:,:5]-p[:,None],axis=-1).min(1);left=np.linalg.norm(hands['left'][:,:5]-p[:,None],axis=-1).min(1)
    rows.append(dict(object_id=key,name=names[key],position_extent_mm=(np.ptp(p,axis=0)*1000).tolist(),right_tip_to_origin_min_mm=float(right.min()*1000),right_tip_to_origin_median_mm=float(np.median(right)*1000),left_tip_to_origin_median_mm=float(np.median(left)*1000)))
    if key=='4':continue
    for start in range(0,120,5):
        sl=slice(start,start+31);motion=float(np.linalg.norm(np.ptp(p[sl],axis=0)));r=float(np.mean(right[sl]<.08));l=float(np.mean(left[sl]<.08))
        windows.append(dict(object_id=key,name=names[key],frames=[start,start+30],duration_s=1.,position_extent_mm=motion*1000,right_proximity_fraction=r,left_proximity_fraction=l,median_right_tip_to_origin_mm=float(np.median(right[sl])*1000),candidate=bool(motion>.02 and r>.8 and l<.2)))
candidates=sorted([w for w in windows if w['candidate']],key=lambda w:(w['median_right_tip_to_origin_mm'],-w['position_extent_mm']))
write_json(RUN/'backup_source_screen.json',dict(clip='HOT3D train_quest3/clip-000001',source_selection_before_new_task_policy_trials=True,object_rows=rows,top_reference_candidates=candidates[:5],status='candidate_requires_surface_and_retarget_check' if candidates else 'No alternate-object window meets prespecified source motion/right-hand/left-hand proximity screen in this one clip.',limitations='Object origin proximity is only a coarse source-quality screen, not annotated contact or physical success. No task defined from policy success.'))
print(json.dumps(dict(rows=rows,candidates=candidates[:5]),indent=2))
