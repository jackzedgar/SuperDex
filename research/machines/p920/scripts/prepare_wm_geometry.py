"""CPU-only scalar SDF grid for the SAME five convex collision pieces."""
import time,json,hashlib
import numpy as np,mujoco,trimesh
from scipy.spatial import ConvexHull
from exploration_common import RUN,DATA,write_json

started=time.perf_counter();out=RUN/'wm';out.mkdir(exist_ok=True);m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
meshes=[trimesh.load(f,force='mesh',process=False) for f in sorted(DATA.glob('object_collision_*.obj'))]
points=np.concatenate([x.vertices for x in meshes]);lo=points.min(0)-.05;hi=points.max(0)+.05;shape=np.ceil((hi-lo)/.003).astype(int)+1;spacing=(hi-lo)/(shape-1)
grid=np.stack(np.meshgrid(*[np.linspace(lo[i],hi[i],shape[i]) for i in range(3)],indexing='ij'),-1).reshape(-1,3);sdf=np.full(len(grid),np.inf)
for mesh_id,mesh in enumerate(meshes):
    hull=ConvexHull(mesh.vertices);planes=hull.equations
    for k in range(0,len(grid),2048):
        p=grid[k:k+2048];_,distance,_=trimesh.proximity.closest_point(mesh,p);inside=(p@planes[:,:3].T+planes[:,3]<=1e-9).all(1);distance[inside]*=-1;sdf[k:k+len(p)]=np.minimum(sdf[k:k+len(p)],distance)
    print('Completed collision piece',mesh_id+1,'of',len(meshes),'CPU seconds',time.perf_counter()-started,flush=True)
np.savez_compressed(out/'sdf_grid.npz',values=sdf.reshape(shape).astype(np.float32),lo=lo,hi=hi,spacing=spacing)
body_ids=np.array([[m.body(f'right_finger{i}_link{j}').id for j in range(1,5)] for i in range(1,6)])
joints=m.body_jntadr[body_ids];tips=[m.site(f'tip_{i}').id for i in range(1,6)]
np.savez(out/'kinematics.npz',body_pos=m.body_pos[body_ids],body_quat=m.body_quat[body_ids],body_ipos=m.body_ipos[body_ids],joint_axis=m.jnt_axis[joints],joint_pos=m.jnt_pos[joints],joint_qpos0=m.qpos0[m.jnt_qposadr[joints]],tip_pos=m.site_pos[tips],palm_point=m.body_ipos[m.body('wrist').id],wrist_base_pos=m.body_pos[1],wrist_base_quat=m.body_quat[1])
write_json(out/'geometry_manifest.json',dict(source='Union of baseline CoACD collision meshes; no visual-mesh replacement or physics change.',sign='negative inside each convex piece, union=min; Euclidean triangle distance, trilinear scalar interpolation',grid_shape=shape.tolist(),spacing_m=spacing.tolist(),bounds_m=[lo.tolist(),hi.tolist()],queries='5 existing fingertip sites + centers of mass of links 1/2/3 for each finger + wrist/palm COM: 21 total.',outside='Scalar value at box projection plus Euclidean distance to projection; gradient differentiates that SAME scalar, including tangential derivatives.',cpu_seconds=time.perf_counter()-started,meshes=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(DATA.glob('object_collision_*.obj'))]))
print('Geometry complete',shape,'points',len(grid),'CPU seconds',time.perf_counter()-started,flush=True)
