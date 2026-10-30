"""Measure source-mesh versus collision-surface discrepancy near the grasp."""
import numpy as np,mujoco,trimesh,json
from scipy.spatial.transform import Rotation
from grasp_lab import DATA,RUN,model_for,fixed_initials,write_json

m=model_for('baseline');d=mujoco.MjData(m);mesh=trimesh.load(DATA/'object_visual.obj',force='mesh');initial=fixed_initials();rows=[]
for trial,q in enumerate(initial['qpos']):
    d.qpos[:]=q;mujoco.mj_forward(m,d);r=Rotation.from_quat(q[29:],scalar_first=True).as_matrix()
    for c in d.contact:
        if {int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}!={1,2}:continue
        sign=1 if m.geom_contype[c.geom2]==2 else -1
        object_surface=c.pos+sign*c.frame[:3]*c.dist/2
        point=((object_surface-q[26:29])@r)[None]
        closest,dist,face=trimesh.proximity.closest_point_naive(mesh,point)
        outward_world=mesh.face_normals[face[0]]@r.T
        angle=np.rad2deg(np.arccos(np.clip(abs(np.dot(outward_world,c.frame[:3])),0,1)))
        rows.append(dict(trial=trial,geom=[int(c.geom1),int(c.geom2)],surface_discrepancy_m=float(dist[0]),unoriented_surface_normal_angle_deg=float(angle),point_world=object_surface.tolist()))
write_json(RUN/'collision_geometry_audit.json',dict(source_mesh_watertight=bool(mesh.is_watertight),mesh_extent_m=mesh.extents.tolist(),object_mass_kg=float(m.body_mass[m.body('object').id]),object_inertia_kg_m2=m.body_inertia[m.body('object').id].tolist(),mass_inertia_status='assumed, not measured',samples=rows,median_surface_discrepancy_m=float(np.median([r['surface_discrepancy_m'] for r in rows])),max_surface_discrepancy_m=max(r['surface_discrepancy_m'] for r in rows),median_normal_angle_deg=float(np.median([r['unoriented_surface_normal_angle_deg'] for r in rows]))))
print(json.dumps({k:v for k,v in json.loads((RUN/'collision_geometry_audit.json').read_text()).items() if k!='samples'},indent=2))
