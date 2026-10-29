"""Exact closest-triangle distance using Warp's CPU BVH; no GPU allocation."""
import numpy as np,warp as wp,trimesh
from scipy.spatial import ConvexHull
@wp.kernel
def distances(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),output:wp.array(dtype=float)):
 i=wp.tid();p=points[i]
 q=wp.mesh_query_point_no_sign(mesh,p,2.0)
 if q.result:
  nearest=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
  output[i]=wp.length(p-nearest)
 else:output[i]=1.0e9
def signed_distance(mesh,points,check=True):
 mesh=mesh.copy();mesh.fix_normals()
 bvh=wp.Mesh(points=wp.array(np.asarray(mesh.vertices,dtype=np.float32),dtype=wp.vec3,device='cpu'),indices=wp.array(np.asarray(mesh.faces,dtype=np.int32).ravel(),dtype=wp.int32,device='cpu'))
 query=wp.array(np.asarray(points,dtype=np.float32),dtype=wp.vec3,device='cpu')
 result=wp.empty(len(points),dtype=float,device='cpu')
 wp.launch(distances,len(points),inputs=[bvh.id,query,result],device='cpu')
 value=result.numpy()
 planes=ConvexHull(mesh.vertices).equations
 for start in range(0,len(points),8192):
  inside=(points[start:start+8192]@planes[:,:3].T+planes[:,3]<=1e-9).all(1)
  block=value[start:start+8192];block[inside]*=-1
 assert np.isfinite(value).all() and abs(value).max()<2
 error=None
 if check:
  ids=np.random.default_rng(280043).choice(len(points),min(256,len(points)),replace=False);p=points[ids]
  _,ref,_=trimesh.proximity.closest_point(mesh,p);planes=ConvexHull(mesh.vertices).equations
  ref[(p@planes[:,:3].T+planes[:,3]<=1e-9).all(1)]*=-1
  error=float(np.max(np.abs(ref-value[ids])));assert error<2e-5,error
 return value,error
