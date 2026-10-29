"""CPU collision SDFs and fixed BPS1024/PCA128 geometry context for full A1."""
import argparse,fcntl,hashlib,json,time
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import ConvexHull,cKDTree
from a1_data_sources import ROOT,RUN,write
from a1_hand_assets import ASSETS

GEOM=ROOT/'data/a1_geometry'
def grid(key):
    out=GEOM/'objects'/key;out.mkdir(parents=True,exist_ok=True)
    with (out/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'grid.json').exists():return
        start=time.monotonic();asset=ASSETS/'objects'/key
        meta=json.loads((asset/'asset.json').read_text())
        meshes=[trimesh.load(asset/f'collision_{i}.obj',force='mesh',process=False) for i in range(meta['parts'])]
        verts=np.concatenate([m.vertices for m in meshes]);lo=verts.min(0)-.05;hi=verts.max(0)+.05
        shape=np.maximum(64,np.ceil((hi-lo)/.003).astype(int)+1)
        assert np.max(shape)<=128,shape
        spacing=(hi-lo)/(shape-1)
        points=np.stack(np.meshgrid(*[np.linspace(lo[i],hi[i],shape[i]) for i in range(3)],indexing='ij'),-1).reshape(-1,3)
        sdf=np.full(len(points),np.inf)
        for i,mesh in enumerate(meshes):
            from a1_mesh_distance import signed_distance
            distance,check_error=signed_distance(mesh,points)
            sdf=np.minimum(sdf,distance)
            print('BVH check max error m',check_error,flush=True)
            print('SDF',key,'piece',i+1,'/',len(meshes),'seconds',time.monotonic()-start,flush=True)
        values=sdf.reshape(shape).astype(np.float32)
        grad=np.stack(np.gradient(values,*spacing),-1)
        np.savez_compressed(out/'sdf.npz',values=values.astype(np.float16),gradient=grad.astype(np.float16),lo=lo,hi=hi,spacing=spacing)
        report=dict(object_key=key,shape=shape.tolist(),spacing_m=spacing.tolist(),padding_m=.05,storage='fp16 scalar+gradient; runtime flattened table',sign='negative inside convex pieces; union=min',distance_backend='Warp CPU BVH, each piece checked on256fixed points against trimesh closest triangles + convex planes within20micrometers',interior_limit='Minimum of component signed distances is an implicit union field; in overlaps its negative magnitude is not globally exact Euclidean distance to the union boundary. Deep negative input is clamped at -3mm.',runtime_derivative='Exact derivative of interpolated scalar including outside tangential projection terms; stored finite-difference gradients are audit data.',source_asset=str(asset.relative_to(ROOT)),cpu_wall_seconds=time.monotonic()-start,sha256=hashlib.sha256((out/'sdf.npz').read_bytes()).hexdigest())
        write(out/'grid.json',report)

def bps():
    from scipy.spatial.transform import Rotation
    start=time.monotonic();GEOM.mkdir(parents=True,exist_ok=True)
    out=GEOM/'bps_pca128.npz';assert not out.exists(),'Preserve frozen geometry basis'
    rng=np.random.default_rng(280043);direction=rng.normal(size=(1024,3));direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    basis=direction*rng.uniform(size=(1024,1))**(1/3)*.2
    paths=[('hot3d_'+p.stem.removeprefix('obj_'),p) for p in sorted((ROOT/'data/raw/hot3d/object_models_eval').glob('*.glb'))]
    paths += [('oakink2_'+p.parent.name.replace('@','_'),p) for p in sorted((ROOT/'data/a1_sources/oakink2/object_repair/align_ds').glob('*/model.obj'))]
    vectors=[];keys=[];rows=[];augmentation=[]
    for key,p in paths:
        m=trimesh.load(p,process=False)
        if isinstance(m,trimesh.Scene):m=m.to_geometry()
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
        surface,_=trimesh.sample.sample_surface(m,12000,seed=seed)
        rotations=np.concatenate([np.eye(3)[None],Rotation.random(7,random_state=np.random.RandomState(seed)).as_matrix()])
        for R in rotations:
            vectors.append(cKDTree(surface@R.T).query(basis)[0]);augmentation.append(R)
        keys.append(key);rows.append(dict(object_key=key,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    x=np.array(vectors,dtype=np.float64);mean=x.mean(0);u,s,vt=np.linalg.svd(x-mean,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10));assert rank>=128,rank
    components=vt[:128];code=(x[::8]-mean)@components.T
    np.savez_compressed(out,basis=basis.astype(np.float32),pca_mean=mean.astype(np.float32),pca_components=components.astype(np.float32),object_keys=np.array(keys),object_codes=code.astype(np.float32),fit_rotations=np.array(augmentation,dtype=np.float32),singular_values=s.astype(np.float32))
    write(GEOM/'bps_pca128.json',dict(seed=280043,objects=len(keys),fit_observations=len(x),numerical_rank=rank,pca_dim=128,bps_points=1024,basis_radius_m=.2,fit_rule='Unlabelled pinned source geometry only; identity plus7fixed random object-frame orientations per geometry. No trajectories, policy results, or prediction targets. Identity-frame code retained for each actual object. Not an object-heldout evaluation.',explained_variance_fraction=float(np.sum(s[:128]**2)/np.sum(s**2)),cpu_wall_seconds=time.monotonic()-start,sources=rows,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    print('BPS complete',len(keys),rank,'seconds',time.monotonic()-start,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--object');p.add_argument('--bps',action='store_true');a=p.parse_args()
    if a.bps:bps()
    if a.object:grid(a.object)
if __name__=='__main__':main()
