"""CPU checks directly relevant to the new differentiable geometry implementation."""
import time,json
import numpy as np,torch,mujoco,trimesh
from scipy.spatial import ConvexHull
from exploration_common import RUN,DATA,write_json
from world_model_mini import Geometry

torch.set_num_threads(1);g=Geometry(RUN/'wm').double();m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));d=mujoco.MjData(m);ref=np.load(DATA/'trajectory.npz')['qpos'];rng=np.random.default_rng(23);max_fk=0
for i in range(20):
    q=ref[i*2].copy();q[6:26]+=rng.normal(0,.01,20);d.qpos[:]=q;mujoco.mj_forward(m,d)
    world=[d.site_xpos[m.site(f'tip_{j}').id] for j in range(1,6)]
    for depth in [1,2,3]:
        for finger in range(1,6):world.append(d.xipos[m.body(f'right_finger{finger}_link{depth}').id])
    world.append(d.xipos[m.body('wrist').id]);obj=m.body('object').id;truth=(np.array(world)-d.xpos[obj])@d.xmat[obj].reshape(3,3)
    actual=g.points(torch.tensor(q[None],dtype=torch.float64)).detach().numpy()[0];max_fk=max(max_fk,float(np.max(abs(actual-truth))))
assert max_fk<2e-6,('FK differs from MuJoCo',max_fk)
p=g.grid_lo+(g.grid_hi-g.grid_lo)*torch.tensor(rng.uniform(.1,.9,(10,3)),dtype=torch.float64);p[0,0]=g.grid_hi[0]+.03;p.requires_grad_();v,gradient=g.sdf(p);autograd=torch.autograd.grad(v.sum(),p)[0];err=float((gradient-autograd).abs().max());assert err<1e-7
assert torch.autograd.gradcheck(lambda x:g.sdf(x)[0],(p,),eps=1e-6,atol=1e-5,rtol=1e-4)
write_json(RUN/'wm/geometry_validation.json',dict(fk_max_coordinate_error_m=max_fk,analytic_scalar_gradient_max_difference=err,finite_difference_passed=True,device='CPU only',scope='FK against MuJoCo and derivatives of SAME trilinear scalar including outside-box extension; not a grasp gate.'))
print('Geometry checks passed',max_fk,err)
