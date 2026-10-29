"""CPU tests of the NEW integrator, not another grasp admission gate."""
import numpy as np,torch,mujoco
from world_model_residual import integrate_object,residual_normalization
from world_model_mini import qexp
from exploration_common import RUN,DATA,read_json,write_json

torch.set_num_threads(1);torch.manual_seed(23);s=torch.zeros(12,143,dtype=torch.float64);s[:,29:33]=qexp(torch.randn(12,3,dtype=torch.float64)*.3);s[:,26:29]=torch.randn(12,3,dtype=torch.float64);s[:,59:65]=torch.randn(12,6,dtype=torch.float64);d=torch.randn(12,64,dtype=torch.float64)*.04
p,q,v,w=integrate_object(s,d);g=s.new_tensor([0.,0.,-9.81]);v_expected=s[:,59:62]+.02*g+d[:,26:29]+d[:,58:61];p_expected=s[:,26:29]+.02*s[:,59:62]+.5*.02*.022*g+.016*d[:,26:29]+.006*d[:,58:61]
assert torch.max(abs(v-v_expected))<1e-12 and torch.max(abs(p-p_expected))<1e-12
m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));maxq=0.
for i in range(12):
    pos=s[i,:33].numpy().copy();vel=s[i,33:65].numpy().copy()
    for offset in [26,58]:
        for _ in range(5):
            vel[26:29]+=np.array([0,0,-9.81])*.002+d[i,offset:offset+3].numpy()/5;vel[29:32]+=d[i,offset+3:offset+6].numpy()/5;mujoco.mj_integratePos(m,pos,vel,.002)
    maxq=max(maxq,float(np.max(abs(pos[29:33]-q[i].numpy()))))
assert maxq<1e-12
z=torch.zeros_like(d);pz,qz,vz,wz=integrate_object(s,z);assert torch.max(abs(vz-s[:,59:62]-.02*g))<1e-12
d.requires_grad_(True);loss=sum(x.square().sum() for x in integrate_object(s,d));grad=torch.autograd.grad(loss,d)[0];assert torch.isfinite(grad).all() and (grad[:,26:32].abs().sum(0)>0).all() and (grad[:,58:64].abs().sum(0)>0).all()
write_json(RUN/'parameterization_check.json',dict(linear_velocity_formula_max_error=float((v-v_expected).abs().max()),linear_position_formula_max_error=float((p-p_expected).abs().max()),quaternion_vs_mujoco_integratePos_max_error=maxq,gravity_prior_check=True,all_object_output_channels_have_gradient=True,scope='CPU algebra and quaternion convention only; no claims of exact contact dynamics.'))
print('CPU parameterization checks passed')
