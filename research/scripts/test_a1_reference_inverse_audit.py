import unittest
import numpy as np,mujoco
from a1_reference_inverse_audit import inverse_snapshot,envelope


class InverseContracts(unittest.TestCase):
 def test_free_body_gravity_and_free_fall(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>');d=mujoco.MjData(m)
  q=d.qpos.copy();v=np.zeros(6);a=np.zeros(6)
  f,_=inverse_snapshot(m,d,q,v,a);np.testing.assert_allclose(f,[0,0,19.62,0,0,0],atol=1e-10)
  a[2]=-9.81;f,_=inverse_snapshot(m,d,q,v,a);np.testing.assert_allclose(f,0,atol=1e-10)
 def test_passive_damping_removed_once(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><body><joint type="slide" damping="2"/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>');d=mujoco.MjData(m)
  f,_=inverse_snapshot(m,d,np.zeros(1),np.array([3.]),np.zeros(1));np.testing.assert_allclose(f,[6.],atol=1e-10)
 def test_contact_forward_inverse_without_integration(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><option tolerance="1e-12" iterations="100"/><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .095"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>');d=mujoco.MjData(m)
  mujoco.mj_forward(m,d);q=d.qpos.copy();v=d.qvel.copy();a=d.qacc.copy();f,_=inverse_snapshot(m,d,q,v,a)
  self.assertGreater(d.ncon,0);np.testing.assert_allclose(f,0,atol=1e-8);np.testing.assert_array_equal(q,d.qpos)
 def test_force_and_joint_limit_envelope(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><body><joint name="j" type="slide" actuatorfrcrange="-2 2"/><geom type="sphere" size=".1" mass="1"/></body></worldbody><actuator><position joint="j" kp="10" kv="2" ctrlrange="-1 1" forcerange="-3 3"/></actuator></mujoco>')
  lo,hi=envelope(m,np.array([.2]),np.array([.5]));np.testing.assert_allclose([lo[0],hi[0]],[-2,2])
  lo,hi=envelope(m,np.array([0.]),np.array([10.]));np.testing.assert_allclose([lo[0],hi[0]],[-2,-2])
  for q,v in [(0.,0.),(.2,.5),(0.,10.)]:
   d=mujoco.MjData(m);d.qpos[0]=q;d.qvel[0]=v;lo,hi=envelope(m,d.qpos,d.qvel)
   for u,want in [(-1.,lo[0]),(1.,hi[0])]:
    d.ctrl[0]=u;mujoco.mj_forward(m,d);self.assertAlmostEqual(float(d.qfrc_actuator[0]),want,places=9)
 def test_discrete_inverse_rejected(self):
  m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>');m.opt.enableflags|=int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE);d=mujoco.MjData(m)
  with self.assertRaises(AssertionError):inverse_snapshot(m,d,d.qpos.copy(),d.qvel.copy(),np.zeros(m.nv))


if __name__=='__main__':unittest.main()
