import unittest
import numpy as np
from dr_model_packets import NAMES,mix_packets
from test_dr_factor_plan import fixture

class Checks(unittest.TestCase):
 def setUp(self):
  self.context=fixture();self.base={k:np.ones(s,np.float64) for k,s in dict(body_mass=(5,),body_inertia=(5,3),body_ipos=(5,3),body_iquat=(5,4),geom_friction=(6,3),actuator_gainprm=(10,10),actuator_biasprm=(10,10)).items()};self.donor={k:v*2 for k,v in self.base.items()}
 def mix(self,g):return mix_packets(self.base,self.donor,*self.context.values(),g,2,[1,3],5,[1,2,7,8])
 def test_parent_is_exact_and_unaliased(self):
  model,c=self.mix(None)
  for k in NAMES:np.testing.assert_array_equal(model[k],self.base[k]);self.assertFalse(np.shares_memory(model[k],self.base[k]))
  np.testing.assert_array_equal(c,self.context[280043])
 def test_object_packet_keeps_mass_com_inertia_together(self):
  m,c=self.mix('object_dynamics')
  for k in NAMES:
   expected=self.base[k].copy()
   if k in ('body_mass','body_inertia','body_ipos'):expected[2]=self.donor[k][2]
   np.testing.assert_array_equal(m[k],expected)
 def test_only_original_friction_components_changed(self):
  m,c=self.mix('friction');expected=self.base['geom_friction'].copy();expected[[1,3,5],0]=2
  np.testing.assert_array_equal(m['geom_friction'],expected)
  for k in NAMES:
   if k!='geom_friction':np.testing.assert_array_equal(m[k],self.base[k])
 def test_gain_pair_changes_only_finger_actuator_columns(self):
  m,c=self.mix('finger_gains')
  for k in NAMES:
   expected=self.base[k].copy()
   if k=='actuator_gainprm':expected[[1,2,7,8],0]=2
   if k=='actuator_biasprm':expected[[1,2,7,8],1:3]=2
   np.testing.assert_array_equal(m[k],expected)
 def test_offsets_change_context_only(self):
  m,c=self.mix('finger_offsets')
  for k in NAMES:np.testing.assert_array_equal(m[k],self.base[k])
  self.assertFalse(np.array_equal(c,self.context[280043]))
 def test_invalid_group_shape_and_nan_fail(self):
  with self.assertRaises(ValueError):self.mix('control_delay')
  self.donor['body_mass']=np.ones(6)
  with self.assertRaises(ValueError):self.mix('object_dynamics')
  self.donor['body_mass']=np.array([1,1,np.nan,1,1])
  with self.assertRaises(ValueError):self.mix('object_dynamics')

if __name__=='__main__':unittest.main()
