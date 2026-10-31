import unittest
import numpy as np
from dr_factor_plan import GROUPS,make_plan

def fixture():
 a=np.zeros(276,np.float32);a[136:139]=[.05,.7,.6];a[142:145]=[.001,.001,.001]
 for h in (0,68):a[h+20:h+40]=2;a[h+40:h+60]=.1;a[h+60:h+62]=[1,0];a[h+62:h+68]=.02
 b=a.copy();b[136]=.06;b[137:139]=[.8,.7];b[139:142]=.005;b[142:145]*=1.2
 for h in (0,68):b[h:h+20]=.002;b[h+20:h+60]*=1.05;b[h+60:h+62]=[2,1]
 return {280043:a,280044:b}

class Checks(unittest.TestCase):
 def test_full_plan_is_bidirectional_bounded_and_reproducible(self):
  p=make_plan(fixture());self.assertEqual(p,make_plan(fixture()));self.assertEqual(len(p['conditions']),12);self.assertEqual(len(p['execution_order']),36)
  for row in p['conditions']:self.assertEqual(sum(x['condition']==row['id'] for x in p['execution_order']),3)
 def test_only_registered_factor_changes_and_input_not_mutated(self):
  ps=fixture();before={k:v.copy() for k,v in ps.items()};p=make_plan(ps)
  for r in p['conditions']:
   c=np.array(r['context'],np.float32);base=ps[r['base_seed']]
   self.assertEqual(np.flatnonzero(c!=base).tolist(),r['changed_indices'])
   if r['group']:
    ids=list(GROUPS[r['group']]);np.testing.assert_array_equal(c[ids],ps[r['donor_seed']][ids]);self.assertTrue(set(r['changed_indices'])<=set(ids))
  for k in ps:np.testing.assert_array_equal(ps[k],before[k])
 def test_development_seed_rejected(self):
  p=fixture();p[280046]=p.pop(280044)
  with self.assertRaises(ValueError):make_plan(p)
 def test_unchanged_factors_not_fabricated(self):
  p=fixture();p[280044]=p[280043].copy();self.assertEqual(len(make_plan(p)['conditions']),2)
 def test_fixed_tau_bps_and_inertia_frame_cannot_change(self):
  for i in (62,130,145,148,275):
   p=fixture();p[280044][i]+=.001
   with self.assertRaises(ValueError):make_plan(p)
 def test_invalid_physical_packet_delay_and_nonfinite_rejected(self):
  for i,v in [(136,-1),(142,1),(60,1),(0,.02),(20,0),(137,1.3),(138,.2),(10,float('nan'))]:
   p=fixture();p[280044][i]=v
   with self.assertRaises(ValueError):make_plan(p)

if __name__=='__main__':unittest.main()
