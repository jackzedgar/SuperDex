import unittest
import numpy as np
from a1_control_witness_math import FixedCEM,knot_bias,rank

class Checks(unittest.TestCase):
    def test_limits_and_inactive_channels(self):
        c=FixedCEM([1,0]);p=c.population();self.assertLessEqual(float(abs(p).max()),.25);self.assertFalse(p[:,:,26:].any());self.assertFalse(p[:2].any())
    def test_deterministic_search_and_baseline(self):
        a,b=FixedCEM([1,1]),FixedCEM([1,1]);p=a.population();np.testing.assert_array_equal(p,b.population())
        rows=[dict(complete=True,controls=i+1,regular_success=False,strict_success=False,return_sum=i) for i in range(16)]
        a.update(p,rows,100);b.update(p,rows,100);q=a.population();np.testing.assert_array_equal(q,b.population());np.testing.assert_array_equal(q[1],p[15]);self.assertFalse(q[0].any())
    def test_interpolation_endpoints_bounds_and_mask(self):
        k=np.random.default_rng(3).uniform(-.25,.25,(4,52));np.testing.assert_allclose(knot_bias(k,0,98,[1,1]),k[0]);np.testing.assert_allclose(knot_bias(k,97,98,[1,1]),k[-1])
        for t in range(98):
            v=knot_bias(k,t,98,[0,1]);self.assertFalse(v[:26].any());self.assertLessEqual(float(abs(v).max()),.25)
    def test_success_rank_is_original_full_episode(self):
        row=dict(complete=True,controls=98,regular_success=True,strict_success=False,return_sum=0)
        fail=dict(complete=True,controls=98,regular_success=False,strict_success=False,return_sum=1e9)
        self.assertGreater(rank(row,98),rank(fail,98))
        row['controls']=20
        with self.assertRaises(AssertionError):rank(row,98)
    def test_partial_generation_cannot_train_search(self):
        c=FixedCEM([1,1]);p=c.population();rows=[dict(complete=False,controls=1,regular_success=False,strict_success=False,return_sum=0)]*16
        with self.assertRaises(AssertionError):c.update(p,rows,98)
if __name__=='__main__':unittest.main(verbosity=2)
