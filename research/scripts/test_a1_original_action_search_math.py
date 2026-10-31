import unittest
import numpy as np
from a1_original_action_search_math import FixedCEM,knot_bias,knot_times,torch_knot_bias,resample_prior,rank

class Checks(unittest.TestCase):
    def test_nonuniform_anchors_and_intervals(self):
        k=np.repeat(np.array([0.,1.,-1.,2.])[:,None],52,axis=1)
        for h in [98,1333]:
            for t,v in zip(knot_times(h),[0,1,-1,2]):np.testing.assert_allclose(knot_bias(k,t,h,[1,1]),v)
            for t,v in [(4,.5),(20,0),(32+(h-1-32)/2,.5)]:np.testing.assert_allclose(knot_bias(k,t,h,[1,1]),v)
    def test_torch_matches_independent_numpy_all_phases(self):
        import torch
        for h in [98,1333]:
            k=np.random.default_rng(16).uniform(-2,2,(h+2,4,52)).astype(np.float32)
            phases=np.arange(-1,h+1);active=np.tile(np.repeat([0.,1.],26),(len(phases),1))
            actual=torch_knot_bias(torch.from_numpy(k),torch.from_numpy(phases),torch.tensor(knot_times(h),dtype=torch.float32),torch.tensor(active,dtype=torch.float32)).numpy()
            expected=np.stack([knot_bias(row,int(t),h,[0,1]) for row,t in zip(k,phases)])
            np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=0)
    def test_prior_resampled_at_new_nodes_without_old_score(self):
        from a1_control_witness_math import knot_bias as legacy
        old=np.random.default_rng(12).uniform(-.25,.25,(4,52))
        for h in [98,1333]:
            new=resample_prior(old,h,[1,0])
            np.testing.assert_allclose(new,np.stack([legacy(old,t,h,[1,0]) for t in knot_times(h)]),atol=2e-8,rtol=0)
            c=FixedCEM([1,0],new);self.assertIsNone(c.best_score);np.testing.assert_array_equal(c.best,new)
    def test_population_bound_baseline_incumbent_mask(self):
        prior=np.zeros((4,52),np.float32);prior[:,:26]=.2;c=FixedCEM([1,0],prior);p=c.population()
        self.assertEqual(p.shape,(64,4,52));self.assertLessEqual(abs(p).max(),2);self.assertFalse(p[:,:,26:].any());self.assertFalse(p[0].any());np.testing.assert_array_equal(p[1],prior)
    def test_reproducibility_and_eight_elites(self):
        a=FixedCEM([1,1],np.zeros((4,52)));b=FixedCEM([1,1],np.zeros((4,52)))
        p=a.population();np.testing.assert_array_equal(p,b.population())
        rows=[dict(complete=True,controls=i+1,regular_success=False,strict_success=False,return_sum=i) for i in range(64)]
        r=a.update(p,rows,98);b.update(p,rows,98);self.assertEqual(r['elite_indices'],list(range(63,55,-1)))
        np.testing.assert_array_equal(a.population(),b.population())
    def test_incomplete_generation_preserves_search_state(self):
        c=FixedCEM([1,1],np.zeros((4,52)));p=c.population();rows=[dict(complete=True,controls=1,regular_success=False,strict_success=False,return_sum=0)]*63
        for partial in [rows,rows+[dict(rows[0],complete=False)]]:
            with self.assertRaises(AssertionError):c.update(p,partial,98)
            self.assertIsNone(c.best_score);self.assertFalse(c.mean.any())
    def test_complete_success_outweighs_return_or_survival(self):
        a=dict(complete=True,controls=98,regular_success=True,strict_success=False,return_sum=0)
        self.assertGreater(rank(a,98),rank(dict(a,regular_success=False,return_sum=1e9),98))
        with self.assertRaises(AssertionError):rank(dict(a,controls=97),98)
    def test_original_clipping_not_expanded(self):
        means=np.array([-1.,-.5,0,.5,1.]);np.testing.assert_array_equal(np.clip(means+2,-1,1),np.ones(5));np.testing.assert_array_equal(np.clip(means-2,-1,1),-np.ones(5))
        k=np.random.default_rng(9).uniform(-2,2,(4,52))
        for t in range(1333):self.assertLessEqual(abs(knot_bias(k,t,1333,[1,1])).max(),2)
if __name__=='__main__':unittest.main(verbosity=2)
