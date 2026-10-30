"""Analytic and replicated-sample oracles for whole-cluster uncertainty."""
import unittest
import numpy as np
from a1_cluster_uncertainty import paired_cluster_intervals,integer_weight_quantile


class Checks(unittest.TestCase):
    def test_integer_quantile_agrees_with_explicit_replication(self):
        rng=np.random.default_rng(301047)
        for _ in range(25):
            v=np.sort(rng.uniform(0,20,20));w=rng.integers(0,5,20)
            np.testing.assert_allclose(integer_weight_quantile(v,w,[0,.5,.9,1]),np.quantile(np.repeat(v,w),[0,.5,.9,1]),atol=1e-12)

    def test_identical_models_have_zero_paired_interval_despite_uncertain_absolute_error(self):
        ids=np.arange(8);p=np.array([1,2,3,4,50,60,70,80.]);r=p/2
        result=paired_cluster_intervals(ids,ids,p,r,p,r,['a']*4+['b']*4,'initialization_family',resamples=1000)
        np.testing.assert_array_equal(result['paired_W1_minus_W0']['percentile95'],np.zeros((5,2)))
        self.assertGreater(result['W0']['percentile95'][0][1],result['W0']['percentile95'][0][0])
        self.assertEqual(result['clusters'],2);self.assertFalse(result['formal_G1_claim'])

    def test_constant_shift_has_expected_paired_sign(self):
        ids=np.arange(6);p=np.arange(1.,7.);r=p+10
        result=paired_cluster_intervals(ids,ids,p,r,p+2,r-3,['a','a','b','b','c','c'],'episode',resamples=1000)
        np.testing.assert_allclose(result['paired_W1_minus_W0']['estimate'],[2,-3,2,-3,0],atol=1e-12)
        np.testing.assert_allclose(result['paired_W1_minus_W0']['percentile95'],[[2,2],[-3,-3],[2,2],[-3,-3],[0,0]],atol=1e-12)

    def test_window_order_mismatch_and_nonfinite_values_are_rejected(self):
        ids=np.arange(4);v=np.arange(4.)
        with self.assertRaises(AssertionError):paired_cluster_intervals(ids,ids[::-1],v,v,v,v,list('abcd'),'episode',resamples=1000)
        v[0]=np.nan
        with self.assertRaises(AssertionError):paired_cluster_intervals(ids,ids,v,v,v,v,list('abcd'),'episode',resamples=1000)

    def test_one_family_does_not_create_false_certainty_from_many_windows(self):
        ids=np.arange(20);v=np.ones(20)
        result=paired_cluster_intervals(ids,ids,v,v,v,v,['same']*20,'initialization_family',resamples=1000)
        self.assertEqual(result['status'],'UNCERTAINTY_UNAVAILABLE');self.assertIsNone(result['W1'])


if __name__=='__main__':unittest.main(verbosity=2)
