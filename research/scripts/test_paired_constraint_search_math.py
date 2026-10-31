import copy,json,unittest
import numpy as np
from a1_paired_constraint_search_math import *

def row(seed=280043,n=98,regular=False,strict=False,error=.07,reward=0):
    return dict(seed=seed,controls=n,complete=True,status='original_termination',phase_start=0,reset_memory='original',regular_success=regular,strict_success=strict,failure_reason=0,mean_errors=[.02,.2,.03,.03,error,error],return_sum=reward)
def pair(**kwargs):return [row(seed=s,**kwargs) for s in OPTIMIZATION_SEEDS]
def generation(p,g=0):
    return [dict(row(seed=s,n=i+1),candidate_index=i,generation=g,knots=p[i].tolist()) for i in range(32) for s in OPTIMIZATION_SEEDS]

class Checks(unittest.TestCase):
    def test_actual_pool_reference_end_label(self):
        rows=[dict(r,status='reference_end') for r in pair(regular=True,error=.05)]
        self.assertEqual(paired_rank(rows,98,[1,1])[0],1)
        for update in [dict(controls=97),dict(failure_reason=1)]:
            invalid=copy.deepcopy(rows);invalid[0].update(update)
            with self.assertRaises(ValueError):paired_rank(invalid,98,[1,1])
    def test_full_success_beats_high_return_or_small_prefix_error(self):
        self.assertGreater(paired_rank(pair(regular=True,error=.05),98,[1,1]),paired_rank(pair(n=97,error=0,reward=1e10),98,[1,1]))
        with self.assertRaises(ValueError):paired_rank(pair(n=97,regular=True),98,[1,1])
    def test_seed_worst_case_not_average_survival(self):
        robust=pair(n=50);uneven=[row(n=97),row(seed=280044,n=10)]
        self.assertGreater(paired_rank(robust,98,[1,1]),paired_rank(uneven,98,[1,1]))
    def test_unmet_original_tip_gate_precedes_return_at_same_survival(self):
        self.assertGreater(paired_rank(pair(error=.065),98,[1,1]),paired_rank(pair(error=.09,reward=1e9),98,[1,1]))
    def test_inactive_hand_does_not_drive_constraint(self):
        p=pair(error=.05)
        for r in p:r['mean_errors'][2]=999;r['mean_errors'][4]=999
        self.assertLess(normalized_error(p[0],[0,1]),1)
        self.assertGreater(normalized_error(p[0],[1,1]),1)
    def test_seed_duplicate_development_and_partial_rejected(self):
        for rows in [[row(),row()], [row(),row(seed=280046)], [row(),dict(row(seed=280044),complete=False)], [row(),dict(row(seed=280044),status='budget_truncation')]]:
            with self.assertRaises(ValueError):paired_rank(rows,98,[1,1])
    def test_invalid_original_flags_or_values_rejected(self):
        for update in [dict(regular_success=True,failure_reason=1),dict(strict_success=True),dict(reset_memory='velocity'),dict(phase_start=5),dict(mean_errors=[0]*5),dict(return_sum=float('nan'))]:
            rows=pair();rows[0].update(update)
            with self.assertRaises(ValueError):paired_rank(rows,98,[1,1])
    def test_condition_order_independent_but_candidate_identity_bound(self):
        c=PairedCEM([1,1],np.zeros((4,52)));p=c.population();rows=generation(p);a=pair_generation(p,rows,98,[1,1],0)
        self.assertEqual(a[1],pair_generation(p,list(reversed(rows)),98,[1,1],0)[1]);rows[0]['knots']=p[2].tolist()
        with self.assertRaises(ValueError):pair_generation(p,rows,98,[1,1],0)
    def test_bad_generation_does_not_mutate_search_state(self):
        c=PairedCEM([1,1],np.zeros((4,52)));p=c.population();base=generation(p);before=(c.mean.copy(),c.std.copy(),c.best.copy(),copy.deepcopy(c.rng.bit_generator.state))
        bad=[];bad.append(base[:-1]);dup=copy.deepcopy(base);dup[-1]=dup[-2];bad.append(dup);cross=copy.deepcopy(base);cross[0]['generation']=1;bad.append(cross)
        partial=copy.deepcopy(base);partial[-1]['complete']=False;bad.append(partial)
        for rows in bad:
            with self.assertRaises(ValueError):c.update(p,rows,98,0)
            for current,old in zip([c.mean,c.std,c.best],before[:3]):np.testing.assert_array_equal(current,old)
            self.assertIsNone(c.best_score);self.assertEqual(c.generations,0);self.assertEqual(c.rng.bit_generator.state,before[3])
    def test_fixed_anchors_mask_bounds_and_reproducible_elites(self):
        prior=np.zeros((4,52),np.float32);prior[:,26:]=.3;a=PairedCEM([0,1],prior);b=PairedCEM([0,1],prior)
        p=a.population();np.testing.assert_array_equal(p,b.population());self.assertFalse(p[0].any());self.assertFalse(p[:,:,:26].any());self.assertLessEqual(abs(p).max(),2);np.testing.assert_array_equal(p[1],prior)
        r=a.update(p,generation(p),98,0);b.update(p,generation(p),98,0);self.assertEqual(r['elite_indices'],[31,30,29,28]);np.testing.assert_array_equal(a.population(),b.population())
        np.testing.assert_array_equal(a.population()[1],prior)
    def test_generation_limit_and_repeated_update_blocked(self):
        c=PairedCEM([1,1],np.zeros((4,52)));p=c.population();rows=generation(p);c.update(p,rows,98,0)
        with self.assertRaises(ValueError):c.update(p,rows,98,0)
        c.generations=12
        with self.assertRaises(ValueError):c.population()
    def test_budget_plan_does_not_mutate_or_borrow_formal_stages(self):
        ledger=dict(cap_seconds=330*3600,phase_caps=dict(implementation=7*3600,collection=145.5*3600,models_initial=30*3600,evaluation=8*3600,revision2=45*3600,revision6_original_action_search=.9*3600),jobs=[dict(status='finished',phase='revision6_original_action_search',seconds=1400.25),dict(status='finished',phase='revision2',seconds=43.6*3600)])
        old=copy.deepcopy(ledger);plan=proposed_budget_caps(ledger);self.assertEqual(ledger,old);self.assertEqual(plan['phase_caps']['revision7_paired_constraint_search'],3240);self.assertEqual(sum(plan['sources_seconds'].values()),3240)
        self.assertEqual(plan['phase_caps']['revision6_original_action_search'],1401);self.assertEqual(plan['rounding_reserve_seconds'],.75)
        for k in ['implementation','collection','models_initial','evaluation']:self.assertEqual(plan['phase_caps'][k],old['phase_caps'][k])
        ledger['jobs'][0]['status']='running'
        with self.assertRaises(ValueError):proposed_budget_caps(ledger)
    def test_empty_masks_nonfinite_parameters_and_wrong_prior_rejected(self):
        for mask in [[0,0],[1],[1,2]]:
            with self.assertRaises(ValueError):PairedCEM(mask,np.zeros((4,52)))
        for prior in [np.ones((4,52))*3,np.ones((3,52)),np.ones((4,52))*np.nan]:
            with self.assertRaises(ValueError):PairedCEM([1,1],prior)

if __name__=='__main__':unittest.main(verbosity=2)
