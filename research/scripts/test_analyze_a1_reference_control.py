"""CPU analytic fixtures for pairing, weighting, clusters, and evidence gates."""
import copy
import unittest
import numpy as np
from analyze_a1_reference_control import analyze, counts, CONTROLLERS, task_bootstrap


def fixture():
    tasks=['a','b','c','d']; seeds=[1,2]
    config=dict(task_ids=tasks,development_seeds=seeds,controllers=list(CONTROLLERS),repeats=2,
                maximum_conditions=32,run='audit',budget_phase='implementation',maximum_gpu_seconds=1800,
                task_strata=dict(zip(tasks,['s0','s0','s1','s1'])),
                analysis_plan=dict(bootstrap_seed=301044,bootstrap_resamples=5000))
    registration=dict(configuration=copy.deepcopy(config),configuration_sha256='config',planned_conditions=32,
                      policy_sha256='policy',checkpoint_sha256='checkpoint',parent_controls=123,
                      endpoint_metric_columns=list('abcdef'),endpoint_metric_scope='endpoints')
    rows=[]
    for task in tasks:
        for seed in seeds:
            for c in CONTROLLERS:
                for repeat in [1,2]:
                    n=2 if c=='zero_residual' else 3
                    e=[.001,.01] if c=='learned' else [.003,.03]
                    errors=[e]*2+([[99.,99.]] if n==3 else [])
                    rows.append(dict(task=task,seed=seed,controller=c,repeat=repeat,stratum=config['task_strata'][task],
                                     complete_episode=True,regular_success=c=='learned',strict_success=c=='learned',
                                     controls=n,start_phase=0,finite=True,failure_reason=0,
                                     initial_state_context_memory_sha256=f'{task}:{seed}',
                                     error_trace_m_rad=errors,endpoint_metrics=[[0.]*6]*n,contact_trace=[[]]*n))
    summary=dict(conditions=32,planned_conditions=32,status='complete',policy_unchanged=True,
                 checkpoint_sha256='checkpoint',counts={c:counts([r for r in rows if r['controller']==c]) for c in CONTROLLERS})
    job=dict(name='audit',phase='implementation',limit_seconds=1800,status='finished',returncode=0)
    return config,registration,rows,summary,job,'config'


class Checks(unittest.TestCase):
    def test_shared_prefix_excludes_unpaired_tail_and_sign_is_correct(self):
        result=analyze(*fixture()); p=result['paired_analysis']
        self.assertEqual(result['status'],'COMPLETE_DEVELOPMENT_DIAGNOSTIC')
        np.testing.assert_allclose(p['overall']['mean'],[-2,-.02*180/np.pi,1,1,0,0,0,0],atol=1e-12)
        self.assertEqual({x['common_controls'] for x in p['pairs']},{2})
        np.testing.assert_allclose(np.asarray(p['overall']['percentile95'])[:,0],p['overall']['mean'],atol=1e-12)
        self.assertFalse(result['G1_claim']); self.assertFalse(result['quality_admission'])

    def test_repeats_are_separate_counts_but_not_extra_task_clusters(self):
        result=analyze(*fixture())
        self.assertEqual(result['by_controller']['learned']['regular'],16)
        self.assertEqual(result['by_repeat']['learned']['1']['regular'],8)
        self.assertEqual(result['paired_analysis']['overall']['task_clusters'],4)
        self.assertEqual(len(result['paired_analysis']['pairs']),8)

    def test_stratified_resampling_preserves_stratum_composition(self):
        result=task_bootstrap({'a':[0.],'b':[0.],'c':[10.],'d':[10.]},dict(zip('abcd',['s0','s0','s1','s1'])),301044,5000)
        self.assertEqual(result['mean'],[5.]); self.assertEqual(result['percentile95'],[[5.,5.]])

    def test_missing_condition_remains_inconclusive(self):
        c,r,rows,s,j,h=fixture(); rows.pop()
        result=analyze(c,r,rows,None,j,h)
        self.assertEqual(result['status'],'INCOMPLETE_INCONCLUSIVE')
        self.assertEqual(len(result['missing_conditions']),1); self.assertIsNone(result['paired_analysis'])

    def test_incomplete_trajectory_not_counted_as_failure_or_success(self):
        c,r,rows,s,j,h=fixture(); rows[0].update(complete_episode=False,regular_success=None,strict_success=None)
        result=analyze(c,r,rows,None,j,h)
        self.assertEqual(result['by_controller']['learned']['complete'],15)
        self.assertEqual(result['by_controller']['learned']['regular'],15)
        self.assertIsNone(result['paired_analysis'])

    def test_duplicate_or_unpaired_initial_state_is_rejected(self):
        c,r,rows,s,j,h=fixture()
        with self.assertRaises(AssertionError):analyze(c,r,rows+[rows[0]],None,j,h)
        rows[0]['initial_state_context_memory_sha256']='changed'
        with self.assertRaises(AssertionError):analyze(c,r,rows,s,j,h)

    def test_budget_supervisor_must_have_completed(self):
        c,r,rows,s,j,h=fixture(); j['status']='running'
        self.assertIsNone(analyze(c,r,rows,s,j,h)['paired_analysis'])
        j.update(status='finished',returncode=1)
        self.assertIsNone(analyze(c,r,rows,s,j,h)['paired_analysis'])

    def test_wrong_config_or_summary_is_rejected(self):
        c,r,rows,s,j,h=fixture()
        with self.assertRaises(AssertionError):analyze(c,r,rows,s,j,'different')
        s['counts']['learned']['regular']+=1
        with self.assertRaises(AssertionError):analyze(c,r,rows,s,j,h)

    def test_nonfinite_errors_are_rejected(self):
        c,r,rows,s,j,h=fixture(); rows[0]['error_trace_m_rad'][0]=[float('nan'),0.]
        with self.assertRaises(AssertionError):analyze(c,r,rows,s,j,h)

    def test_repeat_variation_is_reported_on_same_common_prefix(self):
        c,r,rows,s,j,h=fixture()
        for row in rows:
            if row['controller']=='learned' and row['repeat']==2:
                row['error_trace_m_rad']=[[.002,.02]]*row['controls']
        result=analyze(c,r,rows,s,j,h)
        np.testing.assert_allclose(result['paired_analysis']['overall']['mean'][4:6],[1,.01*180/np.pi],atol=1e-12)

    def test_natural_reference_end_is_not_an_early_failure_index(self):
        _,_,rows,_,_,_=fixture()
        rows[0].update(failure_reason=2,regular_success=False,strict_success=False)
        result=counts(rows)
        self.assertEqual(set(result['first_failure_control_by_reason']),{'2'})
        self.assertEqual(result['first_failure_control_by_reason']['2']['n'],1)
        self.assertEqual(result['first_failure_control_by_reason']['2']['median'],rows[0]['controls'])


if __name__=='__main__':unittest.main(verbosity=2)
