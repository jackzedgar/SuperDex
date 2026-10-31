import unittest,numpy as np,torch
from a1_teacher_coverage_helpers import validate_data,per_source_rmse,coverage_gate,factual_episode
class Tests(unittest.TestCase):
 def test_source_fit_does_not_hide_failed_trace(self):
  ids=torch.repeat_interleave(torch.arange(9),torch.tensor([98]*8+[82]));target=torch.zeros(866,52);prediction=target.clone();prediction[ids==8]=.02
  errors=per_source_rmse(prediction,target,ids);self.assertLess(float(prediction.square().mean().sqrt()),.01);self.assertGreater(errors[8],.01)
 def test_gate_needs_all_nine_source_fits(self):
  row=dict(seed=280044,controls=98,regular_success=True,failure_reason=0)
  self.assertTrue(coverage_gate(2000,[.01]*9,[row]*3));self.assertFalse(coverage_gate(2000,[.001]*8+[.011],[row]*3));self.assertFalse(coverage_gate(2000,[.001]*8,[row]*3))
 def test_gate_rejects_nonfinite_and_partial_updates(self):
  row=dict(seed=280044,controls=98,regular_success=True,failure_reason=0)
  self.assertFalse(coverage_gate(2000,[.001]*8+[float('nan')],[row]*3));self.assertFalse(coverage_gate(1999,[.001]*9,[row]*3))
 def test_gate_rejects_incomplete_wrong_seed_or_failed_replay(self):
  row=dict(seed=280044,controls=98,regular_success=True,failure_reason=0)
  for patch in [dict(seed=280046),dict(controls=97),dict(regular_success=False),dict(failure_reason=1)]:self.assertFalse(coverage_gate(2000,[.001]*9,[row,row,dict(row,**patch)]))
 def test_episode_origin_uses_reset_record_and_preserves_legacy_labels(self):
  ep=dict(lane=1,episode_id=4,seed=82,start_phase=0,reset_mode='uniform_reference',reset_memory_mode='velocity_consistent');origin=dict(seed=82,phase=0,reset_mode='formal_episode_start',training_memory='original')
  out=factual_episode(ep,{(1,4):origin});self.assertEqual(out['reset_mode'],'formal_episode_start');self.assertEqual(out['reset_memory_mode'],'original');self.assertEqual(out['original_runner_labels']['reset_mode'],'uniform_reference');self.assertEqual(ep['reset_mode'],'uniform_reference')
 def test_episode_origin_mismatch_fails(self):
  ep=dict(lane=0,episode_id=2,seed=8,start_phase=0)
  with self.assertRaises(AssertionError):factual_episode(ep,{(0,2):dict(seed=9,phase=0)})
 def fixture(self):
  counts=[98]*8+[82];sources=[dict(controls=n,counterfactual_labels=i in [4,5,8]) for i,n in enumerate(counts)];m=dict(observed_transitions=866,initialization_families=1,optimization_seed=280044,clone_observed_transitions_with_counterfactual_teacher_labels=278,sources=sources)
  d={k:np.zeros(shape,np.float32) for k,shape in [('obs',(866,546)),('goal',(866,1000)),('teacher_target_action',(866,52)),('actual_executed_action',(866,52))]};d['source_index']=np.repeat(np.arange(9),counts);d['phase']=np.concatenate([np.arange(n) for n in counts]);return d,m
 def test_data_preserves_counterfactual_label_separation(self):
  d,m=self.fixture();validate_data(d,m);d.pop('actual_executed_action')
  with self.assertRaises(AssertionError):validate_data(d,m)
 def test_data_rejects_cross_episode_phase_alignment(self):
  d,m=self.fixture();d['phase'][98]=98
  with self.assertRaises(AssertionError):validate_data(d,m)
if __name__=='__main__':unittest.main()
