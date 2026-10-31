import unittest,torch
from a1_teacher_warmstart_helpers import causal_histories,clone_parameters,clone_gate,comparison_gate

class TeacherTest(unittest.TestCase):
 def test_history_has_no_future_state_or_current_action(self):
  s=torch.arange(5.)[:,None].expand(5,287);a=(10+torch.arange(4.))[:,None].expand(4,52)
  sh,ah=causal_histories(s,a)
  self.assertEqual(sh[:,:,0].tolist(),[[0,0,0],[0,0,1],[0,1,2],[1,2,3]])
  self.assertEqual(ah[:,:,0].tolist(),[[0,0,0],[0,0,10],[0,10,11],[10,11,12]])
 def test_trace_boundary_is_not_joined(self):
  s=torch.ones(3,287);a=torch.ones(2,52);_,ah=causal_histories(s,a)
  self.assertEqual(float(ah[0].sum()),0.)
 def test_only_actor_is_updated(self):
  p=torch.nn.Module();p.encoder=torch.nn.Linear(2,2);p.decoder=torch.nn.Linear(2,2);p.critic=torch.nn.Linear(2,1);p.logstd=torch.nn.Parameter(torch.ones(2))
  selected=clone_parameters(p);self.assertEqual(len(selected),4)
  self.assertTrue(all(x.requires_grad for x in selected));self.assertFalse(p.logstd.requires_grad);self.assertTrue(all(not x.requires_grad for x in p.critic.parameters()))
 def test_clone_requires_all_fixed_complete_replays(self):
  r=dict(seed=280044,controls=98,regular_success=True,failure_reason=0)
  self.assertTrue(clone_gate(2000,.02,[r]*3));self.assertFalse(clone_gate(1999,0,[r]*3));self.assertFalse(clone_gate(2000,.021,[r]*3));self.assertFalse(clone_gate(2000,.01,[r]*2))
  self.assertFalse(clone_gate(2000,float('nan'),[r]*3));self.assertFalse(clone_gate(2000,0,[r,r,dict(r,controls=97)]))
 def test_matched_comparison(self):
  def arm(n):return dict(new_controls=131072,updates=64,development=dict(complete=True,conditions=[dict(seed=s,regular_success=i<n) for i,s in enumerate(range(280046,280050))]))
  self.assertTrue(comparison_gate(arm(2),arm(0)));self.assertFalse(comparison_gate(arm(2),arm(1)));self.assertFalse(comparison_gate(arm(1),arm(0)))
  short=arm(2);short['new_controls']=129024;self.assertFalse(comparison_gate(short,arm(0)))
  missing=arm(2);missing['development']['complete']=False;self.assertFalse(comparison_gate(missing,arm(0)))

if __name__=='__main__':unittest.main()
