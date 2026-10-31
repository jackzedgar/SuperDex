import unittest
import numpy as np
import torch
from a1_policy import CollectorPolicy,VALUE_DIM
from a1_original_action_search_math import knot_bias
from a1_exact_feedback_policy import ExactFeedbackPolicy,setup_gate
torch.set_num_threads(1)

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  base=CollectorPolicy(280043)
  with torch.no_grad():base.decoder[-1].weight.normal_(0,.01)
  cls.parent=base.state_dict();cls.knots=np.random.default_rng(2).normal(0,.1,(4,52)).astype(np.float32)
 def policy(self,arm='candidate'):return ExactFeedbackPolicy(self.parent,self.knots,arm)
 def data(self):return torch.randn(4,546),torch.randn(4,1000),torch.tensor([0,8,32,97])
 def test_zero_candidate_exact_value_and_observation_derivative(self):
  m=self.policy();o,g,p=self.data();o.requires_grad_(True);a=m.distribution(o,g,p).mean;b=(m.base.distribution(o,g).mean+m.phase_bias(p)).clamp(-1,1)
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(torch.autograd.grad(a.sum(),o,retain_graph=True)[0],torch.autograd.grad(b.sum(),o)[0]))
 def test_direct_zero_matches_parent(self):
  m=self.policy('direct');o,g,p=self.data();self.assertTrue(torch.equal(m.distribution(o,g,p).mean,m.base.distribution(o,g).mean));self.assertFalse(m.teacher_knots.any())
 def test_phase_bias_all_original_phases(self):
  m=self.policy();p=torch.arange(98);expected=np.stack([knot_bias(self.knots,int(i),98,np.ones(2)) for i in p]);self.assertLess(float(abs(m.phase_bias(p).numpy()-expected).max()),1e-6)
 def test_residual_bound_and_frozen_gradients(self):
  m=self.policy();o,g,p=self.data();m.distribution(o,g,p).mean.sum().backward();self.assertTrue(all(x.grad is None for n,x in m.base.named_parameters() if not n.startswith('critic.')));self.assertTrue(m.residual[-1].weight.grad.abs().max()>0)
  with torch.no_grad():m.residual[-1].bias.fill_(100)
  self.assertLessEqual(float(m.components(o,g,p)[2].abs().max()),.100001);self.assertTrue(m.frozen_actor_unchanged(self.parent))
 def test_fixed_std_and_raw_action_logprob_phase_alignment(self):
  m=self.policy();o,g,p=self.data();d=m.distribution(o,g,p);self.assertTrue(torch.equal(d.stddev,torch.full((4,52),.03)))
  a=d.mean+.04;lp=d.log_prob(a).sum(-1);ids=torch.tensor([2,0,3,1]);self.assertTrue(torch.allclose(lp[ids],m.distribution(o[ids],g[ids],p[ids]).log_prob(a[ids]).sum(-1)))
  self.assertGreater(float((lp-m.distribution(o,g,p.flip(0)).log_prob(a).sum(-1)).abs().max()),1)
 def test_actor_and_critic_update_preserves_original_backbone(self):
  m=self.policy();o,g,p=self.data();opt=torch.optim.Adam([x for x in m.parameters() if x.requires_grad],lr=3e-5);v=torch.randn(4,VALUE_DIM);loss=m.distribution(o,g,p).mean.square().sum()+m.value(v).square().sum();opt.zero_grad();loss.backward();opt.step()
  self.assertTrue(m.frozen_actor_unchanged(self.parent));self.assertTrue(m.residual[-1].weight.abs().max()>0);self.assertTrue(any(not torch.equal(x,self.parent[k]) for k,x in m.base.state_dict().items() if k.startswith('critic.')))
 def test_gate_requires_three_original_successes(self):
  rows=[dict(seed=280044,complete=True,controls=98,failure_reason=0,regular_success=True) for _ in range(3)];self.assertTrue(setup_gate(rows,1e-6));self.assertFalse(setup_gate(rows[:2],1e-6));rows[0]['regular_success']=False;self.assertFalse(setup_gate(rows,0))
 def test_gate_rejects_partial_wrong_seed_and_nonfinite(self):
  rows=[dict(seed=280044,complete=True,controls=98,failure_reason=0,regular_success=True) for _ in range(3)];self.assertFalse(setup_gate(rows,float('nan')));self.assertFalse(setup_gate(rows,1e-4));rows[0]['seed']=280043;self.assertFalse(setup_gate(rows,0));rows[0]['seed']=280044;rows[0]['controls']=97;self.assertFalse(setup_gate(rows,0))
if __name__=='__main__':unittest.main()
