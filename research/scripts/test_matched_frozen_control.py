import unittest,torch
from a1_matched_frozen_control import replace_frozen,interpretation,ORDER
class Tests(unittest.TestCase):
 def test_swap_preserves_graph_parameter_storage(self):
  p=torch.nn.Linear(3,2);ptr={k:v.data_ptr() for k,v in p.state_dict().items()};s={k:torch.ones_like(v) for k,v in p.state_dict().items()}
  self.assertEqual(replace_frozen(p,s),ptr);self.assertTrue(all(not x.requires_grad for x in p.parameters()));self.assertTrue(torch.equal(p(torch.zeros(1,3)),torch.ones(1,2)))
  z={k:torch.zeros_like(v) for k,v in s.items()};self.assertEqual(replace_frozen(p,z),ptr);self.assertTrue(torch.equal(p(torch.zeros(1,3)),torch.zeros(1,2)))
 def test_missing_key_fails(self):
  with self.assertRaises(AssertionError):replace_frozen(torch.nn.Linear(2,1),{})
 def test_fixed_interpretation(self):
  rows=[dict(controller=x,complete=True,regular_success=x=='teacher') for x in ORDER]
  self.assertEqual(interpretation(rows),'CLONE_RETENTION_LOSS_WITH_SHARED_EXECUTOR')
  self.assertEqual(interpretation(rows[:-1]),'INCOMPLETE_NO_CONCLUSION')
  rows[0]['regular_success']=False;self.assertEqual(interpretation(rows),'TEACHER_CONTROL_NOT_RETAINED')
  for r in rows:r['regular_success']=True
  self.assertEqual(interpretation(rows),'EXECUTION_PATH_DEPENDENT_CLONE_RESULT_REQUIRES_REVIEW')
if __name__=='__main__':unittest.main()
