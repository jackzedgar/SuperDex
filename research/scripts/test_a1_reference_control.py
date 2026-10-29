"""CPU tests of diagnostic control selection, pairing and budget interruption."""
import sys,unittest,json,os,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch
from diagnose_a1_reference_control import action_for,trace,order_for,summarize,endpoint_metrics,policy_hash,validate_parent

class Policy(torch.nn.Module):
 def __init__(self):super().__init__();self.register_buffer('value',torch.tensor(.4));self.calls=0
 def distribution(self,obs,goal):self.calls+=1;return SimpleNamespace(mean=torch.full((1,52),float(self.value)))
class Env:
 def __init__(self):
  self.device='cpu';self.task=SimpleNamespace(length=4,mask=torch.tensor([[1.,0.]]));self.actions_seen=[];self.reset()
 def reset(self,phase=0,seed=0,randomize=True):
  assert phase==0 and randomize
  self.steps=0;self.phase=torch.tensor([0]);self.qpos=torch.zeros(1,33);self.qvel=torch.zeros(1,32);self.context=torch.full((1,276),float(seed));self.ctrl=torch.zeros(1,26);self.history=torch.zeros(1,2,3,26);self.servo=torch.zeros(1,2,6);self.last_action=torch.zeros(1,52);self.timeout=False;self.s=torch.zeros(1,287);self.contact_summary=torch.zeros(10)
 def state(self):return self.s
 def step(self,action):
  self.actions_seen.append(action.clone());self.last_action.copy_(action);self.steps+=1;self.phase.fill_(self.steps);self.s+=1;self.timeout=self.steps==3
  info=dict(pos_error=torch.tensor(.001),rot_error=torch.tensor(.02),finite=torch.tensor(True),reason=torch.tensor(0),regular_success=torch.tensor(self.timeout),strict_success=torch.tensor(self.timeout),mean_errors=torch.zeros(1,6))
  return self.s,torch.tensor(1.),torch.tensor(self.timeout),info

class Checks(unittest.TestCase):
 def test_order_balanced_and_alternating(self):
  self.assertEqual(order_for(0,0),('learned','zero_residual','zero_residual','learned'))
  self.assertEqual(order_for(0,1),('zero_residual','learned','learned','zero_residual'))
  counts={c:0 for c in ['learned','zero_residual']}
  for task in range(40):
   for seed in range(4):
    for c in order_for(task,seed):counts[c]+=1
  self.assertEqual(counts,{'learned':320,'zero_residual':320})
 def test_reference_does_not_call_policy(self):
  env=Env();policy=Policy()
  result=trace(env,policy,42,'zero_residual',lambda *_:self.fail('Policy inputs called'),lambda *_:[0.]*6)
  self.assertEqual(policy.calls,0);self.assertTrue(all(torch.equal(a,torch.zeros_like(a)) for a in env.actions_seen));self.assertTrue(result['complete_episode'])
 def test_learned_matches_mean_mask_and_causal_histories(self):
  env=Env();policy=Policy();histories=[];before=policy_hash(policy)
  def inputs(task,sh,ah,phase):histories.append((sh.clone(),ah.clone(),phase.clone()));return sh[:,-1],ah[:,-1]
  result=trace(env,policy,42,'learned',inputs,lambda *_:[0.]*6)
  self.assertEqual(result['controls'],3);self.assertTrue(result['regular_success']);self.assertEqual(policy.calls,3);self.assertEqual(before,policy_hash(policy))
  self.assertTrue(torch.equal(histories[0][1],torch.zeros(1,3,52)))
  self.assertTrue(torch.equal(histories[1][1][:,-1],env.actions_seen[0]));self.assertTrue(torch.all(env.actions_seen[0][:,26:]==0));self.assertTrue(torch.allclose(env.actions_seen[0][:,:26],torch.full((1,26),.4)))
 def test_pairing_includes_physical_context_and_execution_memory(self):
  env=Env();policy=Policy();inputs=lambda *_:(None,None)
  a=trace(env,policy,42,'learned',inputs,lambda *_:[0.]*6);b=trace(env,policy,42,'zero_residual',inputs,lambda *_:[0.]*6);c=trace(env,policy,43,'zero_residual',inputs,lambda *_:[0.]*6)
  self.assertEqual(a['initial_state_context_memory_sha256'],b['initial_state_context_memory_sha256']);self.assertNotEqual(a['initial_state_context_memory_sha256'],c['initial_state_context_memory_sha256'])
 def test_interruption_is_not_failure_or_success(self):
  env=Env();row=trace(env,Policy(),42,'zero_residual',lambda *_:(None,None),lambda *_:[0.]*6,lambda:env.steps>=1)
  self.assertEqual(row['controls'],1);self.assertFalse(row['complete_episode']);self.assertIsNone(row['regular_success']);self.assertIsNone(row['strict_success']);self.assertNotEqual(summarize([row],1)['status'],'complete')
 def test_stop_before_first_step_preserves_incomplete_record(self):
  row=trace(Env(),Policy(),42,'zero_residual',lambda *_:(None,None),lambda *_:[0.]*6,lambda:True)
  self.assertEqual(row['controls'],0);self.assertIsNone(row['finite']);self.assertFalse(row['complete_episode'])
 def test_unknown_controller_is_rejected(self):
  with self.assertRaises(ValueError):action_for(Env(),Policy(),None,None,'other',lambda *_:(None,None))
 def test_force_and_control_occupancy_are_endpoint_metrics(self):
  env=Env();env.qpos[:,6:26]=.2;env.task.qpos=torch.zeros(4,33);env.task.sides=['right'];env.m=SimpleNamespace(actuator_forcerange=np.array([[-1.,1.]]*26),actuator_forcelimited=np.ones(26,dtype=bool));env.wd=SimpleNamespace(actuator_force=torch.tensor([[.5,1.]+[0.]*24]));env.lo=torch.full((26,),-1.);env.hi=torch.ones(26)
  with patch.dict(sys.modules,{'warp':SimpleNamespace(to_torch=lambda x:x)}):values=endpoint_metrics(env,{'wrist_error':torch.tensor(.01)})
  np.testing.assert_allclose(values,[.01,.2,1.,1/26,0.,0.],atol=1e-6)

class LaunchGateChecks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='a1_reference_gate_');self.root=Path(self.temp.name);self.run=self.root/'runs/a1_full_20261028';self.folder=self.run/'parent';self.folder.mkdir(parents=True)
  self.config={'parent_run':'parent','run':'diagnostic','maximum_gpu_seconds':1800}
  self.summary={'status':'steps_complete','bands':{},'real_controls':3}
  self.jobs=[dict(name='worker'+str(i),command=['python','scripts/train_a1_shared_collector.py','--run','parent'],status='finished',returncode=0,reason='completed',supervisor_pid=100000000+i,supervisor_start_ticks='0',child_pid=100000100+i) for i in range(4)]
  self.jobs.append(dict(name='diagnostic',phase='implementation',limit_seconds=1800,status='running',child_pid=os.getpid()))
  self.save()
 def save(self):
  (self.folder/'summary.json').write_text(json.dumps(self.summary));(self.run/'gpu_ledger.json').write_text(json.dumps({'jobs':self.jobs}))
 def tearDown(self):self.temp.cleanup()
 def test_registered_terminal_parent_can_proceed_to_checkpoint_verification(self):self.assertEqual(validate_parent(self.root,self.config),self.folder/'checkpoint_final.pt')
 def test_active_parent_job_blocks_launch(self):
  self.jobs[0]['status']='running';self.save()
  with self.assertRaises(AssertionError):validate_parent(self.root,self.config)
 def test_complete_quality_blocks_unnecessary_diagnostic(self):
  for q in ['20','50','80']:(self.folder/f'quality_{q}_passport.json').write_text('{}')
  with self.assertRaises(AssertionError):validate_parent(self.root,self.config)
 def test_wrong_budget_phase_blocks_launch(self):
  self.jobs[-1]['phase']='revision2';self.save()
  with self.assertRaises(AssertionError):validate_parent(self.root,self.config)
 def test_live_supervisor_overrides_terminal_ledger(self):
  self.jobs[0]['supervisor_pid']=os.getpid();self.jobs[0]['supervisor_start_ticks']=Path('/proc/self/stat').read_text().split()[21];self.save()
  with self.assertRaises(AssertionError):validate_parent(self.root,self.config)

if __name__=='__main__':unittest.main(verbosity=2)
