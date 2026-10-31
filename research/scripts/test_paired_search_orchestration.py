"""CPU synthetic fixtures test persistence/stopping,not physical performance."""
import hashlib,importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np,torch
from a1_paired_constraint_search_math import OPTIMIZATION_SEEDS,DEVELOPMENT_SEEDS

class Checks(unittest.TestCase):
 def execute(self,mode):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name);out=root/'runs/a1_full_20261028/fixture';out.parent.mkdir(parents=True)
  weight={'weight':torch.tensor([1.])};ck=root/'parent.pt';torch.save(dict(configuration=dict(task_id='fixture',scientific_revision=4),smoke=False,policy=weight),ck)
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  prior=root/'prior.json';prior.write_text(json.dumps(dict(checkpoint_sha256=sha(ck),scientific_revision=6,knot_times=[0,8,32,97],knots=np.zeros((4,52)).tolist())))
  config=dict(scientific_revision=7,phase='revision7_paired_constraint_search',job_limit_seconds=1620,population=32,max_generations=12,optimizer_seed=410032,optimization_seeds=list(OPTIMIZATION_SEEDS),development_seeds=list(DEVELOPMENT_SEEDS),authorization_reply='批准第七项≤0.9 GPUh验证',source_hashes={},checkpoint='parent.pt',checkpoint_sha256=sha(ck),prior_selection='prior.json',prior_selection_sha256=sha(prior),run='fixture',task_id='fixture',task_path='dummy',full_horizon=98)
  cp=root/'config.json';cp.write_text(json.dumps(config));calls=[]
  class Policy:
   def __init__(self,*a):self.saved=None
   def cuda(self):return self
   def load_state_dict(self,p):self.saved=p
   def eval(self):pass
   def parameters(self):return []
   def state_dict(self):return self.saved
  class Pool:
   def __init__(self,*a):
    self.horizon=98;self.task=types.SimpleNamespace(mask=torch.tensor([[1.,1.]]));self.controls=0;self.warmup_controls=8;self.envs=[types.SimpleNamespace(physical_controls=1) for _ in range(8)];self.audit={}
   def run(self,knots,seeds,guard,validate=False):
    calls.append(list(seeds));rows=[]
    if seeds==list(DEVELOPMENT_SEEDS):
     selftest.assertTrue((out/'selected_controller.json').exists());selftest.assertEqual(json.loads((out/'selected_controller.json').read_text())['development_not_opened_yet'],True)
    for i,(k,s) in enumerate(zip(knots,seeds)):
     truncated=mode=='truncated' and len(calls)==2 and i==7;n=97 if truncated else 98
     success=mode=='success';rows.append(dict(seed=s,controls=n,return_sum=0.,complete=not truncated,regular_success=success,strict_success=False,failure_reason=0,phase_start=0,reset_memory='original',context={},knots=k.tolist(),mean_errors=[.02,.2,.03,.03,.05 if success else .08,.05 if success else .08],status='budget_truncation' if truncated else 'reference_end',trace={'phase':np.arange(n+1)}))
    count=sum(r['controls'] for r in rows);self.controls+=count;self.envs[0].physical_controls+=count
    if mode=='truncated' and len(calls)==2:mod.STOP=True
    return rows
  selftest=self
  fake_policy=types.ModuleType('a1_policy');fake_policy.CollectorPolicy=Policy
  fake_pool=types.ModuleType('a1_original_action_search_pool');fake_pool.WitnessPool=Pool
  special={'a1_policy':fake_policy,'a1_original_action_search_pool':fake_pool};old={k:sys.modules.get(k) for k in special};sys.modules.update(special)
  try:
   spec=importlib.util.spec_from_file_location('runner_fixture',Path(__file__).with_name('run_a1_paired_constraint_search.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  finally:
   for k,v in old.items():
    if v is None:sys.modules.pop(k,None)
    else:sys.modules[k]=v
  with patch.object(mod,'R',root),patch.object(mod.socket,'gethostname',return_value='Lambda'),patch.object(mod.getpass,'getuser',return_value='youran'),patch.object(sys,'argv',['runner','--config',str(cp)]):mod.main()
  return out,json.loads((out/'summary.json').read_text()),[json.loads(s) for s in (out/'conditions.jsonl').read_text().splitlines()],calls
 def test_twelve_generations_pair_all_seeds_and_freeze_before_development(self):
  out,s,rows,calls=self.execute('failure');self.assertEqual(s['episodes'],786);self.assertEqual(len(s['generations']),12);self.assertEqual(s['development_regular'],0);self.assertFalse(s['witness_gate_passed']);self.assertEqual(s['status'],'COMPLETE')
  for g in range(12):
   selected=[r for r in rows if r.get('generation')==g];self.assertEqual(len(selected),64)
   for i in range(32):self.assertEqual({r['seed'] for r in selected if r['candidate_index']==i},set(OPTIMIZATION_SEEDS))
  self.assertEqual(s['selected_replay_n'],6);self.assertEqual(s['baseline_replay_n'],6);self.assertEqual(calls[-1],list(DEVELOPMENT_SEEDS))
 def test_truncated_generation_never_updates_or_opens_development(self):
  out,s,rows,calls=self.execute('truncated');self.assertEqual(s['status'],'BOUNDED_STOP_INCOMPLETE');self.assertFalse(s['generations']);self.assertFalse((out/'selected_controller.json').exists());self.assertFalse((out/'development.json').exists());self.assertFalse(json.loads((out/'incomplete_generation.json').read_text())['used_for_update'])
 def test_two_seed_success_freezes_after_one_complete_generation(self):
  out,s,rows,calls=self.execute('success');self.assertEqual(s['episodes'],82);self.assertEqual(len(s['generations']),1);self.assertTrue(s['witness_gate_passed']);self.assertEqual(s['selected_replay_regular'],6);self.assertEqual(s['development_regular'],4);self.assertFalse(s['full_A1_complete']);self.assertFalse(s['quality_passports_created'])
if __name__=='__main__':unittest.main(verbosity=2)
