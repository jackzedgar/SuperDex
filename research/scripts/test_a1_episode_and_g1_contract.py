"""CPU regression checks using synthetic records, never opening A1 held-out data."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from a1_episode_data import EpisodeWriter,EpisodeShard,split_family
from a1_episode_contract import complete_controls,is_complete_episode
import a1_window_pool as wp
import freeze_a1_dataset as freeze
from a1_g1_coverage import g1_criteria,REQUIRED_PHASES

class EpisodeChecks(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(prefix='a1_contract_cpu_');self.root=Path(self.temp.name);self.path=self.root/'runs/data/shard'
  self.task=SimpleNamespace(m=SimpleNamespace(nq=2,nv=2),meta={'task_id':'task0','object':{'object_key':'object0'},'stratum':'synthetic'},sides=['right'],mask=torch.tensor([[1.,0.]]))
  self.writer=EpisodeWriter(self.path,self.task,'20',{'sha':'synthetic'},'sequence')
  self.seed=next(i for i in range(1000) if split_family('sequence:'+str(i))=='train')
  for n,reason,tag in [(10,'reference_end',1),(9,'budget_truncation',2),(12,'physics_failure_2',3)]:
   self.writer.append(self.arrays(n,reason,tag),self.seed,0,reason,{'formal':True,'policy_noise_sigma':.1})
  self.meta={'shards':[{'path':'runs/data/shard','task_id':'task0'}],'object_keys':['object0']}
  self.manifest=self.root/'manifest.json';self.manifest.write_text(json.dumps(self.meta))
  self.oldroot=wp.ROOT;wp.ROOT=self.root
 def tearDown(self):wp.ROOT=self.oldroot;self.temp.cleanup()
 def arrays(self,n,reason,tag):
  arrays={}
  for key,spec in self.writer.spec.items():
   size=n+1 if key in ['state','generalized','contact','phase','contact_ranges'] else 1 if key in ['context','initial_history','contact_details'] else n
   arrays[key]=np.zeros((size,spec['width']),dtype=spec['dtype'])
  arrays['state'][:,0]=tag;arrays['terminated'][-1]=int(reason!='budget_truncation')
  return arrays
 def test_complete_counts_and_resume_preserve_partial(self):
  self.assertEqual(complete_controls(self.writer.entries),22);self.assertEqual(sum(e['controls'] for e in self.writer.entries),31)
  before={p.name:p.read_bytes() for p in self.path.iterdir()}
  resumed=EpisodeWriter(self.path,self.task,'20',{'sha':'synthetic'},'sequence')
  self.assertEqual(complete_controls(resumed.entries),22);self.assertEqual(before,{p.name:p.read_bytes() for p in self.path.iterdir()})
  self.assertEqual(len({e['split'] for e in resumed.entries}),1)
 def test_pool_excludes_cutoff_and_rejects_insufficient_distinct_windows(self):
  pool=wp.WindowPool(self.manifest,'train',8);ids=pool.sample_ids(8,np.random.default_rng(1),set())
  self.assertEqual(set(ids[:,1]),{0,2});self.assertEqual(len(set(map(tuple,ids))),8)
  with self.assertRaises(ValueError):pool.sample_ids(9,np.random.default_rng(1),set())
 def test_direct_window_cannot_bypass_complete_episode_contract(self):
  with self.assertRaises(AssertionError):EpisodeShard(self.path).window([1],[0],8)
 def test_terminal_label_mismatch_cannot_commit(self):
  arrays=self.arrays(4,'budget_truncation',8);arrays['terminated'][-1]=1
  with self.assertRaises(AssertionError):self.writer.append(arrays,self.seed,0,'budget_truncation')
  self.assertEqual(len(self.writer.entries),3)
 def test_unknown_termination_rejected(self):
  with self.assertRaises(ValueError):is_complete_episode({'termination':'unknown','controls':10})
 def test_fixed_window_manifest_binding(self):
  pool=wp.WindowPool(self.manifest,'train',8);path=self.root/'fixed.npz';first=pool.fixed_windows(path,per_task=4,natural=4)
  again=pool.fixed_windows(path,per_task=4,natural=4);np.testing.assert_array_equal(first['balanced'],again['balanced'])
  self.meta['changed']=True;self.manifest.write_text(json.dumps(self.meta))
  with self.assertRaises(AssertionError):wp.WindowPool(self.manifest,'train',8).fixed_windows(path)
 def test_freeze_counts_complete_records_and_preserves_all_file_hashes(self):
  (self.root/'configs').mkdir();(self.root/'configs/a1_probe40_v1.json').write_text('{"tasks":[]}')
  (self.root/'configs/a1_full_execution_20261028.json').write_text('{"plan":{"original_G1_thresholds":{}}}')
  geo=self.root/'data/a1_geometry/objects/object0';geo.mkdir(parents=True);(geo/'sdf.npz').write_bytes(b'synthetic')
  old=(freeze.ROOT,freeze.RUN,sys.argv);out=self.root/'frozen.json'
  try:
   freeze.ROOT=self.root;freeze.RUN=self.root/'runs';sys.argv=['freeze','--data-root',str(self.root/'runs/data'),'--output',str(out),'--smoke'];freeze.main()
  finally:freeze.ROOT,freeze.RUN,sys.argv=old
  frozen=json.loads(out.read_text());self.assertEqual(frozen['controls'],22);self.assertEqual(frozen['episodes'],2);self.assertEqual(frozen['recorded_episodes'],3);self.assertEqual(frozen['excluded_incomplete_controls'],9)
  self.assertEqual(frozen['shards'][0]['excluded_incomplete_episode_ids'],[1]);self.assertIn('episodes.jsonl',frozen['shards'][0]['files'])

class CoverageChecks(unittest.TestCase):
 def setUp(self):
  phase={'n':1,'position_p90_mm':8.,'orientation_p90_deg':8.}
  row={'overall':{'position_median_mm':3.,'orientation_median_deg':3.,'position_le15mm_fraction':.8},'phases':{p:phase.copy() for p in [*REQUIRED_PHASES,'switch']},'per_task':{'task0':{}},'per_object':{'object0':{}}}
  self.summaries={f'H{h}_{s}':copy.deepcopy(row) for h in [8,32] for s in ['balanced','natural']}
  self.coverage={h:{'missing_task_ids':[],'expected_task_ids':['task0'],'expected_object_keys':['object0']} for h in [8,32]}
 def test_original_threshold_edges_and_full_coverage_pass(self):self.assertTrue(all(all(v.values()) for v in g1_criteria(self.summaries,self.coverage).values()))
 def test_missing_phase_does_not_pass_with_good_medians(self):
  self.summaries['H32_balanced']['phases']['slip']['n']=0
  self.assertFalse(g1_criteria(self.summaries,self.coverage)['balanced']['all_required_phases_present_8and32'])
 def test_missing_sampled_task_does_not_pass_eligible_pool(self):
  self.summaries['H8_natural']['per_task']={}
  self.assertFalse(g1_criteria(self.summaries,self.coverage)['natural']['all_tasks_and_objects_sampled'])
 def test_original_threshold_is_not_relaxed(self):
  self.summaries['H8_balanced']['overall']['position_median_mm']=3.00001
  self.assertFalse(g1_criteria(self.summaries,self.coverage)['balanced']['position8'])

if __name__=='__main__':unittest.main(verbosity=2)
