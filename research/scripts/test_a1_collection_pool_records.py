import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from a1_collection_pool_records import EpisodePlan,EpisodeBuffer,episode_noise
from a1_episode_data import EpisodeWriter,EpisodeShard
from a1_episode_contract import complete_controls

class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.task=SimpleNamespace(m=SimpleNamespace(nq=2,nv=2),meta={'task_id':'fixture','object':{'object_key':'fixture'},'stratum':'fixture'},sides=['right'],mask=torch.tensor([[1.,0.]]))
    def tearDown(self):self.tmp.cleanup()
    def writer(self):return EpisodeWriter(self.root/'data',self.task,'hardware_smoke_only',{'sha':'fixture'},'family')
    def test_original_fresh_reset_sequence(self):
        plan=EpisodePlan(self.root/'reservations',34,98);rng=np.random.default_rng(34)
        for i in range(64):
            row=plan.reserve();self.assertEqual((row['phase'],row['seed']),(int(rng.integers(98)),int(rng.integers(2**31))))
    def test_resume_skips_inflight_reservations(self):
        path=self.root/'reservations';plan=EpisodePlan(path,34,98);rows=[plan.reserve() for _ in range(5)]
        recovered=EpisodePlan(path,34,98);self.assertEqual(plan.reserve(),recovered.reserve())
        self.assertEqual(recovered.rows[-1]['ordinal'],5)
    def test_noise_is_independent_of_lane_interleaving(self):
        plan=EpisodePlan(self.root/'reservations',34,98);a,b=plan.reserve(),plan.reserve();ra,rb=episode_noise(a),episode_noise(b)
        x=[ra.standard_normal(52) for _ in range(3)];rb.standard_normal(52);ra=episode_noise(a)
        np.testing.assert_array_equal(np.stack(x),np.stack([ra.standard_normal(52) for _ in range(3)]))
        self.assertFalse(np.array_equal(x[0],episode_noise(b).standard_normal(52)))
    def test_torn_or_repurposed_reservations_rejected(self):
        p=self.root/'reservations';plan=EpisodePlan(p,34,98);plan.reserve()
        with self.assertRaises(ValueError):EpisodePlan(p,35,98)
        with p.open('a') as f:f.write('{"ordinal":')
        with self.assertRaises(ValueError):EpisodePlan(p,34,98)
    def buffer(self,row,tag):
        contact=np.zeros(10);contact[8]=2;details=np.full((3,16),tag,dtype=np.float32)
        return EpisodeBuffer(row,np.full(131,tag),np.full(4,tag),np.full(276,tag),np.full(156,tag),contact,details,np.zeros(2),{'seed':row['seed']})
    def step(self,buffer,tag,done=False,reason=0):
        t=buffer.controls+1;contact=np.zeros(10);contact[8]=1;u=np.zeros(52);u[:26]=tag+t
        buffer.step(dict(state=np.full(131,tag+t),generalized=np.full(4,tag+t),request=u,action=u*.01,applied=u,reward_terms=np.zeros(10),contact=contact,phase=buffer.reservation['phase']+t,count=t,finite=True,done=done,reason=reason,regular=False,strict=False),np.full((3,16),tag+t))
    def test_interleaved_complete_episodes_keep_own_data_history_contacts(self):
        w=self.writer();plan=EpisodePlan(w.path/'pool_reservations.jsonl',12,98);a=self.buffer(plan.reserve(),1);b=self.buffer(plan.reserve(),100)
        for _ in range(8):self.step(a,1);self.step(b,100)
        self.step(b,100,True,2);eb=b.finish(w,1,False,'uniform_reference');self.step(a,1,True,1);ea=a.finish(w,0,False,'uniform_reference')
        self.assertEqual(plan.verify_commits(w.entries),{0,1});shard=EpisodeShard(w.path)
        batch=shard.window([ea['id'],eb['id']],[0,0],8)
        self.assertEqual(float(batch['commands'][0,0,0]),2);self.assertEqual(float(batch['commands'][1,0,0]),101)
        self.assertEqual(float(batch['start'][0,53]),1);self.assertEqual(float(batch['start'][1,53]),100)
        self.assertEqual(ea['lengths']['contact_details'],11);self.assertEqual(eb['lengths']['contact_ranges'],10)
        self.assertEqual(complete_controls(w.entries),18)
    def test_budget_truncation_retained_but_ineligible(self):
        w=self.writer();plan=EpisodePlan(w.path/'pool_reservations.jsonl',12,98);b=self.buffer(plan.reserve(),2)
        for _ in range(8):self.step(b,2)
        e=b.finish(w,0,False,'uniform_reference');self.assertEqual(e['termination'],'budget_truncation');self.assertEqual(complete_controls(w.entries),0)
        with self.assertRaises(AssertionError):EpisodeShard(w.path).window([0],[0],8)
    def test_duplicate_or_mismatched_commit_rejected(self):
        plan=EpisodePlan(self.root/'reservations',12,98);r=plan.reserve();e={'seed':r['seed'],'phase_start':r['phase'],'metadata':{'collection_pool_ordinal':0}}
        with self.assertRaises(ValueError):plan.verify_commits([e,e])
        e['seed']+=1
        with self.assertRaises(ValueError):plan.verify_commits([e])
    def test_zero_control_reservation_cannot_be_fake_episode(self):
        w=self.writer();plan=EpisodePlan(w.path/'pool_reservations.jsonl',12,98);b=self.buffer(plan.reserve(),1)
        with self.assertRaises(AssertionError):b.finish(w,0,False,'uniform_reference')
        self.assertFalse(w.entries)
if __name__=='__main__':unittest.main(verbosity=2)
