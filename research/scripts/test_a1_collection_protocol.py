import copy,unittest
from a1_collection_protocol import FROZEN_COLLECTION_PROTOCOL,formal_collection_reset_mode

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ck={'configuration':{'reset_mode':'formal_episode_start','scientific_revision':4,'training_memory':'original'}}
        self.pp={'schema_version':2,'reset_mode':'uniform_reference','training_reset_mode':'formal_episode_start','collection_protocol':dict(FROZEN_COLLECTION_PROTOCOL)}
    def test_legacy_uniform_behavior_preserved(self):
        self.assertEqual(formal_collection_reset_mode({'configuration':{'reset_mode':'uniform_reference'}},{'reset_mode':'uniform_reference'}),'uniform_reference')
    def test_formal_training_does_not_change_collection(self):
        before=copy.deepcopy((self.ck,self.pp));self.assertEqual(formal_collection_reset_mode(self.ck,self.pp),'uniform_reference');self.assertEqual((self.ck,self.pp),before)
    def test_no_implicit_legacy_relabeling(self):
        with self.assertRaises(AssertionError):formal_collection_reset_mode(self.ck,{'reset_mode':'uniform_reference'})
    def test_reject_changed_collection_physics_memory_or_noise(self):
        for key,val in [('reset_mode','formal_episode_start'),('reset_memory_mode','velocity_consistent'),('policy_noise_sigma',0.0)]:
            p=copy.deepcopy(self.pp);p['collection_protocol'][key]=val
            with self.assertRaises(AssertionError):formal_collection_reset_mode(self.ck,p)
    def test_reject_incorrect_training_provenance(self):
        p=copy.deepcopy(self.pp);p['training_reset_mode']='uniform_reference'
        with self.assertRaises(AssertionError):formal_collection_reset_mode(self.ck,p)
    def test_reject_unregistered_mode_or_revision(self):
        for key,val in [('scientific_revision',5),('training_memory','velocity_consistent'),('reset_mode','adaptive_anchor')]:
            c=copy.deepcopy(self.ck);c['configuration'][key]=val
            with self.assertRaises(AssertionError):formal_collection_reset_mode(c,self.pp)
if __name__=='__main__':unittest.main()
