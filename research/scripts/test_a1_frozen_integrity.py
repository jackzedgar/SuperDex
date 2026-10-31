import copy,hashlib,json,os,unittest
from unittest.mock import patch
from pathlib import Path
from test_a1_episode_and_g1_contract import EpisodeChecks
import a1_window_pool as wp

class FrozenChecks(unittest.TestCase):
    def setUp(self):
        self.fixture=EpisodeChecks('test_freeze_counts_complete_records_and_preserves_all_file_hashes');self.fixture.setUp()
        self.fixture.test_freeze_counts_complete_records_and_preserves_all_file_hashes()
        self.root=self.fixture.root;self.manifest=self.root/'frozen.json';self.shard=self.fixture.path
    def tearDown(self):self.fixture.tearDown()
    def load(self):return wp.WindowPool(self.manifest,'train',8)
    def write_meta(self,meta):self.manifest.write_text(json.dumps(meta))
    def test_intact_frozen_data_reads_same_windows(self):
        p=self.load();self.assertEqual(sum(int(v[-1]) for v in p.cdf.values()),8)
    def test_same_size_changed_data_rejected_even_after_cached_read(self):
        self.load();p=self.shard/'state.bin';old=p.read_bytes();stat=p.stat();p.write_bytes(bytes([old[0]^1])+old[1:]);os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_index_mutation_rejected(self):
        p=self.shard/'episodes.jsonl';p.write_text(p.read_text().replace('"seed":','"seed":0,"previous_seed":',1))
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_missing_hash_for_used_column_rejected(self):
        m=json.loads(self.manifest.read_text());del m['shards'][0]['files']['request.bin'];self.write_meta(m)
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_geometry_mutation_rejected(self):
        (self.root/'data/a1_geometry/objects/object0/sdf.npz').write_bytes(b'Synthetic')
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_probe_mutation_rejected(self):
        (self.root/'configs/a1_probe40_v1.json').write_text('{"tasks":[1]}')
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_formal_metadata_cannot_skip_required_hashes(self):
        m=json.loads(self.manifest.read_text());m.update(formal=True,status='FROZEN_FULL_A1_DATA_V1');m.pop('frozen_integrity_version',None);self.write_meta(m)
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_hand_geometry_binding_and_cache(self):
        m=json.loads(self.manifest.read_text());m.update(formal=True,status='FROZEN_FULL_A1_DATA_V1',frozen_integrity_version=1,hand_kinematics={})
        for side in ['right','left']:
            p=self.root/f'data/a1_processed/assets/hands/{side}/kinematics.npz';p.parent.mkdir(parents=True);p.write_bytes(b'fakehand')
            m['hand_kinematics'][side]={'bytes':8,'sha256':hashlib.sha256(b'fakehand').hexdigest()}
        self.write_meta(m);self.load();p.write_bytes(b'FAKEHAND')
        with self.assertRaises((ValueError,AssertionError)):self.load()
    def test_unchanged_file_hash_is_reused_only_in_process(self):
        from a1_frozen_integrity import _VERIFIED
        _VERIFIED.clear();p=self.load();self.assertTrue(p.integrity['verified'])
        with patch('a1_frozen_integrity.hashlib.sha256',side_effect=AssertionError('Unexpected repeated full data hashing')):self.load()
if __name__=='__main__':unittest.main(verbosity=2)
