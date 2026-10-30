import copy,unittest
from a1_remote_budget import make_remote_grant,reconcile_remote_receipt,canonical_hash,reserved_seconds


class RemoteBudgetTests(unittest.TestCase):
    def setUp(self):
        self.phases=dict(implementation=7*3600,collectors=45*3600,collection=150*3600,models_initial=30*3600,evaluation=8*3600,revision1=45*3600,revision2=45*3600)
        self.ledger=dict(cap_seconds=330*3600,phase_caps=self.phases.copy(),jobs=[])
        self.registry=dict(global_cap_seconds=330*3600,maximum_single_validation_seconds=2700,
            authorization_files=['explicit_user_authorization'],allowed_scripts=['scripts/test_job.py'],
            nodes=dict(remote=dict(hostname='remote',account='researcher',working_directory='/owned/work',python='/owned/venv/bin/python',
                                   gpus={'0':'GPU-remote'},max_process_gpu_mib=4096,free_gpu_floor_mib=8192,allow_shared_gpu=True)))
        self.request=dict(name='test_001',node='remote',gpu=0,gpu_uuid='GPU-remote',phase='implementation',limit_seconds=900,
            max_process_gpu_mib=4096,free_gpu_floor_mib=8192,command=['scripts/test_job.py'],source_hashes={'scripts/test_job.py':'a'*64},job_registration_sha256='b'*64)

    def grant(self):return make_remote_grant(self.request,self.ledger,self.phases,self.registry,1000,'nonce')

    def receipt(self,seconds=20):
        g=self.grant()
        return dict(name='test_001',grant_sha256=canonical_hash(g),hostname='remote',account='researcher',gpu_uuid='GPU-remote',
                    status='finished',seconds=seconds,started=1005,ended=1005+seconds,returncode=0,reason='completed',identity_checked_alive=False)

    def test_remote_reservation_is_full_limit_not_spent_seconds(self):
        self.grant();self.assertEqual(reserved_seconds(self.ledger['jobs'],1010),900)
        self.assertEqual(self.ledger['jobs'][0]['status'],'remote_reserved')

    def test_old_local_same_index_different_uuid_does_not_block_remote(self):
        self.ledger['jobs'].append(dict(name='local',gpu=0,gpu_uuid='GPU-local',phase='implementation',status='running',started=999,limit_seconds=900))
        self.grant();self.assertEqual(len(self.ledger['jobs']),2)

    def test_duplicate_attempt_and_duplicate_gpu_rejected(self):
        self.grant()
        with self.assertRaises(AssertionError):self.grant()
        self.request['name']='test_002'
        with self.assertRaises(AssertionError):self.grant()

    def test_stage_cap_uses_pending_remote_limit(self):
        self.ledger['jobs'].append(dict(name='used',phase='implementation',status='finished',seconds=24500))
        with self.assertRaises(AssertionError):self.grant()

    def test_global_cap_keeps_evaluation_reserve(self):
        self.ledger['jobs'].append(dict(name='used',phase='revision1',status='finished',seconds=329*3600-800))
        with self.assertRaises(AssertionError):self.grant()

    def test_gpu_memory_and_script_must_be_registered(self):
        for field,value in [('gpu_uuid','GPU-other'),('max_process_gpu_mib',8192),('command',['/bin/sh'])]:
            old=copy.deepcopy(self.request);self.request[field]=value
            with self.assertRaises(AssertionError):self.grant()
            self.request=old

    def test_final_actual_time_replaces_reservation_and_is_immutable(self):
        r=self.receipt();reconcile_remote_receipt(self.ledger,r,1030)
        self.assertEqual(reserved_seconds(self.ledger['jobs'],1030),20)
        self.assertEqual(reconcile_remote_receipt(self.ledger,r,1031)['status'],'already_reconciled')
        r['seconds']=19
        with self.assertRaises(AssertionError):reconcile_remote_receipt(self.ledger,r,1032)

    def test_overshoot_is_reported_and_fully_charged_not_clamped(self):
        r=self.receipt(901);out=reconcile_remote_receipt(self.ledger,r,2000)
        self.assertTrue(out['budget_violation']);self.assertEqual(self.ledger['jobs'][0]['seconds'],901)

    def test_wrong_identity_or_live_final_receipt_rejected(self):
        r=self.receipt()
        for field,value in [('hostname','other'),('account','other'),('grant_sha256','wrong'),('identity_checked_alive',True)]:
            bad=copy.deepcopy(r);bad[field]=value
            with self.assertRaises(AssertionError):reconcile_remote_receipt(self.ledger,bad,1030)


if __name__=='__main__':unittest.main()
