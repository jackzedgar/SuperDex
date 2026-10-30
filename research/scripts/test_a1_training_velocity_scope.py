"""CPU-only execution of the real reset method against a minimal test environment."""
import ast,types,unittest
from pathlib import Path
import torch

source=(Path(__file__).parent/'a1_env.py').read_text()
cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='A1Env')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset');method.decorator_list=[]


class FakeEnv:
    def __init__(self,allowed=False):
        self.allow_training_velocity_override=allowed;self.device='cpu';self.forward_seen=[];self.randomize_calls=0
        self.task=types.SimpleNamespace(nu=26,qpos=torch.arange(66,dtype=torch.float32).reshape(2,33),qvel=torch.arange(64,dtype=torch.float32).reshape(2,32))
        self.qpos=torch.zeros(1,33);self.qvel=torch.zeros(1,32);self.ctrl=torch.zeros(1,26)
        self.history=torch.zeros(1,2,3,26);self.servo=torch.zeros(1,2,6);self.hold_command=torch.zeros(1,2,26)
        self.hold_mode=torch.ones(1,1,dtype=torch.bool);self.phase=torch.zeros(1,dtype=torch.long);self.episode_start=self.phase.clone()
        self.last_action=torch.zeros(1,52);self.actions=self.last_action.clone();self.sums=torch.zeros(1,6);self.count=torch.zeros(1);self.all_strict=torch.ones(1,dtype=torch.bool)
        self.wd=types.SimpleNamespace(time=torch.ones(1),qacc_warmstart=torch.ones(1,32),qacc=torch.ones(1,32),qfrc_applied=torch.ones(1,32),xfrc_applied=torch.ones(1,2,6))
        self.wm=self
    def randomize(self,*_):self.randomize_calls+=1
    def _world(self,q):return torch.stack([q,torch.zeros_like(q)],1)
    def state(self):return torch.cat([self.qpos,self.qvel],1)


namespace=dict(torch=torch,wp=types.SimpleNamespace(to_torch=lambda x:x),mjw=types.SimpleNamespace(forward=lambda m,d:m.forward_seen.append(m.qvel.clone())))
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'real_reset','exec'),namespace)
FakeEnv.reset=namespace['reset']


class TrainingVelocityScope(unittest.TestCase):
    def test_default_reference_reset_unchanged(self):
        e=FakeEnv();e.reset(1)
        self.assertTrue(torch.equal(e.qpos,e.task.qpos[1:2]));self.assertTrue(torch.equal(e.qvel,e.task.qvel[1:2]));self.assertEqual(len(e.forward_seen),1)

    def test_only_hand_velocity_changes_before_single_forward(self):
        e=FakeEnv(True);q=e.task.qpos.clone();v=e.task.qvel.clone();override=torch.linspace(-1,1,26)
        e.reset(1,training_hand_qvel=override)
        self.assertTrue(torch.equal(e.qvel[:,:26],override[None]));self.assertTrue(torch.equal(e.qvel[:,26:],v[1:2,26:]))
        self.assertTrue(torch.equal(e.qpos,q[1:2]));self.assertTrue(torch.equal(e.task.qpos,q));self.assertTrue(torch.equal(e.task.qvel,v))
        self.assertEqual(len(e.forward_seen),1);self.assertTrue(torch.equal(e.forward_seen[0],e.qvel))

    def test_opt_in_required_before_any_mutation(self):
        e=FakeEnv()
        with self.assertRaises(AssertionError):e.reset(1,training_hand_qvel=torch.zeros(26))
        self.assertEqual(e.randomize_calls,0);self.assertEqual(len(e.forward_seen),0)

    def test_snapshot_and_invalid_override_rejected_before_mutation(self):
        for override,snapshot in [(torch.zeros(25),None),(torch.full((26,),float('nan')),None),(torch.zeros(26),{})]:
            e=FakeEnv(True)
            with self.assertRaises(AssertionError):e.reset(1,snapshot=snapshot,training_hand_qvel=override)
            self.assertEqual(e.randomize_calls,0)

    def test_following_phase0_evaluation_restores_original_velocity(self):
        e=FakeEnv(True);e.reset(1,training_hand_qvel=torch.zeros(26));e.reset(0)
        self.assertTrue(torch.equal(e.qvel,e.task.qvel[:1]));self.assertTrue(torch.equal(e.qpos,e.task.qpos[:1]));self.assertEqual(len(e.forward_seen),2)


if __name__=='__main__':unittest.main()
