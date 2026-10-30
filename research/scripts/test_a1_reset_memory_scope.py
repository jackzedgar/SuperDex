import ast,unittest,types
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[1]
def methods(path,names):
 tree=ast.parse(path.read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in names:nodes.append(n)
  if isinstance(n,ast.ClassDef):nodes.extend(x for x in n.body if isinstance(x,ast.FunctionDef) and x.name in names)
 return ast.Module(body=nodes,type_ignores=[])
ns=dict(torch=torch);exec(compile(methods(ROOT/'scripts/a1_reset_memory.py',{'apply_velocity_reset_memory'}),'helper','exec'),ns)
apply=ns['apply_velocity_reset_memory']
def environment():
 e=types.SimpleNamespace(device='cpu',task=types.SimpleNamespace(nu=52),m=types.SimpleNamespace(actuator_gainprm=np.zeros((52,3)),actuator_biasprm=np.zeros((52,3))))
 e.m.actuator_gainprm[:,0]=10;e.m.actuator_biasprm[:,2]=-2
 e.qpos=torch.zeros(1,59);e.qvel=torch.zeros(1,58);e.qvel[:,:52]=torch.linspace(-.1,.1,52)
 e.context=torch.arange(8);e.lo=torch.full((52,),-1.);e.hi=torch.full((52,),1.)
 e.ctrl=torch.zeros(1,52);e.history=torch.zeros(1,2,3,26);e.servo=torch.zeros(1,2,6);e.current_state=torch.zeros(1,52)
 e._world=lambda target:target.reshape(1,2,26);e.state=lambda:e.ctrl.clone()
 def reset(*args,**kwargs):
  e.ctrl.copy_(e.qpos[:,:52]);e.history.zero_();e.servo.zero_();e.current_state.copy_(e.state())
 e.reset=reset;return e
class ResetMemoryTests(unittest.TestCase):
 def test_unsaturated_initial_pd_force_is_zero_and_physical_state_is_unchanged(self):
  e=environment();q=e.qpos.clone();v=e.qvel.clone();ctx=e.context.clone();apply(e)
  self.assertTrue(torch.allclose(10*(e.ctrl-q[:,:52])-2*v[:,:52],torch.zeros(1,52),atol=1e-7))
  for a,b in [(q,e.qpos),(v,e.qvel),(ctx,e.context)]:self.assertTrue(torch.equal(a,b))
  self.assertTrue(torch.equal(e.history[:,:,0],e.ctrl.reshape(1,2,26)))
  self.assertTrue(torch.equal(e.current_state,e.state()))
 def test_actuator_limits_remain_enforced(self):
  e=environment();e.qpos[:,:26]=3;e.qpos[:,26:52]=-3;apply(e)
  self.assertTrue(torch.equal(e.ctrl[:,:26],torch.ones(1,26)));self.assertTrue(torch.equal(e.ctrl[:,26:],-torch.ones(1,26)))
 def test_factor_matches_the_completed_preregistered_diagnostic(self):
  ref=dict(torch=torch);exec(compile(methods(ROOT/'scripts/diagnose_a1_reset_memory.py',{'velocity_memory'}),'reference','exec'),ref)
  e=environment();expected=ref['velocity_memory'](e.qpos,e.qvel,52,torch.full((52,),10.),torch.full((52,),-2.),e.lo,e.hi);apply(e)
  self.assertTrue(torch.equal(e.ctrl,expected))
 def test_training_resets_change_memory_but_complete_development_evaluation_does_not(self):
  e=environment();r=types.SimpleNamespace(env=e,state_history=torch.zeros(1,3,52),action_history=torch.zeros(1,3,52));seen=[]
  def evaluate(env,policy,seeds):
   env.reset(0,seed=seeds[0],randomize=True);seen.append(torch.equal(env.ctrl,env.qpos[:,:52]));return [dict(controls=1)]
  scope=dict(np=np,torch=torch,apply_velocity_reset_memory=apply,evaluate=evaluate)
  exec(compile(methods(ROOT/'scripts/a1_multitask_runner.py',{'reset','evaluation'}),'runner','exec'),scope)
  obj=types.SimpleNamespace(reset_mode='uniform_reference',reset_memory_mode='velocity_consistent',rng=np.random.default_rng(123),rows=[dict(task_id='task')],policy=None,evaluation_controls=0)
  obj.choose=lambda:('task',0,7,None,False);obj.get=lambda tid:r;obj.reset=types.MethodType(scope['reset'],obj)
  obj.reset();self.assertFalse(torch.equal(e.ctrl,e.qpos[:,:52]))
  scope['evaluation'](obj,0,1,[123]);self.assertEqual(seen,[True]);self.assertFalse(torch.equal(e.ctrl,e.qpos[:,:52]))
if __name__=='__main__':unittest.main()
