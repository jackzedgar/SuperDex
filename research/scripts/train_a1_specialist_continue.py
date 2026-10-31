"""Same-method continuation inside the originally approved one GPUh per-task cap."""
import argparse,datetime,hashlib,json,os,signal,socket,getpass,time,traceback
from pathlib import Path
import numpy as np
import torch
from a1_policy import CollectorPolicy
from a1_multitask_runner import MultiRunner
from train_a1_collector import ppo_update,evaluate

ROOT=Path(__file__).resolve().parents[1]
STOP=False
def on_stop(*_):
    global STOP
    STOP=True
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def assert_same_tree(a,b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:assert_same_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same_tree(x,y)
    else:assert a==b

def assert_original_reset(env):
    world=env._world(env.qpos[:,:env.task.nu])
    assert torch.equal(env.ctrl,env.qpos[:,:env.task.nu])
    assert torch.equal(env.history,world[:,:,None].expand(-1,-1,3,-1))
    assert torch.equal(env.servo,world[:,:,:6])
    assert int(env.phase)==0 and not bool(env.last_action.any())

class EvaluationGuard:
    def __init__(self,env,deadline):self.env=env;self.deadline=deadline;self.controls=0
    def __getattr__(self,name):return getattr(self.env,name)
    def reset(self,*args,**kwargs):
        value=self.env.reset(*args,**kwargs);assert_original_reset(self.env);return value
    def step(self,action):
        if STOP or time.monotonic()>=self.deadline:raise InterruptedError('Bounded evaluation interrupted; no incomplete episode is a success.')
        value=self.env.step(action);self.controls+=1;return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran'
    config_path=ROOT/a.config;c=json.loads(config_path.read_text())
    assert c['scope']=='same_method_remaining_short_budget'
    assert c['max_controls']==65536 and 60<=c['job_limit_seconds']<=3600 and c['training_seconds_cap']<=2400
    assert c['training_reset_memory_mode']=='velocity_consistent' and c['reset_mode']=='uniform_reference'
    assert c['batch_controls']==2048 and c['lr']==3e-5
    for name,digest in c['pinned_sources'].items():assert sha(ROOT/name)==digest,name
    parent_path=ROOT/c['parent_checkpoint'];assert sha(parent_path)==c['parent_checkpoint_sha256']
    manifest_path=ROOT/c['manifest'];assert sha(manifest_path)==c['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());assert len(manifest['tasks'])==40
    task=[row for row in manifest['tasks'] if row['task_id']==c['task']];assert len(task)==1
    selected=dict(manifest);selected['tasks']=task
    out=ROOT/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    torch.set_num_threads(2)
    start=time.monotonic();deadline=start+c['job_limit_seconds']-60
    train_deadline=start+c['training_seconds_cap']
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    assert parent['formal_A1'] is False and parent['new_controls']==32768 and parent['new_updates']==16
    assert parent['configuration']['task']==c['task'] and parent['parent_controls']==2710699
    assert parent['configuration']['training_reset_memory_mode']==c['training_reset_memory_mode']
    rng=dict(rng=parent['worker_rng'],torch_rng=parent['torch_rng'],cuda_rng=parent['cuda_rng'],iteration=parent['new_updates'])
    policy=CollectorPolicy(280043).cuda()
    policy.load_state_dict(parent['policy'])
    opt=torch.optim.Adam(policy.parameters(),lr=3e-5,betas=(.9,.95))
    opt.load_state_dict(parent['optimizer'])
    generator=torch.Generator(device='cuda');generator.set_state(parent['ppo_generator'].cpu())
    assert_same_tree(policy.state_dict(),parent['policy']);assert_same_tree(opt.state_dict(),parent['optimizer'])
    assert torch.equal(generator.get_state().cpu(),parent['ppo_generator'].cpu())
    write(out/'restore_gate.json',dict(status='PASS',policy_optimizer_generator_exact=True,
         parent_sha256=c['parent_checkpoint_sha256'],restored_rng_source='specialist_checkpoint_before_endpoint_evaluation',root_worker_rng_rank=c['parent_worker_rng_rank'],
         source_specialist_updates=int(rng['iteration']),gpu_uuid=os.environ.get('SUPERDEX_GPU_UUID'),time_beijing=now()))
    runner=MultiRunner(selected,policy,280043,cache_limit=1,reset_mode='uniform_reference',
                       determinism='constraint',reset_memory_mode='velocity_consistent')
    runner.load_stats({c['task']:parent['global_sampler'][c['task']]})
    env=runner.active.env
    # Check the original formal initializer before any PPO update; no rollout.
    env.reset(0,seed=c['implementation_reset_seed'],randomize=True);assert_original_reset(env)
    runner.rng.bit_generator.state=rng['rng']
    torch.set_rng_state(rng['torch_rng'].cpu());torch.cuda.set_rng_state(rng['cuda_rng'].cpu())
    runner.reset()
    write(out/'run_configuration.json',dict(c,started_beijing=now(),
        config_sha256=sha(config_path),policy_bank_scope='two selected training tasks only; not a complete40-task bank',
        full_A1=False,quality_passports_allowed=False,restored_rng_not_uninterrupted_physical_trajectory=True))
    # Existing baseline is copied by reference, never rerun or relabeled as new.
    prior=json.loads((ROOT/c['parent_evaluation']).read_text())['all_conditions']
    prior=[r for r in prior if r['task']==c['task']]
    assert len(prior)==4 and {r['seed'] for r in prior}==set(c['development_eval_seeds'])
    write(out/'parent_development_reference.json',dict(source=c['parent_evaluation'],source_sha256=sha(ROOT/c['parent_evaluation']),conditions=prior,new_evaluation=False))
    controls=0;updates=0;uncommitted=0;reason='control_target';evaluation_rows=[]
    def checkpoint(name):
        save(out/name,dict(policy={k:v.detach().cpu() for k,v in policy.state_dict().items()},
            optimizer=opt.state_dict(),new_controls=controls,new_updates=updates,
            parent_controls=parent['parent_controls'],parent_iteration=parent['parent_iteration'],
            prior_task_controls=parent['new_controls'],prior_task_updates=parent['new_updates'],
            cumulative_task_controls=parent['new_controls']+controls,cumulative_task_updates=parent['new_updates']+updates,
            ppo_generator=generator.get_state(),worker_rng=runner.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            global_sampler=runner.stats(),configuration=c,formal_A1=False,quality_passports_allowed=False))
    checkpoint('checkpoint_initial.pt')
    try:
        while controls<c['max_controls'] and not STOP and time.monotonic()<train_deadline:
            runner.episodes=[];before=time.monotonic()
            data=runner.rollout(2048,lambda:STOP or time.monotonic()>=train_deadline)
            if data is None:reason='training_time_or_signal';break
            n=len(data['reward'])
            if n!=2048:
                uncommitted=n;save(out/'pending_rollout.pt',{k:v.cpu() for k,v in data.items()});reason='training_time_or_signal';break
            assert all(bool(torch.isfinite(v).all()) for v in data.values())
            assert torch.equal(data['mask'],env.task.mask.expand(n,2))
            if updates==0:
                # Validate captured behavior probabilities against the unchanged actor.
                with torch.no_grad():
                    ids=torch.arange(0,n,67,device='cuda')
                    dist=policy.distribution(data['obs'][ids],data['goal'][ids])
                    active=data['mask'][ids].repeat_interleave(26,-1)
                    lp=(dist.log_prob(data['action'][ids])*active).sum(-1)
                    difference=float((lp-data['old_logprob'][ids]).abs().max())
                    assert difference<=5e-4,('behavior_logprob_mismatch',difference)
                write(out/'preupdate_gate.json',dict(status='PASS',batch_controls=2048,
                    finite=True,task_mask_exact=True,original_evaluation_reset_checked=True,
                    behavior_logprob_max_difference=difference,time_beijing=now()))
            metrics=ppo_update(policy,opt,data,generator,3e-5)
            controls+=n;updates+=1
            row=dict(time_beijing=now(),task=c['task'],new_controls=controls,updates=updates,cumulative_task_controls=parent['new_controls']+controls,
                seconds=time.monotonic()-start,update_seconds=time.monotonic()-before,
                mean_rollout_reward=float(data['reward'].mean()),
                physical_training_controls=runner.real_controls,**metrics)
            with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            with (out/'training_episodes.jsonl').open('a') as f:
                for episode in runner.episodes:f.write(json.dumps(dict(update=updates,**episode))+'\n')
            checkpoint('checkpoint_latest.pt')
            write(out/'progress.json',dict(status='TRAINING',**row))
            print('UPDATE',updates,controls,flush=True)
        if STOP:reason='budget_or_signal'
        elif controls<c['max_controls']:reason='training_time_cap'
        checkpoint('checkpoint_final.pt')
        guard=EvaluationGuard(env,deadline)
        for seed in c['development_eval_seeds']:
            if STOP or time.monotonic()>=deadline:break
            try:rows=evaluate(guard,policy,[seed])
            except InterruptedError:break
            evaluation_rows += [dict(task=c['task'],complete_episode=True,**r) for r in rows]
            write(out/'evaluation_progress.json',dict(conditions=evaluation_rows,complete=len(evaluation_rows)==4))
        complete=len(evaluation_rows)==4
        write(out/'evaluation_final.json',dict(scope='single_task_development_not_full160',
            task=c['task'],new_controls=controls,conditions=evaluation_rows,complete=complete,
            regular=sum(r['regular_success'] for r in evaluation_rows),strict=sum(r['strict_success'] for r in evaluation_rows),
            n=len(evaluation_rows),original_phase0_memory=True,quality_passport=False))
        write(out/'summary.json',dict(status=reason,task=c['task'],new_controls=controls,updates=updates,
            prior_task_controls=parent['new_controls'],cumulative_task_controls=parent['new_controls']+controls,
            physical_training_controls=runner.real_controls,uncommitted_rollout_controls=uncommitted,
            physical_evaluation_controls=guard.controls,physical_total_env_controls=env.physical_controls,
            full_development_for_this_task=complete,formal_A1_complete=False,
            checkpoint_sha256=sha(out/'checkpoint_final.pt'),seconds=time.monotonic()-start,time_beijing=now()))
    except BaseException as exc:
        checkpoint('checkpoint_failed_last_committed.pt')
        write(out/'fatal.json',dict(error=repr(exc),traceback=traceback.format_exc(),new_controls=controls,updates=updates,time_beijing=now()))
        raise

if __name__=='__main__':main()
