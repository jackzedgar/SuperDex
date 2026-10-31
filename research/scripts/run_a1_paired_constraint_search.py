"""Authorized seventh bounded diagnostic; frozen experts,original physics,no PPO."""
import argparse,datetime,getpass,hashlib,json,os,signal,socket,time,traceback
from pathlib import Path
import numpy as np
import torch
from a1_policy import CollectorPolicy
from a1_original_action_search_pool import WitnessPool
from a1_original_action_search_math import knot_times
from a1_paired_constraint_search_math import PairedCEM,OPTIMIZATION_SEEDS,DEVELOPMENT_SEEDS

R=Path(__file__).resolve().parents[1];STOP=False
def halt(*_):
    global STOP
    STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
    c=json.loads(args.config.read_text());assert c['scientific_revision']==7 and c['phase']=='revision7_paired_constraint_search'
    assert c['job_limit_seconds']==1620 and c['population']==32 and c['max_generations']==12 and c['optimizer_seed']==410032
    assert tuple(c['optimization_seeds'])==OPTIMIZATION_SEEDS and tuple(c['development_seeds'])==DEVELOPMENT_SEEDS
    assert c['authorization_reply']=='批准第七项≤0.9 GPUh验证'
    for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
    parent_path=R/c['checkpoint'];prior_path=R/c['prior_selection']
    assert sha(parent_path)==c['checkpoint_sha256'] and sha(prior_path)==c['prior_selection_sha256']
    out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
    start=time.monotonic();deadline=start+1530;search_deadline=start+1100;pool=None;records=[];generations=[]
    write(out/'run_configuration.json',dict(c,started_beijing=now(),network_updates=False,full_A1_complete=False))
    def guard():return STOP or time.monotonic()>=deadline
    def search_guard():return guard() or time.monotonic()>=search_deadline
    def persist(rows,label,names=None,candidate_indices=None,generation=None):
        for i,row in enumerate(rows):
            trace=row.pop('trace');p=out/f'trace_{label}_{i:02d}.npz'
            assert not p.exists();np.savez_compressed(p,**{k:np.asarray(v) for k,v in trace.items()})
            row.update(label=label,within_batch_index=i,trace_file=p.name,trace_sha256=sha(p),time_beijing=now())
            if names is not None:row['controller']=names[i]
            if candidate_indices is not None:row.update(candidate_index=int(candidate_indices[i]),generation=int(generation))
            with (out/'conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
            records.append(row)
        return rows
    def progress(stage,**fields):
        v=dict(stage=stage,time_beijing=now(),seconds=time.monotonic()-start,physical_controls=pool.controls if pool else 0,episodes_recorded=len(records),**fields)
        write(out/'progress.json',v);print(json.dumps(v),flush=True)
    try:
        parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        assert parent['configuration']['task_id']==c['task_id'] and parent['configuration']['scientific_revision']==4 and not parent['smoke']
        policy=CollectorPolicy(280043).cuda();policy.load_state_dict(parent['policy']);policy.eval()
        for p in policy.parameters():p.requires_grad_(False)
        pool=WitnessPool(R/c['task_path'],policy);assert pool.horizon==c['full_horizon']
        progress('IMPLEMENTATION_GATE')
        impl=persist(pool.run(np.zeros((2,4,52),np.float32),list(OPTIMIZATION_SEEDS),guard,validate=True),'implementation_zero',names=['baseline']*2)
        assert len(impl)==2 and all(r['complete'] for r in impl) and {r['seed'] for r in impl}==set(OPTIMIZATION_SEEDS)
        assert all(not np.asarray(r['knots']).any() for r in impl)
        write(out/'implementation_gate.json',dict(status='PASS',conditions=impl,audit=pool.audit.copy(),original_nworld1=True,source_hashes=c['source_hashes']))
        prior=json.loads(prior_path.read_text());assert prior['checkpoint_sha256']==c['checkpoint_sha256'] and prior['scientific_revision']==6
        np.testing.assert_array_equal(prior['knot_times'],knot_times(pool.horizon))
        initial=np.asarray(prior['knots'],np.float32);cem=PairedCEM(pool.task.mask[0].cpu().numpy(),initial,c['optimizer_seed'])
        write(out/'search_initialization.json',dict(prior_path=c['prior_selection'],prior_sha256=c['prior_selection_sha256'],initial_knots=initial.tolist(),old_score_reused=False,optimizer_seed=c['optimizer_seed'],optimization_seeds=list(OPTIMIZATION_SEEDS),knot_times=knot_times(pool.horizon).tolist()))
        for g in range(12):
            if search_guard():break
            population=cem.population();np.save(out/f'population_{g}.npy',population);rows=[]
            # Each8-lane batch evaluates four candidates on both optimization
            # seeds. Rows retain candidate+generation+seed identities on disk.
            for offset in range(0,32,4):
                if search_guard():break
                ids=np.repeat(np.arange(offset,offset+4),2);seeds=list(OPTIMIZATION_SEEDS)*4
                progress('SEARCH',generation=g,candidate_offset=offset)
                rows+=persist(pool.run(population[ids],seeds,search_guard,validate=False),f'g{g}_b{offset}',candidate_indices=ids,generation=g)
            if len(rows)!=64 or not all(r['complete'] for r in rows):
                write(out/'incomplete_generation.json',dict(generation=g,records=len(rows),complete=sum(r['complete'] for r in rows),used_for_update=False));break
            selected=cem.update(population,rows,pool.horizon,g)
            generations.append(dict(generation=g,**selected,regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),conditions=64))
            write(out/'search_state.json',dict(generations=generations,mean=cem.mean.tolist(),std=cem.std.tolist(),best=cem.best.tolist(),best_score=cem.best_score,rng=cem.rng.bit_generator.state,selection_uses_only_optimization_seeds=list(OPTIMIZATION_SEEDS)))
            progress('GENERATION_COMPLETE',**generations[-1])
            if selected['both_seed_regular_candidates']:break
        replay=[];development=[];selection_sha=None
        if generations and not guard():
            selection=dict(checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],knots=cem.best.tolist(),selected_training_score=cem.best_score,optimization_seeds=list(OPTIMIZATION_SEEDS),knot_times=knot_times(pool.horizon).tolist(),scientific_revision=7,selected_beijing=now(),development_not_opened_yet=True,policy_weights_updated=False,quality_passport=False)
            write(out/'selected_controller.json',selection);selection_sha=sha(out/'selected_controller.json')
            for seed in OPTIMIZATION_SEEDS:
                if guard():break
                names=['baseline','selected','selected','baseline','baseline','selected']
                params=np.stack([np.zeros((4,52),np.float32) if name=='baseline' else cem.best for name in names])
                progress('FROZEN_OPTIMIZATION_SEED_REPLAY',seed=seed)
                replay+=persist(pool.run(params,[seed]*6,guard,validate=False),f'replay_s{seed}',names=names)
            if len(replay)==12 and all(r['complete'] for r in replay) and not guard():
                assert sha(out/'selected_controller.json')==selection_sha
                progress('FIXED_DEVELOPMENT_NO_RETUNING')
                development=persist(pool.run(np.stack([cem.best]*4),list(DEVELOPMENT_SEEDS),guard,validate=False),'development',names=['selected']*4)
                write(out/'development.json',dict(complete=len(development)==4 and all(r['complete'] for r in development),conditions=development,regular=sum(r['complete'] and r['regular_success'] for r in development),strict=sum(r['complete'] and r['strict_success'] for r in development),n=len(development),scope='single_task_original_full_episode_development_not_full160_not_G1',selection_sha256=selection_sha,selection_frozen_before_development=True,retuned=False))
        assert all(torch.equal(v.detach().cpu(),parent['policy'][k]) for k,v in policy.state_dict().items())
        assert sha(parent_path)==c['checkpoint_sha256'] and sha(prior_path)==c['prior_selection_sha256']
        if selection_sha:assert sha(out/'selected_controller.json')==selection_sha
        assert sum(r['controls'] for r in records)==pool.controls and sum(e.physical_controls for e in pool.envs)==pool.controls+pool.warmup_controls
        selected=[r for r in replay if r['controller']=='selected'];baseline=[r for r in replay if r['controller']=='baseline']
        witness=len(selected)==6 and all(sum(r['seed']==s for r in selected)==3 for s in OPTIMIZATION_SEEDS) and all(r['complete'] and r['regular_success'] for r in selected)
        complete=len(development)==4 and {r['seed'] for r in development}==set(DEVELOPMENT_SEEDS) and all(r['complete'] for r in development)
        status='COMPLETE' if complete else 'BOUNDED_STOP_INCOMPLETE'
        summary=dict(status=status,time_beijing=now(),seconds=time.monotonic()-start,task=c['task_id'],generations=generations,physical_controls=pool.controls,warmup_controls=pool.warmup_controls,physical_env_controls=sum(e.physical_controls for e in pool.envs),episodes=len(records),complete_episodes=sum(r['complete'] for r in records),audit=pool.audit,witness_gate_passed=witness,selected_replay_regular=sum(r['complete'] and r['regular_success'] for r in selected),selected_replay_n=len(selected),baseline_replay_regular=sum(r['complete'] and r['regular_success'] for r in baseline),baseline_replay_n=len(baseline),development_complete=complete,development_n=len(development),development_regular=sum(r['complete'] and r['regular_success'] for r in development),development_strict=sum(r['complete'] and r['strict_success'] for r in development),original_policy_checkpoint_sha256=c['checkpoint_sha256'],policy_parameters_unchanged=True,optimizer_steps=0,full_A1_complete=False,quality_passports_created=False,negative_result_proves_infeasibility=False)
        write(out/'summary.json',summary);progress(status)
    except BaseException as exc:
        write(out/'fatal.json',dict(error=repr(exc),traceback=traceback.format_exc(),time_beijing=now(),seconds=time.monotonic()-start,physical_controls=pool.controls if pool else None,episodes_saved=len(records)));raise
if __name__=='__main__':main()
