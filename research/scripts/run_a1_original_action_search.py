"""Authorized sixth bounded diagnostic; does not update a policy or create passports."""
import argparse,datetime,getpass,hashlib,json,signal,socket,time,traceback
from pathlib import Path
import numpy as np
import torch
from a1_policy import CollectorPolicy
from a1_original_action_search_math import FixedCEM,resample_prior,knot_times
from a1_original_action_search_pool import WitnessPool

R=Path(__file__).resolve().parents[1];STOP=False
def stopped(*_):
    global STOP
    STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
    c=json.loads(args.config.read_text());assert c['scientific_revision']==6 and c['job_limit_seconds']==1620 and c['max_generations']==12 and c['population']==64
    assert c['optimization_seed']==280043 and c['development_seeds']==[280046,280047,280048,280049]
    for path,digest in c['source_hashes'].items():assert sha(R/path)==digest,path
    parent_path=R/c['checkpoint'];assert sha(parent_path)==c['checkpoint_sha256']
    out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped);torch.set_num_threads(1)
    start=time.monotonic();deadline=start+1530;search_deadline=start+1100;pool=None;records=[];generations=[]
    write(out/'run_configuration.json',dict(c,started_beijing=now(),updates_network_weights=False,full_A1_complete=False))
    def guard():return STOP or time.monotonic()>=deadline
    def persist(rows,label,names=None):
        result=[]
        for i,row in enumerate(rows):
            trace=row.pop('trace');p=out/f'trace_{label}_{i:02d}.npz'
            np.savez_compressed(p,**{k:np.asarray(v) for k,v in trace.items()})
            row.update(label=label,within_batch_index=i,trace_file=p.name,trace_sha256=sha(p),time_beijing=now())
            if names is not None:row['controller']=names[i]
            with (out/'conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            records.append(row);result.append(row)
        return result
    def progress(stage,**fields):
        value=dict(stage=stage,time_beijing=now(),seconds=time.monotonic()-start,physical_controls=pool.controls if pool else 0,episodes_recorded=len(records),**fields)
        write(out/'progress.json',value);print(json.dumps(value),flush=True)
    try:
        parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        assert parent['configuration']['task_id']==c['task_id'] and parent['configuration']['scientific_revision']==4 and not parent['smoke']
        policy=CollectorPolicy(280043).cuda();policy.load_state_dict(parent['policy']);policy.eval()
        for parameter in policy.parameters():parameter.requires_grad_(False)
        pool=WitnessPool(R/c['task_path'],policy);assert pool.horizon==c['full_horizon']
        progress('IMPLEMENTATION_GATE')
        zeros=np.zeros((2,4,52),dtype=np.float32)
        impl=persist(pool.run(zeros,[280043]*2,guard,validate=True),'implementation_zero',names=['baseline']*2)
        assert len(impl)==2 and all(r['complete'] for r in impl),'Implementation check did not complete inside its bound'
        assert all(np.max(np.abs(r['knots']))==0 for r in impl)
        write(out/'implementation_gate.json',dict(status='PASS',audit=pool.audit.copy(),zero_offset_episodes=impl,original_nworld1=True,source_hashes=c['source_hashes'],physical_controls=pool.controls,seconds=time.monotonic()-start))
        prior_path=R/c['prior_selection'];assert sha(prior_path)==c['prior_selection_sha256']
        prior=json.loads(prior_path.read_text());assert prior['checkpoint_sha256']==c['checkpoint_sha256']
        initial=resample_prior(prior['knots'],pool.horizon,pool.task.mask[0].cpu().numpy())
        cem=FixedCEM(pool.task.mask[0].cpu().numpy(),initial,410031)
        write(out/'search_initialization.json',dict(prior_path=c['prior_selection'],prior_sha256=c['prior_selection_sha256'],initial_knots=initial.tolist(),knot_times=knot_times(pool.horizon).tolist(),old_score_reused=False,optimizer_seed=410031))
        for g in range(12):
            if STOP or time.monotonic()>=search_deadline:break
            population=cem.population();np.save(out/f'population_{g}.npy',population)
            rows=[]
            for offset in range(0,64,8):
                if STOP or time.monotonic()>=search_deadline:break
                progress('SEARCH',generation=g,batch_offset=offset)
                batch=pool.run(population[offset:offset+8],[280043]*8,lambda:STOP or time.monotonic()>=search_deadline,validate=False)
                rows+=persist(batch,f'g{g}_b{offset}')
            if len(rows)!=64 or not all(r['complete'] for r in rows):
                write(out/'incomplete_generation.json',dict(generation=g,records=len(rows),complete=sum(r['complete'] for r in rows),used_for_search_update=False));break
            selected=cem.update(population,rows,pool.horizon)
            generations.append(dict(generation=g,**selected,regular=sum(r['regular_success'] for r in rows),strict=sum(r['strict_success'] for r in rows),max_controls=max(r['controls'] for r in rows)))
            write(out/'search_state.json',dict(generations=generations,mean=cem.mean.tolist(),std=cem.std.tolist(),best=cem.best.tolist(),best_score=cem.best_score,rng=cem.rng.bit_generator.state,selection_uses_only_seed280043=True))
            progress('GENERATION_COMPLETE',**generations[-1])
            if any(r['regular_success'] for r in rows):break
        replay=[];development=[]
        if generations and not guard():
            selection=dict(checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256'],knots=cem.best.tolist(),selected_training_score=cem.best_score,
                selection_seed=280043,knot_times=knot_times(pool.horizon).tolist(),scientific_revision=6,selected_beijing=now(),development_not_opened_yet=True,policy_weights_updated=False,quality_passport=False)
            write(out/'selected_controller.json',selection);selection_sha=sha(out/'selected_controller.json')
            names=['baseline','selected','selected','baseline','baseline','selected']
            params=np.stack([np.zeros((4,52),dtype=np.float32) if n=='baseline' else cem.best for n in names])
            progress('FROZEN_TRAINING_SEED_REPLAY')
            replay=persist(pool.run(params,[280043]*6,guard,validate=False),'replay',names=names)
            if len(replay)==6 and all(r['complete'] for r in replay) and not guard():
                assert sha(out/'selected_controller.json')==selection_sha
                progress('FIXED_DEVELOPMENT_NO_RETUNING')
                development=persist(pool.run(np.stack([cem.best]*4),c['development_seeds'],guard,validate=False),'development',names=['selected']*4)
                write(out/'development.json',dict(complete=len(development)==4 and all(r['complete'] for r in development),conditions=development,
                    regular=sum(r['regular_success'] and r['complete'] for r in development),strict=sum(r['strict_success'] and r['complete'] for r in development),n=len(development),
                    scope='single_task_original_full_episode_development_not_full160_not_G1',selection_sha256=selection_sha,selection_frozen_before_development=True,retuned=False))
        assert all(torch.equal(v.detach().cpu(),parent['policy'][k]) for k,v in policy.state_dict().items()),'Frozen network changed'
        assert sha(parent_path)==c['checkpoint_sha256']
        total=sum(r['controls'] for r in records);assert total==pool.controls
        selected_replays=[r for r in replay if r['controller']=='selected'];baseline_replays=[r for r in replay if r['controller']=='baseline']
        complete=len(development)==4 and all(r['complete'] for r in development)
        status='COMPLETE' if complete else 'BOUNDED_STOP_INCOMPLETE'
        summary=dict(status=status,time_beijing=now(),seconds=time.monotonic()-start,task=c['task_id'],generations=generations,
            physical_controls=pool.controls,warmup_controls=pool.warmup_controls,physical_env_controls=sum(e.physical_controls for e in pool.envs),
            episodes=len(records),complete_episodes=sum(r['complete'] for r in records),audit=pool.audit,
            witness_gate_passed=len(selected_replays)==3 and all(r['complete'] and r['regular_success'] for r in selected_replays),
            selected_replay_regular=sum(r['complete'] and r['regular_success'] for r in selected_replays),selected_replay_n=len(selected_replays),
            baseline_replay_regular=sum(r['complete'] and r['regular_success'] for r in baseline_replays),baseline_replay_n=len(baseline_replays),
            development_complete=complete,development_n=len(development),development_regular=sum(r['complete'] and r['regular_success'] for r in development),development_strict=sum(r['complete'] and r['strict_success'] for r in development),
            original_policy_checkpoint_sha256=c['checkpoint_sha256'],policy_parameters_unchanged=True,optimizer_steps=0,
            full_A1_complete=False,quality_passports_created=False,negative_result_proves_infeasibility=False)
        write(out/'summary.json',summary);progress(status)
    except BaseException as exc:
        write(out/'fatal.json',dict(error=repr(exc),traceback=traceback.format_exc(),time_beijing=now(),seconds=time.monotonic()-start,physical_controls=pool.controls if pool else None,episodes_saved=len(records)))
        raise

if __name__=='__main__':main()
