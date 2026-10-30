"""All-phase CPU preprocessing for a single revision3 candidate; no rollout."""
import os,socket,getpass,json,time,hashlib,datetime,argparse
from pathlib import Path
import numpy as np
import mujoco
from a1_contact_velocity_projection import contact_system,project_hand_velocity


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True,type=Path);a=p.parse_args()
    assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    os.nice(10);os.chdir(Path(__file__).resolve().parents[1]);reg=json.loads(a.registration.read_text())
    assert digest(__file__)==reg['builder_sha256']
    assert digest('scripts/a1_contact_velocity_projection.py')==reg['projector_sha256']
    original=json.loads(Path('reports/a1_reference_inverse_registration_20261030.json').read_text())
    for f,h in original['protected_source_hashes'].items():assert digest(f)==h
    rows=json.loads(Path('configs/a1_probe40_v1.json').read_text())['tasks']
    assert [r['task_id'] for r in rows]==original['task_ids']
    out=Path(reg['output_directory']);out.mkdir(exist_ok=False)
    def write(name,value):
        p=out/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)
    ledger=Path('runs/a1_full_20261028/gpu_ledger.json');lh=digest(ledger);start=time.monotonic()
    write('registration.json',dict(registration=reg,registration_sha256=digest(a.registration),pid=os.getpid(),started_unix=time.time(),ledger_sha256=lh))
    records=[];complete=True;total=0
    status_codes=dict(identity_feasible=0,projected=1,infeasible_coordinate_bounds=2,optimizer_or_constraint_failure=3)
    for row in rows:
        folder=Path(row['task_path'])
        for f,h in original['task_source_hashes'][row['task_id']].items():assert digest(folder/f)==h
        with np.load(folder/'reference.npz') as z:
            # Match actual TaskTensors/A1Env storage before any projection.
            q32=np.asarray(z['qpos'],dtype=np.float32);v32=np.asarray(z['qvel'],dtype=np.float32)
        q=q32.astype(np.float64);v=v32.astype(np.float64);count=len(q)-1
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m);nu=m.nu;ob=m.body('object').id
        assert all(m.body_rootid[m.jnt_bodyid[j]]!=ob for j in range(nu))
        bound=np.max(abs(v[:-1,:nu]),axis=0)
        projected=v32[:-1,:nu].copy();status=np.full(count,-1,dtype=np.int8);contacts=np.zeros(count,np.int32)
        minimum_before=np.zeros(count);minimum_after=np.zeros(count);energy=np.zeros(count);iterations=np.zeros(count,np.int32);rounding_pass=np.zeros(count,bool)
        failure_records=[]
        for t in range(count):
            if time.monotonic()-start>=reg['maximum_cpu_wall_seconds']:complete=False;break
            d.qpos[:]=q[t];d.qvel[:]=v[t];mujoco.mj_forward(m,d)
            aa,cc,hh,ids=contact_system(m,d);candidate,info=project_hand_velocity(v[t,:nu],hh,aa,cc,bound)
            stored=candidate.astype(np.float32);post=aa@stored.astype(np.float64)+cc
            projected[t]=stored;status[t]=status_codes[info['status']];contacts[t]=len(aa)
            minimum_before[t]=info['minimum_before'] if len(aa) else 0
            minimum_after[t]=float(post.min()) if len(aa) else 0
            iterations[t]=info['optimizer_iterations'];delta=stored.astype(np.float64)-v[t,:nu];energy[t]=.5*delta@hh@delta
            rounding_pass[t]=np.isfinite(stored).all() and np.all(abs(stored)<=bound) and (not len(aa) or post.min()>=-reg['stored_float32_contact_tolerance_m_s'])
            if info['fallback_to_original']:assert np.array_equal(stored,v32[t,:nu])
            if info['fallback_to_original'] or not rounding_pass[t]:failure_records.append(dict(phase=t,projection=info,stored_float32_minimum_normal_velocity=minimum_after[t],stored_float32_check=bool(rounding_pass[t])))
            assert np.array_equal(d.qpos,q[t]) and np.array_equal(d.qvel,v[t])
            total+=1
        f=out/(row['task_id']+'.npz')
        np.savez_compressed(f,hand_qvel=projected,status=status,contact_count=contacts,minimum_normal_velocity_before=minimum_before,
                            minimum_normal_velocity_after=minimum_after,metric_change_energy=energy,optimizer_iterations=iterations,
                            stored_float32_constraint_pass=rounding_pass,absolute_coordinate_bound=bound)
        valid=status>=0
        r=dict(task=row['task_id'],stratum=row['stratum'],stored_frames=len(q),expected_phases=count,recorded_phases=int(valid.sum()),
               task_source_hashes=original['task_source_hashes'][row['task_id']],qpos_runtime_float32_sha256=hashlib.sha256(q32.tobytes()).hexdigest(),
               qvel_runtime_float32_sha256=hashlib.sha256(v32.tobytes()).hexdigest(),file=f.name,sha256=digest(f),
               status_counts={k:int(np.sum(status==code)) for k,code in status_codes.items()},
               stored_float32_constraint_failures=int(np.sum(valid&~rounding_pass)),failures=failure_records,
               maximum_abs_hand_velocity_change=float(np.max(abs(projected[valid].astype(float)-v[:-1,:nu][valid]))) if valid.any() else None,
               separate_hand_and_object_inertia_roots=True)
        records.append(r);write('task_records.json',records);write('progress.json',dict(tasks=len(records),phases=total,last_task=row['task_id'],elapsed_seconds=time.monotonic()-start))
        if not complete:break
    complete=complete and len(records)==40 and all(x['recorded_phases']==x['expected_phases'] for x in records)
    counts={k:sum(r['status_counts'][k] for r in records) for k in status_codes}
    for f,h in original['protected_source_hashes'].items():assert digest(f)==h
    for row in rows:
        for f,h in original['task_source_hashes'][row['task_id']].items():assert digest(Path(row['task_path'])/f)==h
    assert digest(ledger)==lh
    summary=dict(status='complete' if complete else 'CPU_LIMIT_INCOMPLETE',tasks=len(records),phases=total,status_counts=counts,
                 stored_float32_constraint_failures=sum(r['stored_float32_constraint_failures'] for r in records),
                 all_task_nonterminal_phase_coverage=complete,default_training_unchanged=True,gpu_validation=False,new_ppo_started=False,
                 gpu_process_hours_added=0,ledger_sha256_before=lh,ledger_sha256_after=digest(ledger),
                 elapsed_seconds=time.monotonic()-start,finished_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                 caveat='A table is not permission to launch PPO. Review every fallback and float32 failure; paired frozen-policy GPU usefulness validation is still required.')
    write('summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
