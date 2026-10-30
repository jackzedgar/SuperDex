"""Bounded CPU prototype preflight; sampled static states, no integration."""
import os,socket,getpass,json,time,hashlib,copy,datetime
from pathlib import Path
import numpy as np
import mujoco
from a1_contact_velocity_projection import contact_system,project_hand_velocity


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    os.nice(10);root=Path(__file__).resolve().parents[1];os.chdir(root)
    registered=Path('reports/a1_revision3_cpu_prototype_registration_20261030.json')
    reg=json.loads(registered.read_text())
    for n,h in reg['source_hashes'].items():assert digest(Path('scripts')/n)==h
    out=Path('runs/a1_full_20261028/revision3_contact_velocity_cpu_preflight_20261030_001');out.mkdir(exist_ok=False)
    ledger=Path('runs/a1_full_20261028/gpu_ledger.json');ledger_before=digest(ledger)
    original=json.loads(Path('reports/a1_reference_inverse_registration_20261030.json').read_text())
    for p,h in original['protected_source_hashes'].items():assert digest(p)==h
    start=time.monotonic()
    def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
    write('registration.json',dict(registration=reg,registration_sha256=digest(registered),audit_source_sha256=digest(__file__),started_unix=time.time(),pid=os.getpid(),ledger_sha256_before=ledger_before))
    rows=json.loads(Path('configs/a1_probe40_v1.json').read_text())['tasks'];results=[];dr_checks=[]
    assert [x['task_id'] for x in rows]==original['task_ids']
    complete=True
    for row in rows:
        folder=Path(row['task_path'])
        for n,h in original['task_source_hashes'][row['task_id']].items():assert digest(folder/n)==h
        with np.load(folder/'reference.npz') as z:q=z['qpos'].copy();v=z['qvel'].copy()
        m=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));d=mujoco.MjData(m)
        bound=np.max(abs(v[:-1,:m.nu]),axis=0);last=len(q)-2
        phases=[0,last//4,last//2,3*last//4,last];assert len(set(phases))==5
        for t in phases:
            if time.monotonic()-start>reg['maximum_cpu_wall_seconds']:complete=False;break
            d.qpos[:]=q[t];d.qvel[:]=v[t];mujoco.mj_forward(m,d)
            a,c,h,ids=contact_system(m,d)
            projected,info=project_hand_velocity(v[t,:m.nu],h,a,c,bound)
            candidate=v[t].copy();candidate[:m.nu]=projected
            assert np.array_equal(candidate[m.nu:],v[t,m.nu:])
            assert np.array_equal(d.qpos,q[t]) and np.array_equal(d.qvel,v[t])
            d.qvel[:]=candidate;mujoco.mj_forward(m,d)
            a2,c2,h2,ids2=contact_system(m,d)
            assert ids2==ids and np.allclose(a,a2,rtol=0,atol=1e-10)
            assert np.array_equal(d.qpos,q[t]) and np.array_equal(d.qvel[m.nu:],v[t,m.nu:])
            recomputed=a2@candidate[:m.nu]+c2
            if not info['fallback_to_original'] and len(recomputed):assert recomputed.min()>=-1e-7
            result=dict(task=row['task_id'],stratum=row['stratum'],phase=t,stored_frames=len(q),
                        projection=info,hand_velocity_before=v[t,:m.nu].tolist(),hand_velocity_after=projected.tolist(),
                        minimum_normal_velocity_recomputed=float(recomputed.min()) if len(recomputed) else None,
                        normal_velocities_before=(a@v[t,:m.nu]+c).tolist(),normal_velocities_after=recomputed.tolist(),
                        qpos_unchanged=True,object_velocity_unchanged=True)
            results.append(result)
            if t in [0,last//2]:
                for sign in [-1,1]:
                    mm=copy.copy(m);ob=mm.body('object').id
                    # Two fixed extremes inside the original independent DR ranges.
                    mass=.7 if sign<0 else 1.4;inertia=.7 if sign<0 else 1.3
                    mm.body_mass[ob]*=mass;mm.body_inertia[ob]*=mass*inertia
                    mm.body_ipos[ob]+=sign*.01
                    mm.geom_friction[mm.geom_contype==1,0]=.4 if sign<0 else 1.2
                    mm.geom_friction[mm.geom('floor').id,0]=.3 if sign<0 else .9
                    for base in range(0,m.nu,26):
                        ss=slice(base+6,base+26);gain=.9 if sign<0 else 1.1
                        mm.actuator_gainprm[ss,0]*=gain;mm.actuator_biasprm[ss,1:3]*=gain
                    dd=mujoco.MjData(mm);mujoco.mj_setConst(mm,dd)
                    dd.qpos[:]=q[t];dd.qvel[:]=v[t];mujoco.mj_forward(mm,dd)
                    aa,cc,hh,jj=contact_system(mm,dd)
                    same_geometry=ids==jj and aa.shape==a.shape
                    error_a=float(np.max(abs(aa-a))) if same_geometry and aa.size else (0.0 if same_geometry else None)
                    error_c=float(np.max(abs(cc-c))) if same_geometry and cc.size else (0.0 if same_geometry else None)
                    error_h=float(np.max(abs(hh-h)))
                    passed=same_geometry and error_a<=1e-9 and error_c<=1e-9 and error_h<=1e-9
                    dr_checks.append(dict(task=row['task_id'],phase=t,variant=sign,same_contact_geometry=same_geometry,
                                          jacobian_max_error=error_a,offset_max_error=error_c,hand_metric_max_error=error_h,passed=passed))
                    # copy.copy(MjModel) must be independent; verify native original.
                    dn=mujoco.MjData(m);dn.qpos[:]=q[t];dn.qvel[:]=v[t];mujoco.mj_forward(m,dn)
                    an,cn,hn,jn=contact_system(m,dn)
                    assert jn==ids and np.array_equal(an,a) and np.array_equal(hn,h)
        write('progress.json',dict(tasks=len({x['task'] for x in results}),conditions=len(results),dr_checks=len(dr_checks),elapsed_seconds=time.monotonic()-start))
        if not complete:break
    complete=complete and len(results)==200 and len(dr_checks)==160
    write('conditions.json',results);write('dr_invariance_checks.json',dr_checks)
    assert digest(ledger)==ledger_before
    for p,h in original['protected_source_hashes'].items():assert digest(p)==h
    for row in rows:
        for n,h in original['task_source_hashes'][row['task_id']].items():assert digest(Path(row['task_path'])/n)==h
    summary=dict(status='complete' if complete else 'CPU_LIMIT_INCOMPLETE',tasks=len({x['task'] for x in results}),conditions=len(results),
                 projection_status_counts={s:sum(x['projection']['status']==s for x in results) for s in sorted({x['projection']['status'] for x in results})},
                 dr_checks=len(dr_checks),dr_checks_passed=sum(x['passed'] for x in dr_checks),
                 all_task_all_phase_validation=False,gpu_validation=False,new_ppo_started=False,G1_or_full_A1_pass=False,
                 gpu_process_hours_added=0,ledger_sha256_before=ledger_before,ledger_sha256_after=digest(ledger),
                 elapsed_seconds=time.monotonic()-start,finished_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
    write('summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
