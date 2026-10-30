"""Read-only nominal CPU inverse dynamics; never a rollout or A1 pass test."""
import argparse,datetime,getpass,hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
import mujoco


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def envelope(model,q,v):
    n=model.nu
    assert np.all(model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)
    assert np.array_equal(model.actuator_trnid[:,0],np.arange(n))
    assert np.all(model.actuator_gear[:,0]==1) and np.all(model.actuator_gear[:,1:]==0)
    assert np.array_equal(model.jnt_dofadr[:n],np.arange(n)) and np.array_equal(model.jnt_qposadr[:n],np.arange(n))
    assert np.all(model.actuator_gaintype==mujoco.mjtGain.mjGAIN_FIXED) and np.all(model.actuator_gainprm[:,0]>0)
    assert np.all(model.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE) and np.all(model.actuator_dyntype==mujoco.mjtDyn.mjDYN_NONE)
    assert np.all(model.actuator_ctrllimited) and np.all(model.actuator_forcelimited)
    assert not np.any(model.jnt_actgravcomp)
    gain=model.actuator_gainprm[:,0];b=model.actuator_biasprm
    bias=b[:,0]+b[:,1]*q[:n]+b[:,2]*v[:n]
    lower=np.clip(gain*model.actuator_ctrlrange[:,0]+bias,*model.actuator_forcerange.T)
    upper=np.clip(gain*model.actuator_ctrlrange[:,1]+bias,*model.actuator_forcerange.T)
    limited=model.jnt_actfrclimited[:n].astype(bool)
    lower[limited]=np.clip(lower[limited],*model.jnt_actfrcrange[:n][limited].T)
    upper[limited]=np.clip(upper[limited],*model.jnt_actfrcrange[:n][limited].T)
    assert np.all(lower<=upper)
    return lower,upper


def inverse_snapshot(model,data,q,v,acceleration):
    assert not model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE)
    data.qpos[:]=q;data.qvel[:]=v;data.qacc[:]=acceleration
    data.qfrc_applied[:]=0;data.xfrc_applied[:]=0
    mujoco.mj_inverse(model,data)
    ma=np.empty(model.nv);mujoco.mj_mulM(model,data,ma,acceleration)
    required=ma+data.qfrc_bias-data.qfrc_passive-data.qfrc_constraint
    actual=data.qfrc_inverse.copy()
    assert np.isfinite(actual).all()
    error=float(np.max(abs(required-actual)))
    assert error<=1e-9*(1+float(np.max(abs(required))))
    assert np.array_equal(data.qpos,q) and np.array_equal(data.qvel,v) and np.array_equal(data.qacc,acceleration)
    return actual,error


def stats(values):
    x=np.asarray(values);assert len(x) and np.isfinite(x).all()
    return dict(n=len(x),mean=float(np.mean(x)),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);args=p.parse_args()
    assert socket.gethostname()=='Lambda' and getpass.getuser()=='youran' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    os.nice(10);root=Path(__file__).resolve().parents[1];os.chdir(root)
    reg=json.loads(args.registration.read_text());assert reg['status']=='PREREGISTERED_NOT_LAUNCHED'
    assert reg['method']=='native_continuous_inverse_nominal_reference_v1' and reg['maximum_cpu_wall_seconds']==900
    assert mujoco.__version__==reg['mujoco_version']
    for f,h in reg['protected_source_hashes'].items():assert digest(f)==h,f
    assert digest(__file__)==reg['audit_source_sha256']
    out=root/reg['output_directory'];out.mkdir(exist_ok=False)
    def write(name,value):
        target=out/name;tmp=target.with_suffix(target.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(target)
    started=time.monotonic();write('registration.json',dict(registration=reg,registration_sha256=digest(args.registration),pid=os.getpid(),started_unix=time.time()))
    rows=json.loads((root/reg['manifest']).read_text())['tasks'];assert [r['task_id'] for r in rows]==reg['task_ids']
    results=[];complete=True;total_states=0
    for row in rows:
        folder=root/row['task_path']
        for f,h in reg['task_source_hashes'][row['task_id']].items():assert digest(folder/f)==h
        with np.load(folder/'reference.npz') as raw:q=raw['qpos'].copy();v=raw['qvel'].copy()
        model=mujoco.MjModel.from_xml_path(str(folder/'scene.xml'));data=mujoco.MjData(model);nu=model.nu;count=len(q)-2
        assert model.nq==nu+7 and model.nv==nu+6 and model.na==0
        assert model.opt.disableflags==0 and model.opt.enableflags==0 and model.opt.timestep==.002
        assert model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert np.allclose(model.geom_friction[model.geom_contype==1,0],.8) and model.geom_friction[model.geom('floor').id,0]==.6
        assert np.array_equal(model.actuator_biasprm[:,1],-model.actuator_gainprm[:,0])
        object_body=model.body('object').id;mg=model.body_mass[object_body]*np.linalg.norm(model.opt.gravity)
        meta=json.loads((folder/'qualification.json').read_text());length=.5*np.linalg.norm(meta['object']['extents_m']);assert mg>0 and length>0
        inv=np.zeros((2,count,model.nv));acc=np.zeros_like(inv);lo=np.zeros((count,nu));hi=lo.copy();valid=np.zeros((2,count),dtype=bool)
        errors=np.zeros((2,count));contacts=np.zeros((2,count),dtype=int)
        for t in range(count):
            if time.monotonic()-started>=reg['maximum_cpu_wall_seconds']:complete=False;break
            lo[t],hi[t]=envelope(model,q[t],v[t])
            for mode in [0,1]:
                if mode==1 and t==0:continue
                aa=(v[t+1]-v[t])/.02 if mode==0 else (v[t+1]-v[t-1])/.04
                inv[mode,t],errors[mode,t]=inverse_snapshot(model,data,q[t],v[t],aa)
                acc[mode,t]=aa;contacts[mode,t]=data.ncon;valid[mode,t]=True;total_states+=1
        file=out/(row['task_id']+'.npz')
        np.savez_compressed(file,phase=np.arange(count),mode=np.array(['forward_velocity_difference','central_velocity_difference']),valid=valid,
            inverse_generalized_force=inv,reference_acceleration=acc,actuator_minimum_force=lo,actuator_maximum_force=hi,equation_identity_max_error=errors,contact_geometry_count=contacts)
        result=dict(task=row['task_id'],stratum=row['stratum'],stored_frames=len(q),expected_states=count+max(count-1,0),recorded_states=int(valid.sum()),
                    nominal_mass_kg=float(model.body_mass[object_body]),weight_N=float(mg),object_half_diagonal_m=float(length),arrays=file.name,arrays_sha256=digest(file),methods={})
        for mode,label in enumerate(['forward_velocity_difference','central_velocity_difference']):
            for segment in ['all_eligible_phases','first32_reference_phases']:
                selected=valid[mode] & ((np.arange(count)<32) if segment=='first32_reference_phases' else True)
                if not selected.any():continue
                forces=inv[mode,selected];excess=np.maximum(np.maximum(lo[selected]-forces[:,:nu],forces[:,:nu]-hi[selected]),0)
                result['methods'][label+'/'+segment]=dict(object_extra_force_N=stats(np.linalg.norm(forces[:,nu:nu+3],axis=1)),
                    object_extra_torque_Nm=stats(np.linalg.norm(forces[:,nu+3:nu+6],axis=1)),
                    object_extra_force_over_weight=stats(np.linalg.norm(forces[:,nu:nu+3],axis=1)/mg),
                    object_extra_torque_over_weight_length=stats(np.linalg.norm(forces[:,nu+3:nu+6],axis=1)/(mg*length)),
                    fraction_frames_any_actuator_envelope_excess=float(np.mean(np.max(excess,axis=1)>1e-8)),
                    actuator_envelope_excess_over_nominal_force_limit=stats(np.max(excess/np.max(abs(model.actuator_forcerange),axis=1),axis=1)),
                    equation_identity_max_error=float(np.max(errors[mode,selected])))
        results.append(result);write('progress.json',dict(tasks=len(results),states=total_states,elapsed_seconds=time.monotonic()-started,last_task=row['task_id']))
        write('task_results.json',results)
        if not complete:break
    complete=complete and len(results)==40 and all(r['recorded_states']==r['expected_states'] for r in results)
    write('summary.json',dict(status='complete' if complete else 'CPU_LIMIT_INCOMPLETE',tasks=len(results),states=total_states,elapsed_seconds=time.monotonic()-started,
        gpu_process_hours_added=0,training_started=False,formal_A1_complete=False,G1_claim=False,
        limitations=reg['limitations'],finished_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()))
    print(json.dumps(dict(status='complete' if complete else 'CPU_LIMIT_INCOMPLETE',tasks=len(results),states=total_states,elapsed_seconds=time.monotonic()-started)))


if __name__=='__main__':main()
