"""Paired grasp diagnosis. Baseline data and checkpoints are read-only.

No attachment, object wrench, or in-episode state overwrite is allowed here.
"""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation
from train_real_ppo import Policy
from real_env import ROOT,DATA

RUN=Path(os.environ.get('SUPERDEX_GRASP_RUN',ROOT/'runs/grasp_repair_20261026'))
OLD=ROOT/'runs/real_track_20261026/ppo'
VARIANTS={
    'baseline':{},
    'contact4':{'condim':4},
    'wrist_ff':{'wrist_ff_s':.02},
    'wrist_ff35':{'wrist_ff_s':.035},
    'zero_velocity':{'zero_velocity':True},
    'grip_balance':{'preload_multipliers':[3.3669817389251144,1,.10909114616803642,.48660065493105115,1]},
    'index_contact':{'controller_file':'contact_completion_targets.npz'},
    'grip_balance_index':{'controller_file':'contact_completion_targets.npz','preload_multipliers':[3.3669817389251144,1,.10909114616803642,.48660065493105115,1]},
}


def write_json(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def quat_angle(q,r):
    q=np.asarray(q,dtype=np.float64);r=np.asarray(r,dtype=np.float64)
    return 2*np.arccos(np.clip(abs(np.dot(q,r))/(np.linalg.norm(q)*np.linalg.norm(r)),0,1))


def fixed_initials():
    RUN.mkdir(parents=True,exist_ok=True)
    dest=RUN/'fixed_initials.npz'
    if dest.exists():return np.load(dest)
    q=[];v=[];files=[]
    for i in range(8):
        suffix='' if i==0 else f'_{i:02d}'
        f=OLD/f'rollout_before{suffix}.npz';a=np.load(f);b=np.load(OLD/f'rollout_after{suffix}.npz')
        assert np.array_equal(a['qpos'][0],b['qpos'][0]) and np.array_equal(a['qvel'][0],b['qvel'][0])
        q.append(a['qpos'][0]);v.append(a['qvel'][0]);files.append({'path':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    np.savez_compressed(dest,qpos=q,qvel=v,seeds=np.arange(20261026,20261034))
    write_json(RUN/'fixed_initials_manifest.json',dict(source_commit='0667e7d39dbdc07b47c90a92f7031ba04f284443',before_after_identical=True,sources=files,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    return np.load(dest)


def model_for(variant):
    m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'))
    opt=VARIANTS[variant]
    if opt.get('condim'):
        # Finite-area soft-finger spin friction. Translational friction unchanged.
        m.geom_condim[(m.geom_contype==1)|(m.geom_contype==2)]=opt['condim']
    return m


def get_target(ref,target,phase,variant,static=False):
    c=target[phase].copy()
    if 'controller_file' in VARIANTS[variant]:
        c=np.load(RUN/VARIANTS[variant]['controller_file'])['ctrl'][phase].copy()
    if 'preload_multipliers' in VARIANTS[variant]:
        bias=c-ref['qpos'][phase,:26]
        for i,s in enumerate(VARIANTS[variant]['preload_multipliers']):bias[6+4*i:10+4*i]*=s
        c=ref['qpos'][phase,:26]+bias
    if not static:c[:6]+=VARIANTS[variant].get('wrist_ff_s',0)*ref['qvel'][phase,:6]
    return c


def observation(q,v,ref,target,phase,last):
    current=ref['qpos'][phase];nxt=target[min(phase+1,len(target)-1)]
    quat=q[29:]*(-1 if np.dot(q[29:],current[29:])<0 else 1)
    return np.clip(np.r_[q[:26],v[:26]*.1,nxt-q[:26],(q[26:29]-current[26:29])*10,quat-current[29:],v[26:]*.1,phase/(len(target)-1),last],-20,20).astype(np.float32)


def cpu_contacts(m,d):
    contacts=[]
    for i,c in enumerate(d.contact):
        types={int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}
        if 2 not in types:continue
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        contacts.append(dict(geom=[int(c.geom1),int(c.geom2)],kind='hand_object' if 1 in types else 'floor_object',pos=c.pos.tolist(),dist=float(c.dist),frame=c.frame.tolist(),force_contact=force.tolist(),force_world=(c.frame.reshape(3,3).T@force[:3]).tolist(),dim=int(c.dim)))
    return contacts


def metrics(m,q,v,reference,contacts,time):
    ep=float(np.linalg.norm(q[26:29]-reference[26:29]));er=float(quat_angle(q[29:],reference[29:]));hand=[c for c in contacts if c['kind']=='hand_object']
    return dict(time_s=time,position_m=ep,rotation_rad=er,wrist_position_m=float(np.linalg.norm(q[:3]-reference[:3])),wrist_joint_angle_rmse=float(np.sqrt(np.mean((q[3:6]-reference[3:6])**2))),finger_rmse_rad=float(np.sqrt(np.mean((q[6:26]-reference[6:26])**2))),object_linear_speed_m_s=float(np.linalg.norm(v[26:29])),object_angular_speed_rad_s=float(np.linalg.norm(v[29:])),contact_count=len(hand),normal_force_sum_n=float(sum(max(0,c['force_contact'][0]) for c in hand)),max_hand_penetration_m=max([max(0,-c['dist']) for c in hand]+[0]))


def summarize(rows,mode,stop_reason):
    def first(predicate):return next((r['time_s'] for r in rows if predicate(r)),None)
    allgood=all(r['position_m']<=.03 and r['rotation_rad']<=np.pi/6 for r in rows)
    return dict(task=mode,steps=len(rows),duration_s=len(rows)*.02,first_error_10mm_or_10deg_s=first(lambda r:r['position_m']>.01 or r['rotation_rad']>np.pi/18),
                first_no_contact_s=first(lambda r:r['contact_count']==0),first_strict_violation_s=first(lambda r:r['position_m']>.03 or r['rotation_rad']>np.pi/6),
                termination_time_s=len(rows)*.02 if stop_reason!='horizon' else None,reason=stop_reason,
                object_position_mean_m=float(np.mean([r['position_m'] for r in rows])),object_rotation_mean_deg=float(np.rad2deg(np.mean([r['rotation_rad'] for r in rows]))),
                object_position_final_m=rows[-1]['position_m'],object_rotation_final_deg=float(np.rad2deg(rows[-1]['rotation_rad'])),
                wrist_position_mean_m=float(np.mean([r['wrist_position_m'] for r in rows])),finger_rmse_mean_rad=float(np.mean([r['finger_rmse_rad'] for r in rows])),
                max_penetration_m=max(r['max_hand_penetration_m'] for r in rows),no_contact_steps=sum(r['contact_count']==0 for r in rows),
                dynamic_success=mode!='A' and len(rows)==51 and allgood and stop_reason=='horizon',
                static_hold_2s_within_10mm_15deg=(len(rows)==100 and all(r['position_m']<=.01 and r['rotation_rad']<=np.pi/12 for r in rows)) if mode=='A' else None)


def cpu_run(variant,mode,trial,policy=None,start_phase=0):
    m=model_for(variant);d=mujoco.MjData(m);ref=np.load(DATA/'trajectory.npz');target=np.load(DATA/'controller_reference.npz')['ctrl'];initial=fixed_initials()
    # The eight evaluation perturbations are exact saved tensors, not regenerated RNG.
    d.qpos[:]=initial['qpos'][trial] if trial>=0 else ref['qpos'][start_phase]
    d.qvel[:]=0 if mode=='A' or VARIANTS[variant].get('zero_velocity') else initial['qvel'][trial] if trial>=0 else ref['qvel'][start_phase]
    qs=[d.qpos.copy()];vs=[d.qvel.copy()];controls=[];actions=[];rows=[];contactlog=[];last=np.zeros(26);reason='horizon';n=100 if mode=='A' else 51-start_phase
    for step in range(n):
        phase=0 if mode=='A' else start_phase+step;nextphase=0 if mode=='A' else start_phase+step+1
        action=np.zeros(26)
        if mode=='C':
            with torch.no_grad():action=policy.actor(torch.tensor(observation(d.qpos,d.qvel,ref,target,phase,last))).numpy()
        base=get_target(ref,target,nextphase,variant,static=mode=='A');ctrl=base+np.clip(action,-1,1)*np.r_[[.025]*3,[.18]*3,[.25]*20]
        d.ctrl[:]=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        for sub in range(10):
            if sub==9:contact_q=d.qpos.copy();contact_v=d.qvel.copy()
            mujoco.mj_step(m,d)
        # These contacts/forces belong to t_end - 2 ms, BEFORE the last integration.
        cs=cpu_contacts(m,d);rows.append(metrics(m,d.qpos,d.qvel,ref['qpos'][nextphase],cs,(step+1)*.02))
        contactlog.append(dict(time_s=(step+1)*.02-.002,prestep_qpos=contact_q.tolist(),prestep_qvel=contact_v.tolist(),contacts=cs))
        qs.append(d.qpos.copy());vs.append(d.qvel.copy());controls.append(d.ctrl.copy());actions.append(action.copy());last=np.clip(action,-1,1)
        if mode!='A':
            r=rows[-1]
            if not np.isfinite(d.qpos).all():reason='nonfinite'
            elif np.any(d.qpos[:26]<m.jnt_range[:26,0]-.15) or np.any(d.qpos[:26]>m.jnt_range[:26,1]+.15):reason='joint_limit'
            elif r['position_m']>.15:reason='position_gt_15cm'
            elif r['rotation_rad']>1.4:reason='rotation_gt_1.4rad'
            if reason!='horizon':break
    taskdir=mode if start_phase==0 else mode+f'_phase{start_phase:02d}'
    folder=RUN/'cpu'/variant/taskdir;folder.mkdir(parents=True,exist_ok=True);stem=f'trial_{trial:02d}'
    np.savez_compressed(folder/f'{stem}.npz',qpos=qs,qvel=vs,ctrl=controls,action=actions)
    write_json(folder/f'{stem}_contacts.json',contactlog);write_json(folder/f'{stem}_metrics.json',rows)
    result=summarize(rows,mode,reason);result.update(trial=trial,variant=variant,engine='CPU MuJoCo',parameter_change=VARIANTS[variant],start_phase=start_phase,segment_completed=len(rows)==n and reason=='horizon',first_10_controls_position_mean_m=float(np.mean([r['position_m'] for r in rows[:10]])),first_10_controls_rotation_mean_deg=float(np.rad2deg(np.mean([r['rotation_rad'] for r in rows[:10]]))));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=list(VARIANTS));p.add_argument('--tasks',nargs='+',default=['A','B']);p.add_argument('--nominal-only',action='store_true');args=p.parse_args();fixed_initials();policy=None
    if 'C' in args.tasks:
        c=torch.load(OLD/'checkpoint.pt',map_location='cpu',weights_only=True);policy=Policy(118);policy.load_state_dict(c['policy']);policy.eval()
    results=[]
    for variant in args.variants:
        for task in args.tasks:
            for trial in [-1] if args.nominal_only else range(8):
                r=cpu_run(variant,task,trial,policy);results.append(r)
            current=[r for r in results if r['variant']==variant and r['task']==task]
            print(json.dumps(dict(variant=variant,task=task,n=len(current),success=sum(r['dynamic_success'] if task!='A' else r['static_hold_2s_within_10mm_15deg'] for r in current),position_mean_m=np.mean([r['object_position_mean_m'] for r in current]),rotation_mean_deg=np.mean([r['object_rotation_mean_deg'] for r in current]),length=np.mean([r['steps'] for r in current]))),flush=True)
    name='cpu_nominal.json' if args.nominal_only else 'cpu_paired_'+'_'.join(args.variants)+'_'+'_'.join(args.tasks)+'.json'
    write_json(RUN/name,results)


if __name__=='__main__':main()
