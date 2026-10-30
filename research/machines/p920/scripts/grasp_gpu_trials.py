"""GPU A/B/C trials, CPU actor to avoid allocating a CUDA BLAS workspace.

GPU physics and actual solver contacts are logged. Initial states are immutable.
"""
import argparse,json,subprocess
import numpy as np,mujoco,mujoco_warp as mjw,warp as wp,torch
from real_env import RealEnv
from train_real_ppo import Policy
from grasp_lab import RUN,OLD,DATA,fixed_initials,get_target,metrics,summarize,write_json,VARIANTS
from grasp_batch_audit import gpu_contacts,compare_contacts,cpu_contacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',default='baseline');p.add_argument('--tasks',nargs='+',default=['A','B','C']);args=p.parse_args()
    assert not VARIANTS[args.variant].get('condim'),'This runner keeps baseline physical parameters.'
    env=RealEnv(1);remaining=torch.cuda.mem_get_info()[0]/2**20
    if remaining<64:raise RuntimeError(f'GPU resource margin only {remaining:.1f} MiB; not an algorithm failure')
    initial=fixed_initials();ref=np.load(DATA/'trajectory.npz');target=np.load(DATA/'controller_reference.npz')['ctrl'];cpu=mujoco.MjData(env.m)
    ckpt=torch.load(OLD/'checkpoint.pt',map_location='cpu',weights_only=True);policy=Policy(118);policy.load_state_dict(ckpt['policy']);policy.eval()
    with wp.ScopedCapture() as c9:
        for _ in range(9):mjw.step(env.wm,env.wd)
    with wp.ScopedCapture() as c1:mjw.step(env.wm,env.wd)
    results=[]
    for task in args.tasks:
        for trial in range(8):
            env.reset(torch.tensor([0],device=env.device),False);env.qpos[0]=torch.tensor(initial['qpos'][trial],device=env.device);env.qvel[0]=torch.tensor(initial['qvel'][trial] if task!='A' else np.zeros(32,dtype=np.float32),device=env.device)
            qs=[env.qpos[0].cpu().numpy().copy()];vs=[env.qvel[0].cpu().numpy().copy()];controls=[];actions=[];rows=[];contactlog=[];comp=[];reason='horizon'
            for step in range(100 if task=='A' else 51):
                k=0 if task=='A' else step+1
                with torch.no_grad():action=policy.actor(env.observe().cpu())[0].numpy() if task=='C' else np.zeros(26,dtype=np.float32)
                command=get_target(ref,target,k,args.variant,static=task=='A')+np.clip(action,-1,1)*env.scale.cpu().numpy()
                env.ctrl[0]=torch.tensor(np.clip(command,env.m.actuator_ctrlrange[:,0],env.m.actuator_ctrlrange[:,1]),device=env.device,dtype=torch.float32)
                wp.capture_launch(c9.graph);before=env.qpos[0].cpu().numpy().copy();bv=env.qvel[0].cpu().numpy().copy();ctrl=env.ctrl[0].cpu().numpy().copy()
                wp.capture_launch(c1.graph);q=env.qpos[0].cpu().numpy().copy();v=env.qvel[0].cpu().numpy().copy();cs,_=gpu_contacts(env)
                row=metrics(env.m,q,v,ref['qpos'][k],cs,(step+1)*.02);rows.append(row)
                cpu.qpos[:]=before;cpu.qvel[:]=bv;cpu.ctrl[:]=ctrl;cpu.qacc_warmstart[:]=0;mujoco.mj_forward(env.m,cpu);reconstructed=cpu_contacts(env.m,cpu)
                comp.append(compare_contacts(cs,reconstructed));contactlog.append(dict(time_s=(step+1)*.02-.002,prestep_qpos=before.tolist(),prestep_qvel=bv.tolist(),contacts=cs))
                qs.append(q);vs.append(v);controls.append(ctrl);actions.append(action);env.phase[:]=k;env.last_action[0]=torch.tensor(np.clip(action,-1,1),device=env.device)
                if task!='A':
                    if not np.isfinite(q).all():reason='nonfinite'
                    elif np.any(q[:26]<env.m.jnt_range[:26,0]-.15) or np.any(q[:26]>env.m.jnt_range[:26,1]+.15):reason='joint_limit'
                    elif row['position_m']>.15:reason='position_gt_15cm'
                    elif row['rotation_rad']>1.4:reason='rotation_gt_1.4rad'
                    if reason!='horizon':break
            folder=RUN/'gpu'/args.variant/task;folder.mkdir(parents=True,exist_ok=True);stem=f'trial_{trial:02d}'
            np.savez_compressed(folder/f'{stem}.npz',qpos=qs,qvel=vs,ctrl=controls,action=actions)
            write_json(folder/f'{stem}_metrics.json',rows);write_json(folder/f'{stem}_contacts.json',contactlog);write_json(folder/f'{stem}_cpu_contact_comparison.json',comp)
            result=summarize(rows,task,reason);result.update(variant=args.variant,trial=trial,engine='GPU MuJoCo Warp; CPU actor for C');results.append(result)
        current=[r for r in results if r['task']==task]
        print(json.dumps(dict(variant=args.variant,task=task,n=8,success=sum(r['dynamic_success'] if task!='A' else r['static_hold_2s_within_10mm_15deg'] for r in current),mean_length=np.mean([r['steps'] for r in current]),mean_position_m=np.mean([r['object_position_mean_m'] for r in current]),mean_rotation_deg=np.mean([r['object_rotation_mean_deg'] for r in current]))),flush=True)
    write_json(RUN/f'gpu_trials_{args.variant}.json',results)


if __name__=='__main__':main()
