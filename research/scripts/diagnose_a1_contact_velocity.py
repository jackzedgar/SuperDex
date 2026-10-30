"""Frozen-policy paired training-reset validation. No optimizer, passport or final test."""
import argparse,gc,hashlib,json,signal,time
from pathlib import Path
import numpy as np,torch,warp as wp,mujoco
from a1_data_sources import ROOT,RUN,write
from a1_env import A1Env
from a1_policy import CollectorPolicy,policy_inputs
from a1_reset_memory import apply_velocity_reset_memory

stop=False
def halt(*_):
    global stop;stop=True


def tensor_hash(values):return hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for v in values)).hexdigest()


@torch.no_grad()
def gpu_contact_velocity(env):
    c=env.wd.contact;n=int(wp.to_torch(env.wd.nacon)[0]);assert n<=env.contacts.cap
    pair=wp.to_torch(c.geom)[:n].long();dist=wp.to_torch(c.dist)[:n];margin=wp.to_torch(c.includemargin)[:n]
    addresses=wp.to_torch(c.efc_address)[:n].reshape(n,-1) if n else torch.empty((0,1),device=env.device)
    valid=(pair>=0).all(-1)&(dist<=margin)&(addresses>=0).any(-1)
    safe=pair.clamp(0,env.m.ngeom-1);types=env.contacts.types[safe];valid&=(types==1).any(-1)&(types==2).any(-1)
    bid=env.contacts.bodies[safe];pos=wp.to_torch(c.pos)[:n];normal=wp.to_torch(c.frame)[:n,0,:]
    vel=wp.to_torch(env.wd.cvel)[0,bid];com=wp.to_torch(env.wd.subtree_com)[0,env.contacts.roots[bid]]
    point_velocity=vel[...,3:]+torch.linalg.cross(vel[...,:3],pos[:,None,:]-com)
    projected=((point_velocity[:,1]-point_velocity[:,0])*normal).sum(-1)
    values=projected[valid].cpu().numpy();points=pos[valid].cpu().numpy();normals=normal[valid].cpu().numpy();bodies=bid[valid].cpu().numpy()
    native=mujoco.MjData(env.m);native.qpos[:]=env.qpos.cpu().numpy()[0];native.qvel[:]=env.qvel.cpu().numpy()[0];mujoco.mj_forward(env.m,native)
    differences=[]
    for value,point,direction,(b0,b1) in zip(values,points,normals,bodies):
        j0=np.zeros((3,env.m.nv));j1=np.zeros_like(j0)
        mujoco.mj_jac(env.m,native,j0,None,point,int(b0));mujoco.mj_jac(env.m,native,j1,None,point,int(b1))
        differences.append(abs(float(direction@(j1-j0)@native.qvel)-float(value)))
    return dict(active_hand_object_contacts=len(values),minimum_normal_m_s=float(values.min()) if len(values) else None,
                maximum_native_jacobian_disagreement_m_s=max(differences,default=0.))


@torch.no_grad()
def trace(env,policy,phase,seed,override):
    env.reset(phase,seed=seed,randomize=True,training_hand_qvel=override);apply_velocity_reset_memory(env)
    matched=tensor_hash([env.qpos,env.qvel[:,env.task.nu:],env.context])
    full=tensor_hash([env.qpos,env.qvel,env.context,env.ctrl,env.history,env.servo])
    velocity=gpu_contact_velocity(env)
    sh=env.state()[:,None].repeat(1,3,1);ah=torch.zeros(1,3,52,device=env.device);errors=[];rewards=[]
    assert env.task.length-1-phase>=32
    for _ in range(32):
        obs,goal=policy_inputs(env.task,sh,ah,env.phase);action=policy.distribution(obs,goal).mean*env.task.mask.repeat_interleave(26,-1)
        state,reward,done,info=env.step(action);errors.append([float(info['pos_error']),float(info['rot_error'])]);rewards.append(float(reward))
        sh=torch.cat([sh[:,1:],state[:,None]],1);ah=torch.cat([ah[:,1:],env.last_action[:,None]],1)
        if bool(done):break
    return dict(matched_physical_context_sha256=matched,full_initial_state_sha256=full,start_phase=phase,requested_controls=32,
                controls=len(errors),survived_requested_prefix=len(errors)==32 and not bool(info['failure']),full32_window=True,
                failure_reason=int(info['reason']),finite=bool(info['finite']),error_trace_m_rad=errors,reward_sum=sum(rewards),initial_contacts=velocity)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',type=Path,required=True);a=parser.parse_args()
    config=json.loads(a.configuration.read_text());signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(2)
    out=RUN/config['run'];out.mkdir(exist_ok=False);start=time.monotonic();checkpoint_path=ROOT/config['checkpoint']
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==config['checkpoint_sha256']
    for relative,expected in config['source_hashes'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False);assert checkpoint['formal']
    policy=CollectorPolicy().cuda();policy.load_state_dict(checkpoint['policy']);policy.eval()
    for parameter in policy.parameters():parameter.requires_grad_(False)
    write(out/'registration.json',dict(configuration=config,configuration_sha256=hashlib.sha256(a.configuration.read_bytes()).hexdigest(),
           parent_controls=checkpoint['controls'],formal=False,new_PPO_started=False,scope='960 paired 32-control development prefixes; not complete collector success, G1, policy training or fixed-test evaluation'))
    records=[];mode_names=['v5_velocity_memory','contact_velocity_projection']
    for ti,task in enumerate(config['planned_tasks']):
        if stop:break
        table_path=ROOT/config['cache_directory']/task['cache_file'];assert hashlib.sha256(table_path.read_bytes()).hexdigest()==task['cache_sha256']
        with np.load(table_path) as table:hand=table['hand_qvel'].copy()
        env=A1Env(ROOT/task['path'],allow_training_velocity_override=True)
        assert tensor_hash([env.task.qpos])==task['qpos_runtime_float32_sha256'] and tensor_hash([env.task.qvel])==task['qvel_runtime_float32_sha256']
        for pi,phase in enumerate(task['phases']):
            for si,seed in enumerate(config['dr_seeds']):
                if stop:break
                order=[0,1,1,0] if (ti+pi+si)%2==0 else [1,0,0,1];repeats=[0,0];matched=None;full={}
                for mode in order:
                    if stop:break
                    result=trace(env,policy,phase,int(seed),hand[phase] if mode else None)
                    if matched is None:matched=result['matched_physical_context_sha256']
                    assert matched==result['matched_physical_context_sha256'],'Matched pose, object velocity or DR changed'
                    if mode in full:assert full[mode]==result['full_initial_state_sha256'],'Within-mode repeated reset changed'
                    full[mode]=result['full_initial_state_sha256'];repeats[mode]+=1
                    records.append(dict(task=task['task'],stratum=task['stratum'],seed=seed,mode=mode_names[mode],repeat=repeats[mode],**result))
                    write(out/'conditions.json',records)
                write(out/'progress.json',dict(conditions=len(records),tasks_started=ti+1,seconds=time.monotonic()-start))
            if stop:break
        env.contacts.e=None;del env;gc.collect();torch.cuda.empty_cache()
        print(json.dumps(dict(task=task['task'],conditions=len(records))),flush=True)
    counts={mode:dict(n=sum(r['mode']==mode for r in records),survived_32=sum(r['mode']==mode and r['survived_requested_prefix'] for r in records)) for mode in mode_names}
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==config['checkpoint_sha256']
    write(out/'summary.json',dict(status='complete' if len(records)==960 else 'budget_stop',conditions=len(records),counts=counts,
          seconds=time.monotonic()-start,formal_A1_complete=False,new_PPO_started=False,fixed_test_opened=False,
          interpretation='Apply all preregistered usefulness and GPU contact consistency gates before any revision3 PPO. No automatic training launch.'))


if __name__=='__main__':main()
