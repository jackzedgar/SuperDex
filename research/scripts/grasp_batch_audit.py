"""Actual GPU contacts and physics-timestamp-aligned CPU reconstruction.

Run each world count in its own process to release GPU memory between tests.
"""
import argparse,json,time,subprocess
import numpy as np
import mujoco,mujoco_warp as mjw,warp as wp,torch
from grasp_lab import RUN,write_json,cpu_contacts,quat_angle
from real_env import RealEnv


def gpu_contacts(env):
    wp.synchronize();n=int(env.wd.nacon.numpy()[0]);cap=len(env.wd.contact.dist)
    assert n<=cap,(n,cap)
    if not n:return [],n
    ids=wp.array(np.arange(n,dtype=np.int32),device='cuda:0');force=wp.zeros(n,dtype=wp.spatial_vector,device='cuda:0')
    mjw.contact_force(env.wm,env.wd,ids,False,force);wp.synchronize()
    c=env.wd.contact
    fields={name:getattr(c,name).numpy()[:n] for name in ['worldid','geom','pos','dist','frame','dim','efc_address']};fs=force.numpy();result=[]
    for i in range(n):
        g=fields['geom'][i];types={int(env.m.geom_contype[x]) for x in g}
        if 2 not in types:continue
        frame=fields['frame'][i].reshape(3,3)
        result.append(dict(world=int(fields['worldid'][i]),geom=g.tolist(),kind='hand_object' if 1 in types else 'floor_object',pos=fields['pos'][i].tolist(),dist=float(fields['dist'][i]),frame=frame.ravel().tolist(),force_contact=fs[i].tolist(),force_world=(frame.T@fs[i,:3]).tolist(),dim=int(fields['dim'][i]),efc_address=fields['efc_address'][i].tolist()))
    return result,n


def compare_contacts(gpu,cpu):
    g=[c for c in gpu if c['kind']=='hand_object'];h=[c for c in cpu if c['kind']=='hand_object']
    distances=[];normal=[]
    for x in g:
        matches=[c for c in h if sorted(c['geom'])==sorted(x['geom'])]
        if not matches:continue
        y=min(matches,key=lambda c:np.linalg.norm(np.array(c['pos'])-x['pos']))
        distances.append(float(np.linalg.norm(np.array(y['pos'])-x['pos'])))
        ng=np.asarray(x['frame'][:3]);nc=np.asarray(y['frame'][:3]);normal.append(float(np.rad2deg(np.arccos(np.clip(abs(np.dot(ng,nc)),0,1)))))
    return dict(gpu_count=len(g),cpu_count=len(h),count_delta=len(g)-len(h),matched=len(distances),
                nearest_point_max_m=max(distances,default=None),normal_angle_max_deg=max(normal,default=None),
                gpu_normal_force_n=sum(c['force_contact'][0] for c in g),cpu_normal_force_n=sum(c['force_contact'][0] for c in h))


def run(nworld):
    out=RUN/f'batch_{nworld}';out.mkdir(parents=True,exist_ok=True)
    resource=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used','--format=csv,noheader'],text=True)
    (out/'resources.txt').write_text(resource)
    free=int(resource.splitlines()[0].split(',')[1].strip().split()[0])
    if free<350:
        write_json(out/'summary.json',dict(status='not_run_resource',free_mb=free,reason='Less than 350 MiB free for physics-only diagnostic; no actor/optimizer is allocated.'));return
    try:env=RealEnv(nworld)
    except RuntimeError as e:
        if 'alloc' in str(e).lower() or 'memory' in str(e).lower():
            write_json(out/'summary.json',dict(status='resource_allocation_failure',free_before_mb=free,error=str(e),algorithm_conclusion=False));return
        raise
    remaining=torch.cuda.mem_get_info()[0]/2**20
    (out/'resources_after_init.txt').write_text(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True))
    if remaining<64:
        write_json(out/'summary.json',dict(status='stopped_resource_margin',free_after_init_mb=remaining,reason='Require at least 64 MiB remaining after physics-only initialization.'));return
    allids=torch.arange(nworld,device=env.device)
    # Single-substep graph exposes the pre-integration state for contact attribution.
    mjw.step(env.wm,env.wd);wp.synchronize()
    with wp.ScopedCapture() as c:mjw.step(env.wm,env.wd)
    graph=c.graph;env.reset(allids,False)
    initial=env.qpos.cpu().numpy().copy();initialv=env.qvel.cpu().numpy().copy();assert np.array_equal(initial,np.broadcast_to(initial[0],initial.shape))
    pre=[];post=[];prevel=[];postvel=[];controls=[];comparisons=[];contactlog=[];maxn=0;first_pair_difference=None
    cpu=mujoco.MjData(env.m);start=time.perf_counter()
    for s in range(510):
        phase=s//10+1;env.ctrl[:]=env.target[phase]
        before=env.qpos.cpu().numpy().copy();bv=env.qvel.cpu().numpy().copy();bc=env.ctrl.cpu().numpy().copy()
        wp.capture_launch(graph)
        after=env.qpos.cpu().numpy().copy();av=env.qvel.cpu().numpy().copy();actual,n=gpu_contacts(env);maxn=max(maxn,n)
        pre.append(before);prevel.append(bv);post.append(after);postvel.append(av);controls.append(bc)
        for world in range(nworld):
            gs=[r for r in actual if r['world']==world]
            cpu.qpos[:]=before[world];cpu.qvel[:]=bv[world];cpu.ctrl[:]=bc[world];cpu.qacc_warmstart[:]=0;mujoco.mj_forward(env.m,cpu);cs=cpu_contacts(env.m,cpu)
            comp=compare_contacts(gs,cs);comp.update(physics_step=s+1,world=world,contact_time_s=s*.002,state_time_s=(s+1)*.002)
            if s%10==9:
                # Quantify the old diagnostic's 2 ms timing mismatch too.
                cpu.qpos[:]=after[world];cpu.qvel[:]=av[world];mujoco.mj_forward(env.m,cpu)
                comp['poststate_cpu_contact_count']=len([x for x in cpu_contacts(env.m,cpu) if x['kind']=='hand_object'])
            comparisons.append(comp)
            if comp['count_delta'] and first_pair_difference is None:first_pair_difference=dict(**comp,qpos=before[world].tolist(),qvel=bv[world].tolist(),ctrl=bc[world].tolist(),gpu_contacts=gs,cpu_contacts=cs)
        contactlog.append(dict(physics_step=s+1,time_s=s*.002,contacts=actual))
    elapsed=time.perf_counter()-start
    np.savez_compressed(out/'trajectory.npz',qpos_before=pre,qvel_before=prevel,qpos_after=post,qvel_after=postvel,ctrl=controls,initial_qpos=initial,initial_qvel=initialv)
    write_json(out/'gpu_contacts.json',contactlog);write_json(out/'cpu_comparison.json',comparisons);write_json(out/'first_contact_difference.json',first_pair_difference)
    # One world's reset must not write into another world's state/control/history.
    reset_checks={}
    if nworld>1:
        for name,array in [('qpos',env.qpos),('qvel',env.qvel),('ctrl',env.ctrl),('warmstart',wp.to_torch(env.wd.qacc_warmstart)),('time',wp.to_torch(env.wd.time)),('phase',env.phase),('action',env.last_action)]:
            reset_checks[name]=array.clone()
        env.reset(torch.tensor([0],device=env.device),False)
        resetdiff={name:float((array[1:]-reset_checks[name][1:]).abs().max()) for name,array in [('qpos',env.qpos),('qvel',env.qvel),('ctrl',env.ctrl),('warmstart',wp.to_torch(env.wd.qacc_warmstart)),('time',wp.to_torch(env.wd.time)),('phase',env.phase),('action',env.last_action)]}
        reset_exact=all(x==0 for x in resetdiff.values())
    else:resetdiff=None;reset_exact=None
    # Explicit synchronization must agree with normal shared-stream execution.
    env.reset(allids,False);syncq=[]
    for s in range(30):
        env.ctrl[:]=env.target[s//10+1];torch.cuda.synchronize();wp.capture_launch(graph);wp.synchronize();syncq.append(env.qpos.cpu().numpy().copy())
    syncerr=float(np.max(abs(np.asarray(syncq)-np.asarray(post)[:30])))
    # Reset world 0 at 60 ms, then verify other worlds advance as in the untouched trace.
    env.reset(torch.tensor([0],device=env.device),False);env.ctrl[:]=env.target[4];wp.capture_launch(graph);wp.synchronize()
    post_reset_other_error=float(np.max(abs(env.qpos.cpu().numpy()[1:]-np.asarray(post)[30,1:]))) if nworld>1 else None
    first_spread={}
    if nworld>1:
        a=np.asarray(post);pos=np.linalg.norm(a[:,:,26:29]-a[:,0:1,26:29],axis=-1).max(1)
        for threshold in [1e-6,1e-4,.001,.005]:
            found=np.flatnonzero(pos>threshold);first_spread[str(threshold)]=dict(physics_step=int(found[0]+1),time_s=float((found[0]+1)*.002),position_spread_m=float(pos[found[0]])) if len(found) else None
        maxspread=float(pos.max())
    else:maxspread=0
    report=dict(status='completed',nworld=nworld,physics_steps=510,noise=False,same_initial_copies=True,world_reset_other_fields_exact=reset_exact,reset_field_max_changes=resetdiff,
                post_reset_other_world_one_step_qpos_max_difference=post_reset_other_error,explicit_sync_max_qpos_difference=syncerr,
                first_interworld_position_differences=first_spread,maximum_interworld_object_position_spread_m=maxspread,
                max_total_gpu_contacts=maxn,allocated_contact_capacity=len(env.wd.contact.dist),
                contact_comparison_count=len(comparisons),different_contact_count_fraction=sum(c['count_delta']!=0 for c in comparisons)/len(comparisons),
                elapsed_with_logging_s=elapsed,shared_resource_note='Diagnostic with CPU transfers and shared GPU; not throughput evidence.',
                contact_timestamp_note='GPU contact/force from actual mjw.step belongs to pre-integration state. CPU is reconstructed at that same qpos/qvel/ctrl; warmstart may differ, so force equality is not asserted.')
    write_json(out/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,choices=[1,2,8],required=True);args=p.parse_args();run(args.worlds)
