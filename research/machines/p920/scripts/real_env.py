"""Single-clip MuJoCo Warp environment. All runtime state and actions stay on GPU.

Only reset writes the object's free-joint state. step writes servo controls only.
The six wrist joints form a position-controlled virtual mount, not a robot arm.
"""
import os,json,time
from pathlib import Path
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
import torch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/clip000000_obj000004_right'
GPU='GPU-810d1bb5-dcdc-3d20-c3c5-22bffb1fcf9d'


def quat_error(q,r):
    # q and r are unit wxyz. abs handles the quaternion double cover.
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-8)
    return 2*torch.acos(torch.sum(q*r,dim=-1).abs().clamp(max=1))


class RealEnv:
    def __init__(self,nworld=1,run=None):
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
        assert torch.cuda.device_count()==1 and str(torch.cuda.get_device_properties(0).uuid)==GPU[4:]
        check=json.loads((ROOT/'runs/real_track_20261026/qualification_cpu.json').read_text())
        assert check['ready_for_bounded_ppo'],check
        self.device='cuda:0';self.n=nworld
        wp.config.kernel_cache_dir=str(ROOT/'cache/warp');wp.init();wp.set_device(self.device)
        assert len(wp.get_cuda_devices())==1
        # Shared CUDA stream establishes the Torch write -> Warp read ordering.
        # CUDA graphs cannot capture Torch's legacy default stream (stream 0).
        self.torch_stream=torch.cuda.Stream(device=0)
        torch.cuda.set_stream(self.torch_stream)
        self.stream=wp.stream_from_torch(self.torch_stream)
        wp.set_stream(self.stream)
        self.m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));initial=mujoco.MjData(self.m)
        raw=np.load(DATA/'trajectory.npz');control=np.load(DATA/'controller_reference.npz')['ctrl']
        self.ref=torch.tensor(raw['qpos'],device=self.device,dtype=torch.float32)
        self.refvel=torch.tensor(raw['qvel'],device=self.device,dtype=torch.float32)
        self.target=torch.tensor(control,device=self.device,dtype=torch.float32)
        self.length=len(self.ref)
        initial.qpos[:]=raw['qpos'][0];initial.ctrl[:]=control[0];mujoco.mj_forward(self.m,initial)
        self.wm=mjw.put_model(self.m)
        # Batched contact sensitivity is unresolved. The qualified learning pilot
        # uses ONE world; ATOMICS added cost without eliminating batch differences.
        # Another MNR job was later found occupying ~23.3 GiB on physical GPU 0;
        # the earlier OOM does not measure the intrinsic dedicated-device capacity.
        self.wm.opt.deterministic=mjw.DeterminismType.CONSTRAINT
        self.wd=mjw.put_data(self.m,initial,nworld=nworld,nconmax=256,njmax=768)
        self.qpos=wp.to_torch(self.wd.qpos);self.qvel=wp.to_torch(self.wd.qvel);self.ctrl=wp.to_torch(self.wd.ctrl)
        assert self.qpos.data_ptr()==self.wd.qpos.ptr and self.ctrl.data_ptr()==self.wd.ctrl.ptr
        self.lo=torch.tensor(self.m.actuator_ctrlrange[:,0],device=self.device,dtype=torch.float32)
        self.hi=torch.tensor(self.m.actuator_ctrlrange[:,1],device=self.device,dtype=torch.float32)
        self.scale=torch.tensor([.025]*3+[.18]*3+[.25]*20,device=self.device)
        self.phase=torch.zeros(nworld,dtype=torch.long,device=self.device)
        self.last_action=torch.zeros(nworld,26,device=self.device)
        self.steps=torch.zeros(nworld,dtype=torch.long,device=self.device)
        self.returns=torch.zeros(nworld,device=self.device)
        self.possum=self.returns.clone();self.rotsum=self.returns.clone()
        self.all_good=torch.ones(nworld,dtype=torch.bool,device=self.device)
        self.episode_start=self.phase.clone()
        started=time.perf_counter()
        self._physics();wp.synchronize()
        with wp.ScopedCapture() as capture:self._physics()
        self.graph=capture.graph
        self.compile_seconds=time.perf_counter()-started
        self.reset(torch.arange(nworld,device=self.device),random_phase=False)
        self.obs_dim=self.observe().shape[1]

    def _physics(self):
        for _ in range(10):mjw.step(self.wm,self.wd)

    def reset(self,ids,random_phase=True,noise=False):
        count=len(ids)
        phase=torch.randint(0,self.length-10,(count,),device=self.device) if random_phase else torch.zeros(count,dtype=torch.long,device=self.device)
        self.phase[ids]=phase;self.episode_start[ids]=phase
        self.qpos[ids]=self.ref[phase];self.qvel[ids]=self.refvel[phase]
        if noise:
            self.qpos[ids,26:29]+=torch.randn(count,3,device=self.device)*.001
            self.qpos[ids,6:26]+=torch.randn(count,20,device=self.device)*.002
        self.ctrl[ids]=self.target[phase]
        for name in ['qacc_warmstart','qacc','qfrc_applied','xfrc_applied']:
            if hasattr(self.wd,name):wp.to_torch(getattr(self.wd,name))[ids]=0
        wp.to_torch(self.wd.time)[ids]=0
        self.last_action[ids]=0;self.steps[ids]=0;self.returns[ids]=0;self.possum[ids]=0;self.rotsum[ids]=0;self.all_good[ids]=True

    def observe(self):
        current=self.ref[self.phase];nextref=self.target[(self.phase+1).clamp(max=self.length-1)]
        # Quaternion sign-invariant orientation observation.
        q=self.qpos[:,29:33];r=current[:,29:33];q=q*torch.where((q*r).sum(-1,keepdim=True)<0,-1.,1.)
        return torch.cat([self.qpos[:,:26],self.qvel[:,:26]*.1,(nextref-self.qpos[:,:26]),
                          (self.qpos[:,26:29]-current[:,26:29])*10,q-r,self.qvel[:,26:]*.1,
                          self.phase[:,None]/(self.length-1),self.last_action],dim=-1).clamp(-20,20)

    @torch.no_grad()
    def step(self,raw_action):
        action=raw_action.clamp(-1,1)
        nextphase=(self.phase+1).clamp(max=self.length-1)
        self.ctrl[:]=torch.maximum(self.lo,torch.minimum(self.hi,self.target[nextphase]+action*self.scale))
        wp.capture_launch(self.graph)
        self.phase[:]=nextphase;self.steps+=1
        ref=self.ref[self.phase]
        ep=torch.linalg.vector_norm(self.qpos[:,26:29]-ref[:,26:29],dim=-1)
        er=quat_error(self.qpos[:,29:33],ref[:,29:33])
        eq=torch.mean((self.qpos[:,6:26]-ref[:,6:26])**2,dim=-1)
        ew=torch.sum((self.qpos[:,:3]-ref[:,:3])**2,dim=-1)
        reward=.02*(5*torch.exp(-(ep/.04)**2)+2*torch.exp(-(er/.5)**2)+.5*torch.exp(-eq/.1)+.5*torch.exp(-ew/.001)-.03*action.square().mean(-1)-.05*(action-self.last_action).square().mean(-1))
        finite=torch.isfinite(self.qpos).all(-1)&torch.isfinite(self.qvel).all(-1)
        joint_bad=((self.qpos[:,:26]<self.lo-.15)|(self.qpos[:,:26]>self.hi+.15)).any(-1)
        failure=(ep>.15)|(er>1.4)|(~finite)|joint_bad
        timeout=self.phase==self.length-1
        done=failure|timeout
        self.returns+=reward;self.possum+=ep;self.rotsum+=er;self.all_good&=(ep<=.03)&(er<=np.pi/6)&finite
        self.last_action[:]=action
        info=dict(pos_error=ep,rot_error=er,failure=failure,timeout=timeout,finite=finite,
                  reason=torch.where(~finite,4,torch.where(joint_bad,3,torch.where(ep>.15,1,torch.where(er>1.4,2,0)))),
                  episode_return=self.returns.clone(),episode_length=self.steps.clone(),
                  mean_pos=self.possum/self.steps,mean_rot=self.rotsum/self.steps,
                  success=timeout&~failure&self.all_good&(self.episode_start==0))
        # No automatic reset: caller records final state and bootstraps before reset.
        return self.observe(),reward,done,info


def qualify_gpu(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    env=RealEnv(1);ctrl_before=env.ctrl[0,0].clone();env.ctrl[0,0]=.123
    wp.synchronize();alias=abs(float(env.wd.ctrl.numpy()[0,0])-.123)<1e-6;env.ctrl[0,0]=ctrl_before
    env.reset(torch.arange(env.n,device=env.device),False)
    states=[];errors=[];start=time.perf_counter()
    for t in range(env.length-1):
        _,_,_,info=env.step(torch.zeros(env.n,26,device=env.device));states.append(env.qpos[0].cpu().numpy().copy());errors.append([info['pos_error'][0].item(),info['rot_error'][0].item()])
        assert info['finite'].all().item()
    wp.synchronize();elapsed=time.perf_counter()-start
    # Independent CPU replay of exactly the same servo targets and initial state.
    d=mujoco.MjData(env.m);ref=env.ref.cpu().numpy();d.qpos[:]=ref[0];d.qvel[:]=env.refvel[0].cpu().numpy();cpu=[]
    for i in range(1,env.length):
        d.ctrl[:]=env.target[i].cpu().numpy()
        for _ in range(10):mujoco.mj_step(env.m,d)
        cpu.append(d.qpos.copy())
    cpu=np.array(cpu);states=np.array(states);errors=np.array(errors)
    same_worlds=(env.qpos-env.qpos[0]).abs().max().item()
    world_pos_spread=torch.linalg.vector_norm(env.qpos[:,26:29]-env.qpos[0,26:29],dim=-1).max().item()
    world_rot_spread=quat_error(env.qpos[:,29:33],env.qpos[0,29:33]).max().item()
    delta=np.linalg.norm(states[:,26:29]-cpu[:,26:29],axis=1)
    # Determinism after reset also validates warmstart clearing.
    env.reset(torch.arange(env.n,device=env.device),False)
    env.step(torch.zeros(env.n,26,device=env.device));reset_delta=np.max(abs(env.qpos[0].cpu().numpy()-states[0]))
    first_all=env.qpos.clone();repeat=[env.qpos[0].cpu().numpy().copy()]
    for _ in range(env.length-2):
        env.step(torch.zeros(env.n,26,device=env.device));repeat.append(env.qpos[0].cpu().numpy().copy())
    repeat=np.asarray(repeat)
    repeat_pos=np.linalg.norm(repeat[:,26:29]-states[:,26:29],axis=1).max()
    repeat_angle=2*np.arccos(np.clip(np.abs(np.sum(repeat[:,29:33]*states[:,29:33],axis=1))/(np.linalg.norm(repeat[:,29:33],axis=1)*np.linalg.norm(states[:,29:33],axis=1)),0,1)).max()
    env.reset(torch.arange(env.n,device=env.device),False)
    perturb=torch.zeros(env.n,26,device=env.device);perturb[0]=1
    env.step(perturb)
    other_world_change=None
    controlled_world_change=(env.qpos[0]-first_all[0]).abs().max().item()
    checks=dict(gpu_uuid_only_0=True,zero_copy_alias=alias,finite=True,
                complete_replay_repeat_under_5mm_1deg=bool(repeat_pos<.005 and repeat_angle<np.pi/180),
                action_changes_physical_state=controlled_world_change>.001,
                reset_reproducible=bool(reset_delta<1e-5),cpu_gpu_object_max_difference_under_3cm=bool(delta.max()<.03))
    report=dict(checks=checks,ready_for_bounded_ppo=all(checks.values()),num_envs=env.n,steps=env.length-1,
                compile_seconds=env.compile_seconds,measured_seconds=elapsed,control_steps_per_second=env.n*(env.length-1)/elapsed,
                cpu_gpu_object_mean_difference_m=float(delta.mean()),cpu_gpu_object_max_difference_m=float(delta.max()),
                reference_control_object_pos_mean_m=float(errors[:,0].mean()),reference_control_object_rot_mean_deg=float(np.rad2deg(errors[:,1].mean())),
                reset_max_qpos_difference=float(reset_delta),peak_torch_memory_mb=torch.cuda.max_memory_allocated()/1e6,
                interworld_qpos_max_difference=same_worlds,interworld_object_pos_spread_m=world_pos_spread,interworld_rotation_spread_deg=float(np.rad2deg(world_rot_spread)),
                unaffected_world_max_qpos_change=other_world_change,controlled_world_max_qpos_change=controlled_world_change,
                repeat_full_trajectory_position_max_m=float(repeat_pos),repeat_full_trajectory_angle_max_deg=float(np.rad2deg(repeat_angle)),
                determinism_note='CONSTRAINT enabled; complete single-world repetition measured. Multiworld consistency failed earlier; retained separately, not claimed solved. CONTACTS is a no-op in this release.',
                gpu_uuid=GPU,interop='wp.to_torch zero copy; Torch and Warp share one CUDA stream',scope='ONE real-mesh world; not D0 throughput or fidelity acceptance')
    np.savez_compressed(out/'gpu_reference_control.npz',qpos=states,cpu_qpos=cpu,errors=errors)
    (out/'qualification_gpu.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':qualify_gpu(ROOT/'runs/real_track_20261026')
