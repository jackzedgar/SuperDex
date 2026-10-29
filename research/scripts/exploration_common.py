"""Shared budget, GPU-0 check and actual-contact logging for exploratory learning."""
import os,json,time,subprocess
from pathlib import Path
import numpy as np,torch,warp as wp,mujoco_warp as mjw
from real_env import ROOT,DATA,GPU

RUN=Path(os.environ.get('SUPERDEX_EXPLORE_RUN',ROOT/'runs/explore_20261026'))
RUN.mkdir(parents=True,exist_ok=True)
def write_json(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def read_json(p):return json.loads(p.read_text())
def activate(env):
    # Two independent single-world buffers run serially. Finish previous policy /
    # physics writes before changing the current stream for checkpoint evaluation.
    wp.synchronize();torch.cuda.synchronize();torch.cuda.set_stream(env.torch_stream);wp.set_stream(env.stream)
def resource_check(min_mb=1200):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(x for x in rows.splitlines() if GPU in x);free=int(row.split(',')[2]);assert free>=min_mb,f'Selected GPU {GPU} has {free} MiB free, requires {min_mb}; resource skip, not algorithm failure.'
    return rows


class Contacts:
    def __init__(self,env):
        self.e=env;self.ids=wp.array(np.arange(256,dtype=np.int32),device='cuda:0');self.forces=wp.zeros(256,dtype=wp.spatial_vector,device='cuda:0')
    def read(self):
        e=self.e;count=int(e.wd.nacon.numpy()[0]);assert count<=256
        if not count:return 0,0,0.,np.zeros(3,dtype=np.float32)
        mjw.contact_force(e.wm,e.wd,self.ids,False,self.forces);wp.synchronize()
        g=e.wd.contact.geom.numpy()[:count];f=self.forces.numpy()[:count,0];p=e.wd.contact.pos.numpy()[:count]
        ty=e.m.geom_contype[g];hand=((ty==1).any(1)&(ty==2).any(1)&(f>.01));floor=((ty==4).any(1)&(ty==2).any(1)&(f>.01))
        centroid=np.average(p[hand],axis=0,weights=f[hand]) if hand.any() else np.zeros(3)
        return int(hand.sum()),int(floor.sum()),float(f[hand].sum()),centroid.astype(np.float32)


class EpisodeLog:
    def __init__(self,path,env,meta):
        self.path=path;self.meta=meta;self.q=[env.qpos[0].cpu().numpy().copy()];self.v=[env.qvel[0].cpu().numpy().copy()];self.rows=[]
        self.initial_ctrl=env.ctrl[0].cpu().numpy().copy()
    def add(self,env,action,phase,reward,info,contacts):
        self.q.append(env.qpos[0].cpu().numpy().copy());self.v.append(env.qvel[0].cpu().numpy().copy())
        self.rows.append((env.ctrl[0].cpu().numpy().copy(),action[0].cpu().numpy().copy(),phase,float(reward[0]),float(info['pos_error'][0]),float(info['rot_error'][0]),*contacts))
    def save(self,reason,complete):
        if not self.rows:return
        a=list(zip(*self.rows));self.path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(self.path,qpos=self.q,qvel=self.v,ctrl=a[0],raw_action=a[1],phase=a[2],reward=a[3],position_error=a[4],rotation_error=a[5],hand_contact=a[6],floor_contact=a[7],normal_force=a[8],contact_centroid=a[9],initial_ctrl=self.initial_ctrl)
        write_json(self.path.with_suffix('.json'),dict(**self.meta,steps=len(self.rows),reason=reason,complete_episode=complete,contact_timestamp='For transition t->t+1: loaded actual GPU contacts at t+0.018 s; states at t and t+0.020 s. Normal force >0.01 N per point.',source='MuJoCo Warp physical transitions, object only assigned at reset'))


def finish_timing(name,started,details):
    seconds=time.perf_counter()-started
    write_json(RUN/(name+'_timing.json'),dict(gpu_process_seconds=seconds,**details));return seconds
