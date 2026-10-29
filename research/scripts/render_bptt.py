"""Render preselected actual physical trajectories, not reference playback."""
import os,time,itertools,json
os.environ['MUJOCO_GL']='egl';os.environ['MUJOCO_EGL_DEVICE_ID']='0'
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from bptt_common import ROOT,RUN,write
def bounds(m,d,opt):
    allpoints=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g]==0 or not opt.geomgroup[m.geom_group[g]]:continue
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=m.geom_dataid[g];a=m.mesh_vertadr[mesh];v=m.mesh_vert[a:a+m.mesh_vertnum[mesh]]
        else:v=np.array(list(itertools.product([-1,1],repeat=3)))*m.geom_rbound[g]
        allpoints.append(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    x=np.concatenate(allpoints);return x.min(0),x.max(0)
def pair(models,logs,summaries,labels,path):
    ds=[mujoco.MjData(m) for m in models];rs=[mujoco.Renderer(m,360,480) for m in models]
    cams=[mujoco.MjvCamera() for _ in models];opt=mujoco.MjvOption();opt.geomgroup[1]=1
    for c in cams:c.azimuth=110;c.elevation=-20
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    writer=imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=1)
    for k in range(max(len(x['qpos']) for x in logs)):
        canvas=Image.new('RGB',(960,440),'white');shared_bounds=None
        if models[0] is models[1]:
            limits=[]
            for m,d,log in zip(models,ds,logs):
                d.qpos[:]=log['qpos'][min(k,len(log['qpos'])-1)];mujoco.mj_forward(m,d);limits.extend(bounds(m,d,opt))
            shared_bounds=(np.min(limits,axis=0),np.max(limits,axis=0))
        for j,(m,d,r,c,log,summary,label) in enumerate(zip(models,ds,rs,cams,logs,summaries,labels)):
            end=len(log['qpos'])-1;at=min(k,end);d.qpos[:]=log['qpos'][at];mujoco.mj_forward(m,d);lo,hi=shared_bounds if shared_bounds is not None else bounds(m,d,opt)
            c.lookat[:]=(lo+hi)/2;c.distance=1.12*np.linalg.norm(hi-lo)/2/np.sin(np.deg2rad(m.vis.global_.fovy/2))
            r.update_scene(d,camera=c,scene_option=opt);panel=Image.new('RGB',(480,440),'white');panel.paste(Image.fromarray(r.render()),(0,80));draw=ImageDraw.Draw(panel)
            draw.text((7,4),label,fill='black',font=font)
            draw.text((7,23),f'Actual logged t={at*.02:.2f}s | 0.5x playback',fill='black',font=font)
            if at:draw.text((7,42),f'Pos {log["position_error"][at-1]*1000:.1f}mm | Rot {np.rad2deg(log["rotation_error"][at-1]):.1f}deg',fill='black',font=font)
            draw.text((7,61),'Full-task strict success: '+str(summary['success']),fill='black',font=font)
            if k>end:draw.text((7,87),('PHYSICAL TERMINATION - final state held' if summary['reason'] else 'REFERENCE ENDED - final state held'),fill='white',font=font)
            canvas.paste(panel,(j*480,0))
        writer.append_data(np.asarray(canvas))
        if k in [0,25,50]:canvas.save(path.with_name(path.stem+f'_frame{k:02d}.png'))
    writer.close()
    for r in rs:r.close()
def main():
    started=time.time();out=RUN/'videos';out.mkdir(exist_ok=False);created=[]
    m=mujoco.MjModel.from_xml_path(str(ROOT/'runtime/lambda/processed/clip000000_obj000004_right/scene.xml'))
    for seed in [43,59]:
        common=[]
        for step in range(0,524289,65536):
            if all((RUN/f'core_{method}_{seed}/fixed_evaluation/new32/step_{step:06d}/summary.json').exists() for method in ['P','S']):common.append(step)
        if not common:continue
        step=max(common)
        for trial in [0,7]:
            folders=[RUN/f'core_{method}_{seed}/fixed_evaluation/new32/step_{step:06d}' for method in ['P','S']]
            logs=[np.load(p/f'trial_{trial:02d}.npz') for p in folders];summary=[json.loads((p/'summary.json').read_text())['trials'][trial] for p in folders]
            path=out/f'seed{seed}_initial{trial:02d}_step{step:06d}.mp4'
            pair([m,m],logs,summary,[f'{method}, seed{seed}, initial{trial}, {step} controls' for method in ['P','S']],path);created.append(path.name)
    p=RUN/'shared_two_objects'
    if (p/'summary.json').exists():
        step=json.loads((p/'summary.json').read_text())['real_controls'];folders=[p/f'development/step_{step:06d}/task_{i}' for i in [0,1]]
        if all((f/'summary.json').exists() for f in folders):
            m1=mujoco.MjModel.from_xml_path(str(ROOT/'runtime/lambda/processed/backup_clip000001_obj000033_right/scene.xml'))
            logs=[np.load(f/'trial_00.npz') for f in folders];summaries=[json.loads((f/'summary.json').read_text())['trials'][0] for f in folders]
            path=out/'shared_two_objects_initial00.mp4';pair([m,m1],logs,summaries,['Shared policy: spoon, dev initial0','Shared policy: DVD remote, dev initial0'],path);created.append(path.name)
    write(out/'manifest.json',dict(files=created,source='Previously recorded real MuJoCo Warp policy execution. Render-only qpos playback; no new physics or imagined/reference trajectories.',selection='Predeclared seed43/59,new conditions0 and7; latest COMMON completed scheduled checkpoint. Shared task dev condition0.',seconds=time.time()-started,physics_controls=0,early_termination='Final recorded state held with explicit physical/reference-end label, no extrapolation',camera='Common union-bounds camera for P/S; separate tracking cameras for different objects',playback_speed=.5))
if __name__=='__main__':main()

