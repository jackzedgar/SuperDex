"""Render stored physical trajectories for preselected cases; no physics stepping."""
import os,time,argparse,itertools
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from exploration_common import RUN,DATA,read_json,write_json,resource_check

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);p.add_argument('--output',default='videos');a=p.parse_args();start=time.perf_counter();assert os.environ['MUJOCO_EGL_DEVICE_ID']=='0';assert '/' not in a.output and '..' not in a.output
    out=RUN/a.output;out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check(1200));selection=read_json(RUN/'video_selection_rule.json');completion=read_json(RUN/'core_completion.json');step=completion['steps_per_arm'];base=RUN/'core_heldout'
    model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));data=mujoco.MjData(model);renderer=mujoco.Renderer(model,480,640);camera=mujoco.MjvCamera();camera.distance=.55;camera.azimuth=110;camera.elevation=-20;option=mujoco.MjvOption();option.geomgroup[1]=1;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);completed=[]
    visible=[]
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom]==0 or not option.geomgroup[model.geom_group[geom]]:continue
        if model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=model.geom_dataid[geom];offset=model.mesh_vertadr[mesh];vertices=model.mesh_vert[offset:offset+model.mesh_vertnum[mesh]]
        else:vertices=np.array(list(itertools.product([-1,1],repeat=3)))*model.geom_rbound[geom]
        visible.append((geom,vertices))
    for initial in selection['initial_ids']:
        if time.perf_counter()-start+20>a.seconds:break
        arrays={arm:np.load(base/f'{arm}_step_{step:06d}_repeat_0'/f'trial_{initial:02d}.npz') for arm in ['P','M']};summaries={arm:read_json(base/f'{arm}_step_{step:06d}_repeat_0'/'summary.json')['trials'][initial] for arm in arrays}
        writer=imageio.get_writer(out/f'heldout_initial_{initial:02d}.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1)
        for k in range(52):
            combined=Image.new('RGB',(1280,580))
            # Frame the union of actual visible mesh bounds in both panels.
            # Wrist qpos are joint coordinates, not an absolute world-space point.
            bounds=[]
            for log in arrays.values():
                data.qpos[:]=log['qpos'][min(k,len(log['qpos'])-1)];mujoco.mj_forward(model,data)
                points=np.concatenate([vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom] for geom,vertices in visible]);bounds.extend([points.min(0),points.max(0)])
            lo=np.min(bounds,axis=0);hi=np.max(bounds,axis=0);camera.lookat[:]=(lo+hi)/2;camera.distance=1.1*np.linalg.norm(hi-lo)/2/np.sin(np.deg2rad(model.vis.global_.fovy/2))
            for j,arm in enumerate(['P','M']):
                log=arrays[arm];last=len(log['qpos'])-1;at=min(k,last);data.qpos[:]=log['qpos'][at];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=option);panel=Image.new('RGB',(640,580),'white');panel.paste(Image.fromarray(renderer.render()),(0,100));draw=ImageDraw.Draw(panel)
                title='P: pure PPO' if arm=='P' else 'M: PPO + frozen W1-mini'
                draw.text((10,5),f'{title} | fixed held-out initial {initial}',fill='black',font=font);draw.text((10,29),f'Logged t={at*.02:.2f}s | 0.5x | shared tracking camera',fill='black',font=font)
                if at:
                    pos=log['position_error'][at-1]*1000;rot=np.rad2deg(log['rotation_error'][at-1]);draw.text((10,53),f'Position {pos:.1f} mm | orientation {rot:.1f} deg',fill='black',font=font)
                draw.text((10,77),'Complete-trajectory strict success: '+('YES' if summaries[arm]['success'] else 'NO'),fill='black',font=font)
                if k>last:draw.text((10,115),'TERMINATED: final recorded state held',fill='white',font=font)
                combined.paste(panel,(j*640,0))
            writer.append_data(np.asarray(combined))
            if k in [0,26,51]:combined.save(out/f'initial_{initial:02d}_frame_{k:02d}.png')
        writer.close();completed.append(initial)
    renderer.close();write_json(out/'summary.json',dict(completed_initial_ids=completed,planned_initial_ids=selection['initial_ids'],checkpoint_steps=step,source='All stored physical states from final core checkpoint, held-out repeat0. Selection fixed before held-out evaluation.',rendering_only=True,physics_controls=0,physical_controls=0,model_predictions=0,process_seconds=time.perf_counter()-start,playback_speed=0.5,early_termination='Final recorded state held with explicit label; no extrapolated physics.'))

if __name__=='__main__':main()
