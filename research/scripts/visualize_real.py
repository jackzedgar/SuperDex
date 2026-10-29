"""Preserved reference versus physical logs; all plotted units are explicit."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
from pathlib import Path
import argparse,json,csv
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed/clip000000_obj000004_right'
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#333333']


def main():
    p=argparse.ArgumentParser();p.add_argument('--ppo',type=Path,default=ROOT/'runs/real_track_20261026/ppo');p.add_argument('--plots-only',action='store_true');args=p.parse_args();out=args.ppo/'visuals';out.mkdir(parents=True,exist_ok=True)
    a=np.load(DATA/'trajectory.npz');t=a['time_s'];processing=json.loads((DATA/'processing_report.json').read_text());before=np.load(args.ppo/'rollout_before.npz');after=np.load(args.ppo/'rollout_after.npz')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for j in range(5):axes[0,0].plot(t,a['keypoint_error_m'][:,j]*1000,label=['Thumb','Index','Middle','Ring','Little'][j],color=COLORS[j],linestyle=['-','--','-.',':','-'][j])
    axes[0,0].set(title='Retargeting: human to Wuji fingertips',xlabel='Time (s)',ylabel='Euclidean error (mm)');axes[0,0].legend(ncol=3,fontsize=8,loc='upper center');axes[0,0].set_ylim(0,5.3)
    axes[0,1].plot(t,a['penetration_m']*1000,color=COLORS[1]);axes[0,1].axhline(2,color='#555555',ls='--',label='Pilot admission limit');axes[0,1].set(title='Reference hand/object/floor penetration',xlabel='Time (s)',ylabel='Maximum depth (mm)',ylim=(0,2.3));axes[0,1].legend(fontsize=8)
    for run,label,color,ls in [(before,'Before PPO',COLORS[0],'--'),(after,'After PPO',COLORS[1],'-')]:
        tm=(run['phase']+1)*.02
        axes[1,0].plot(tm,run['errors'][:,0]*100,label=label,color=color,ls=ls)
        axes[1,1].plot(tm,np.rad2deg(run['errors'][:,1]),label=label,color=color,ls=ls)
    axes[1,0].axhline(3,color='#555555',ls=':',label='Success threshold');axes[1,1].axhline(30,color='#555555',ls=':',label='Success threshold')
    axes[1,0].set(title='Physical execution: paired episode 0',xlabel='Time (s)',ylabel='Object position error (cm)',xlim=(0,t[-1]));axes[1,1].set(title='Physical execution: paired episode 0',xlabel='Time (s)',ylabel='Object rotation error (degrees)',xlim=(0,t[-1]))
    for ax in axes[1]:ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    fig.suptitle('HOT3D clip 000000 / right hand / wooden spoon\nSingle-clip pilot; terminated episodes stop at their actual end',fontsize=12)
    fig.savefig(out/'tracking_diagnostics.png');fig.savefig(out/'tracking_diagnostics.pdf');plt.close(fig)
    fig=plt.figure(figsize=(12,4),layout='constrained')
    allpoints=np.concatenate([a['human_landmarks'].reshape(-1,3),a['robot_keypoints'].reshape(-1,3)]);center=(allpoints.max(0)+allpoints.min(0))/2;radius=np.ptp(allpoints,axis=0).max()/2+.01
    for j,k in enumerate([0,len(t)//2,len(t)-1]):
        ax=fig.add_subplot(1,3,j+1,projection='3d');h=a['human_landmarks'][k];r=a['robot_keypoints'][k]
        ax.scatter(*h[:5].T,color=COLORS[0],marker='o',s=35,label='Human fingertips')
        ax.scatter(*r[:5].T,color=COLORS[1],marker='x',s=45,label='Wuji fingertips')
        for finger in range(5):
            ax.plot(*np.vstack([r[finger+5],r[finger]]).T,color=COLORS[1],lw=1)
        ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=f't = {t[k]:.2f} s',xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius));ax.set_box_aspect([1,1,1]);ax.view_init(20,-65)
        if j==0:ax.legend(fontsize=8,loc='upper left')
    fig.suptitle('Original human landmarks vs. retargeted Wuji; shared metric axes')
    fig.savefig(out/'retarget_overlay.png',bbox_inches='tight',pad_inches=.3);plt.close(fig)
    rows=list(csv.DictReader((args.ppo/'training.csv').open()));x=np.array([float(r['environment_steps']) for r in rows])
    fig,ax=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for axes,key,label in zip(ax,['episode_return','object_position_mean_m','object_rotation_mean_deg'],['Episode return','Object position error (cm)','Object rotation error (degrees)']):
        y=np.array([float(r[key]) if r[key] else np.nan for r in rows])
        if key=='object_position_mean_m':y*=100
        axes.plot(x,y,color=COLORS[0],lw=1);axes.set(xlabel='Environment control steps',ylabel=label);axes.set_ylim(bottom=0)
    fig.suptitle('PPO training, one seed; random reference start phases (not evaluation success)')
    fig.savefig(out/'training_curves.png');fig.savefig(out/'training_curves.pdf');plt.close(fig)
    if args.plots_only:return
    # Render stored states. Left forcibly places reference object and hand for
    # visualization; right renders logged physics. Never labels left as execution.
    m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,480,640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.02,-.02,.2];camera.distance=.85;camera.azimuth=110;camera.elevation=-20
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    opt=mujoco.MjvOption();opt.geomgroup[1]=1
    writer=imageio.get_writer(out/'reference_vs_policy.mp4',fps=50,codec='libx264',quality=8)
    physical=np.vstack([after['qpos'],after['final_qpos']]);last=len(physical)-1
    for k in range(len(t)):
        panels=[]
        for q,label in [(a['qpos'][k],'REFERENCE PLAYBACK (kinematic placement)'),(physical[min(k,last)],'POLICY: logged physical execution')]:
            d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=opt)
            panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel);draw.rectangle((0,0,640,58),fill='white');draw.text((12,7),label,fill='black',font=font)
            draw.text((12,31),f't = {min(k,last)*.02 if label.startswith("POLICY") else t[k]:.2f} s',fill='black',font=font);panels.append(panel)
        combined=Image.new('RGB',(1280,480));combined.paste(panels[0],(0,0));combined.paste(panels[1],(640,0))
        if k>last:ImageDraw.Draw(combined).text((655,70),'EPISODE ENDED - final state frozen',fill='white',font=font)
        writer.append_data(np.asarray(combined))
        if k in [0,len(t)//2,len(t)-1]:combined.save(out/f'comparison_{k:03d}.png')
    writer.close();renderer.close()
    (out/'visualization_notes.json').write_text(json.dumps(dict(raw='trajectory.npz and rollout_*.npz',reference='Kinematic visualization only; no success claim',policy='Stored GPU physics trajectory; if terminated early, final state freezes with a visible label',smoothing='none',statistics='One training seed; no confidence intervals or multi-task claims',source_frame_range=processing['source_frame_range'],units='m internally; panels explicitly show mm, cm, degrees and seconds',citation='Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065'),indent=2)+'\n')
    print('Saved',out)


if __name__=='__main__':main()
