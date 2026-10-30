"""Truthful paired figures and videos of stored physical states; no simulation.

One clip, eight fixed initial conditions, no independent-training-seed inference.
"""
import os,argparse,json
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
import numpy as np,mujoco,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from grasp_lab import RUN,DATA,write_json

BLUE='#0072B2';ORANGE='#B34E00';GREEN='#007856'
OUT=RUN/'visuals';OUT.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text())
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(name+'.pdf'),facecolor='white');plt.close(fig)


def plots():
    with plt.rc_context({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42}):
        fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for n,c,ls in [(2,BLUE,'-'),(8,ORANGE,'--')]:
            a=np.load(RUN/f'batch_{n}/trajectory.npz')['qpos_after'];y=np.linalg.norm(a[:,:,26:29]-a[:,0:1,26:29],axis=-1).max(1)*1000
            ax[0].plot(np.arange(1,511)*.002,y,label=f'{n} identical worlds',color=c,ls=ls)
        ax[0].axhline(1,color='#555555',ls=':',label='1 mm reference');ax[0].set(xlabel='Simulated time (s)',ylabel='Maximum difference from world 0 (mm)',ylim=(0,28),title='A. Identical inputs, different world states');ax[0].legend(fontsize=9)
        cpu=load(RUN/'cpu_microperturb.json')['cases'][2:];gpu=load(RUN/'gpu_microperturb.json')['cases'][2:];x=np.arange(6)
        ax[1].plot(x,[r['position_max_m']*1000 for r in cpu],'o-',label='CPU double',color=BLUE)
        ax[1].plot(x,[r['max_position_difference_m']*1000 for r in gpu],'s--',label='GPU float / one world',color=ORANGE)
        ax[1].set(xticks=x,xticklabels=['x -','x +','y -','y +','z -','z +'],yscale='log',xlabel='Object reset offset: nominal +/- 0.1 micrometre',ylabel='Max position difference from unperturbed (mm)',title='B. GPU sensitivity also occurs in one world');ax[1].legend(fontsize=9)
        fig.suptitle('Numerical/contact sensitivity remains unresolved; no shared-load throughput claim',fontsize=12);save(fig,'batch_and_sensitivity')
        fig,ax=plt.subplots(2,2,figsize=(12,7.2),layout='constrained')
        for variant,label,c,ls in [('baseline','Baseline',BLUE,'-'),('grip_balance','Preload candidate',ORANGE,'--')]:
            for i in range(8):
                rows=load(RUN/'gpu'/variant/'A'/f'trial_{i:02d}_metrics.json');t=[r['time_s'] for r in rows]
                for axes,key,mul in [(ax[0,0],'position_m',1000),(ax[0,1],'rotation_rad',180/np.pi)]:axes.plot(t,[r[key]*mul for r in rows],color=c,ls=ls,alpha=.5,lw=1,label=label if i==0 else None)
        ax[0,0].axhline(10,color='#555555',ls=':');ax[0,1].axhline(15,color='#555555',ls=':')
        ax[0,0].set(title='A. Static hold: all eight paired trials',ylabel='Object position error (mm)',ylim=(0,None));ax[0,1].set(title='B. Static hold: all eight paired trials',ylabel='Object rotation error (degrees)',ylim=(0,180))
        pairs=load(RUN/'fixed_evaluation.json')['pairs']['checkpoint_control_ablation']['trials']
        for axes,key,label in [(ax[1,0],'position','Position error (mm)'),(ax[1,1],'rotation','Rotation error (degrees)')]:
            mul=1000 if key=='position' else 1
            for p in pairs:axes.plot([0,1],[p['baseline_'+key]*mul,p['candidate_'+key]*mul],'-o',color='#777777',alpha=.7,ms=4,lw=1)
            means=[np.mean([p[s+'_'+key]*mul for p in pairs]) for s in ['baseline','candidate']];axes.plot([0,1],means,'s--',color=ORANGE,ms=8,lw=2,label='Mean of 8 paired errors')
            axes.set(xticks=[0,1],xticklabels=['Checkpoint\noriginal control','Checkpoint\npreload candidate'],ylabel=label,ylim=(0,None),title='C. Paired position error' if key=='position' else 'D. Paired rotation error');axes.legend(fontsize=8)
        for axes in ax[0]:axes.set_xlabel('Simulated time (s)');axes.legend(fontsize=9)
        fig.suptitle('Static hold: 0/8 for both. Dynamic checkpoint: 0/8 for both.\nBottom: SAME execution interval within each pair; lines identify trials, not independent seeds.',fontsize=12);save(fig,'paired_grasp_diagnostics')
        fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for variant,label,c,ls in [('baseline','Baseline',BLUE,'-'),('grip_balance','Preload candidate',ORANGE,'--')]:
            rows=load(RUN/'gpu'/variant/'A'/'trial_00_contact_aligned_metrics.json');t=[r['time_s'] for r in rows];tc=[r['contact_time_s'] for r in rows]
            for axes,key,mul,ts in [(ax[0,0],'wrist_position_m',1000,t),(ax[0,1],'finger_rmse_rad',1,t),(ax[1,0],'normal_force_sum_n',1,tc),(ax[1,1],'force_weighted_tangential_speed_m_s',1000,tc)]:axes.plot(ts,[r[key]*mul for r in rows],color=c,ls=ls,label=label)
        for axes,title,y in zip(ax.flat,['Wrist position tracking','Finger tracking to original reference','Actual GPU contact forces','Contact tangent speed proxy'],['Error (mm)','RMSE (rad)','Sum of normal forces (N)','Force-weighted speed (mm/s)']):axes.set(title=title,xlabel='Simulated time (s)',ylabel=y,ylim=(0,None));axes.legend(fontsize=8)
        fig.suptitle('Static hold trial 0: small wrist error does not establish stable grasp\nContact timestamps precede returned state by 2 ms; zero speed without loaded contact is not sticking.',fontsize=12);save(fig,'tracking_and_contact')


def videos():
    m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,480,640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.02,-.02,.2];camera.distance=.85;camera.azimuth=110;camera.elevation=-20
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);opt=mujoco.MjvOption();opt.geomgroup[1]=1
    for task in ['A','B','C']:
        if task=='C':paths=[RUN/'gpu_original_graph'/x for x in ['baseline_checkpoint','grip_balance_checkpoint']]
        else:paths=[RUN/'gpu'/x/task for x in ['baseline','grip_balance']]
        data=[np.load(x/'trial_00.npz') for x in paths];rows=[load(x/'trial_00_metrics.json') for x in paths];count=101 if task=='A' else 52
        writer=imageio.get_writer(OUT/f'comparison_{task}.mp4',fps=25,codec='libx264',quality=8)
        for k in range(count):
            canvas=Image.new('RGB',(1280,544),'white');dc=ImageDraw.Draw(canvas)
            title={'A':'A: Independent 2 s static hold | original vs preload candidate','B':'B: Free-object reference control | original vs preload candidate','C':'C: Existing PPO checkpoint, GPU actor | original vs preload candidate'}[task]
            dc.text((16,8),title,fill='black',font=font);dc.text((16,34),'Stored GPU physics | trial 0 | 0.5x playback | both variants fail; 0/8 passing initial states',fill='black',font=font)
            for j in range(2):
                last=len(data[j]['qpos'])-1;index=min(k,last);d.qpos[:]=data[j]['qpos'][index];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=opt)
                panel=Image.fromarray(renderer.render());draw=ImageDraw.Draw(panel);draw.rectangle((0,0,640,82),fill='white')
                draw.text((12,6),('BASELINE' if j==0 else 'PRELOAD CANDIDATE')+f' | state t={index*.02:.2f} s',fill='black',font=font)
                if index:
                    r=rows[j][index-1];error=f'Object error: {r["position_m"]*1000:.1f} mm / {np.rad2deg(r["rotation_rad"]):.1f} deg'
                    draw.text((12,31),error,fill='black',font=font)
                note='STATIC DIAG: 10 mm / 15 deg; separate from dynamic success' if task=='A' else 'STRICT: every control <=30 mm / 30 deg, full 51 steps'
                draw.text((12,57),note,fill='#333333',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14))
                if k>last:draw.rectangle((0,87,640,118),fill='white');draw.text((12,91),'TERMINATED: final state frozen, no later simulation',fill='#9B3300',font=font)
                canvas.paste(panel,(j*640,64))
            writer.append_data(np.asarray(canvas))
            if k in [0,count//2,count-1]:canvas.save(OUT/f'comparison_{task}_{k:03d}.png')
        writer.close()
    renderer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plots-only',action='store_true');args=p.parse_args();plots()
    if not args.plots_only:videos()
    write_json(OUT/'visualization_provenance.json',dict(destination='Provisional research diagnostic report, not journal submission',unit='One development clip; eight paired initial conditions, one saved policy seed',uncertainty='No CI; raw trial lines retained',transforms='No smoothing. GPU batch positions are max norm from world 0. Micro-offset panel uses log y with all six positive values. Dynamic errors use each pair\'s common execution prefix.',videos='Stored physics states only, trial 0 selected by index. Identical camera, 50 Hz states played at 25 fps. A is a separate static task. Early terminated panel explicitly freezes.',source='runs/grasp_repair_20261026: raw NPZ/JSON files and fixed_evaluation.json',citation='Kassis et al. (2026), Scientific Agent Skills. https://doi.org/10.48550/arXiv.2609.00065'))
    print('Saved plots and comparison videos:',OUT)


if __name__=='__main__':main()
