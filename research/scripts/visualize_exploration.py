"""Standalone learning curves and prediction figures; raw data retained."""
import json,csv
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exploration_common import RUN,read_json,write_json

OUT=RUN/'visuals';OUT.mkdir(exist_ok=True);COLORS={'uniform_rsi':'#0072B2','start50':'#B34E00','W0':'#0072B2','W1':'#B34E00','persistence':'#555555','constant_velocity':'#007856','ballistic':'#8B5AA0'}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(name+'.pdf'),facecolor='white');plt.close(fig)
def csvrows(path):return list(csv.DictReader(path.open()))


def main():
    ppo=read_json(RUN/'ppo_comparison.json');curve=ppo['learning_curve'];base=next(x for x in curve if x['arm']=='original');wm=read_json(RUN/'wm/evaluation.json');cfg=read_json(RUN/'wm/config.json')
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for arm,label,ls,marker in [('uniform_rsi','Uniform RSI','-','o'),('start50','50% start + 50% uniform RSI','--','s')]:
            rows=[x for x in curve if x['arm']==arm];x=np.array([20480]+[r['total_steps'] for r in rows]);c=COLORS[arm]
            for ax,key,title,ylabel in [(axes[0,0],'mean_return','Fixed evaluation return','Return'),(axes[0,1],'first_failure_median_s','First strict failure (median of 8)','Time (s)'),(axes[1,0],'successes','Full-trajectory strict successes','Successes / 8'),(axes[1,1],'mean_length','Actual executed episode length','50 Hz control steps')]:
                ax.plot(x,[base[key]]+[r[key] for r in rows],color=c,ls=ls,marker=marker,label=label);ax.set(title=title,xlabel='Total training steps including original 20,480',ylabel=ylabel);ax.legend(fontsize=8)
        # Set limits only AFTER both arms; setting ylim during the first arm
        # disables autoscaling and would silently clip the second arm's gains.
        for a,key in [(axes[0,0],'mean_return'),(axes[0,1],'first_failure_median_s')]:
            a.set_ylim(0,1.12*max(r[key] for r in curve))
        axes[1,0].set_ylim(-.15,8.2);axes[1,0].set_yticks(range(0,9,2));axes[1,1].set_ylim(0,54)
        fig.suptitle('One continuation seed, one wooden-spoon clip, the SAME eight initial states\nAll checkpoints evaluated with one common physical graph; success thresholds unchanged.',fontsize=12);save(fig,'ppo_fixed_learning_curve')
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        keys=[('mean_reward','Mean step reward'),('approx_kl','Approximate KL'),('ppo_clip_fraction','PPO clip fraction'),('advantage_std','GAE advantage SD'),('explained_variance_before_update','Value explained variance'),('contact_fraction','Loaded hand-contact fraction')]
        for arm,ls in [('uniform_rsi','-'),('start50','--')]:
            rows=csvrows(RUN/arm/'training.csv');x=np.array([float(r['additional_steps']) for r in rows])
            for ax,(key,title) in zip(axes.flat,keys):
                y=np.array([float(r[key]) for r in rows]);smooth=np.convolve(y,np.ones(10)/10,mode='valid');ax.plot(x,y,color=COLORS[arm],alpha=.18,lw=.7);ax.plot(x[9:],smooth,color=COLORS[arm],ls=ls,label=arm);ax.set(title=title,xlabel='Additional training steps');ax.legend(fontsize=8)
        fig.suptitle('Training diagnostics: faint raw updates + trailing mean of 10 updates\nTraining uses different reset distributions; these curves are not fixed-evaluation success.',fontsize=12);save(fig,'ppo_training_diagnostics')
        fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');rows=csvrows(RUN/'wm/training.csv');history=read_json(RUN/'wm/validation_history.json')
        for name,ls in [('W0','-'),('W1','--')]:
            r=[x for x in rows if x['model']==name and x['training_loss']];x=np.array([float(y['update']) for y in r]);y=np.array([float(z['training_loss']) for z in r]);ax[0].plot(x,y,color=COLORS[name],alpha=.2,lw=.7);ax[0].plot(x[19:],np.convolve(y,np.ones(20)/20,'valid'),color=COLORS[name],ls=ls,label=name+'-mini')
            h=[x for x in history if x['model']==name];vx=[x['update'] for x in h];vy=[np.mean([v['position_mean_mm'] for v in r['stats'].values() if v['windows']]) for r in h]
            if vx[-1]!=cfg['paired_updates']:vx.append(cfg['paired_updates']);vy.append(np.mean([v['position_mean_mm'] for v in wm[name]['8'].values() if v['windows']]))
            ax[1].plot(vx,vy,color=COLORS[name],ls=ls,marker='o' if name=='W0' else 's',label=name+'-mini')
        ax[0].set(title='8-step autoregressive fitting',xlabel='Paired optimizer update',ylabel='Normalized weighted MSE',yscale='log');ax[1].set(title='Held-out 8-step position error',xlabel='Paired optimizer update',ylabel='Unweighted mean of 3 regime means (mm)',ylim=(0,None))
        for a in ax:a.legend()
        fig.suptitle('W0/W1-mini: same episodes, sampled batches and updates\nLoss: faint raw values + trailing mean of 20; validation has no smoothing.',fontsize=12);save(fig,'world_model_learning_curve')
        names=['W0','W1','persistence','constant_velocity','ballistic'];labels=['W0','W1','Hold','Const. v','Gravity'];fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
        for row,h in enumerate(['8','32']):
            for col,regime in enumerate(['no_contact','sustained_contact','contact_switch']):
                a=axes[row,col];n=wm['W0'][h][regime]['windows'];a.set_title(f'{h} steps / {regime.replace("_"," ")}\n{n} held-out windows')
                if not n:a.text(.5,.5,'N/A: no supported windows',ha='center',va='center',transform=a.transAxes);a.set_axis_off();continue
                med=[wm[name][h][regime]['position_median_mm'] for name in names];p90=[wm[name][h][regime]['position_p90_mm'] for name in names];a.bar(np.arange(5),med,color=[COLORS[x] for x in names],alpha=.8);a.scatter(np.arange(5),p90,marker='_',s=140,color='black',label='90th percentile');a.set(xticks=np.arange(5),xticklabels=labels,ylabel='Position error (mm)',ylim=(0,max(p90)*1.12));a.tick_params(axis='x',labelrotation=25);a.legend(fontsize=8)
        fig.suptitle('Endpoint prediction error: bar = median, black mark = P90 (NOT a confidence interval)\nAxes have explicitly different ranges. Persistent contact is often floor contact; only 10 no-contact 8-step windows.',fontsize=12);save(fig,'world_model_regime_errors')
        da=read_json(RUN/'wm/prediction_analysis.json');fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');x=np.arange(2);bottom=np.zeros(2)
        for mode,label,color,hatch in [(0,'No loaded contact','#777777','//'),(1,'Hand only','#0072B2',''),(2,'Floor only','#B34E00','..'),(3,'Hand + floor','#007856','xx')]:
            y=np.array([r['fractions'][str(mode)] for r in da['data_distribution']])*100;axes[0].bar(x,y,bottom=bottom,label=label,color=color,hatch=hatch);bottom+=y
        axes[0].set(xticks=x,xticklabels=['Training','Validation'],ylabel='Transitions (%)',ylim=(0,100),title='Data composition');axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
        for name,ls,marker in [('W0','-','o'),('W1','--','s')]:
            d=[r for r in da['weighted_metrics'] if r['model']==name and r['subset']=='hand_contact_in_window'];axes[1].plot([r['horizon'] for r in d],[r['position_median_mm'] for r in d],color=COLORS[name],ls=ls,marker=marker,label=name+'-mini')
        axes[1].set(xticks=[8,32],xlabel='Prediction horizon (controls)',ylabel='Weighted position median (mm)',ylim=(0,None),title='Windows involving loaded hand contact');axes[1].legend();fig.suptitle('Floor-dominated data and the hand-involving subset are reported separately',fontsize=12);save(fig,'data_composition_and_hand_subset')
    write_json(OUT/'provenance.json',dict(audience='Research exploration report, not publication submission',units='mm, degrees, seconds, 50Hz controls; GPU-process wall seconds in compute ledger',uncertainty='No confidence intervals. One PPO continuation seed and one WM seed; eight fixed initial conditions are not independent tasks. Prediction windows overlap only within validation episodes.',smoothing='Training figures explicitly retain faint raw values and trailing 10-update PPO / 20-update WM means. Evaluation and quantile figures are unsmoothed.',missing='32-step no-contact group has no windows and is N/A, never zero.',source='ppo_comparison.json, wm/evaluation.json, wm/prediction_analysis.json, training.csv',citation='Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills. https://doi.org/10.48550/arXiv.2609.00065'))
    print('Saved five standalone PNG/PDF figure pairs.')


if __name__=='__main__':main()
