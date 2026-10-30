"""Learning amount on unchanged data/architecture, with raw data and limitations."""
import csv
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FuncFormatter
from exploration_common import RUN,read_json,write_json

OUT=RUN/'visuals';OUT.mkdir(exist_ok=True)
COLORS={'uniform_rsi':'#0072B2','start50':'#B34E00','W0':'#0072B2','W1':'#B34E00','persistence':'#555555','constant_velocity':'#007856','ballistic':'#8B5AA0'}
def save(fig,name):fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(name+'.pdf'),facecolor='white');plt.close(fig)
def rows(path):return list(csv.DictReader(path.open()))


def main():
    p=read_json(RUN/'ppo_analysis.json');wm=read_json(RUN/'wm/trend_analysis.json');curve=p['curve'];original=next(x for x in curve if x['arm']=='original')
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        keys=[('mean_return','Fixed evaluation return','Return'),('successes','Strict full-trajectory successes','Successes / 8'),('first_failure_median_s','First failure: median of FAILED trials','Time (s)'),('mean_length','Executed episode length','50 Hz controls')]
        for arm,ls,m in [('uniform_rsi','-','o'),('start50','--','s')]:
            rs=[original]+[r for r in curve if r['arm']==arm];x=[r['total_training_steps'] for r in rs]
            for ax,(key,title,unit) in zip(axes.flat,keys):ax.plot(x,[np.nan if r[key] is None else r[key] for r in rs],color=COLORS[arm],ls=ls,marker=m,label=arm);ax.set(title=title,ylabel=unit)
            for ax,key,title in [(axes[1,1],'position_mean_mm','Position error over paired common interval'),(axes[1,2],'rotation_mean_deg','Orientation error over paired common interval')]:
                mx=[];my=[]
                for pair in p['matched_checkpoints']:
                    mx.append(pair['total_training_steps']);my.append(next(r[key] for r in pair['comparison']['means'] if r['arm']==arm))
                ax.plot(mx,my,color=COLORS[arm],ls=ls,marker=m,label=arm);ax.set(title=title,ylabel='mm' if 'position' in key else 'degrees')
        for ax in axes.flat:ax.axvline(61440,color='#666666',ls=':',lw=1);ax.set_xlabel('Cumulative training controls');ax.legend(fontsize=8);ax.set_ylim(bottom=0)
        axes[0,1].set_ylim(-.15,8.2);axes[0,1].set_yticks([0,2,4,6,8]);axes[1,0].set_ylim(0,54)
        fig.suptitle('Single seed, unchanged eight initial states / success rule; one common physical graph\nDotted line: this continuation begins. Successes are right-censored in failure-time analysis, not invented failures.',fontsize=12);save(fig,'ppo_complete_curve')
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');keys=[('mean_reward','Mean training reward'),('approx_kl','Approximate KL'),('ppo_clip_fraction','PPO clip fraction'),('advantage_std','GAE advantage SD'),('action_clip_fraction','Raw action clipping fraction'),('explained_variance_before_update','Value explained variance')]
        for arm,ls in [('uniform_rsi','-'),('start50','--')]:
            rr=rows(RUN/arm/'training.csv');x=np.array([float(r['total_steps_with_original']) for r in rr])
            for ax,(key,title) in zip(axes.flat,keys):
                y=np.array([float(r[key]) for r in rr]);ax.plot(x,y,color=COLORS[arm],alpha=.15,lw=.6);ax.plot(x[19:],np.convolve(y,np.ones(20)/20,'valid'),color=COLORS[arm],ls=ls,label=arm);ax.set(title=title,xlabel='Cumulative training controls');ax.legend(fontsize=8)
        for ax in axes.flat:ax.xaxis.set_major_locator(MaxNLocator(5));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v/1000:g}k'))
        fig.suptitle('PPO continuation diagnostics: faint raw values + trailing 20-update mean\nDifferent reset distributions imply different training-state mixtures; use fixed evaluation for task performance.',fontsize=12);save(fig,'ppo_diagnostics')
        fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained');rr=rows(RUN/'wm/training.csv')
        for name,ls,m in [('W0','-','o'),('W1','--','s')]:
            training=[r for r in rr if r['model']==name and r['training_loss']];x=np.array([int(r['update']) for r in training]);y=np.array([float(r['training_loss']) for r in training]);axes[0,0].plot(x,y,color=COLORS[name],alpha=.15,lw=.6);axes[0,0].plot(x[39:],np.convolve(y,np.ones(40)/40,'valid'),color=COLORS[name],ls=ls,label=name)
            loss=sorted([r for r in wm['validation_loss'] if r['model']==name],key=lambda r:r['update']);axes[0,1].plot([r['update'] for r in loss],[r['normalized_8step_validation_loss'] for r in loss],color=COLORS[name],ls=ls,marker=m,label=name)
            for ax,h,subset,title in [(axes[0,2],8,'natural','8-step position / natural mixture'),(axes[1,0],32,'natural','32-step position / natural mixture'),(axes[1,1],8,'hand','8-step position / hand involved'),(axes[1,2],32,'hand','32-step position / hand involved')]:
                rs=sorted([r for r in wm['metrics'] if r['model']==name and r['horizon']==h and r['subset']==subset],key=lambda r:r['update']);ax.plot([r['update'] for r in rs],[r['position_mean_mm'] for r in rs],color=COLORS[name],ls=ls,marker=m,label=name);ax.set(title=title,ylabel='Weighted endpoint mean (mm)')
        axes[0,0].set(title='Training objective',ylabel='Normalized weighted MSE',yscale='log');axes[0,1].set(title='Held-out 8-step objective',ylabel='Equal-regime normalized weighted MSE',yscale='log')
        for ax in axes.flat:ax.axvline(590,color='#666666',ls=':',lw=1);ax.set_xlabel('Paired update');ax.legend(fontsize=8)
        fig.suptitle('Frozen data / split / seeds / minibatches, same W0/W1-mini architecture\nDotted line: previous run stopped at 590. Training: raw + trailing 40-update mean. Validation: unsmoothed.',fontsize=12);save(fig,'wm_learning_curves')
        fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
        for row,h in enumerate([8,32]):
            for col,subset in enumerate(['no_contact','sustained_contact','contact_switch']):
                ax=axes[row,col];supported=next(r['windows'] for r in wm['metrics'] if r['horizon']==h and r['subset']==subset);ax.set_title(f'{h} steps / {subset.replace("_"," ")} / N={supported}')
                if not supported:ax.text(.5,.5,'N/A: no held-out windows',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
                for name,ls,m in [('W0','-','o'),('W1','--','s')]:
                    rs=sorted([r for r in wm['metrics'] if r['model']==name and r['horizon']==h and r['subset']==subset],key=lambda r:r['update']);ax.plot([r['update'] for r in rs],[r['position_p50_mm'] for r in rs],color=COLORS[name],ls=ls,marker=m,label=name)
                for name,label in [('persistence','Hold'),('constant_velocity','Constant v'),('ballistic','Gravity')]:
                    r=next(r for r in wm['metrics'] if r['model']==name and r['horizon']==h and r['subset']==subset);ax.axhline(r['position_p50_mm'],color=COLORS[name],ls=':',label=label)
                ax.set(xlabel='Paired update',ylabel='Endpoint position P50 (mm)',yscale='log');ax.legend(fontsize=8)
        fig.suptitle('Contact-regime learning curves and unchanged simple baselines\nLogarithmic y axes; ranges differ. N counts overlapping held-out windows, not independent trials.',fontsize=12);save(fig,'wm_contact_regime_curves')
    write_json(OUT/'provenance.json',dict(source=['ppo_analysis.json','wm/trend_analysis.json','uniform_rsi/training.csv','start50/training.csv','wm/training.csv'],raw_retained=True,smoothing='PPO training: trailing20 updates; WM training: trailing40 updates; all evaluation unsmoothed',uncertainty='One training seed and eight reused initial conditions; no confidence intervals or independent-task claims',missing='32-step no-contact has zero windows, N/A',citation='Kassis et al. (2026), Scientific Agent Skills. https://doi.org/10.48550/arXiv.2609.00065'))
    print('Four learning-curve PNG/PDF pairs saved.')


if __name__=='__main__':main()
