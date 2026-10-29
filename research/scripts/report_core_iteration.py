"""Learning/control evidence, unsmoothed evaluations and explicit process costs."""
import base64,html,csv
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from exploration_common import RUN,read_json,write_json

COL={'P':'#0072B2','M':'#B34E00'}
STYLE={'P':dict(linestyle='-',marker='o'),'M':dict(linestyle='--',marker='s')}
OUT=RUN/'visuals';OUT.mkdir(exist_ok=True)
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor='white');fig.savefig(OUT/(name+'.pdf'),facecolor='white');plt.close(fig)
def figure(name,caption):
    return '<figure><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/(name+'.png')).read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'

def main():
    d=read_json(RUN/'iteration_analysis.json');recipe=read_json(RUN/'frozen_recipe.json');es=d['evaluations'];interpret=read_json(RUN/'interpretation.json') if (RUN/'interpretation.json').exists() else dict(status='IN PROGRESS — no final research conclusion yet')
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.15}):
        for split,n in [('seen8',8),('heldout16',16)]:
            fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
            if not any(r['stage']=='core' and r['split']==split for r in es):plt.close(fig);continue
            matched_steps={0}|{r['steps'] for r in d['pairs'] if r['stage']=='core' and r['split']==split and r['repeat']==0}
            for arm in ['P','M']:
                available=[r for r in es if r['stage']=='core' and r['split']==split and r['arm'] in [arm,'shared_start'] and r['repeat']==0]
                rows=sorted([r for r in available if r['steps'] in matched_steps],key=lambda r:r['steps']);extra=[r for r in available if r['steps'] not in matched_steps];x=np.array([r['steps'] for r in rows])/1000
                for ax,key,title,ylabel in [(axs[0,0],'successes','Full-trajectory strict success',f'Successes / {n}'),(axs[0,1],'restricted_mean_failure_s','First strict failure (censored at 1.02 s)','Mean min(failure, horizon) (s)'),(axs[0,2],'mean_return','Original task return','Undiscounted return'),(axs[1,2],'mean_length','Episode length','Real control steps')]:
                    ax.plot(x,[r[key] for r in rows],color=COL[arm],label=arm,**STYLE[arm]);ax.set(title=title,ylabel=ylabel)
                    if extra:ax.scatter(np.array([r['steps'] for r in extra])/1000,[r[key] for r in extra],marker='^',facecolors='none',edgecolors=COL[arm],s=35,label=arm+' unpaired (secondary)')
                pairs=sorted([r for r in d['pairs'] if r['stage']=='core' and r['split']==split and r['repeat']==0],key=lambda r:r['steps'])
                for ax,key,title,ylabel in [(axs[1,0],'position_mm','Position error / common executed prefix','Mean position error (mm)'),(axs[1,1],'rotation_deg','Orientation error / common executed prefix','Mean orientation error (degrees)')]:
                    ax.plot(np.array([r['steps'] for r in pairs])/1000,[r[arm+'_'+key] for r in pairs],color=COL[arm],label=arm,**STYLE[arm]);ax.set(title=title,ylabel=ylabel)
            axs[0,0].set_ylim(-.3,n+.3);axs[0,1].set_ylim(0,1.06);axs[1,2].set_ylim(0,53)
            xmax=max(r['steps'] for r in es if r['stage']=='core' and r['split']==split)/1000
            for ax in axs.flat:ax.set_xlabel('Retained-policy real controls per arm (thousands)');ax.set_xlim(0,xmax*1.04);ax.legend()
            fig.suptitle('P: pure PPO; M: PPO + frozen W1-mini, four imagined steps\n'+('Eight reused DEVELOPMENT conditions; hollow triangles are unpaired recovery/current evaluations' if split=='seen8' else 'Sixteen HELD-OUT conditions; first replay, evaluated after recipe/core completion')+' · seed 43 · no smoothing',fontsize=11);save(fig,'learning_'+split)
        fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for arm in ['P','M']:
            rows=d['model'].get('core/'+arm,[]);x=np.array([r['steps'] for r in rows])/1000
            for ax,key,title,ylabel in [(axs[0,0],'position_4_mm','Model error on newly visited states','4-step position error (mm)'),(axs[0,1],'reward_bias_4','Predicted minus real reward','4-step reward bias'),(axs[0,2],'reward_mae_4','Prediction accuracy for actual controls','4-step reward absolute error'),(axs[1,0],'input_over_3std_fraction','Frozen input normalization','Fraction of features outside 3 training SD')]:
                y=np.array([r.get(key,np.nan) for r in rows]);ax.plot(x,y,color=COL[arm],alpha=.25,lw=.6);window=16
                if len(y)>=window:ax.plot(x[window-1:],np.convolve(y,np.ones(window)/window,'valid'),color=COL[arm],ls=STYLE[arm]['linestyle'],label=arm)
                ax.set(title=title,ylabel=ylabel)
        m=d['training'].get('core/M',[]);x=np.array([r['steps'] for r in m])/1000
        axs[1,1].plot(x,[r['model_scaled_ratio'] for r in m],color=COL['M']);axs[1,1].axhline(recipe['rho'],color='#555555',ls=':');axs[1,1].set(title='Auxiliary gradient / PPO actor gradient',ylabel='Norm ratio before combined clipping',ylim=(0,max(.15,recipe['rho']*1.5)));axs[1,1].ticklabel_format(axis='y',style='plain',useOffset=False)
        cos=np.array([r['gradient_cosine'] for r in m]);axs[1,2].plot(x,cos,color=COL['M'],alpha=.3,lw=.7)
        if len(cos)>=16:axs[1,2].plot(x[15:],np.convolve(cos,np.ones(16)/16,'valid'),color=COL['M'])
        axs[1,2].axhline(0,color='#555555',lw=.8);axs[1,2].set(title='PPO / model actor gradient agreement',ylabel='Cosine similarity',ylim=(-1,1))
        axs[0,1].axhline(0,color='#555555',lw=.8)
        for ax in axs.flat:ax.set_xlabel('Additional real controls (thousands)')
        for ax in list(axs.flat)[:4]:ax.legend()
        fig.suptitle('Frozen-model diagnostics are evaluated on executed training controls\nRaw points + trailing 16 recorded observations; samples overlap, no independence claim. No held-out fitting.',fontsize=12);save(fig,'model_policy_diagnostics')
        fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for arm in ['P','M']:
            rows=d['training'].get('core/'+arm,[]);x=np.array([r['steps'] for r in rows])/1000
            for ax,key,title in zip(axs.flat,['mean_reward','approx_kl','action_clip_fraction','explained_variance','reward_obj_pos','reward_obj_rot'],['Rollout reward / mixed reset phases','PPO approximate KL','Raw action clipping fraction','Critic explained variance','Object-position reward per step','Object-orientation reward per step']):
                y=np.array([r[key] for r in rows]);ax.plot(x,y,color=COL[arm],alpha=.2,lw=.6)
                if len(y)>=16:ax.plot(x[15:],np.convolve(y,np.ones(16)/16,'valid'),color=COL[arm],ls=STYLE[arm]['linestyle'],label=arm)
                ax.set(title=title,xlabel='Additional real controls (thousands)');ax.legend()
        for branch in d.get('aborted_branches',[]):
            rows=branch['training'];x=np.array([r['steps'] for r in rows])/1000
            for ax,key in zip(axs.flat,['mean_reward','approx_kl','action_clip_fraction','explained_variance','reward_obj_pos','reward_obj_rot']):
                ax.plot(x,[r[key] for r in rows],color='#666666',ls=':',lw=1,label='SIGTERM-aborted branch');ax.legend(fontsize=7)
        fig.suptitle('PPO training diagnostics: raw + trailing 16 recorded observations\nReward growth alone is not strict task success; full task evaluation appears separately.',fontsize=12);save(fig,'ppo_training_diagnostics')
        fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        for arm in ['P','M']:
            rows=sorted([r for r in es if r['stage']=='confirmation' and r['split']=='seen8' and r['arm']==arm],key=lambda r:r['steps']);x=np.array([r['steps'] for r in rows])/1000
            for ax,key,title in zip(axs,['successes','restricted_mean_failure_s','mean_return'],['Strict success / 8 seen conditions','Restricted first-failure time (s)','Original task return']):ax.plot(x,[r[key] for r in rows],color=COL[arm],label=arm,**STYLE[arm]);ax.set(title=title,xlabel='Additional controls (thousands)');ax.legend()
        axs[0].set_ylim(-.3,8.3);axs[1].set_ylim(0,1.06);fig.suptitle('Separate confirmation run; shorter matched budget, never merged with core seed',fontsize=12);save(fig,'confirmation_learning')
        endpoints=[r for r in es if r['stage']=='core' and r['split']=='heldout16' and r['arm'] in ['P','M']]
        if endpoints:
            final=max(r['steps'] for r in endpoints);endpoints=[r for r in endpoints if r['steps']==final]
            fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
            for arm in ['P','M']:
                rows=sorted([r for r in endpoints if r['arm']==arm],key=lambda r:r['repeat'])
                for ax,key,title in zip(axs[0],['successes','restricted_mean_failure_s','mean_return'],['Same endpoint / repeated replay','Restricted first failure (s)','Original task return']):
                    ax.plot([r['repeat']+1 for r in rows],[r[key] for r in rows],color=COL[arm],label=arm,**STYLE[arm]);ax.set(title=title,xlabel='Replay index (same 16 conditions)');ax.legend()
                first=rows[0];x=np.arange(6)+(-.17 if arm=='P' else .17)
                axs[1,0].bar(x,first['component_returns'],width=.34,color=COL[arm],label=arm,hatch='' if arm=='P' else '//')
                for ax,key,title in [(axs[1,1],'hand_contact_fraction','Hand contact / each reserved condition'),(axs[1,2],'floor_contact_fraction','Floor support / each reserved condition')]:
                    ax.plot([t['trial'] for t in first['trials']],[t[key] for t in first['trials']],color=COL[arm],label=arm,**STYLE[arm]);ax.set(title=title,xlabel='Initial-condition ID',ylabel='Fraction of executed controls',ylim=(-.03,1.03));ax.legend()
            axs[0,0].set(ylabel='Strict successes / 16',ylim=(-.3,16.3));axs[0,1].set_ylim(0,1.06)
            axs[1,0].set_xticks(np.arange(6),['Object\npos','Object\nrot','Finger','Wrist','Action','Change']);axs[1,0].axhline(0,color='#555555',lw=.8);axs[1,0].set(title='Reward decomposition / first replay',ylabel='Mean full-episode component return');axs[1,0].legend()
            fig.suptitle('Final core checkpoint: all reserved conditions, no best-replay selection\nContact timestamps lag state by one 2-ms substep; force threshold 0.01 N. Replays do not increase independent n.',fontsize=12);save(fig,'endpoint_replays_components')
    rows=[]
    for r in es:
        rows.append([r['stage'],r['split'],r['arm'],r['steps'],r['repeat'],f"{r['successes']}/{r['n']}",f"{r['restricted_mean_failure_s']:.3f}",f"{r['mean_return']:.3f}",f"{r['mean_length']:.1f}",str(r['failure_types'])])
    body='<h1>SuperDex: real PPO versus model-assisted PPO</h1><p>Full original wooden-spoon task; unchanged success thresholds: every control ≤30 mm and ≤30°, all 51 controls, no physical termination. Original eight conditions are development observations. Sixteen held-out conditions are a narrow perturbation family, not new tasks.</p>'
    body+='<h2>Research interpretation</h2><pre>'+html.escape(__import__('json').dumps(interpret,ensure_ascii=False,indent=2))+'</pre>'
    body+='<h2>Budget and method</h2><p>Total charged GPU process time: '+f"{d['cost']['total_gpu_process_seconds']/3600:.3f} / 6.000 hours"+'. Import, startup, diagnostics and evaluation are included; CPU-only preparation/plotting is separate. Only physical GPU 0, one physical environment.</p>'
    body+='<p>Both arms: same source policy/Adam, lr 1e-4, 50% phase-zero resets, unchanged original reward/physics/action limits. M adds four-step frozen-W1 policy loss on 16 rollout starts once per 128-control PPO update; model/actor gradient norm target '+str(recipe['rho'])+'. Model weights are never fitted on evaluation data.</p>'
    body+='<p>Development candidates: rho 0.1 and 0.3, 16,384 controls each; P control also 16,384. Initial common learning-rate change and model strength selection are disclosed, not independently attributed. The two old initial model variants were not mixed.</p>'
    for name,caption in [('learning_seen8','All core checkpoints on eight reused development conditions. First-failure time treats success as right-censored at 1.02 seconds; no smoothing.'),('learning_heldout16','All core checkpoints on sixteen reserved conditions, first replay. These conditions did not select or stop training.'),('model_policy_diagnostics','Prediction error, reward bias, distribution shift and gradient balance on newly visited training states.'),('ppo_training_diagnostics','PPO update stability and reward components. Smoothed curves disclose their trailing window.'),('confirmation_learning','Separate confirmation seed and budget. Shorter confirmation is not pooled with the primary learning curve.'),('endpoint_replays_components','Final fixed-policy replays, reward decomposition and actual hand/floor contact. Every reserved initial condition is retained.')]:
        if (OUT/(name+'.png')).exists():body+=figure(name,caption)
    body+='<h2>Every fixed evaluation</h2>'+table(['Stage','Split','Arm','Steps','Replay','Strict','First failure, restricted mean s','Return','Length','Failure types'],rows)
    video_folder=RUN/'videos_delivery'
    if video_folder.exists():
        body+='<h2>Preselected physical policy replays</h2><p>Initial conditions 0 and 1 were fixed before held-out evaluation. Both panels show stored physical policy trajectories, with a shared camera and 0.5x playback. No new physics is generated for rendering.</p>'
        for initial in [0,1]:
            picture=video_folder/f'initial_{initial:02d}_frame_26.png'
            body+='<figure><img alt="Preselected physical P/M comparison, initial '+str(initial)+', t=0.52 seconds" src="data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode()+'"><figcaption>Initial '+str(initial)+'; midpoint snapshot. <a href="videos_delivery/heldout_initial_'+str(initial).zfill(2)+'.mp4">Open all 52 logged frames</a>.</figcaption></figure>'
    body+='<h2>GPU process ledger</h2><p>Main curves use the controls that led to retained policy/optimizer checkpoints. Aborted branches are additional real sampling expenditure, shown as bounds where a signal arrived between durable log writes. They are not silently included in a matched-policy x coordinate.</p>'+table(['Process','Category','Seconds','Retained training controls','Eval controls','Auxiliary predictions','Diagnostic predictions','PPO updates','Aborted controls lower','Aborted controls upper'],[[r[k] for k in ['name','category','wall_seconds','training_interactions','evaluation_interactions','auxiliary_model_predictions','diagnostic_model_predictions','ppo_optimizer_updates','aborted_interactions_lower_bound','aborted_interactions_upper_bound']] for r in d['cost']['processes']])
    body+='<h2>Limits and reproducibility</h2><p>Contact solver sensitivity remains. Endpoint repeats share initial conditions and are not independent tests; training seeds are reported separately. Reward and strict-success statistics are not interchangeable. Neither task completion nor prediction improvement certifies formal D0/G1/G2/G3.</p><p>Code, exact source checkpoint identifiers, frozen recipe, per-step evaluation logs, sampled training episodes, all checkpoints and negative development candidates are retained in the iteration directory. The backup remote-control task was selected from source geometry before policy trials and is not used to replace spoon outcomes.</p><p>Visualization workflow: Kassis et al. (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>. General research figures; no journal-specific compliance claim.</p>'
    (RUN/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>SuperDex core learning iteration</title><style>body{font:16px/1.55 system-ui;max-width:1280px;margin:35px auto;padding:0 20px;color:#20252b}img{max-width:100%}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto}td,th{border:1px solid #ccc;padding:7px}th{background:#edf2f7}figure{margin:30px 0}figcaption{font-size:14px;color:#444}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px}h2{margin-top:36px}</style>'+body+'</html>',encoding='utf-8')
    manifest=[]
    for p in sorted(OUT.glob('*.png')):
        im=Image.open(p);manifest.append(dict(path=p.name,pixels=im.size,mode=im.mode,dpi=im.info.get('dpi')))
    write_json(OUT/'provenance.json',dict(source='iteration_analysis.json; raw logs retained',smoothing='Evaluations: none. Training/model diagnostics: raw plus trailing 16 recorded observations.',uncertainty='No statistical significance claims. Conditions and repeated replays are not training-seed replication.',missing='Unavailable curves omitted and tables retain all available evaluations.',metadata=manifest))
    print('Saved report and',len(manifest),'PNG/PDF figures')

if __name__=='__main__':main()
