"""Numerical findings for the research judgment; no automatic claim of superiority."""
import numpy as np
from exploration_common import RUN,ROOT,read_json,write_json

def main():
    a=read_json(RUN/'iteration_analysis.json');findings=dict(stages={},cost={},diagnostics={},limitations=[])
    for stage in ['core','confirmation']:
        p=RUN/(stage+'_completion.json')
        if not p.exists():continue
        completion=read_json(p);end=completion['steps_per_arm'];result=dict(steps_per_arm=end,seed=completion['seed'],splits={})
        for split in ['seen8','heldout16']:
            rows=[r for r in a['evaluations'] if r['stage']==stage and r['split']==split and r['steps']==end]
            if not rows:continue
            pair=next(r for r in a['pairs'] if r['stage']==stage and r['split']==split and r['steps']==end and r['repeat']==0)
            result['splits'][split]=dict(common_prefix={k:v for k,v in pair.items() if k!='per_initial'})
            arms={}
            for arm in ['P','M']:
                rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['repeat']);first=rr[0]
                arms[arm]=dict(primary={k:v for k,v in first.items() if k not in ['folder','trials']},replay_success_counts=[r['successes'] for r in rr],replay_returns=[r['mean_return'] for r in rr],replay_restricted_failure_s=[r['restricted_mean_failure_s'] for r in rr],mean_hand_contact=float(np.mean([t['hand_contact_fraction'] for t in first['trials']])),mean_floor_contact=float(np.mean([t['floor_contact_fraction'] for t in first['trials']])))
            firsts={arm:next(r for r in rows if r['arm']==arm and r['repeat']==0) for arm in ['P','M']};truth=np.array([[t['success'] for t in firsts[arm]['trials']] for arm in ['P','M']],dtype=bool)
            result['splits'][split].update(arms=arms,paired_success_discordance=dict(both=int((truth[0]&truth[1]).sum()),P_only=int((truth[0]&~truth[1]).sum()),M_only=int((~truth[0]&truth[1]).sum()),neither=int((~truth[0]&~truth[1]).sum())))
        findings['stages'][stage]=result
        for arm in ['P','M']:
            costs=[r for r in a['cost']['processes'] if r['name'].startswith(stage+'/'+arm+'/')]
            findings['cost'][stage+'/'+arm]={k:sum(r[k] for r in costs) for k in ['wall_seconds','training_interactions','evaluation_interactions','auxiliary_model_predictions','diagnostic_model_predictions','ppo_optimizer_updates','aborted_interactions_lower_bound','aborted_interactions_upper_bound','aborted_auxiliary_predictions_lower_bound','aborted_auxiliary_predictions_upper_bound','aborted_optimizer_updates_lower_bound','aborted_optimizer_updates_upper_bound']}
            rows=a['model'].get(stage+'/'+arm,[]);t=a['training'].get(stage+'/'+arm,[])
            if rows:
                n=min(16,len(rows));keys=['position_4_mm','rotation_4_deg','reward_bias_4','reward_mae_4','input_over_3std_fraction']
                findings['diagnostics'][stage+'/'+arm]=dict(first_16={k:float(np.mean([r[k] for r in rows[:n] if k in r])) for k in keys},last_16={k:float(np.mean([r[k] for r in rows[-n:] if k in r])) for k in keys})
            if arm=='M' and t:
                findings['diagnostics'][stage+'/'+arm].update(gradient_cosine_mean=float(np.mean([r['gradient_cosine'] for r in t])),gradient_cosine_negative_fraction=float(np.mean([r['gradient_cosine']<0 for r in t])),model_scaled_ratio_max=max(r['model_scaled_ratio'] for r in t))
    processes=a['cost']['processes'];dev={r['name'].split('/')[-1]:r for r in processes if r['name'] in ['development/P_attempt2','development/rho10']}
    if 'P_attempt2' in dev and 'rho10' in dev:findings['uncontended_development_M_over_P_wall_ratio']=dev['rho10']['wall_seconds']/dev['P_attempt2']['wall_seconds']
    findings['total_gpu_process_seconds']=a['cost']['total_gpu_process_seconds'];findings['cap_seconds']=21600
    retained=sum(r['training_interactions']+r['evaluation_interactions'] for r in processes);findings['total_physical_controls_excluding_initialization_bounds']=[retained+sum(r['aborted_interactions_lower_bound'] for r in processes),retained+sum(r['aborted_interactions_upper_bound'] for r in processes)]
    if (RUN/'external_sigterm_record.json').exists():
        findings['external_interruption']=read_json(RUN/'external_sigterm_record.json');findings['limitations'].append('A SIGTERM-aborted M branch lost unsaved optimizer updates. Main curves compare equal real controls on retained policy lineages; all extra aborted sampling and wall time are disclosed separately, not silently called matched total expenditure.')
    if (RUN/'resource_adaptation.json').exists():findings['resource_adaptation']=read_json(RUN/'resource_adaptation.json');findings['limitations'].append('External GPU 0 training appeared after the first P block. Raw core wall-time differences include unequal contention exposure and are not an isolated algorithm-cost benchmark.')
    if (RUN/'core_administrative_resets.json').exists():findings['administrative_resets']=read_json(RUN/'core_administrative_resets.json')
    findings['limitations']+=['Sixteen held-out conditions perturb the same task; no object/task generalization claim.','Endpoint replays are solver-repeatability observations, not independent training seeds.','Confirmation has a shorter matched budget; do not call it full-length replication.','Initialization and graph compilation are excluded from real learning-control counts but included in charged process time.','P also executes the same read-only world-model diagnostics; extra auxiliary computations are listed separately.']
    write_json(RUN/'research_findings.json',findings);print('Saved numerical research findings; human/model interpretation remains explicit.')

if __name__=='__main__':main()
