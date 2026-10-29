"""Choose the reserved follow-up from development evidence, never held-out results."""
import numpy as np
from exploration_common import RUN,read_json,write_json

def main():
    assert not (RUN/'confirmation_plan.json').exists()
    core=read_json(RUN/'core_completion.json');progress=read_json(RUN/'core_progress.json')['checkpoints'];recipe=read_json(RUN/'frozen_recipe.json')
    development=read_json(RUN/'development/rho10/model_diagnostics.json')
    baseline={k:float(np.mean([r[k] for r in development if k in r])) for k in ['position_4_mm','reward_mae_4']}
    records=[];first=progress[0]['summaries']['M']
    for i in range(max(0,len(progress)-2),len(progress)):
        rows=[r for folder in sorted((RUN/'core/M').glob(f'block_{i:02d}*')) for r in read_json(folder/'model_diagnostics.json')];metrics={k:float(np.mean([r[k] for r in rows if k in r])) for k in baseline};s=progress[i]['summaries']['M']
        errors_double=all(metrics[k]>2*baseline[k] for k in baseline)
        # A zero-success first checkpoint must not make degradation impossible.
        # Require non-improving success plus material deterioration of continuous
        # real metrics; this condition is not a gate for the ongoing PPO run.
        degraded=bool(s['successes']<=first['successes'] and s['mean_return']<.9*first['mean_return'] and (s['median_first_failure_s'] or 1.02)<.8*(first['median_first_failure_s'] or 1.02))
        records.append(dict(block=i,means=metrics,ratios={k:metrics[k]/baseline[k] for k in baseline},both_errors_over_twice_development=errors_double,real_degraded_vs_first_core_checkpoint=degraded))
    trigger=len(records)==2 and all(r['both_errors_over_twice_development'] and r['real_degraded_vs_first_core_checkpoint'] for r in records)
    decision=dict(development_model_baseline=baseline,last_two_blocks=records,refresh_trigger=trigger,heldout_accessed=False,scope='Mean model diagnostics over each complete block; real degradation requires no higher success, >10% lower return and >20% earlier failed-trial median than the first core checkpoint. This conservative trigger does not prove absence of distribution shift.')
    write_json(RUN/'confirmation_decision.json',decision)
    if trigger:
        print('Evidence supports reviewing a separately labeled one-refresh continuation; no automatic fitting or held-out access.');return
    steps=read_json(RUN/'resource_adaptation.json')['confirmation_target_per_arm'] if (RUN/'resource_adaptation.json').exists() else 131072
    if (RUN/'resource_recovery.json').exists():steps=read_json(RUN/'resource_recovery.json')['restored_confirmation_target_per_arm']
    write_json(RUN/'confirmation_plan.json',dict(kind='second_seed_replication',seed=59,steps_per_arm=steps,rho=recipe['rho'],source_policy=recipe['source_policy'],source_model=recipe['source_model'],heldout_accessed_for_selection=False,reason='Predeclared default; joint model-error and real-performance drift trigger absent. Repeat the same frozen P/M recipe from the original source with seed59.'))
    print('Frozen confirmation: seed59,',steps,'additional controls per arm; no model refresh.')

if __name__=='__main__':main()
