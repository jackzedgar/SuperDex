"""Freeze a resource-based confirmation target before any reserved evaluation."""
import math
from exploration_common import RUN,read_json,write_json

def main():
    assert (RUN/'core_completion.json').exists()
    assert not (RUN/'core_heldout').exists()
    assert not (RUN/'confirmation_budget_schedule.json').exists()
    plan=read_json(RUN/'confirmation_plan.json')
    assert plan['kind']=='second_seed_replication'
    cfg=read_json(RUN/'budget.json');rows=read_json(RUN/'process_ledger.json')['processes']
    used={cat:sum(r['wall_seconds'] for r in rows if r['category']==cat) for cat in cfg['allocation_seconds']}
    remaining=21600-sum(used.values());rates={};evidence={}
    for arm in ['P','M']:
        candidates=[];all_completed=[]
        for row in rows:
            if not row['name'].startswith('core/'+arm+'/') or row['returncode']!=0:continue
            p=RUN/row['name']/'summary.json'
            if p.exists() and read_json(p)['training_interactions']>0:
                all_completed.append(row)
                if read_json(p)['training_interactions']==65536:candidates.append(row)
        row=(candidates or all_completed)[-1];count=read_json(RUN/row['name']/'summary.json')['training_interactions'];rates[arm]=row['wall_seconds']/count;evidence[arm]=row['name']
    reserves=dict(core_heldout=600,confirmation_heldout=400,render=120,exception=100)
    # Full-process rates already include initialization and each block's seen-8
    # evaluation. Allow 8% load variation and another 100 s for a partial block.
    candidates=[]
    for target in range(65536,131072+1,16384):
        estimated=1.08*sum(rates.values())*target+100
        candidates.append(dict(steps_per_arm=target,estimated_training_seconds=estimated,
                               fits=estimated+sum(reserves.values())<=remaining))
    feasible=[c for c in candidates if c['fits']]
    assert feasible,'Preserve evaluation reserve; no unbounded confirmation launch.'
    selected=feasible[-1]
    allocation=dict(development=math.ceil(used['development'])+1,
                    core=math.ceil(used['core'])+reserves['core_heldout']+50)
    allocation['confirmation']=21600-sum(allocation.values())
    record=dict(charged_before=sum(used.values()),remaining_before=remaining,
                previous_allocation=cfg['allocation_seconds'],new_allocation=allocation,
                rates_seconds_per_control=rates,rate_evidence=evidence,reserves=reserves,
                candidates=candidates,selected=selected,heldout_accessed=False,
                rule='Largest feasible common 16384-control increment in [65536,131072], fixed before training and reserved evaluation; no outcome-based extension.')
    write_json(RUN/'confirmation_budget_schedule.json',record)
    cfg['allocation_seconds']=allocation;write_json(RUN/'budget.json',cfg)
    plan['original_max_steps_per_arm']=plan['steps_per_arm'];plan['steps_per_arm']=selected['steps_per_arm']
    plan['budget_schedule']='confirmation_budget_schedule.json';write_json(RUN/'confirmation_plan.json',plan)
    print(record)

if __name__=='__main__':main()
