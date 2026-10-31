"""Causal saved-trace inputs and fixed ninth-experiment decision rules."""
import torch

def causal_histories(state,actions):
    n=len(actions)
    assert state.shape==(n+1,287) and actions.shape==(n,52) and n>0
    t=torch.arange(n,device=state.device)[:,None]
    si=(t+torch.tensor([-2,-1,0],device=t.device)).clamp(min=0)
    ai=t+torch.tensor([-3,-2,-1],device=t.device)
    sh=state[si];ah=actions[ai.clamp(min=0)]*(ai>=0)[...,None]
    return sh,ah

def clone_parameters(policy):
    chosen=[]
    for name,p in policy.named_parameters():
        active=name.startswith('encoder.') or name.startswith('decoder.')
        p.requires_grad_(active)
        if active:chosen.append(p)
    assert chosen
    return chosen

def clone_gate(updates,rmse,rows,horizon=98):
    return bool(updates==2000 and 0<=rmse<=.02 and len(rows)==3 and all(
        r['seed']==280044 and r['controls']==horizon and r['regular_success'] and r['failure_reason']==0 for r in rows))

def comparison_gate(candidate,control):
    for r in [candidate,control]:
        if not r or r.get('new_controls')!=131072 or r.get('updates')!=64:return False
        d=r.get('development')
        if not d or not d.get('complete') or len(d['conditions'])!=4:return False
        if sorted(x['seed'] for x in d['conditions'])!=[280046,280047,280048,280049]:return False
    c=sum(bool(x['regular_success']) for x in candidate['development']['conditions'])
    b=sum(bool(x['regular_success']) for x in control['development']['conditions'])
    return bool(c>=2 and c-b>=2)
