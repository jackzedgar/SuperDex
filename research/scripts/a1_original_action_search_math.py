"""Sixth registered finite search. Original environment limits are unchanged."""
import numpy as np
from a1_control_witness_math import rank

def knot_times(horizon):
    assert isinstance(horizon,int) and horizon>33
    return np.array([0,8,32,horizon-1],dtype=np.float64)

def knot_bias(knots,phase,horizon,mask):
    assert np.shape(knots)==(4,52)
    times=knot_times(horizon);phase=np.clip(phase,0,horizon-1)
    left=min(int(np.searchsorted(times,phase,side='right'))-1,2)
    weight=(phase-times[left])/(times[left+1]-times[left])
    active=np.repeat(np.asarray(mask,dtype=np.float32),26)
    return ((1-weight)*knots[left]+weight*knots[left+1])*active

def torch_knot_bias(knots,phases,times,mask):
    left=((phases[:,None]>=times[None,:]).sum(-1)-1).clamp(0,2)
    weight=((phases.float()-times[left])/(times[left+1]-times[left])).clamp(0,1)[:,None]
    a=knots.gather(1,left[:,None,None].expand(-1,1,52))[:,0]
    b=knots.gather(1,(left+1)[:,None,None].expand(-1,1,52))[:,0]
    return ((1-weight)*a+weight*b)*mask

def resample_prior(old_knots,horizon,mask):
    old_knots=np.asarray(old_knots,dtype=np.float64)
    assert old_knots.shape==(4,52) and np.isfinite(old_knots).all() and abs(old_knots).max()<=.25000001
    old_times=np.linspace(0,horizon-1,4);times=knot_times(horizon)
    active=np.repeat(np.asarray(mask,dtype=np.float32),26)
    return (np.stack([np.interp(times,old_times,old_knots[:,j]) for j in range(52)],axis=1)*active).astype(np.float32)

class FixedCEM:
    def __init__(self,mask,prior,seed=410031):
        self.rng=np.random.default_rng(seed);self.active=np.repeat(np.asarray(mask,dtype=np.float32),26)
        assert self.active.shape==(52,) and set(self.active)<={0,1}
        prior=np.asarray(prior,dtype=np.float32)
        assert prior.shape==(4,52) and np.isfinite(prior).all() and abs(prior).max()<=2 and not np.any(prior*(1-self.active))
        self.mean=prior.astype(np.float64).copy();self.std=np.full((4,52),.6)
        self.best=prior.copy();self.best_score=None
    def population(self):
        x=np.clip(self.rng.normal(self.mean,self.std,size=(64,4,52)),-2,2)*self.active
        x=x.astype(np.float32);x[0]=0;x[1]=self.best
        return x
    def update(self,population,rows,horizon):
        assert np.shape(population)==(64,4,52) and len(rows)==64
        scores=[rank(r,horizon) for r in rows];order=sorted(range(64),key=lambda i:scores[i],reverse=True);top=order[0]
        if self.best_score is None or scores[top]>self.best_score:self.best=population[top].copy();self.best_score=scores[top]
        elite=population[order[:8]].astype(np.float64)
        self.mean=.5*self.mean+.5*elite.mean(0)
        self.std=np.clip(.5*self.std+.5*elite.std(0),.1,1.)
        return dict(best_index=top,best_score=list(self.best_score),elite_indices=order[:8])
