"""Fixed, bounded search parameters; no environment or network changes."""
import numpy as np

def knot_bias(knots,phase,horizon,mask):
    assert horizon>=2 and np.shape(knots)==(4,52)
    active=np.repeat(np.asarray(mask,dtype=np.float32),26)
    x=np.clip(phase/(horizon-1),0,1)*3;i=min(int(x),2);w=x-i
    return ((1-w)*knots[i]+w*knots[i+1])*active

def rank(row,horizon):
    assert row['complete'] and 0<row['controls']<=horizon
    assert not row['regular_success'] or row['controls']==horizon
    return (int(row['regular_success']),int(row['strict_success']),row['controls']/horizon,float(row['return_sum']))

class FixedCEM:
    def __init__(self,mask,seed=410030):
        self.rng=np.random.default_rng(seed);self.active=np.repeat(np.asarray(mask,dtype=np.float32),26)
        assert self.active.shape==(52,) and set(self.active)<= {0,1}
        self.mean=np.zeros((4,52));self.std=np.full((4,52),.1);self.best=np.zeros((4,52),dtype=np.float32);self.best_score=None
    def population(self):
        x=np.clip(self.rng.normal(self.mean,self.std,size=(16,4,52)),-.25,.25)*self.active
        x=x.astype(np.float32);x[0]=0;x[1]=self.best
        return x
    def update(self,population,rows,horizon):
        assert np.shape(population)==(16,4,52) and len(rows)==16
        scores=[rank(r,horizon) for r in rows];order=sorted(range(16),key=lambda i:scores[i],reverse=True)
        top=order[0]
        if self.best_score is None or scores[top]>self.best_score:self.best=population[top].copy();self.best_score=scores[top]
        elite=population[order[:4]].astype(np.float64)
        self.mean=.5*self.mean+.5*elite.mean(0)
        self.std=np.clip(.5*self.std+.5*elite.std(0),.02,.15)
        return dict(best_index=top,best_score=list(self.best_score),elite_indices=order[:4])
