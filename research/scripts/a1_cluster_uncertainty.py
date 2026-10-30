"""Paired cluster uncertainty for fixed A1 prediction windows, CPU only.

Call separately with immutable episode IDs and initialization-family IDs.
Family-level resampling is primary; episode-only resampling is a sensitivity
analysis when multiple episodes share an initialization family. This utility
does not admit data, open test sets, aggregate six models, or declare G1 passed.
"""
import numpy as np

METRICS = ('position_median_mm', 'orientation_median_deg', 'position_p90_mm',
           'orientation_p90_deg', 'position_le15mm_fraction')


def integer_weight_quantile(sorted_values, sorted_weights, q):
    """Exactly NumPy's linear quantile for integer replicated observations."""
    counts = np.asarray(sorted_weights)
    assert np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
    total = int(counts.sum()); assert total > 0
    ranks = (total-1)*np.asarray(q, dtype=float)
    assert ((ranks >= 0) & (ranks <= total-1)).all()
    lower = np.floor(ranks).astype(np.int64); upper = np.ceil(ranks).astype(np.int64)
    cumulative = np.cumsum(counts)
    a = np.asarray(sorted_values)[np.searchsorted(cumulative, lower, side='right')]
    b = np.asarray(sorted_values)[np.searchsorted(cumulative, upper, side='right')]
    return a+(b-a)*(ranks-lower)


def paired_cluster_intervals(ids_w0, ids_w1, position_w0, rotation_w0,
                             position_w1, rotation_w1, cluster_ids,
                             cluster_kind, seed=301046, resamples=5000):
    ids_w0=np.asarray(ids_w0);ids_w1=np.asarray(ids_w1)
    assert ids_w0.shape==ids_w1.shape and np.array_equal(ids_w0,ids_w1), 'W0/W1 fixed window order mismatch'
    assert cluster_kind in ['episode','initialization_family']
    values = [np.asarray(v,dtype=float) for v in [position_w0,rotation_w0,position_w1,rotation_w1]]
    n=len(ids_w0);assert all(v.shape==(n,) and np.isfinite(v).all() and (v>=0).all() for v in values)
    clusters=np.asarray(cluster_ids,dtype=str);assert clusters.shape==(n,)
    assert resamples>=1000
    unique, index=np.unique(clusters,return_inverse=True)
    base=dict(status='UNCERTAINTY_UNAVAILABLE',windows=n,clusters=len(unique),cluster_kind=cluster_kind,
              seed=seed,resamples=resamples,metrics=list(METRICS),formal_G1_claim=False,
              interpretation='Paired descriptive percentile intervals, conditional on the frozen sampled windows. '
              'W1 minus W0: negative error changes or positive <=15mm-fraction changes favor W1. '
              'No claim of independent windows, new task generalization, or A2 control performance.')
    if n<2 or len(unique)<2:
        base['reason']='Fewer than two windows or independent clusters; do not report a degenerate interval as certainty.'
        base.update(W0=None,W1=None,paired_W1_minus_W0=None)
        return base
    assert (clusters!='').all()
    order=[np.argsort(v,kind='stable') for v in values]
    sorted_values=[v[o] for v,o in zip(values,order)]

    def statistics(weights):
        result=[]
        for start in [0,2]:
            p=integer_weight_quantile(sorted_values[start],weights[order[start]],[.5,.9])
            r=integer_weight_quantile(sorted_values[start+1],weights[order[start+1]],[.5,.9])
            fraction=np.dot(weights,(values[start]<=15).astype(float))/weights.sum()
            result.append([p[0],r[0],p[1],r[1],fraction])
        return np.asarray(result)

    point=statistics(np.ones(n,dtype=np.int64));rng=np.random.default_rng(seed)
    draws=np.empty((resamples,2,len(METRICS)))
    for i in range(resamples):
        multiplicity=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        draws[i]=statistics(multiplicity[index])
    base['status']='PAIRED_CLUSTER_INTERVALS_COMPUTED'
    base['resampling']='Uniform whole-cluster resampling with replacement; all observed windows of a selected cluster retained, using identical draws for W0 and W1.'
    for j,name in enumerate(['W0','W1']):
        base[name]=dict(estimate=point[j].tolist(),percentile95=np.quantile(draws[:,j,:],[.025,.975],axis=0).T.tolist())
    base['paired_W1_minus_W0']=dict(estimate=(point[1]-point[0]).tolist(),percentile95=np.quantile(draws[:,1,:]-draws[:,0,:],[.025,.975],axis=0).T.tolist())
    return base
