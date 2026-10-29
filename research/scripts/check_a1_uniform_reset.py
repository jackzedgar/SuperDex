"""CPU contract checks for B1 uniform-reference reset selection; no GPU execution."""
import json, pathlib, collections, types
import numpy as np
from a1_reset import uniform_reference_state
from a1_data_sources import ROOT, RUN

def check():
    manifest=json.loads((ROOT/'configs/a1_probe40_v1.json').read_text())
    rows=manifest['tasks'];rng=np.random.default_rng(280053);ids=[r['task_id'] for r in rows]
    lengths={}
    for row in rows:
        with np.load(ROOT/row['task_path']/'reference.npz') as z: n=len(z['qpos'])
        lengths[row['task_id']]=n
        # Deliberately no anchor/fallback/reward attributes: uniform selection
        # must remain possible even when those unavailable states have no coverage.
        # Selector receives only task IDs and lengths, never learned statistics.
    counts=collections.Counter();first=collections.Counter();phase_sums=collections.Counter();nonbin=0
    n=200000
    for _ in range(n):
        tid,phase=uniform_reference_state(ids,lengths,rng)
        assert 0<=phase<lengths[tid]-1
        counts[tid]+=1;first[tid]+=int(phase==0);phase_sums[tid]+=phase;nonbin+=int(phase%25!=0)
    expected=n/len(lengths)
    assert len(counts)==40 and max(abs(x/expected-1) for x in counts.values())<.10
    assert nonbin/n>.9
    # Check selection range/determinism without stochastic demands that rare
    # long-trajectory phase0 must appear a particular number of times.
    saved=rng.bit_generator.state
    a=[uniform_reference_state(ids,lengths,rng) for _ in range(100)]
    rng.bit_generator.state=saved
    assert a==[uniform_reference_state(ids,lengths,rng) for _ in range(100)]
    result=dict(status='PASS',scope='CPU reset selector only; GPU rollout checked separately',
                draws=n,tasks=len(counts),non_bin_boundary_fraction=nonbin/n,
                max_task_relative_count_deviation=max(abs(x/expected-1) for x in counts.values()),
                task_counts=dict(counts),phase0_counts=dict(first),
                mean_phase={k:phase_sums[k]/counts[k] for k in counts},
                no_anchor_or_replay_access=True,deterministic_replay=True)
    out=RUN/'uniform_reference_reset_cpu_check.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,dict)}))
if __name__=='__main__': check()
