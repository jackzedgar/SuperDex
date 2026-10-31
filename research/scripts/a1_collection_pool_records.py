"""Collection scheduling/provenance only; no physics, reward or policy changes."""
import hashlib,json,os
from pathlib import Path
import numpy as np

class EpisodePlan:
    def __init__(self,path,seed,horizon):
        self.path=Path(path);self.seed=int(seed);self.horizon=int(horizon);assert self.horizon>0
        self.rng=np.random.default_rng(self.seed);self.rows=[]
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line:continue
                row=json.loads(line);expected=self._next(len(self.rows))
                if row!=expected:raise ValueError('Reservation sequence does not match task seed/horizon')
                self.rows.append(row)
    def _next(self,ordinal):
        # Same reset draw sequence as a fresh original uniform-reference Runner.
        return dict(ordinal=ordinal,phase=int(self.rng.integers(self.horizon)),seed=int(self.rng.integers(2**31)),noise_entropy=[self.seed,ordinal,0xA1C0])
    def reserve(self):
        row=self._next(len(self.rows))
        with self.path.open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
        self.rows.append(row);return row.copy()
    def verify_commits(self,entries):
        used=set()
        for e in entries:
            ordinal=e['metadata']['collection_pool_ordinal']
            if ordinal in used or not 0<=ordinal<len(self.rows):raise ValueError('Invalid or duplicate committed reservation')
            row=self.rows[ordinal];used.add(ordinal)
            if (e['seed'],e['phase_start'])!=(row['seed'],row['phase']):raise ValueError('Committed reset differs from reserved reset')
        return used

def episode_noise(row):
    # Independent Gaussian streams per episode, unaffected by lane completion order.
    return np.random.default_rng(np.random.SeedSequence(row['noise_entropy']))

class EpisodeBuffer:
    def __init__(self,reservation,state,generalized,context,history,contact,details,warmstart,context_record):
        self.reservation=reservation.copy();self.context_record=context_record;self.warmstart=np.array(warmstart,copy=True);self.controls=0;self.last=None
        self.arrays={k:[] for k in ['state','request','action','applied','generalized','contact','phase','reward_terms','terminated','contact_details','contact_ranges']}
        self.arrays['context']=[np.array(context,copy=True)];self.arrays['initial_history']=[np.array(history,copy=True)]
        self._state(state,generalized,contact,details,reservation['phase'])
    def _state(self,state,generalized,contact,details,phase):
        a=self.arrays;n=int(contact[8]);assert n>=0 and n<=len(details) and not contact[9]
        for key,value in [('state',state),('generalized',generalized),('contact',contact)]:a[key].append(np.array(value,copy=True))
        a['phase'].append(int(phase));a['contact_ranges'].append([len(a['contact_details']),n]);a['contact_details'].extend(np.array(details[:n],copy=True))
    def step(self,row,details):
        assert int(row['phase'])==self.reservation['phase']+self.controls+1
        assert int(row['count'])==self.controls+1 and row['finite'] and not row['contact'][9]
        self._state(row['state'],row['generalized'],row['contact'],details,row['phase'])
        for key in ['request','action','applied','reward_terms']:self.arrays[key].append(np.array(row[key],copy=True))
        self.arrays['terminated'].append(int(row['done']));self.controls+=1;self.last=row
    def finish(self,writer,lane,formal,training_reset_mode):
        assert self.controls and self.last is not None
        r=self.last;reason=('physics_failure_'+str(int(r['reason']))) if r['reason'] else 'reference_end' if r['done'] else 'budget_truncation'
        arrays={k:np.asarray(v) for k,v in self.arrays.items()}
        return writer.append(arrays,self.reservation['seed'],self.reservation['phase'],reason,metadata=dict(formal=formal,reset_mode='uniform_reference',training_reset_mode=training_reset_mode,collection_reset_memory_mode='original',initial_solver_warmstart=self.warmstart.tolist(),context=self.context_record,policy_noise_sigma=.1,full_reference_start=self.reservation['phase']==0,regular_success=bool(r['regular']),strict_success=bool(r['strict']),bootstrap=False,collection_pool_ordinal=self.reservation['ordinal'],collection_pool_lane=lane,noise_entropy=self.reservation['noise_entropy'],noise_algorithm='NumPy PCG64 per episode; independent N(0,1), multiplied by original policy std0.1; original environment clips total action',collection_execution='independent_original_nworld1'))
