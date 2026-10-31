"""Unapproved static DR intervention plan. No simulator, GPU or launch API."""
import numpy as np

GROUPS={
 'object_dynamics':(136,*range(139,145)),
 'friction':(137,138),
 'finger_gains':tuple(i for h in (0,68) for i in range(h+20,h+60)),
 'finger_offsets':tuple(i for h in (0,68) for i in range(h,h+20)),
 'control_delay':tuple(i for h in (0,68) for i in range(h+60,h+62)),
}

def validate(c):
 c=np.asarray(c,dtype=np.float32)
 if c.shape!=(276,) or not np.isfinite(c).all():raise ValueError('Expected complete finite original context')
 if c[136]<=0 or not np.all(c[142:145]>0) or c[142:145].max()>=c[142:145].sum()/2:raise ValueError('Invalid coupled mass/inertia packet')
 if not .4<=c[137]<=1.2 or not .3<=c[138]<=.9:raise ValueError('Friction outside original range')
 for h in (0,68):
  if np.max(abs(c[h:h+20]))>.010000001:raise ValueError('Offsets outside original range')
  if not np.all(c[h+20:h+60]>0):raise ValueError('Nonpositive finger gains')
  if tuple(c[h+60:h+62]) not in ((1,0),(2,1)):raise ValueError('Invalid paired delay')
  if not np.array_equal(c[h+62:h+68],np.full(6,.02,np.float32)):raise ValueError('Tau is fixed, not a varied factor')
 return c.copy()

def make_plan(parents):
 if set(parents)!={280043,280044}:raise ValueError('Only original optimization conditions may define this draft')
 ps={k:validate(v) for k,v in parents.items()}
 varying=set(i for ids in GROUPS.values() for i in ids);fixed=sorted(set(range(276))-varying)
 if not np.array_equal(ps[280043][fixed],ps[280044][fixed]):raise ValueError('Unexpected differences outside declared factors')
 rows=[]
 for base in (280043,280044):
  rows.append(dict(id=f'parent_{base}',base_seed=base,donor_seed=None,group=None,context=ps[base].tolist(),changed_indices=[]))
  donor=280044 if base==280043 else 280043
  for group,ids in GROUPS.items():
   c=ps[base].copy();c[list(ids)]=ps[donor][list(ids)];validate(c)
   changed=np.flatnonzero(c!=ps[base]).tolist()
   if not changed:continue
   assert set(changed)<=set(ids)
   rows.append(dict(id=f'{base}_with_{donor}_{group}',base_seed=base,donor_seed=donor,group=group,context=c.tolist(),changed_indices=changed))
 rng=np.random.default_rng(410033);order=[]
 for repeat in range(3):
  for index in rng.permutation(len(rows)):order.append(dict(condition=rows[index]['id'],repeat=repeat))
 return dict(status='DRAFT_UNAPPROVED_STATIC_CONTEXTS_ONLY',groups={k:list(v) for k,v in GROUPS.items()},conditions=rows,execution_order=order,order_seed=410033,repetitions=3,original_initialization=True,original_qualifying_thresholds=True,new_GPU_work=0,network_updates=0,no_development_or_heldout_seed=True,max_proposed_gpu_hours=.15,no_auto_PPO=True)
