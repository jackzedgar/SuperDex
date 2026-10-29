"""Compare fixed beginning with the exact support of old uniform RSI, on CPU.

No noise in either arm; old training reset also used no noise. The eight noisy
evaluation states are separate and are not replaced by these phase probes.
"""
import torch,numpy as np,json
from grasp_lab import OLD,RUN,cpu_run,write_json
from train_real_ppo import Policy

c=torch.load(OLD/'checkpoint.pt',map_location='cpu',weights_only=True);p=Policy(118);p.load_state_dict(c['policy']);p.eval();rows=[]
for phase in range(42):
    r=cpu_run('baseline','C',-1,p,start_phase=phase);rows.append(r)
    if phase%10==0:print('phase',phase,'steps',r['steps'],'reason',r['reason'],flush=True)
summary=dict(start_of_clip=rows[0],uniform_RSI=dict(support=list(range(42)),segment_completion_fraction=float(np.mean([r['segment_completed'] for r in rows])),mean_first10_position_m=float(np.mean([r['first_10_controls_position_mean_m'] for r in rows])),mean_first10_rotation_deg=float(np.mean([r['first_10_controls_rotation_mean_deg'] for r in rows]))),
             phase_rows=rows,caveat='Different reference phases are different remaining segments, not paired full-trajectory success. Common first-10-control errors are diagnostic; early terminations remain explicitly recorded.',
             reward_change='none; contact reward not added because a stable reference grasp is not established')
write_json(RUN/'initialization_audit.json',summary);print(json.dumps(summary['uniform_RSI'],indent=2))
