"""Frozen controller replacement and pre-registered six-episode interpretation."""
import torch
ORDER=['teacher','clone','clone','teacher','teacher','clone']

@torch.no_grad()
def replace_frozen(policy,state):
 before={k:v.data_ptr() for k,v in policy.state_dict().items()}
 assert set(before)==set(state)
 policy.load_state_dict(state)
 assert before=={k:v.data_ptr() for k,v in policy.state_dict().items()}
 assert all(torch.equal(v.detach().cpu(),state[k].detach().cpu()) for k,v in policy.state_dict().items())
 for p in policy.parameters():p.requires_grad_(False)
 return before

def interpretation(rows):
 if len(rows)!=6 or [r['controller'] for r in rows]!=ORDER or not all(r['complete'] for r in rows):return 'INCOMPLETE_NO_CONCLUSION'
 t=sum(r['regular_success'] for r in rows if r['controller']=='teacher');c=sum(r['regular_success'] for r in rows if r['controller']=='clone')
 if t<3:return 'TEACHER_CONTROL_NOT_RETAINED'
 if c<3:return 'CLONE_RETENTION_LOSS_WITH_SHARED_EXECUTOR'
 return 'EXECUTION_PATH_DEPENDENT_CLONE_RESULT_REQUIRES_REVIEW'
