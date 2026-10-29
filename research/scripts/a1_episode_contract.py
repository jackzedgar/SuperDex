"""Eligibility of complete physical episodes; interrupted records stay auditable."""
COMPLETE_TERMINATIONS={'reference_end','physics_failure_1','physics_failure_2','physics_failure_3','physics_failure_4'}

def is_complete_episode(entry):
 reason=entry['termination']
 if reason not in COMPLETE_TERMINATIONS and reason!='budget_truncation':
  raise ValueError('Unknown episode termination: '+str(reason))
 if not isinstance(entry['controls'],int) or entry['controls']<=0:
  raise ValueError('Episode controls must be a positive integer')
 return reason in COMPLETE_TERMINATIONS

def complete_controls(entries):
 return sum(e['controls'] for e in entries if is_complete_episode(e))
