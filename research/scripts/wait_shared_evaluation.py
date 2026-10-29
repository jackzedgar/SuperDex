import time,json,subprocess,pathlib
R=pathlib.Path('runs/bptt_iteration_20261027')
for _ in range(800):
 jobs={j['name']:j for j in json.loads((R/'ledger.json').read_text())['jobs']}
 if 'shared_fixed' in jobs:break
 j=jobs.get('shared_two_objects')
 if j and j['status']=='finished':
  if (R/'shared_two_objects/summary.json').exists():
   log=open(R/'shared_fixed.supervisor.log','w')
   subprocess.run(['python3','scripts/bptt_budget.py','shared_fixed','2','1400','final_evaluation','.venv-mjw/bin/python','scripts/evaluate_bptt_multitask.py'],stdout=log,stderr=subprocess.STDOUT)
  break
 time.sleep(30)
