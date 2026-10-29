import json,statistics,collections,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
r=Path('runs/a1_full_20261028/collector_full_v3_lr_001')
m=json.loads(Path('configs/a1_probe40_v1.json').read_text()); tasks={x['task_id']:x for x in m['tasks']}
out={'scope':'Read-only development failure diagnosis; no locked test, no training change','evaluations':[],'reference_audit':[]}
for p in sorted(r.glob('evaluation_????????.json')):
 d=json.loads(p.read_text());groups={}
 for name in sorted({x['stratum'] for x in tasks.values()}):
  rows=[x for x in d['all_conditions'] if tasks[x['task']]['stratum']==name]
  groups[name]={'n':len(rows),'regular':sum(x['regular_success'] for x in rows),'strict':sum(x['strict_success'] for x in rows),'mean_controls':statistics.mean(x['controls'] for x in rows),'failure_reasons':dict(collections.Counter(x['failure_reason'] for x in rows)),'first_failure_median':statistics.median(x['first_strict_failure_s'] if x['first_strict_failure_s'] is not None else x['controls']*.02 for x in rows),'position_mm_first3':np.mean([np.mean(np.asarray(x['error_trace_m_rad'])[:3,0])*1000 for x in rows]),'orientation_deg_first3':np.mean([np.mean(np.asarray(x['error_trace_m_rad'])[:3,1])*180/np.pi for x in rows]),'reward_terms_mean':np.mean([x['reward_terms_mean'] for x in rows],0).tolist()}
 out['evaluations'].append({'controls':d['controls'],'regular':d['regular_success_rate'],'strict':d['strict_success_rate'],'strata':groups,'comparison':d.get('continuation_comparison')})
latest=json.loads(sorted(r.glob('evaluation_????????.json'))[-1].read_text())
out['per_task_latest']=[]
for tid,t in tasks.items():
 p=Path(t['task_path']);q=json.loads((p/'qualification.json').read_text());z=np.load(p/'reference.npz');nu=26*len(q['hands']);vel=z['qvel'];x=ET.parse(p/'scene.xml').getroot()
 out['reference_audit'].append({'task':tid,'stratum':t['stratum'],'duration_s':q['duration_s'],'initial_penetration_mm':q['initial_penetration_mm'],'max_penetration_mm':q['max_penetration_mm'],'physical_qualification':q.get('physical_qualification'),'max_object_speed_mps':float(np.linalg.norm(vel[:,nu:nu+3],axis=1).max()),'initial_object_speed_mps':float(np.linalg.norm(vel[0,nu:nu+3])),'initial_object_rotation_rps':float(np.linalg.norm(vel[0,nu+3:nu+6])),'max_wrist_speed_mps':max(float(np.linalg.norm(vel[:,h*26:h*26+3],axis=1).max()) for h in range(len(q['hands']))),'timestep':x.find('option').get('timestep'),'wrist_actuators':[dict(a.attrib) for a in list(x.find('actuator'))[:6]]})
 rows=[a for a in latest['all_conditions'] if a['task']==tid]
 out['per_task_latest'].append({'task':tid,'successes':sum(a['regular_success'] for a in rows),'controls':[a['controls'] for a in rows],'first_failure':[a['first_strict_failure_s'] for a in rows],'reasons':[a['failure_reason'] for a in rows],'errors_first_step':[a['error_trace_m_rad'][0] for a in rows],'contact_summary_mean':np.mean([a['contact_summary_mean'] for a in rows],0).tolist()})
print(json.dumps(out,indent=2))
