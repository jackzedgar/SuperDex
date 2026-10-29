"""Original G1 thresholds with explicit missing-task/phase rejection."""
REQUIRED_PHASES=('free','onset','sustained','break_contact','slip')

def g1_criteria(summaries,coverage):
 verdict={}
 for sampling in ['balanced','natural']:
  a=summaries[f'H8_{sampling}'];b=summaries[f'H32_{sampling}'];switch=a['phases']['switch']
  def covered(h,s):
   c=coverage[h]
   return not c['missing_task_ids'] and set(s['per_task'])==set(c['expected_task_ids']) and set(s['per_object'])==set(c['expected_object_keys'])
  def le(value,threshold):return value is not None and value<=threshold
  verdict[sampling]=dict(all_tasks_have8and32windows=not coverage[8]['missing_task_ids'] and not coverage[32]['missing_task_ids'],all_tasks_and_objects_sampled=covered(8,a) and covered(32,b),all_required_phases_present_8and32=all(s['phases'].get(p,{}).get('n',0)>0 for s in [a,b] for p in REQUIRED_PHASES),position8=le(a['overall']['position_median_mm'],3),orientation8=le(a['overall']['orientation_median_deg'],3),switch_position8=le(switch['position_p90_mm'],8) if switch['n'] else False,switch_orientation8=le(switch['orientation_p90_deg'],8) if switch['n'] else False,position32=b['overall']['position_le15mm_fraction'] is not None and b['overall']['position_le15mm_fraction']>=.8)
 return verdict
