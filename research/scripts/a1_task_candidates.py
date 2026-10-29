"""Source-only candidate order, fixed before retargeting or learned outcomes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from a1_data_sources import ROOT,RUN,write
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--screen',default='hot3d_reference_screen.json');parser.add_argument('--version',default='v1');parser.add_argument('--velocity-constrained',action='store_true');parser.add_argument('--source-context-filter',action='store_true');parser.add_argument('--max-candidates',type=int,default=20);a=parser.parse_args()
    screen=json.loads((RUN/'public_catalog'/a.screen).read_text())
    mapping={'stable_grasp_transport_candidate':'stable_grasp_transport','in_hand_reorientation_candidate':'in_hand_reorientation_or_regrasp','bimanual_handover_candidate':'bimanual_handover'}
    source_exclusions=[]
    if a.source_context_filter:
        valid=[]
        for r in screen['rows']:
            if r['object_name'] in ['coffee_pot','whiteboard_eraser']:
                if r['candidate_strata']:source_exclusions.append(dict(clip=r['clip'],object=r['object_name'],reason='Observed lid articulation or vertical-board interaction absent from rigid single-object+horizontal support model'))
                continue
            r=dict(r);kinds=r['candidate_strata'].copy()
            other=min(r['right_two_tip_proximity_fraction'],r['left_two_tip_proximity_fraction'])
            if other>.05:
                kinds=[k for k in kinds if k=='bimanual_handover_candidate']
            r['candidate_strata']=kinds;valid.append(r)
        screen['rows']=valid
    candidates=[];reserved={r['clip'] for r in screen['rows'] if 'bimanual_handover_candidate' in r['candidate_strata']}
    for label,stratum in mapping.items():
        rows=[r for r in screen['rows'] if label in r['candidate_strata'] and (stratum=='bimanual_handover' or r['clip'] not in reserved)]
        # Longest qualifying source segment per clip/object, not best policy.
        rows.sort(key=lambda r:(r['clip'],r['object_id'],-r['duration_s'],r['start_frame']))
        unique={}
        for r in rows:unique.setdefault((r['clip'],r['object_id']),r)
        ordered=sorted(unique.values(),key=lambda r:hashlib.sha256(f"280043:{stratum}:{r['clip']}:{r['object_id']}".encode()).hexdigest())
        selected=[];clips=set()
        for r in ordered:
            if r['clip'] in clips:continue
            clips.add(r['clip']);selected.append(r)
            if len(selected)>=a.max_candidates:break
        for r in selected:
            side='right' if r['right_two_tip_proximity_fraction']>=r['left_two_tip_proximity_fraction'] else 'left'
            candidates.append(dict(**r,source='hot3d',stratum=stratum,hands=['right','left'] if stratum=='bimanual_handover' else [side],object_key=f"hot3d_{r['object_id']:06d}",task_id=f"{stratum}_{r['clip']}_o{r['object_id']:03d}_f{r['start_frame']:03d}-{r['end_frame']:03d}"))
    oak=json.loads((RUN/'public_catalog/source_pool_v2_preregister.json').read_text())['oakink_tool_segments']
    for r in sorted(oak,key=lambda r:hashlib.sha256(('280043:'+r['sequence']).encode()).hexdigest()):
        oid=r['program']['obj_list_rh'][0]
        candidates.append(dict(**r,source='oakink2',stratum='light_object_tool_use_below100g',hands=['right'],object_key='oakink2_'+oid.replace('@','_'),task_id='light_tool_oakink_'+hashlib.sha256((r['sequence']+r['interval']).encode()).hexdigest()[:12]))
    if a.velocity_constrained:
        for r in candidates:
            r['original_task_id']=r['task_id'];r['task_id']+='_rv2';r['retarget_version']=2
    out=RUN/f'public_catalog/task_candidates_{a.version}.json'
    assert not out.exists(),'Preserve previous source-only order'
    write(out,dict(status='Candidate order only, not frozen Probe40',source_context_filter=a.source_context_filter,source_exclusions=source_exclusions,source_context_rule='If enabled, other-hand two-tip proximity<=5percent for single-hand segments; exclude observed articulated coffee-pot and vertical-board eraser contexts uniformly, based on source RGB only. This does not select on policy outcomes.',seed=280043,selection='Source semantic/proximity labels, one candidate per clip per stratum, longest qualifying segment, hash order; no policy outcomes',desired_count_per_stratum=10,physical_filters='finite, tip95<=20mm, initial penetration<=3mm, max penetration<=5mm, joint speed<=25rad/s; physical support/reach checks required next; no hold-success gate',candidates=candidates))
    print('candidates',len(candidates),{s:sum(r['stratum']==s for r in candidates) for s in [*mapping.values(),'light_object_tool_use_below100g']})
if __name__=='__main__':main()
