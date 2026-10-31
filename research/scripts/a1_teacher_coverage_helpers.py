"""Eleventh source-wise fit gate and factual episode origin metadata."""
import numpy as np,torch

def validate_data(data,manifest):
 assert manifest['observed_transitions']==866 and manifest['initialization_families']==1 and manifest['optimization_seed']==280044
 assert manifest['clone_observed_transitions_with_counterfactual_teacher_labels']==278
 shapes={'obs':(866,546),'goal':(866,1000),'teacher_target_action':(866,52),'actual_executed_action':(866,52),'source_index':(866,),'phase':(866,)}
 assert set(data)==set(shapes)
 for k,shape in shapes.items():assert data[k].shape==shape and np.isfinite(data[k]).all(),k
 assert np.max(abs(data['teacher_target_action']))<=1 and np.max(abs(data['actual_executed_action']))<=1
 assert set(data['source_index'].tolist())==set(range(9))
 for i,s in enumerate(manifest['sources']):
  ids=data['source_index']==i;assert ids.sum()==s['controls'];np.testing.assert_array_equal(data['phase'][ids],np.arange(s['controls']))
 assert len(manifest['sources'])==9 and sum(s['controls'] for s in manifest['sources'] if s['counterfactual_labels'])==278

def per_source_rmse(prediction,target,source_index):
 return [float((prediction[source_index==i]-target[source_index==i]).square().mean().sqrt()) for i in range(9)]

def coverage_gate(updates,errors,rows):
 return bool(updates==2000 and len(errors)==9 and all(0<=x<=.01 for x in errors) and len(rows)==3 and all(r['seed']==280044 and r['controls']==98 and r['regular_success'] and r['failure_reason']==0 for r in rows))

def factual_episode(ep,reset_index):
 origin=reset_index[(ep['lane'],ep['episode_id'])]
 assert ep['seed']==origin['seed'] and ep['start_phase']==origin['phase']
 result=dict(ep);result['original_runner_labels']={k:ep[k] for k in ['reset_mode','reset_memory_mode']}
 result['reset_mode']=origin['reset_mode'];result['reset_memory_mode']=origin['training_memory'];result['origin_verified_from_reset_record']=True
 return result
