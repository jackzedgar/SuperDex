"""Fixed-window free autoregression, G1 endpoints and simple physical baselines."""
import json,math
from pathlib import Path
import numpy as np,torch
from a1_window_pool import WindowPool
from a1_data_sources import write
from a1_g1_coverage import g1_criteria,REQUIRED_PHASES
from world_model_mini import qmul,qconj,qlog,qexp
def errors(pred,target):
 pos=torch.linalg.vector_norm(pred[...,274:277]-target[...,274:277],dim=-1)*1000
 angle=torch.linalg.vector_norm(qlog(qmul(qconj(target[...,277:281]),pred[...,277:281])),dim=-1)*180/math.pi
 return pos,angle
def describe(pos,rot):
 pos=np.asarray(pos);rot=np.asarray(rot)
 if not len(pos):return dict(n=0,position_median_mm=None,orientation_median_deg=None,position_p90_mm=None,orientation_p90_deg=None,position_le15mm_fraction=None)
 return dict(n=len(pos),position_median_mm=float(np.median(pos)),orientation_median_deg=float(np.median(rot)),position_p90_mm=float(np.quantile(pos,.9)),orientation_p90_deg=float(np.quantile(rot,.9)),position_le15mm_fraction=float(np.mean(pos<=15)))
def phases(contacts):
 hand=contacts[:,:,:2].sum(-1)>0;floor=contacts[:,:,2]>0
 onset=(~hand[:,:-1]&hand[:,1:]).any(1);loss=(hand[:,:-1]&~hand[:,1:]).any(1);slip=(contacts[:,:,6]>0).any(1)
 labels=dict(free=(~hand).all(1),onset=onset,break_contact=loss,switch=onset|loss,sustained=hand.all(1),slip=slip,hand_involved=hand.any(1),support_only=(~hand).all(1)&floor.any(1),unsupported_free=(~hand).all(1)&(~floor).all(1))
 return labels
@torch.no_grad()
def run_evaluation(model,manifest,fixed_dir,out,split='development',per_task=64,natural=4096):
 fixed_dir=Path(fixed_dir);fixed_dir.mkdir(exist_ok=True);out=Path(out);out.mkdir(exist_ok=True)
 summaries={};coverage={}
 for horizon in [8,32]:
  pool=WindowPool(manifest,split,horizon);windows=pool.fixed_windows(fixed_dir/f'{split}_H{horizon}.npz',280050+(split=='test')+horizon,per_task,natural)
  expected_tasks=sorted({r['task_id'] for r in pool.meta['shards']});missing=sorted(set(expected_tasks)-set(pool.task_groups))
  coverage[horizon]=dict(expected_task_count=len(expected_tasks),expected_task_ids=expected_tasks,expected_object_keys=pool.object_keys,eligible_task_count=len(pool.task_groups),missing_task_ids=missing,required_phases=list(REQUIRED_PHASES),windows_per_task={k:int(sum(r[2] for r in v)) for k,v in pool.task_groups.items()})
  for sampling in ['balanced','natural']:
   ids=windows[sampling];allpos=[];allrot=[];pathmean=[];pathmax=[];baseline={k:[[],[]] for k in ['persistence','constant_velocity','gravity']}
   for start in range(0,len(ids),512):
    batch=pool.batch(ids[start:start+512]);b={k:v.to('cuda') for k,v in batch.items()}
    pred=model.rollout(b['start'],b['commands'],b['mask'],b['context'],b['object_id']);p,r=errors(pred,b['target'])
    assert torch.isfinite(pred).all(),'Nonfinite prediction'
    allpos.extend(p[:,-1].cpu().tolist());allrot.extend(r[:,-1].cpu().tolist());pathmean.extend(p.mean(1).cpu().tolist());pathmax.extend(p.max(1).values.cpu().tolist())
    s=b['start'];target=b['target'][:,-1];dt=horizon*.02
    for name in baseline:
     base=s.clone()
     if name!='persistence':
      base[:,274:277]=s[:,274:277]+dt*s[:,281:284]
      base[:,277:281]=qmul(qexp(dt*s[:,284:287]),s[:,277:281])
     if name=='gravity':base[:,276]-=.5*9.81*dt*dt
     bp,br=errors(base,target);baseline[name][0].extend(bp.cpu().tolist());baseline[name][1].extend(br.cpu().tolist())
   pp=np.asarray(allpos);rr=np.asarray(allrot);labels=phases(pool.contacts(ids,horizon))
   tasks=np.asarray([pool.shards[int(s)].meta['task_id'] for s,e,t in ids]);objects=np.asarray([pool.shards[int(s)].meta['object_key'] for s,e,t in ids]);strata=np.asarray([pool.shards[int(s)].meta['stratum'] for s,e,t in ids])
   summary=dict(coverage=coverage[horizon],horizon=horizon,sampling=sampling,split=split,overall=describe(pp,rr),path_position_mean_mm=float(np.mean(pathmean)),path_position_worst_mean_mm=float(np.mean(pathmax)),phases={key:describe(pp[mask],rr[mask]) for key,mask in labels.items()},per_task={key:describe(pp[tasks==key],rr[tasks==key]) for key in np.unique(tasks)},per_object={key:describe(pp[objects==key],rr[objects==key]) for key in np.unique(objects)},per_stratum={key:describe(pp[strata==key],rr[strata==key]) for key in np.unique(strata)},baselines={key:dict(overall=describe(*value),phases={phase:describe(np.asarray(value[0])[mask],np.asarray(value[1])[mask]) for phase,mask in labels.items()}) for key,value in baseline.items()},phase_rule='Actual endpoint GPU contact force>0.02N and distance<2mm; hand-object onset/break over window, slip speed>1cm/s. Multi-label: switch/onset/break may overlap slip. Support-only reported separately.',eligible_windows=sum(int(x[-1]) for x in pool.cdf.values()),eligible_episodes=sum(len(v) for v in pool.groups.values()))
   file=f'H{horizon}_{sampling}';write(out/(file+'.json'),summary)
   np.savez_compressed(out/(file+'.npz'),ids=ids,position_mm=pp,orientation_deg=rr,path_position_mean_mm=np.asarray(pathmean),path_position_max_mm=np.asarray(pathmax),tasks=tasks,objects=objects,strata=strata,**{f'phase_{k}':v for k,v in labels.items()},**{f'baseline_{key}_{which}':np.asarray(value[i]) for key,value in baseline.items() for i,which in enumerate(['position_mm','orientation_deg'])})
   summaries[file]=summary
 verdict=g1_criteria(summaries,coverage)
 result=dict(split=split,coverage=coverage,criteria=verdict,all_present_criteria_pass={k:all(v is True for v in row.values()) for k,row in verdict.items()},formal_G1_claim=False,reason='A single checkpoint evaluation is not the complete three-seed A1 decision.')
 write(out/'criteria.json',result);return summaries,result
