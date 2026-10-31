"""Authorized eighth bounded frozen-controller validation; no PPO or optimization."""
import argparse,copy,datetime,getpass,hashlib,json,os,signal,socket,time,traceback,types
from pathlib import Path
import mujoco,numpy as np,torch,warp as wp
from a1_policy import CollectorPolicy
from a1_original_action_search_pool import WitnessPool
from dr_model_packets import NAMES,ALLOWED,mix_packets

R=Path(__file__).resolve().parents[1];STOP=False
def halt(*_):
 global STOP;STOP=True
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True,type=Path);args=ap.parse_args();c=json.loads(args.config.read_text())
 assert(socket.gethostname(),getpass.getuser())==('Lambda','youran')
 assert c['authorization_reply']=='批准第八项≤0.15 GPUh验证' and c['scientific_revision']==8 and c['phase']=='revision8_grouped_dr_validation' and c['job_limit_seconds']==540
 for p,h in c['source_hashes'].items():assert sha(R/p)==h,p
 parent_path=R/c['checkpoint'];selected_path=R/c['selected_controller'];assert sha(parent_path)==c['checkpoint_sha256'] and sha(selected_path)==c['selected_controller_sha256']
 plan=c['plan'];specs={s['id']:s for s in plan['conditions']};assert len(specs)==10 and len(plan['execution_order'])==30 and all(s['group'] in (*ALLOWED,None) for s in specs.values())
 assert {s['base_seed'] for s in specs.values()}=={280043,280044} and all(sum(x['condition']==s for x in plan['execution_order'])==3 for s in specs)
 out=R/'runs/a1_full_20261028'/c['run'];out.mkdir(exist_ok=False);start=time.monotonic();deadline=start+450;pool=None;records=[];packet_records={}
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(1)
 write(out/'run_configuration.json',dict(c,started_beijing=now(),network_updates=False,full_A1_complete=False))
 def guard():return STOP or time.monotonic()>=deadline
 def progress(stage):write(out/'progress.json',dict(stage=stage,time_beijing=now(),seconds=time.monotonic()-start,conditions=len(records),physical_controls=pool.controls if pool else 0))
 try:
  parent=torch.load(parent_path,map_location='cpu',weights_only=False);assert parent['configuration']['task_id']==c['task_id'] and parent['configuration']['scientific_revision']==4 and not parent['smoke']
  policy=CollectorPolicy(280043).cuda();policy.load_state_dict(parent['policy']);policy.eval()
  for parameter in policy.parameters():parameter.requires_grad_(False)
  chosen=json.loads(selected_path.read_text());assert chosen['scientific_revision']==7 and chosen['checkpoint_sha256']==c['checkpoint_sha256'];knots=np.asarray(chosen['knots'],np.float32)
  pool=WitnessPool(R/c['task_path'],policy);assert pool.horizon==98
  for e in pool.envs:
   original=e.randomize;e.factor_spec=None
   def override(self,seed,enabled=True,_original=original):
    spec=self.factor_spec;assert spec is not None and enabled and seed==spec['base_seed'];_original(seed,enabled)
    base={k:getattr(self.m,k).copy() for k in NAMES};base_record=copy.deepcopy(self.context_record);base_context=np.asarray(base_record['context'],np.float32)
    np.testing.assert_array_equal(base_context,np.asarray(specs[f'parent_{seed}']['context'],np.float32))
    donor=base;donor_record=base_record;donor_context=base_context
    if spec['group'] is not None:
     _original(spec['donor_seed'],enabled);donor={k:getattr(self.m,k).copy() for k in NAMES};donor_record=copy.deepcopy(self.context_record);donor_context=np.asarray(donor_record['context'],np.float32)
     np.testing.assert_array_equal(donor_context,np.asarray(specs[f"parent_{spec['donor_seed']}"]['context'],np.float32));_original(seed,enabled)
     for k in NAMES:np.testing.assert_array_equal(getattr(self.m,k),base[k])
    finger_ids=[i for k in range(len(self.task.sides)) for i in range(k*26+6,k*26+26)]
    mixed,context=mix_packets(base,donor,base_context,donor_context,spec['group'],self.object_body,self.hand_geom,self.floor_geom,finger_ids)
    np.testing.assert_array_equal(context,np.asarray(spec['context'],np.float32))
    if spec['group'] is not None:
     for k,v in mixed.items():getattr(self.m,k)[:]=v
     mujoco.mj_setConst(self.m,mujoco.MjData(self.m))
     for k in [*NAMES,'body_subtreemass','body_invweight0','dof_invweight0']:
      if not hasattr(self.wm,k):continue
      target=wp.to_torch(getattr(self.wm,k));value=torch.as_tensor(getattr(self.m,k),device=self.device,dtype=target.dtype);target.copy_(value.expand_as(target))
     self.context.copy_(torch.as_tensor(context,device=self.device)[None])
    for k,v in mixed.items():np.testing.assert_array_equal(getattr(self.m,k),v)
    np.testing.assert_array_equal(self.context[0].cpu().numpy(),context)
    for k in [*NAMES,'body_subtreemass','body_invweight0','dof_invweight0']:
     if hasattr(self.wm,k):
      target=wp.to_torch(getattr(self.wm,k));expected=torch.as_tensor(getattr(self.m,k),device=self.device,dtype=target.dtype).expand_as(target);assert torch.equal(target,expected),k
    snapshot={f'{prefix}__{k}':v for prefix,packet in [('base',base),('donor',donor),('mixed',mixed)] for k,v in packet.items()};snapshot.update(base_context=base_context,donor_context=donor_context,mixed_context=context)
    path=out/f"packet_{spec['id']}.npz"
    if path.exists():
     with np.load(path) as f:
      assert set(f.files)==set(snapshot)
      for k,v in snapshot.items():np.testing.assert_array_equal(f[k],v)
    else:np.savez_compressed(path,**snapshot)
    meta=dict(condition=spec['id'],group=spec['group'],base_seed=seed,donor_seed=spec['donor_seed'],packet_file=path.name,packet_sha256=sha(path),object_body=self.object_body,hand_geoms=self.hand_geom.tolist(),floor_geom=self.floor_geom,finger_ids=finger_ids,model_and_context_verified=True,full_precision_model_packets=True)
    packet_records[spec['id']]=meta
    self.context_record=copy.deepcopy(base_record);self.context_record.update(context=context.tolist(),grouped_factor=meta)
    if spec['group']=='object_dynamics':
     self.context_record['mass_scale']=donor_record['mass_scale'];self.context_record['inertia_draw_rejections']=donor_record['inertia_draw_rejections']
   e.randomize=types.MethodType(override,e)
  write(out/'implementation_static_gate.json',dict(status='STATIC_SOURCE_AND_CONTROLLER_VERIFIED',runtime_model_and_context_guard_each_reset=True,original_reset_history_guard=True))
  order=plan['execution_order']
  for offset in range(0,len(order),8):
   if guard():break
   batch=order[offset:offset+8]
   for lane,x in enumerate(batch):pool.envs[lane].factor_spec=specs[x['condition']]
   progress('GROUPED_FACTOR_REPLAY')
   rows=pool.run(np.stack([knots]*len(batch)),[specs[x['condition']]['base_seed'] for x in batch],guard,validate=True)
   for lane,row in enumerate(rows):
    spec=specs[batch[lane]['condition']];tr=row.pop('trace');path=out/f'trace_{offset+lane:02d}.npz';np.savez_compressed(path,**{k:np.asarray(v) for k,v in tr.items()})
    assert row['context']['grouped_factor']['condition']==spec['id']
    row.update(condition=spec['id'],repeat=batch[lane]['repeat'],group=spec['group'],base_seed=spec['base_seed'],donor_seed=spec['donor_seed'],trace_file=path.name,trace_sha256=sha(path),time_beijing=now())
    with (out/'conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    records.append(row)
   write(out/'packet_manifest.json',packet_records)
  grouped={key:dict(n=sum(r['condition']==key for r in records),complete=sum(r['condition']==key and r['complete'] for r in records),regular=sum(r['condition']==key and r['regular_success'] for r in records),strict=sum(r['condition']==key and r['strict_success'] for r in records),controls=[r['controls'] for r in records if r['condition']==key]) for key in specs}
  complete=len(records)==30 and all(r['complete'] for r in records);baseline=complete and grouped['parent_280043']['regular']==0 and grouped['parent_280044']['regular']==3
  factors={g:bool(baseline and grouped[f'280043_with_280044_{g}']['regular']==3 and grouped[f'280044_with_280043_{g}']['regular']==0) for g in ALLOWED}
  assert sha(parent_path)==c['checkpoint_sha256'] and sha(selected_path)==c['selected_controller_sha256']
  assert all(torch.equal(v.detach().cpu(),parent['policy'][k]) for k,v in policy.state_dict().items())
  assert sum(r['controls'] for r in records)==pool.controls and sum(e.physical_controls for e in pool.envs)==pool.controls+pool.warmup_controls
  result=dict(status='COMPLETE' if complete else 'BOUNDED_STOP_INCOMPLETE',time_beijing=now(),seconds=time.monotonic()-start,conditions=len(records),grouped=grouped,baseline_reproduced=baseline,local_bidirectional_factor_gates=factors,physical_controls=pool.controls,warmup_controls=pool.warmup_controls,physical_env_controls=sum(e.physical_controls for e in pool.envs),audit=pool.audit,full_A1_complete=False,no_auto_PPO=True,network_updates=0,policy_parameters_unchanged=True,original_nominal_physics_unchanged=True,development_seeds_used=False,scope='Local intervention on one frozen task and two optimization contexts; not collector quality or G1')
  write(out/'summary.json',result);progress(result['status']);print(json.dumps(result),flush=True)
 except BaseException as ex:
  write(out/'fatal.json',dict(type=type(ex).__name__,error=str(ex),traceback=traceback.format_exc(),physical_controls=pool.controls if pool else None,conditions=len(records),time_beijing=now()));raise

if __name__=='__main__':main()
