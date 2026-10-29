"""Matched full W0/W1 training on frozen actual-physics episode data."""
import argparse,json,time,math,signal,hashlib
from pathlib import Path
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_window_pool import WindowPool,save_model,load_model_state
from a1_world_model import A1WorldModel
from a1_prediction_eval import run_evaluation
stop=False
def halt(*_):
 global stop;stop=True
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True);p.add_argument('--kind',choices=['W0','W1'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--run',required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--microbatches',type=int,default=40);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
 signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 from torch._inductor import config as inductor_config
 inductor_config.compile_threads=4
 out=RUN/a.run;out.mkdir(exist_ok=False);pool=WindowPool(a.manifest,'train',8)
 if not a.smoke:assert pool.meta['formal'] and (a.updates,a.microbatches,a.batch_size)==(2000,40,512)
 norm=torch.load(a.normalization,map_location='cpu',weights_only=False);manifest_sha=hashlib.sha256(a.manifest.read_bytes()).hexdigest();assert norm['metadata']['manifest_sha256']==manifest_sha
 geom=pool.geometry() if a.kind=='W1' else None;model=A1WorldModel(a.kind,geom,a.seed).cuda()
 for key in ['input_mean','input_std','delta_std','loss_std']:getattr(model,key).copy_(norm[key].cuda())
 opt=torch.optim.Adam(model.net.parameters(),lr=2e-4);start_update=0
 if a.resume:
  checkpoint=torch.load(a.resume,map_location='cuda',weights_only=False);assert checkpoint['manifest_sha256']==manifest_sha;load_model_state(model,checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);start_update=checkpoint['update']
 config=dict(**model.config(),seed=a.seed,updates=a.updates,microbatches=a.microbatches,microbatch_size=a.batch_size,effective_batch=a.batch_size*a.microbatches,horizon=8,learning_rate=2e-4,clip=1,normalization=str(a.normalization),manifest=str(a.manifest),manifest_sha256=manifest_sha,formal=not a.smoke,batch_seed_rule='np.default_rng(1000003*model_seed+update); identical W0/W1 object-balanced distinct-window order',precision='float32, no AMP or TF32',validation='development only; fixed test unopened',checkpoint_geometry='Shared immutable SDF files, not copied into checkpoints')
 write(out/'configuration.json',config)
 original=model.forward;model.forward=torch.compile(original,fullgraph=True,dynamic=False);started=time.monotonic();seen_updates=start_update
 fixed=out.parent/('fixed_windows_smoke_v1' if a.smoke else 'fixed_windows_v1');fixed.mkdir(exist_ok=True)
 def save(update,label=None):
  save_model(model,out/(label or f'checkpoint_{update:04d}.pt'),optimizer=opt.state_dict(),update=update,seed=a.seed,manifest_sha256=manifest_sha,normalization_metadata=norm['metadata'],training_configuration=config,formal=not a.smoke)
 def evaluate(update):
  model.forward=original
  try:
   summaries,criteria=run_evaluation(model,a.manifest,fixed,out/f'validation_{update:04d}',per_task=8 if a.smoke else 64,natural=32 if a.smoke else 4096)
   compact={k:dict(overall=v['overall'],switch=v['phases']['switch'],hand_involved=v['phases']['hand_involved'],support_only=v['phases']['support_only']) for k,v in summaries.items()}
   write(out/f'validation_summary_{update:04d}.json',dict(update=update,metrics=compact,criteria=criteria))
  finally:model.forward=torch.compile(original,fullgraph=True,dynamic=False)
 save(start_update);evaluate(start_update)
 for update in range(start_update+1,a.updates+1):
  if stop:break
  before=time.monotonic();rng=np.random.default_rng(1000003*a.seed+update)
  seen=None if a.smoke else set();ids=pool.sample_ids(a.batch_size*a.microbatches,rng,seen);digest=hashlib.sha256(ids.tobytes()).hexdigest()
  fraction=min(1,update/(a.updates*.8));lr=2e-4*(.03+.97*.5*(1+math.cos(math.pi*fraction)))
  for group in opt.param_groups:group['lr']=lr
  opt.zero_grad(set_to_none=True);loss_sum=0.;complete=True
  for micro in range(a.microbatches):
   if stop:complete=False;break
   b={k:v.cuda() for k,v in pool.batch(ids[micro*a.batch_size:(micro+1)*a.batch_size]).items()}
   pred=model.rollout(b['start'],b['commands'],b['mask'],b['context'],b['object_id']);loss=model.dynamics_loss(pred,b['target'],b['mask'])
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite full model loss')
   (loss/a.microbatches).backward();loss_sum+=float(loss.detach())/a.microbatches
  if not complete:opt.zero_grad(set_to_none=True);break
  normval=torch.nn.utils.clip_grad_norm_(model.net.parameters(),1.)
  if not torch.isfinite(normval):raise RuntimeError('Nonfinite model gradient')
  opt.step();seen_updates=update
  row=dict(update=update,loss=loss_sum,preclip_gradient_norm=float(normval),learning_rate=lr,seconds=time.monotonic()-before,cumulative_seconds=time.monotonic()-started,training_windows=update*a.batch_size*a.microbatches,model_training_transitions=update*a.batch_size*a.microbatches*8,batch_ids_sha256=digest,peak_torch_bytes=torch.cuda.max_memory_allocated())
  with (out/'learning_curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if update<=5 or update%50==0:print(json.dumps(row),flush=True)
  if update in [250,500,1000,1500,2000] or update==a.updates:
   save(update);evaluate(update)
  elif update%50==0:save(update,'checkpoint_latest.pt')
 save(seen_updates,'checkpoint_final.pt')
 write(out/'summary.json',dict(status='budget_stop' if stop else 'updates_complete',updates=seen_updates,training_windows=seen_updates*a.batch_size*a.microbatches,model_training_transitions=seen_updates*a.batch_size*a.microbatches*8,seconds=time.monotonic()-started,peak_torch_bytes=torch.cuda.max_memory_allocated(),formal=not a.smoke,test_opened=False))
if __name__=='__main__':main()
