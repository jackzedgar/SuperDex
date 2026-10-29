"""Final locked evaluation only after the model-selection decision is frozen."""
import argparse,json,hashlib,time
from pathlib import Path
import torch
from a1_data_sources import ROOT,RUN,write
from a1_window_pool import WindowPool,load_model_state
from a1_world_model import A1WorldModel
from a1_prediction_eval import run_evaluation
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--decision',type=Path,required=True);p.add_argument('--run',required=True);a=p.parse_args()
 decision=json.loads(a.decision.read_text());assert decision['status']=='FROZEN_BEFORE_LOCKED_TEST'
 checksum=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();assert checksum in decision['checkpoint_sha256s']
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert checkpoint['formal'];assert checkpoint['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest()
 pool=WindowPool(a.manifest,'test',8);kind=checkpoint['model_configuration']['kind'];geom=pool.geometry() if kind=='W1' else None
 model=A1WorldModel(kind,geom,checkpoint['seed']).cuda();load_model_state(model,checkpoint['model']);model.eval();torch.set_num_threads(4)
 started=time.monotonic();out=RUN/a.run;out.mkdir(exist_ok=False)
 summaries,criteria=run_evaluation(model,a.manifest,RUN/'fixed_windows_v1',out,split='test')
 write(out/'evaluation_provenance.json',dict(checkpoint=str(a.checkpoint),checkpoint_sha256=checksum,decision=str(a.decision),decision_sha256=hashlib.sha256(a.decision.read_bytes()).hexdigest(),manifest_sha256=checkpoint['manifest_sha256'],kind=kind,seed=checkpoint['seed'],update=checkpoint['update'],seconds=time.monotonic()-started,peak_torch_bytes=torch.cuda.max_memory_allocated(),model_transitions=sum(v['overall']['n']*v['horizon'] for v in summaries.values()),physical_steps=0,criteria=criteria))
if __name__=='__main__':main()
