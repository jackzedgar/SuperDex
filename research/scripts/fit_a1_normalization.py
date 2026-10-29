"""CPU normalization fitted exclusively on frozen training episodes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from a1_window_pool import WindowPool
from a1_world_model import StreamingMoments,common_features,COMMON_INPUT
from a1_state import delta_between,encode_dynamics,DELTA,DYNAMICS
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=524288);a=p.parse_args()
 assert not a.output.exists();torch.set_num_threads(4);pool=WindowPool(a.manifest,'train',1);rng=np.random.default_rng(280052)
 inp=StreamingMoments(COMMON_INPUT);delta=StreamingMoments(DELTA);loss=StreamingMoments(DYNAMICS);digest=hashlib.sha256()
 for start in range(0,a.samples,1024):
  ids=pool.sample_ids(min(1024,a.samples-start),rng);digest.update(ids.tobytes());b=pool.batch(ids)
  inp.add(common_features(b['start'],b['commands'][:,0],b['mask'],b['context']));delta.add(delta_between(b['start'],b['target'][:,0],b['mask']));loss.add(encode_dynamics(b['target'][:,0]))
 input_std=inp.std(1e-3);raw_std=inp.std(1e-8)
 # Inertia is measured in kg*m^2 (~1e-7..1e-3), not metres/radians.
 # A universal1e-3 floor would erase most of its physical context variation.
 inertia_slice=slice(COMMON_INPUT-276+142,COMMON_INPUT-276+145);input_std[inertia_slice]=raw_std[inertia_slice]
 torch.save(dict(input_mean=inp.mean.float(),input_std=input_std,delta_std=delta.std(1e-4),loss_std=loss.std(1e-3),metadata=dict(samples=a.samples,seed=280052,split='train',sampling='object-balanced physical transitions',window_ids_sha256=digest.hexdigest(),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),floors=dict(common_input=1e-3,object_inertia_kg_m2=1e-8,delta=1e-4,loss=1e-3),normalization_version=2,scope=pool.meta['status'])),a.output)
 print(a.output,flush=True)
if __name__=='__main__':main()
