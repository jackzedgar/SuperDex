"""Isolated copies of frozen inputs for tiny existing training-entry checks."""
import json,shutil,hashlib,argparse,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',default='runs/lambda_migration_20261027');args=parser.parse_args()
 run=ROOT/args.run;run.mkdir(parents=True,exist_ok=True);core=ROOT/'runs/core_iteration_20261027';out=run/'smoke';out.mkdir(exist_ok=False)
 cfg=json.loads((core/'budget.json').read_text());cfg.update(stage='migration_smoke_only',source_gpu_uuid=cfg['gpu_uuid'],gpu_uuid=os.environ['SUPERDEX_GPU_UUID'],total_gpu_process_seconds_cap=1200,original_budget_is_not_authorization_for_training=True)
 (out/'budget.json').write_text(json.dumps(cfg,indent=2)+'\n');shutil.copy2(core/'development_initials.npz',out/'development_initials.npz')
 wm=run/'wm_smoke';(wm/'wm').mkdir(parents=True,exist_ok=False);original=ROOT/'runs/learning_trend_20261027';manifest=json.loads((original/'input_manifest.json').read_text())
 for row in manifest['frozen_wm_files']:
  name=row['name'];source=original/'wm'/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256'];shutil.copy2(source,wm/'wm'/name)
 (wm/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(ppo=str(out),world_model=str(wm),frozen_wm_files=len(manifest['frozen_wm_files']),no_original_checkpoint_overwrite=True)))
if __name__=='__main__':main()
