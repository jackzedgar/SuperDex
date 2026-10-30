"""Export source, git history, processed references, checkpoint and evidence.

The 102 MB original tar remains on the workstation; the manifest gives a pinned,
verified download. Original Wuji assets remain in the parent installation.
"""
from pathlib import Path
import subprocess,zipfile,json,hashlib

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/real_track_20261026'
OUT=RUN/'delivery';OUT.mkdir(exist_ok=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit source changes before exporting a Git-linked artifact.'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
subprocess.run(['git','bundle','create',str(OUT/'superdex.git.bundle'),'--all'],cwd=ROOT,check=True)
files={}
for rel in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'):
    if rel:files['source/'+rel]=ROOT/rel
for f in (ROOT/'data/processed/clip000000_obj000004_right').glob('*'):
    if f.is_file():files['processed/'+f.name]=f
for f in RUN.rglob('*'):
    if f.is_file() and OUT not in f.parents and f.name!='delivery_export.json':files['results/'+str(f.relative_to(RUN))]=f
raw=ROOT/'data/raw/hot3d'
for rel in ['clip_splits.json','object_models_eval/models_info.json','object_models_eval/obj_000004.glb','hot3d_dataset_license_agreement.pdf','download_clip_000000.json','download_mesh_000004.json']:
    files['raw_metadata/'+rel]=raw/rel
files['superdex.git.bundle']=OUT/'superdex.git.bundle'
manifest=dict(git_commit=commit,base_commit='8b6637221b3fa2f73f5c4a3362790fa8d388145b',scope='one clip, one object, one environment, one bounded PPO seed',
              omitted=['Original 102440960-byte clip tar stays at data/raw/hot3d/train_quest3/clip-000000.tar; download via fetch_minimal.py.',
                       'Wuji URDF and STL assets are in the existing parent installation; generated scene contains absolute workstation paths. Regenerate it with prepare_real.py on another installation.'],
              files=[dict(path=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for name,f in sorted(files.items())])
(OUT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files['artifact_manifest.json']=OUT/'artifact_manifest.json'
archive=OUT/'superdex_real_pilot_20261026.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,f in sorted(files.items()):z.write(f,name)
print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),git_commit=commit),indent=2))
