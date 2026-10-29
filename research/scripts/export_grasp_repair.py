"""Export the complete current evidence, preserved baseline and a Git bundle."""
import subprocess,hashlib,json,zipfile
from grasp_lab import ROOT,RUN,OLD,DATA,write_json

OUT=RUN/'delivery';OUT.mkdir(exist_ok=True)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit before exporting.'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
bundle=OUT/'superdex.git.bundle';subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True)
files={}
for rel in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'):
    if rel:files['source/'+rel]=ROOT/rel
for f in RUN.rglob('*'):
    if f.is_file() and OUT not in f.parents:files['results/'+str(f.relative_to(RUN))]=f
for f in DATA.glob('*'):
    if f.is_file():files['processed/'+f.name]=f
for f in OLD.glob('*'):
    if f.is_file() and f.suffix in ['.npz','.json','.csv','.pt']:files['baseline_ppo/'+f.name]=f
files['superdex.git.bundle']=bundle
manifest=dict(git_commit=commit,baseline_commit='0667e7d39dbdc07b47c90a92f7031ba04f284443',status='diagnostic corrections verified; grasp unresolved; no new PPO training',new_training_updates=0,
    run_directory=str(RUN),gpu='physical GPU 0 only',requirements='Existing /lost+found/nvme/dexmimic installation with .venv-mjw and parent Wuji assets. Generated scene.xml references these absolute assets. Original 102 MB source tar remains on the workstation, hash/source in existing data manifest.',
    files=[dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in sorted(files.items())])
write_json(OUT/'artifact_manifest.json',manifest);files['artifact_manifest.json']=OUT/'artifact_manifest.json'
archive=OUT/'superdex_grasp_repair_20261026.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in sorted(files.items()):z.write(p,name)
write_json(OUT/'archive_integrity.json',dict(name=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),git_commit=commit))
print((OUT/'archive_integrity.json').read_text())
