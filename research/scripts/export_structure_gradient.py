"""All frozen inputs, checkpoints, physical probes and recoverable Git history."""
import json,hashlib,subprocess,zipfile
from exploration_common import RUN,ROOT,read_json,write_json
def main():
    audit=read_json(RUN/'artifact_validation.json');assert all(audit['checks'].values())
    out=RUN.with_name(RUN.name+'_delivery');out.mkdir(exist_ok=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();bundle=out/'superdex.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True);subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True)
    paths={ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p};paths.update(p for p in RUN.rglob('*') if p.is_file());paths.add(bundle)
    inputs=read_json(RUN/'input_manifest.json');paths.update(ROOT/r['source'] for r in inputs['files']);paths.update(ROOT/r['path'] for r in inputs['unchanged_sources']);paths.update(p for p in (ROOT/'data/processed/clip000000_obj000004_right').rglob('*') if p.is_file())
    for p in ['runs/real_track_20261026/qualification_cpu.json','runs/real_track_20261026/qualification_gpu.json']:paths.add(ROOT/p)
    entries=[]
    for p in sorted(paths):
        assert p.is_file() and p.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(out/'delivery_manifest.json',dict(commit=commit,source_commit=inputs['source_commit'],run=str(RUN.relative_to(ROOT)),files=entries,omissions=['Parent Wuji installation, Python environment, CUDA driver and original HOT3D raw tar are not embedded. Existing workstation installation is required for physics reproduction.']))
    ledger=read_json(RUN/'compute_ledger.json')
    (out/'REPRODUCE.txt').write_text(f'''SuperDex single-seed structure and gradient test
Commit: {commit}
GPU-process wall time: {ledger['total_charged_seconds']:.3f} / 3600 seconds.
A is reused from the exact previous 2000-update W1 run; its historical GPU time is not charged again.

ssh root@47.100.227.205 -p 9091
cd /lost+found/nvme/dexmimic/superdex
source activate_mjw.sh
export SUPERDEX_EXPLORE_RUN="$PWD/runs/structure_gradient_20261027"
python scripts/validate_structure_gradient.py

Fresh bounded reproduction (new directory):
bash scripts/reproduce_structure_gradient.sh

Only GPU 0 and one physical world. No second seed or policy training.
The shell wrapper was syntax-checked; constituent commands ran in this completed round.
The new output must receive a new interpretation, not automatically inherit this result.
The ZIP keeps project-relative paths. It contains frozen data, baseline source files, B weights,
all proposals and physical replays, training/evaluation CSV, report/figures and Git bundle.
Existing parent Wuji assets and independent .venv-mjw are needed for physics reruns.
''')
    archive=out/'SuperDex-structure-gradient-20261027.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
        for n in ['delivery_manifest.json','REPRODUCE.txt']:z.write(out/n,n)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest();(out/'SHA256SUMS.txt').write_text(sha+'  '+archive.name+'\n');print(json.dumps(dict(commit=commit,files=len(entries),bytes=archive.stat().st_size,sha256=sha)))
if __name__=='__main__':main()
