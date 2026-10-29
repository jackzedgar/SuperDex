"""One end-of-stage archive; preserve negative runs and the actual learning system."""
import hashlib,json,subprocess,zipfile
from exploration_common import RUN,ROOT,read_json,write_json

def main():
    assert all(read_json(RUN/'artifact_validation.json')['checks'].values())
    out=RUN.with_name(RUN.name+'_delivery');out.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();bundle=out/'superdex.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True);subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True)
    paths={ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p};paths.add(bundle);paths.update(p for p in RUN.rglob('*') if p.is_file())
    cfg=read_json(RUN/'budget.json');paths.update(ROOT/r['path'] for r in read_json(RUN/'input_manifest.json')['files'])
    for folder in ['data/processed/clip000000_obj000004_right','data/processed/backup_clip000001_obj000033_right']:
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    geom=ROOT/cfg['model_geometry']
    for name in ['kinematics.npz','sdf_grid.npz','geometry.json','geometry_validation.json','dataset_manifest.json','config.json','normalization.npz']:
        if (geom/name).exists():paths.add(geom/name)
    for name in ['runs/real_track_20261026/qualification_cpu.json','runs/real_track_20261026/qualification_gpu.json','runs/grasp_repair_20261026/fixed_initials.npz']:
        paths.add(ROOT/name)
    entries=[]
    for p in sorted(paths):
        assert p.is_file() and p.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(out/'delivery_manifest.json',dict(commit=commit,source_commit=cfg['source_commit'],run=str(RUN.relative_to(ROOT)),files=entries,scope='All new policies, optimizer states, curves, fixed evaluations and sampled training episodes; shared source policy/model/geometry, processed spoon and unused backup reference, code and Git history.',omissions=['Existing parent Wuji asset installation, .venv-mjw, CUDA driver and public raw video tar files are not embedded. Physics reproduction uses the existing workstation. No claimed portable container.']))
    cost=read_json(RUN/'iteration_analysis.json')['cost']
    (out/'REPRODUCE.txt').write_text(f'''SuperDex real model-assisted PPO iteration
Commit: {commit}
Charged GPU-process time: {cost['total_gpu_process_seconds']:.3f} / 21600 seconds.

ssh root@47.100.227.205 -p 9091
cd /lost+found/nvme/dexmimic/superdex
source activate_mjw.sh
export SUPERDEX_EXPLORE_RUN="$PWD/runs/core_iteration_20261027"
python scripts/analyze_core_iteration.py
python scripts/validate_core_iteration.py
python scripts/report_core_iteration.py

Fresh bounded reproduction: bash scripts/reproduce_core_iteration.sh
Only physical GPU 0, one physical world. The wrapper creates new output, refuses to overwrite
existing runs, charges child-process wall time and stops at its allocation. Individual executed
commands, checkpoints, all three RNG streams and frozen settings are retained. Fresh physical
contact trajectories are not promised bitwise identical. The new run needs its own interpretation.

The ZIP uses project-relative paths. Open runs/core_iteration_20261027/report.html.
Pure P and model-assisted M use identical base PPO and real interaction budgets. M's additional
model predictions and process cost are listed separately. No old-model and refreshed-model stages
are silently merged. Original and failed development evidence is retained.
''',encoding='utf-8')
    archive=out/'SuperDex-core-iteration-20261027.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
        for name in ['delivery_manifest.json','REPRODUCE.txt']:z.write(out/name,name)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest();(out/'SHA256SUMS.txt').write_text(sha+'  '+archive.name+'\n')
    print(json.dumps(dict(commit=commit,files=len(entries),bytes=archive.stat().st_size,sha256=sha)))

if __name__=='__main__':main()
