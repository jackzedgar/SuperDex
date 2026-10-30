"""Export this run plus required frozen inputs and a recoverable Git bundle."""
import json,hashlib,subprocess,zipfile
from pathlib import Path
from exploration_common import RUN,ROOT,read_json,write_json


def main():
    assert read_json(RUN/'artifact_validation.json')['checks']['budget_and_process_completion']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();out=RUN.with_name(RUN.name+'_delivery');out.mkdir(exist_ok=True)
    bundle=out/'superdex.bundle';subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True);subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True)
    paths=set(ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT,text=True).split('\0') if p);paths.update(p for p in RUN.rglob('*') if p.is_file());paths.add(bundle)
    manifest=read_json(RUN/'input_manifest.json');paths.update(ROOT/r['path'] for r in manifest['source_files']);paths.update(ROOT/r['source'] for r in manifest['frozen_wm_files']);old=ROOT/manifest['source_run']
    for arm in ['uniform_rsi','start50']:paths.update((old/arm).glob('checkpoint*.pt'))
    for name in ['W0_final.pt','W1_final.pt','normalization.npz','config.json','training.csv','evaluation.json']:paths.add(old/'wm'/name)
    paths.update(p for p in (ROOT/'data/processed/clip000000_obj000004_right').rglob('*') if p.is_file())
    paths.update(ROOT/p for p in ['runs/real_track_20261026/ppo/checkpoint.pt','runs/real_track_20261026/qualification_cpu.json','runs/real_track_20261026/qualification_gpu.json'])
    entries=[]
    for p in sorted(paths):
        assert p.is_file() and p.resolve().is_relative_to(ROOT)
        entries.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(out/'delivery_manifest.json',dict(commit=commit,baseline_commit=manifest['baseline_commit'],run=str(RUN.relative_to(ROOT)),files=entries,scope='All new training/evaluation logs and checkpoints, frozen WM inputs, prior checkpoints needed for full-curve replay, processed reference/collision assets, tracked code and Git history.',omissions=['Parent Wuji installation, Python environment, CUDA driver and original raw HOT3D tar are not embedded. Physics reproduction uses the existing workstation installation.']))
    ledger=read_json(RUN/'compute_ledger.json')
    (out/'REPRODUCE.txt').write_text(f'''SuperDex bounded learning trend, 2026-10-27\nCommit: {commit}\nSource experiment: b93903f / runs/explore_20261026\nGPU-process cap: 3600 s; observed use: {ledger['total_charged_seconds']:.3f} s.\n\nExisting workstation:\nssh root@47.100.227.205 -p 9091\ncd /lost+found/nvme/dexmimic/superdex\nsource activate_mjw.sh\nexport SUPERDEX_EXPLORE_RUN="$PWD/runs/learning_trend_20261027"\npython scripts/validate_learning_trend.py\n\nFresh bounded reproduction:\nbash scripts/reproduce_learning_trend.sh\n\nOnly GPU 0; only one physics world executes at a time. The script restores each previous PPO policy/optimizer, preserves frozen WM inputs, creates fresh outputs, and does not automatically add experiments. Child-process hard limits enforce the allocation. The shell wrapper was syntax-checked; its constituent learning and evaluation commands were executed in this completed run. Fresh runs require fresh interpretation.\n\nThis ZIP preserves project-relative paths. Open the self-contained HTML report under runs/learning_trend_20261027/report.html. All data/weights/source files have SHA-256 entries in delivery_manifest.json. Existing parent Wuji assets and independent .venv-mjw are required for physics reruns; they are not embedded. New PPO episodes are retained separately and were not mixed into this WM dataset.\n''')
    archive=out/'SuperDex-learning-trend-20261027.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(paths):z.write(p,str(p.relative_to(ROOT)))
        for name in ['delivery_manifest.json','REPRODUCE.txt']:z.write(out/name,name)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest();(out/'SHA256SUMS.txt').write_text(sha+'  '+archive.name+'\n');print(json.dumps(dict(commit=commit,files=len(entries),bytes=archive.stat().st_size,sha256=sha),indent=2))


if __name__=='__main__':main()
