"""Read-only CPU progress monitor for the currently running bounded stage."""
import argparse,os,json,csv,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',default='core');a=p.parse_args();root=Path(os.environ['SUPERDEX_EXPLORE_RUN'])
while True:
    try:
        files=list((root/a.stage).glob('*/block_*/training.csv'));f=max(files,key=lambda p:p.stat().st_mtime);rows=list(csv.DictReader(f.open()));r=rows[-1]
        data=dict(stage=a.stage,active=str(f.parent.relative_to(root)),steps=int(float(r['steps'])),child_seconds=round(float(r['process_seconds']),1),kl=round(float(r['approx_kl']),5),clipped=round(float(r['action_clip_fraction']),4),model_ratio=round(float(r['model_scaled_ratio']),4))
        path=root/(a.stage+'_progress.json')
        if path.exists():
            q=json.loads(path.read_text())['checkpoints'][-1];data.update(matched_steps=q['steps'],P_success=q['summaries']['P']['successes'],M_success=q['summaries']['M']['successes'])
        data['completed_process_seconds']=round(sum(x['wall_seconds'] for x in json.loads((root/'process_ledger.json').read_text())['processes']),1)
        done=root/(a.stage+'_completion.json')
        if done.exists():data['completion']=json.loads(done.read_text())
        print(json.dumps(data),flush=True)
        if done.exists():break
    except (ValueError,IndexError,FileNotFoundError):pass
    time.sleep(55)
