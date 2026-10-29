"""Read-only telemetry and predeclared deterioration alerts for owned runs."""
import time,json,csv,subprocess,shutil,os,signal
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/bptt_iteration_20261027'
def main():
    started=time.time();seen=set()
    f=(RUN/'gpu_telemetry.csv').open('a');w=csv.writer(f)
    if f.tell()==0:w.writerow(['unix_time','gpu','uuid','memory_used_mib','utilization_percent','temperature_c','power_w','free_disk_bytes'])
    while time.time()-started<23000:
        try:
            text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True)
            for line in text.splitlines():w.writerow([time.time(),*[a.strip() for a in line.split(',')],shutil.disk_usage(ROOT).free])
            f.flush()
            ledger=json.loads((RUN/'ledger.json').read_text())
            for arm in ['core_P_43','core_S_43','core_P_59','core_S_59']:
                path=RUN/arm/'development_curve.json'
                if not path.exists():continue
                curve=json.loads(path.read_text());last=curve[-1]
                if last['steps']<262144 or (arm,last['steps']) in seen:continue
                seen.add((arm,last['steps']));base=curve[0];checks=[]
                for r in curve[-3:]:
                    a=[];b=[]
                    for i in range(8):
                        old=np.load(RUN/arm/'development/step_000000'/f'trial_{i:02d}.npz')
                        new=np.load(RUN/arm/f'development/step_{r["steps"]:06d}'/f'trial_{i:02d}.npz')
                        n=min(len(old['position_error']),len(new['position_error']))
                        a.append([old['position_error'][:n].mean(),old['rotation_error'][:n].mean()])
                        b.append([new['position_error'][:n].mean(),new['rotation_error'][:n].mean()])
                    ratio=np.mean(b,0)/np.maximum(np.mean(a,0),1e-8)
                    first=np.mean([x['first_failure_s'] or 1.02 for x in r['trials']]);first0=np.mean([x['first_failure_s'] or 1.02 for x in base['trials']])
                    bad=r['successes']==0 and first<.8*first0 and bool((ratio>1.2).all()) and r['mean_length']<.8*base['mean_length']
                    checks.append(dict(step=r['steps'],success=r['successes'],first=first,initial_first=first0,prefix_error_ratio=ratio.tolist(),length=r['mean_length'],bad=bool(bad)))
                record=dict(arm=arm,step=last['steps'],checks=checks,trigger=len(checks)==3 and all(x['bad'] for x in checks),action='Alert; operator applies registered branch stop, no automatic recipe mutation')
                if record['trigger']:
                    job=next((j for j in ledger['jobs'] if j['name']==arm and j['status']=='running'),None)
                    if job:
                        parent=job['pid'];cp=Path(f'/proc/{parent}/task/{parent}/children');stopped=[]
                        if cp.exists():
                            for child in cp.read_text().split():
                                command=Path(f'/proc/{child}/cmdline').read_bytes().decode().replace(chr(0),' ')
                                if 'scripts/train_bptt.py --name '+arm+' ' in command:
                                    os.kill(int(child),signal.SIGTERM);stopped.append(int(child))
                        record['action']='Requested graceful stop of verified OWN training child after registered3-checkpoint deterioration';record['stopped_child_pids']=stopped
                with (RUN/'stopping_audit.jsonl').open('a') as g:g.write(json.dumps(record)+'\n')
            if (RUN/'queue_finished.json').exists():break
        except Exception as e:
            with (RUN/'telemetry_errors.log').open('a') as g:g.write(repr(e)+'\n')
        time.sleep(30)
    f.close()
if __name__=='__main__':main()

