"""Hard shared 1200-second process budget for migration-only GPU validation."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
 folder=ROOT/'runs/lambda_migration_20261027';folder.mkdir(parents=True,exist_ok=True);ledger=folder/'gpu_ledger.json'
 records=json.loads(ledger.read_text()) if ledger.exists() else []
 assert not any(r.get('status')=='running' for r in records),'Resolve any interrupted accounting record before another launch.'
 used=sum(r['process_seconds'] for r in records)
 assert a.seconds>0 and used+a.seconds+6<=1200,(used,a.seconds)
 assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
 assert not any(os.environ['CUDA_VISIBLE_DEVICES'] in line for line in apps.splitlines()),'Selected GPU has an active compute process; do not interrupt it.'
 command=a.command[1:] if a.command[:1]==['--'] else a.command
 record=dict(name=a.name,command=command,limit_seconds=a.seconds,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running',process_seconds=0,gpu=os.environ['CUDA_VISIBLE_DEVICES']);records.append(record)
 def save():
  temp=ledger.with_suffix('.tmp');temp.write_text(json.dumps(records,indent=2)+'\n');temp.replace(ledger)
 start=time.monotonic();save()
 with (folder/(a.name+'.stdout.log')).open('w') as out:
  proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=proc.pid;save()
  try:code=proc.wait(timeout=a.seconds);reason='completed'
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:code=proc.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
   reason='hard_time_budget'
  finally:record.update(process_seconds=time.monotonic()-start,status='finished',returncode=proc.returncode,stop=locals().get('reason','exception'));save()
 print(json.dumps(record,indent=2));print('total_gpu_process_seconds',sum(r['process_seconds'] for r in records))
 raise SystemExit(code)
if __name__=='__main__':main()
