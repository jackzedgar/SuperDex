"""Execute the user's explicit date-bounded cleanup on Lambda only."""
import os,pathlib,json,time,datetime,stat,shutil,subprocess,fcntl
ROOT=pathlib.Path(__file__).resolve().parents[1];HOME=pathlib.Path('/home/youran');OUT=ROOT/'runs/storage_cleanup_20261028'
assert subprocess.check_output(['hostname'],text=True).strip()=='Lambda'
assert subprocess.check_output(['whoami'],text=True).strip()=='youran'
cut=datetime.datetime(2026,5,1,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
plan=json.loads((OUT/'deletion_plan.json').read_text());assert not plan['scan_errors'];assert not plan['references']
allow={'RLControl2','TorchRBM.Jan25.bk','ekanshTest','ollama.md','dma','nlab-content','bfed','nltk_data','Downloads','TransformerAsOn2','ollama-install.sh','odbc18.sh','old_netket_codes','ReviewNetKet.ipynb','python-src','TorchRBM','repgpt','LLaMA-Factory','ekanshenv','data','TransformerAsOn','ScalingLaw2','Text-Embedding-Model-Renormalization'}
trash=HOME/'.local/share/Trash';dev=HOME.stat().st_dev
def check_path(p):
 assert p.is_absolute() and p!=HOME and p!=ROOT
 assert (p.parent==HOME and p.name in allow) or p==trash
 assert p.resolve()==p and not p.is_symlink()
def scan(target):
 paths=[target]
 if target.is_dir():
  for base,dirs,files in os.walk(target,followlinks=False):
   paths.extend(pathlib.Path(base,n) for n in dirs+files)
   dirs[:]=[n for n in dirs if not pathlib.Path(base,n).is_symlink()]
 for p in paths:
  s=p.lstat()
  if s.st_uid!=os.getuid() or s.st_dev!=dev:raise RuntimeError('Ownership/mount mismatch:'+str(p))
  if max(s.st_mtime,s.st_ctime)>=cut:raise RuntimeError('Protected recent change:'+str(p))
 return len(paths)
def active_reference(target):
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   args=(proc/'cmdline').read_bytes().decode(errors='replace')
   if str(target) in args:return 'active command PID'+proc.name
   maps=(proc/'maps').read_text()
   if str(target)+'/' in maps:return 'active mapping PID'+proc.name
   for fd in [proc/'cwd',proc/'exe',*list((proc/'fd').iterdir())]:
    try:
     p=fd.resolve()
     if p==target or target in p.parents:return 'active path PID'+proc.name
    except (OSError,RuntimeError):pass
  except OSError:pass
 return None
lock=open(OUT/'cleanup.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=dict(started=time.time(),host='Lambda',account='youran',cutoff=plan['cutoff'],free_before=shutil.disk_usage(ROOT).free,deleted=[],skipped=[],errors=[],authorization='User explicitly requested removing other old files while preserving anything changed after May2026; conservative cutoff2026-05-01',scope=plan['scope'])
rp=OUT/'execution_receipt.json'
def save():rp.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
save()
# Validate every exact target before the first irreversible action.
approved=[]
for item in plan['eligible']:
 p=pathlib.Path(item['path'])
 try:
  check_path(p);count=scan(p);why=active_reference(p)
  if why:raise RuntimeError(why)
  approved.append((p,item,count))
 except Exception as e:receipt['skipped'].append({'path':str(p),'reason':str(e)})
save()
for p,item,count in approved:
 try:
  check_path(p);why=active_reference(p)
  if why:raise RuntimeError(why)
  scan(p)
  if p.is_dir():shutil.rmtree(p)
  else:p.unlink()
  receipt['deleted'].append(dict(path=str(p),file_count=item['files'],preflight_entries=count,estimated_file_blocks_bytes=item['allocated_unique_file_bytes'],finished=time.time()))
  save();print('REMOVED',str(p),flush=True)
 except Exception as e:
  receipt['errors'].append({'path':str(p),'error':str(e),'partial_deletion_possible':True});save();print('STOP_ERROR',str(p),str(e),flush=True);break
os.sync();receipt.update(finished=time.time(),free_after=shutil.disk_usage(ROOT).free)
receipt['actual_free_increase']=receipt['free_after']-receipt['free_before']
receipt['all_eligible_removed']=len(receipt['deleted'])==len(plan['eligible'])
save();os.sync();print(json.dumps({k:v for k,v in receipt.items() if k!='deleted'},ensure_ascii=False),flush=True)
