"""Delete only audited old, unreferenced, public-wheel uv cache entries."""
import os,pathlib,json,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/a1_budget_20261028'
BASE=pathlib.Path('/home/youran/.uv_cache').resolve();CACHE=BASE/'archive-v0'
a=json.loads((OUT/'cache_candidates.json').read_text());assert not a['active_account_references'];assert a['symlink_scan_errors']==0
links=a['environment_symlinks_into_cache'];chosen=[]
for e in a['entries']:
 p=pathlib.Path(e['path']);assert p.resolve().parent==CACHE and not p.is_symlink()
 refs=[x for x in links if x['target']==str(p) or x['target'].startswith(str(p)+'/')]
 external=[x for x in refs if not pathlib.Path(x['path']).is_relative_to(BASE)]
 public=[x for x in refs if pathlib.Path(x['path']).is_relative_to(BASE/'wheels-v6/pypi') or pathlib.Path(x['path']).is_relative_to(BASE/'wheels-v6/index')]
 if e['age_days']<60 or not e['packages'] or e['external_linked'] or e['exclusive']!=e['allocated'] or external or not public:continue
 if any('/sdists-v9/editable/' in x['path'] or '/sdists-v9/git/' in x['path'] for x in refs):continue
 # Recheck owner, links, and modified time immediately before deletion.
 safe=True
 for base,dirs,files in os.walk(p,followlinks=False):
  for n in files:
   f=pathlib.Path(base)/n;s=f.lstat()
   if s.st_uid!=os.getuid() or (f.is_file() and not f.is_symlink() and s.st_nlink!=1) or time.time()-s.st_mtime<60*86400:safe=False;break
  if not safe:break
 if safe:chosen.append(dict(**e,index_symlinks=[x['path'] for x in refs if pathlib.Path(x['path']).is_relative_to(BASE)]))
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  args=(proc/'cmdline').read_bytes().decode().split('\0')
  if args and pathlib.Path(args[0]).name in ['uv','pip','pip3']:raise RuntimeError('Active package installer; no cleanup')
  maps=(proc/'maps').read_text()
  for e in chosen:
   if e['path']+'/' in maps:raise RuntimeError('Active cached library; no cleanup')
  for fd in (proc/'fd').iterdir():
   try:
    target=str(fd.resolve())
    if any(target.startswith(e['path']+'/') for e in chosen):raise RuntimeError('Open cache entry; no cleanup')
   except (FileNotFoundError,PermissionError):pass
 except (FileNotFoundError,PermissionError):pass
before=shutil.disk_usage(ROOT).free
plan=dict(created=time.time(),authorized_scope='User permitted old reconstructable, inactive cache cleanup on Lambda only',entries=chosen,bytes_before=before,expected_allocated_release=sum(e['allocated'] for e in chosen),deleted=[])
path=OUT/'cleanup_receipt.json';path.write_text(json.dumps(plan,indent=2)+'\n')
for e in chosen:
 p=pathlib.Path(e['path']);assert p.resolve().parent==CACHE and str(p).startswith('/home/youran/.uv_cache/archive-v0/')
 for x in e['index_symlinks']:
  q=pathlib.Path(x);assert q.is_relative_to(BASE)
  if q.is_symlink() and (str(q.resolve())==str(p) or str(q.resolve()).startswith(str(p)+'/')):q.unlink()
 shutil.rmtree(p);plan['deleted'].append(e['path']);path.write_text(json.dumps(plan,indent=2)+'\n')
os.sync();plan.update(free_after=shutil.disk_usage(ROOT).free,observed_free_increase=shutil.disk_usage(ROOT).free-before,completed=time.time(),preserved='All external-hardlinked entries, environment references, editable/git cache, environments, experiments, data and checkpoints',recovery='Re-download the exact named/versioned wheel through uv/pip; package/version/index symlink recorded per entry')
path.write_text(json.dumps(plan,indent=2)+'\n');os.sync()
print(json.dumps({k:v for k,v in plan.items() if k not in ['entries','deleted']}));print('deleted_entries',len(plan['deleted']))
