"""Read-only metadata audit of user's reconstructable cache; no deletion."""
import os,pathlib,json,time,collections
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/a1_budget_20261028';CACHE=pathlib.Path('/home/youran/.uv_cache/archive-v0')
start=time.time();entries=[]
for folder in CACHE.iterdir():
 if not folder.is_dir():continue
 ino={};newest=0
 for base,dirs,files in os.walk(folder,followlinks=False):
  for name in files:
   p=pathlib.Path(base)/name
   try:
    s=p.lstat()
    if not p.is_file() or p.is_symlink():continue
    newest=max(newest,s.st_mtime);k=(s.st_dev,s.st_ino)
    if k not in ino:ino[k]=[s.st_blocks*512,s.st_nlink,0]
    ino[k][2]+=1
   except OSError:pass
 entries.append(dict(path=str(folder),packages=[p.name[:-10] for p in folder.glob('*.dist-info')],allocated=sum(v[0] for v in ino.values()),exclusive=sum(v[0] for v in ino.values() if v[1]==v[2]),external_linked=sum(v[0] for v in ino.values() if v[1]>v[2]),newest_mtime=newest,age_days=(time.time()-newest)/86400 if newest else None))
links=[];errors=0;scanned=0
for base,dirs,files in os.walk('/home/youran',followlinks=False):
 if pathlib.Path(base)==CACHE:dirs[:]=[];continue
 for name in dirs+files:
  p=pathlib.Path(base)/name
  if p.is_symlink():
   try:
    target=p.resolve()
    if target.is_relative_to(CACHE):links.append(dict(path=str(p),target=str(target)))
   except (OSError,RuntimeError):errors+=1
 scanned+=1
opens=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  for fd in (proc/'fd').iterdir():
   try:
    target=str(fd.resolve())
    if target.startswith(str(CACHE)+'/'):opens.append(dict(pid=int(proc.name),target=target))
   except OSError:pass
  for line in (proc/'maps').read_text().splitlines():
   if str(CACHE)+'/' in line:opens.append(dict(pid=int(proc.name),map=line))
 except (OSError,PermissionError):pass
result=dict(entries=sorted(entries,key=lambda e:e['exclusive'],reverse=True),environment_symlinks_into_cache=links,active_account_references=opens,symlink_scan_errors=errors,directories_scanned=scanned,cpu_seconds=time.time()-start,deleted=False)
(OUT/'cache_candidates.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='entries'}));print(json.dumps(result['entries'][:18],indent=2))
