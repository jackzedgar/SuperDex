"""Prepare date-bounded cleanup. Read-only except this task's audit files."""
import os,pathlib,json,datetime,time,stat,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1];HOME=pathlib.Path('/home/youran');OUT=ROOT/'runs/storage_cleanup_20261028'
assert subprocess.check_output(['hostname'],text=True).strip()=='Lambda'
assert subprocess.check_output(['whoami'],text=True).strip()=='youran'
cut=datetime.datetime(2026,5,1,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
protected_names={'projects','miniconda3','snap','Desktop'}
candidates=[];protected=[];dev=HOME.stat().st_dev;started=time.time()
for target in [p for p in HOME.iterdir() if not p.name.startswith('.')]+[HOME/'.local/share/Trash']:
 if target.name in protected_names:
  protected.append({'path':str(target),'reason':'Current project, account/base runtime or desktop directory'});continue
 if target.is_symlink():
  protected.append({'path':str(target),'reason':'Top-level symlink; no target traversal'});continue
 st=target.lstat()
 if max(st.st_mtime,st.st_ctime)>=cut:
  protected.append({'path':str(target),'reason':'Root modified or metadata-changed at/after2026-05-01'});continue
 reasons=[];count=0;allocated=0;logical=0;newest=0;newest_path=None;inodes=set()
 paths=[target]
 if target.is_dir():
  for base,dirs,files in os.walk(target,followlinks=False):
   paths.extend(pathlib.Path(base,n) for n in dirs+files)
   dirs[:]=[d for d in dirs if not pathlib.Path(base,d).is_symlink()]
 for p in paths:
  try:s=p.lstat()
  except OSError as e:reasons.append('metadata error:'+str(p)+':'+str(e));continue
  if s.st_uid!=os.getuid():reasons.append('other owner:'+str(p))
  if s.st_dev!=dev:reasons.append('other mount:'+str(p))
  changed=max(s.st_mtime,s.st_ctime)
  if changed>newest:newest=changed;newest_path=str(p)
  if changed>=cut:reasons.append('recent descendant:'+str(p))
  if stat.S_ISREG(s.st_mode):
   count+=1;logical+=s.st_size;key=(s.st_dev,s.st_ino)
   if key not in inodes:allocated+=s.st_blocks*512;inodes.add(key)
 if reasons:protected.append({'path':str(target),'reason_count':len(reasons),'reason_examples':reasons[:8]});continue
 candidates.append(dict(path=str(target),type='directory' if target.is_dir() else 'file',files=count,logical_bytes=logical,allocated_unique_file_bytes=allocated,newest_change_timestamp=newest,newest_change_path=newest_path))
roots=[pathlib.Path(c['path']) for c in candidates]
def candidate_for(path):
 for p in roots:
  if path==p or p in path.parents:return str(p)
 return None
references=[];scan_errors=[]
for base,dirs,files in os.walk(HOME,followlinks=False):
 if candidate_for(pathlib.Path(base)):dirs[:]=[];continue
 dirs[:]=[n for n in dirs if n not in ['.ssh','.aws','.gnupg','.git'] and not pathlib.Path(base,n).is_symlink()]
 for n in files:
  p=pathlib.Path(base,n)
  try:
   if p.is_symlink():
    resolved=p.resolve();target=candidate_for(resolved)
    if target:references.append(dict(target=target,source=str(p),kind='symlink'))
   elif n in ['pyvenv.cfg','direct_url.json'] or p.suffix in ['.pth','.egg-link']:
    if p.stat().st_size<100000:
     text=p.read_text(errors='replace')
     for target in roots:
      if str(target) in text:references.append(dict(target=str(target),source=str(p),kind='environment_dependency'))
  except (OSError,RuntimeError) as e:scan_errors.append(str(p)+':'+str(e))
 # Check preserved directory symlinks too, without following them.
 try:
  for q in pathlib.Path(base).iterdir():
   if q.is_symlink():
    target=candidate_for(q.resolve())
    if target:references.append(dict(target=target,source=str(q),kind='symlink'))
 except (OSError,RuntimeError):pass
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  args=(proc/'cmdline').read_bytes().decode(errors='replace')
  for target in roots:
   if str(target) in args:references.append(dict(target=str(target),pid=int(proc.name),kind='active_command'))
  for q in [proc/'cwd',proc/'exe',*list((proc/'fd').iterdir())]:
   try:
    target=candidate_for(q.resolve())
    if target:references.append(dict(target=target,pid=int(proc.name),kind='active_path'))
   except (OSError,RuntimeError):pass
  maps=(proc/'maps').read_text()
  for target in roots:
   if str(target)+'/' in maps:references.append(dict(target=str(target),pid=int(proc.name),kind='active_mapping'))
 except OSError:pass
blocked={r['target'] for r in references}
for c in candidates:
 if c['path'] in blocked:protected.append(dict(path=c['path'],reason='Referenced by retained environment/path or running process'))
eligible=[c for c in candidates if c['path'] not in blocked]
plan=dict(host='Lambda',account='youran',user_instruction='2026年5月后改动过的别动，其他全部清理',scope='/home/youran non-hidden old project/data/source items and old desktop trash only',cutoff='2026-05-01T00:00:00+08:00',cutoff_note='Conservative: preserve all May2026 changes as well. Inspect mtime AND ctime, including Git internals; any recent descendant preserves whole target.',created=time.time(),eligible=eligible,protected=protected,references=references,scan_errors=scan_errors,estimated_unique_allocated_file_bytes=sum(e['allocated_unique_file_bytes'] for e in eligible),cpu_seconds=time.time()-started,deleted=False)
(OUT/'deletion_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in plan.items() if k not in ['protected','references']},ensure_ascii=False,indent=2))
print('PROTECTED',json.dumps(protected,ensure_ascii=False))
print('REFERENCES',json.dumps(references,ensure_ascii=False))
