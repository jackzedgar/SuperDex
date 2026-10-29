"""Lambda account metadata inventory only. Never follows links or removes files."""
import os,pathlib,json,time,datetime,collections,stat,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1];HOME=pathlib.Path('/home/youran')
assert subprocess.check_output(['hostname'],text=True).strip().lower()=='lambda'
assert subprocess.check_output(['whoami'],text=True).strip()=='youran'
OUT=ROOT/'runs/storage_inventory_20261028';start=time.time();dev=HOME.stat().st_dev
tz=datetime.timezone(datetime.timedelta(hours=8))
cutoffs={label:datetime.datetime.fromisoformat(date).replace(tzinfo=tz).timestamp() for label,date in [('before_2025_10_28','2025-10-28'),('before_2026_04_28','2026-04-28'),('before_2026_07_28','2026-07-28')]}
groups={};large=[];seen=set();age={k:{'files':0,'logical_bytes':0,'unique_allocated_bytes':0} for k in cutoffs};errors=collections.Counter();total_files=0
skip={'.ssh','.aws','.gnupg','.kube','.git'}
for base,dirs,files in os.walk(HOME,followlinks=False):
 good=[]
 for name in dirs:
  p=pathlib.Path(base)/name
  try:
   if name not in skip and not p.is_symlink() and p.stat().st_dev==dev:good.append(name)
  except OSError:errors['directory_stat']+=1
 dirs[:]=good
 for name in files:
  p=pathlib.Path(base)/name
  try:
   s=p.lstat()
   if not stat.S_ISREG(s.st_mode) or s.st_dev!=dev or s.st_uid!=os.getuid():continue
   total_files+=1;key=(s.st_dev,s.st_ino);unique=key not in seen;seen.add(key)
   rel=p.relative_to(HOME);parts=rel.parts
   for level in [1,2,3]:
    if len(parts)<=level:continue
    group=str(HOME.joinpath(*parts[:level]));g=groups.setdefault(group,{'files':0,'logical_bytes':0,'allocated_file_blocks_bytes':0,'oldest_mtime':s.st_mtime,'newest_mtime':s.st_mtime,'before_2026_04_28_bytes':0,'before_2025_10_28_bytes':0,'hardlinked_logical_bytes':0})
    g['files']+=1;g['logical_bytes']+=s.st_size;g['allocated_file_blocks_bytes']+=s.st_blocks*512;g['oldest_mtime']=min(g['oldest_mtime'],s.st_mtime);g['newest_mtime']=max(g['newest_mtime'],s.st_mtime)
    for label in ['before_2026_04_28','before_2025_10_28']:
     if s.st_mtime<cutoffs[label]:g[label+'_bytes']+=s.st_size
    if s.st_nlink>1:g['hardlinked_logical_bytes']+=s.st_size
   for label,cut in cutoffs.items():
    if s.st_mtime<cut:age[label]['files']+=1;age[label]['logical_bytes']+=s.st_size;age[label]['unique_allocated_bytes']+=s.st_blocks*512 if unique else 0
   if s.st_size>=100_000_000:
    large.append(dict(path=str(p),bytes=s.st_size,allocated_bytes=s.st_blocks*512,mtime=s.st_mtime,modified=datetime.datetime.fromtimestamp(s.st_mtime,tz).isoformat(),inode_change_time=datetime.datetime.fromtimestamp(s.st_ctime,tz).isoformat(),hardlink_count=s.st_nlink))
  except OSError:errors['file_stat']+=1
for g in groups.values():
 for label in ['oldest','newest']:g[label+'_modified']=datetime.datetime.fromtimestamp(g[label+'_mtime'],tz).isoformat()
result=dict(host='Lambda',account='youran',scope='/home/youran only; no symlinks, other mounts, other user or other node traversed',created=datetime.datetime.now(tz).isoformat(),basis='File mtime is last content modification, not last use or creation. ctime is inode metadata change, not birth. Directory dates derive from contained files; no deletion decision from age alone.',counts=total_files,errors=dict(errors),cpu_seconds=time.time()-start,age_groups=age,groups=groups,large_files=sorted(large,key=lambda x:x['bytes'],reverse=True),deleted_files=0)
(OUT/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['groups','large_files']}))
print('OLD_LARGE_FILES')
print(json.dumps([x for x in result['large_files'] if x['mtime']<cutoffs['before_2026_04_28']][:35],indent=2))
print('OLDER_DIRECTORIES')
candidates=[dict(path=p,**g) for p,g in groups.items() if g['logical_bytes']>=500_000_000 and g['newest_mtime']<cutoffs['before_2026_04_28']]
print(json.dumps(sorted(candidates,key=lambda x:x['logical_bytes'],reverse=True)[:25],indent=2))
print('LARGEST_ROOTS')
print(json.dumps([dict(path=p,**g) for p,g in sorted(groups.items(),key=lambda z:z[1]['logical_bytes'],reverse=True) if len(pathlib.Path(p).relative_to(HOME).parts)==1][:16],indent=2))
