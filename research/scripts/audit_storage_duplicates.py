"""Read-only exact duplicate checks and open-deleted-file accounting on Lambda."""
import pathlib,json,hashlib,os,collections,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/storage_inventory_20261028';start=time.time()
old=json.loads((OUT/'inventory.json').read_text());groups=collections.defaultdict(list)
for e in old['large_files']:
 p=pathlib.Path(e['path'])
 if '/site-packages/' in str(p) or '/.uv_cache/' in str(p) or p.parts[3] not in ['TorchRBM','TorchRBM.Jan25.bk','ScalingLaw2','ekanshTest','miniconda3']:continue
 groups[e['bytes']].append(p)
results=[]
for size,paths in groups.items():
 if len(paths)<2:continue
 hashes=collections.defaultdict(list)
 for p in paths:
  h=hashlib.sha256()
  with p.open('rb') as f:
   while True:
    b=f.read(8*1024*1024)
    if not b:break
    h.update(b)
  st=p.stat();hashes[h.hexdigest()].append(dict(path=str(p),inode=st.st_ino,device=st.st_dev,hardlinks=st.st_nlink,allocated=st.st_blocks*512))
 for sha,files in hashes.items():
  if len(files)<2:continue
  inodes={(x['device'],x['inode']) for x in files}
  results.append(dict(bytes_per_file=size,sha256=sha,files=files,distinct_inodes=len(inodes),duplicate_logical_bytes=max(0,len(inodes)-1)*size,note='Exact bytes equal, but path may be required by an experiment/environment. No removal or relinking performed.'))
prefix=[]
large=pathlib.Path('/home/youran/old_netket_codes/tomography_data/3x3_nx1024x2k.shot')
for suffix in ['.75','.625']:
 small=pathlib.Path(str(large)+suffix);equal=True;compared=0
 with large.open('rb') as a,small.open('rb') as b:
  while True:
   z=b.read(8*1024*1024)
   if not z:break
   w=a.read(len(z));compared+=len(z)
   if z!=w:equal=False;break
 prefix.append(dict(source=str(large),candidate=str(small),candidate_bytes=small.stat().st_size,checked_bytes=compared,exact_prefix=equal,scope='Only exact-prefix identity, not scientific dispensability; no file changed'))
opened=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  for fd in (proc/'fd').iterdir():
   try:
    target=os.readlink(fd)
    if target.endswith(' (deleted)'):
     st=fd.stat()
     if st.st_size>=100_000_000:opened.append(dict(pid=int(proc.name),target=target,bytes=st.st_size))
   except OSError:pass
 except OSError:pass
r=dict(cpu_seconds=time.time()-start,exact_duplicate_groups=results,exact_prefix_checks=prefix,open_deleted_files_over100MB=opened,deleted_files=0)
(OUT/'duplicate_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
