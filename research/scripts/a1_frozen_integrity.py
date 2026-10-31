"""Verify frozen data bytes before reading windows; no held-out metric access."""
import hashlib,json
from pathlib import Path

_VERIFIED={}

def _path(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Frozen path must be relative and contained: '+str(relative))
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Frozen path escapes root: '+str(relative))
    return path

def _stamp(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def verify_file(path,digest,size=None):
    path=Path(path);before=_stamp(path)
    if size is not None and before[2]!=size:raise ValueError('Frozen size mismatch: '+str(path))
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid frozen digest: '+str(path))
    key=(str(path.resolve()),digest)
    if _VERIFIED.get(key)==before:return
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    if _stamp(path)!=before:raise ValueError('Frozen file changed while hashing: '+str(path))
    if h.hexdigest()!=digest:raise ValueError('Frozen SHA256 mismatch: '+str(path))
    _VERIFIED[key]=before

def verify_frozen_manifest(root,meta):
    root=Path(root)
    formal=bool(meta.get('formal'))
    if not formal and meta.get('status')!='FROZEN_SMOKE_DATA':
        return {'verified':False,'reason':'Unfrozen nonformal fixture; not accepted as formal A1 data'}
    if formal and (meta.get('status')!='FROZEN_FULL_A1_DATA_V1' or meta.get('frozen_integrity_version')!=1):
        raise ValueError('Formal data requires complete frozen integrity metadata')
    if not meta.get('shards'):raise ValueError('Frozen manifest has no shards')
    count=0
    for row in meta['shards']:
        shard=_path(root,row['path']);files=row['files']
        record=files['schema.json'];verify_file(shard/'schema.json',record['sha256'],record['bytes'])
        schema=json.loads((shard/'schema.json').read_text())
        expected={'schema.json','episodes.jsonl'}|{key+'.bin' for key in schema['columns']}
        if set(files)!=expected:raise ValueError('Frozen shard does not bind every column: '+str(shard))
        if any(Path(name).name!=name for name in expected):raise ValueError('Invalid column name')
        for name,r in files.items():verify_file(shard/name,r['sha256'],r['bytes']);count+=1
        for key in ['task_id','object_key','quality']:
            if key in row and schema[key]!=row[key]:raise ValueError('Frozen shard identity mismatch: '+key)
    if set(meta['geometry'])!=set(meta['object_keys']):raise ValueError('Frozen object geometry coverage mismatch')
    for key,digest in meta['geometry'].items():
        verify_file(_path(root,'data/a1_geometry/objects/'+key+'/sdf.npz'),digest);count+=1
    verify_file(root/'configs/a1_probe40_v1.json',meta['probe40_sha256']);count+=1
    hand=meta.get('hand_kinematics',{})
    if formal and set(hand)!={'right','left'}:raise ValueError('Formal geometry must bind both hand kinematics')
    for side,r in hand.items():
        if side not in ['right','left']:raise ValueError('Unknown hand asset')
        verify_file(root/f'data/a1_processed/assets/hands/{side}/kinematics.npz',r['sha256'],r['bytes']);count+=1
    return {'verified':True,'formal':formal,'files':count,'cache':'Per process only; inode,size,mtime_ns,ctime_ns rechecked on every verification'}
