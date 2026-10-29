"""Pinned public source retrieval for full A1; no images or whole-dataset mirrors."""
import argparse,hashlib,json,time,urllib.request,urllib.error,tarfile,io,pickle,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/a1_full_20261028'
SOURCES={'oakink2':('kelvin34501/OakInk-v2','21705616140d726607027e70d58b7837f442ffd8'),'hot3d':('bop-benchmark/hot3d','30fe9674782f32e1e5edba98476b6ff4300132c5')}
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');t.replace(path)
def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'SuperDex-A1-research/1.0'}),timeout=90) as f:
        return json.load(f),f.headers.get('Link','')
def tree(source,folder=''):
    repo,rev=SOURCES[source];url=f'https://huggingface.co/api/datasets/{repo}/tree/{rev}'
    if folder:url+='/'+folder
    url+='?limit=1000';rows=[]
    while url:
        data,link=get_json(url);rows.extend(data);url=None
        for item in link.split(','):
            if 'rel="next"' in item:url=item.split('<')[1].split('>')[0]
    write(RUN/'public_catalog'/f'{source}_{folder.replace("/","_") or "root"}.json',dict(repo=repo,revision=rev,entries=rows))
    return rows
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def fetch(source,entry):
    import shutil
    repo,rev=SOURCES[source];rel=Path(entry['path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    dest=ROOT/'data/raw'/source/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{entry["path"]}'
    start=time.monotonic();partial=dest.with_name(dest.name+'.partial')
    if not dest.exists():
        for attempt in range(4):
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                if shutil.disk_usage(ROOT).free<5_000_000_000+entry['size']-offset:raise OSError('A1 disk reserve')
                headers={'User-Agent':'SuperDex-A1-research/1.0'}
                if offset:headers['Range']=f'bytes={offset}-'
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as src:
                    append=offset>0 and src.status==206
                    if append:assert src.headers['Content-Range'].startswith(f'bytes {offset}-')
                    with partial.open('ab' if append else 'wb') as out:
                        while b:=src.read(2**20):out.write(b)
                assert partial.stat().st_size==entry['size'],(partial.stat().st_size,entry['size'])
                partial.replace(dest);break
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    assert dest.stat().st_size==entry['size'],str(dest)
    sha=digest(dest);expected=entry.get('lfs',{}).get('oid')
    if expected:assert sha==expected,(dest,sha,expected)
    else:
        b=dest.read_bytes();assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==entry['oid']
    receipt={'source':source,'url':url,'revision':rev,'bytes':dest.stat().st_size,'sha256':sha,'expected_lfs_sha256':expected,'download_and_verify_cpu_wall_seconds':time.monotonic()-start,'anonymous':True}
    write(dest.with_name(dest.name+'.receipt.json'),receipt)
    print(json.dumps(receipt),flush=True);return dest
def extract_data_tar(path,out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=True);manifest=[]
    with tarfile.open(path) as tf:
        for member in tf:
            if not member.isfile():continue
            rel=Path(member.name)
            assert not rel.is_absolute() and '..' not in rel.parts
            dest=(out/rel).resolve();assert dest.is_relative_to(out)
            data=tf.extractfile(member).read();dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert digest(dest)==hashlib.sha256(data).hexdigest()
            else:dest.write_bytes(data)
            manifest.append({'path':str(dest.relative_to(ROOT)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    write(out/(path.name+'.extraction_manifest.json'),manifest);return manifest
def safe_pickle(path):
    # Public annotations are data. Reject arbitrary classes and nested unsafe Torch pickle loads.
    import numpy as np,torch
    allowed={('collections','OrderedDict'):collections.OrderedDict,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
    for module in ['numpy.core.multiarray','numpy._core.multiarray']:
        allowed[(module,'_reconstruct')]=np._core.multiarray._reconstruct
        allowed[(module,'scalar')]=np._core.multiarray.scalar
    for name in ['_rebuild_tensor','_rebuild_tensor_v2','_rebuild_parameter']:
        allowed[('torch._utils',name)]=getattr(torch._utils,name)
    allowed[('torch.storage','_load_from_bytes')]=lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
    class Reader(pickle.Unpickler):
        def find_class(self,module,name):
            if (module,name) not in allowed:raise pickle.UnpicklingError(f'Blocked class {module}.{name}')
            return allowed[module,name]
    with path.open('rb') as f:return Reader(f).load()
def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',action='store_true');p.add_argument('--oakink-prerequisites',action='store_true');p.add_argument('--inspect-annotation',type=Path);a=p.parse_args()
    if a.catalog:
        for source,folders in [('oakink2',['','anno_preview']),('hot3d',['','train_quest3','object_models_eval'])]:
            for folder in folders:
                rows=tree(source,folder);print(source,folder,'entries',len(rows),flush=True)
    if a.oakink_prerequisites:
        rows=tree('oakink2');wanted=['program.tar','program_extension.tar','object_raw.tar','object_repair.tar']
        for name in wanted:
            entry=next(r for r in rows if r['path']==name);dest=fetch('oakink2',entry);extract_data_tar(dest,ROOT/'data/a1_sources/oakink2')
        ann=tree('oakink2','anno_preview');entry=min([r for r in ann if r['type']=='file'],key=lambda r:r['size'])
        path=fetch('oakink2',entry);d=safe_pickle(path)
        summary={k:{'type':type(v).__name__,'length':len(v) if hasattr(v,'__len__') else None,'example_keys':list(v)[:4] if isinstance(v,dict) else None} for k,v in d.items()}
        write(RUN/'public_catalog/oakink_annotation_schema.json',{'source':str(path.relative_to(ROOT)),'schema':summary})
        print(json.dumps(summary),flush=True)
    if a.inspect_annotation:
        d=safe_pickle(a.inspect_annotation);print({k:type(v).__name__ for k,v in d.items()})
if __name__=='__main__':main()
