"""Stream verified HOT3D train tars to compact source annotation packets; never count these as physics transitions."""
import argparse,concurrent.futures,gzip,hashlib,json,tarfile,time,urllib.request,shutil
from pathlib import Path
import numpy as np
from a1_data_sources import ROOT,RUN,SOURCES,tree,fetch,write
class DigestStream:
    def __init__(self,src):self.src=src;self.sha=hashlib.sha256();self.size=0
    def read(self,n=-1):
        b=self.src.read(n);self.sha.update(b);self.size+=len(b);return b
def collect(entry):
    repo,rev=SOURCES['hot3d'];stem=Path(entry['path']).stem
    root=ROOT/'data/a1_sources/hot3d'/stem;root.mkdir(parents=True,exist_ok=True)
    dest=root/'annotations.json.gz';receipt=root/'source_receipt.json'
    if dest.exists() and receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==entry['lfs']['oid']
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==r['packet_sha256'];return r
    url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{entry["path"]}'
    for attempt in range(3):
        start=time.monotonic()
        try:
            assert shutil.disk_usage(ROOT).free>5_000_000_000
            payload={};samples={}
            with urllib.request.urlopen(url,timeout=120) as src:
                digest=DigestStream(src)
                with tarfile.open(fileobj=digest,mode='r|') as tf:
                    for m in tf:
                        if not m.isfile():continue
                        if m.name.endswith('.json') or m.name=='__hand_shapes.json__':
                            payload[m.name]=json.loads(tf.extractfile(m).read())
                        elif any(m.name==f'{i:06d}.image_1201-1.jpg' for i in [0,50,100,149]):
                            samples[m.name]=tf.extractfile(m).read()
                while digest.read(2**20):pass
            assert digest.size==entry['size'],(digest.size,entry['size'])
            assert digest.sha.hexdigest()==entry['lfs']['oid']
            assert '__hand_shapes.json__' in payload and any(k.endswith('.hands.json') for k in payload)
            raw=json.dumps(payload,separators=(',',':')).encode();compressed=gzip.compress(raw,compresslevel=4,mtime=0)
            tmp=dest.with_suffix('.tmp');tmp.write_bytes(compressed);tmp.replace(dest)
            for name,b in samples.items():(root/name).write_bytes(b)
            r={'source_url':url,'revision':rev,'source_bytes_streamed':digest.size,'source_sha256':digest.sha.hexdigest(),'full_source_stream_verified':True,'retained':'All JSON annotation members plus four source RGB samples; other RGB bytes streamed for hash only','restore_full_source':url,'packet_sha256':hashlib.sha256(compressed).hexdigest(),'packet_bytes':len(compressed),'annotation_members':len(payload),'cpu_wall_seconds':time.monotonic()-start,'images':{k:hashlib.sha256(v).hexdigest() for k,v in samples.items()}}
            write(receipt,r);print(json.dumps({'clip':stem,'packet_bytes':len(compressed),'seconds':r['cpu_wall_seconds']}),flush=True);return r
        except Exception as e:
            if attempt==2:return {'source':entry['path'],'error':repr(e),'attempts':3}
            time.sleep(2**attempt)
def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=64);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    existing=RUN/'public_catalog/hot3d_train_quest3.json'
    entries=json.loads(existing.read_text())['entries'] if existing.exists() else tree('hot3d','train_quest3')
    entries=[e for e in entries if e['path'].endswith('.tar')]
    rng=np.random.default_rng(280043);idx=rng.choice(len(entries),size=min(a.count,len(entries)),replace=False)
    selected=[entries[int(i)] for i in sorted(idx)]
    write(RUN/'public_catalog/hot3d_screen_selection.json',{'seed':280043,'count':len(selected),'rule':'Uniform without replacement among pinned public training Quest3 clips, before any policy/model outcomes. Screening pool only, not frozen Probe40.','sources':selected})
    root=tree('hot3d');fetch('hot3d',next(e for e in root if e['path']=='clip_definitions.json'))
    for e in tree('hot3d','object_models_eval'):
        if e['path'].endswith('.glb'):fetch('hot3d',e)
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(collect,selected))
    write(RUN/'public_catalog/hot3d_screen_download_receipt.json',{'cpu_wall_seconds':time.monotonic()-started,'completed':sum('error' not in r for r in rows),'failed':sum('error' in r for r in rows),'records':rows})
if __name__=='__main__':main()
