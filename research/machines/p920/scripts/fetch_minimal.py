"""Anonymous, revision-pinned downloads with content verification."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "bop-benchmark/hot3d"
REVISION = "30fe9674782f32e1e5edba98476b6ff4300132c5"


def fetch(path, root, expected=None):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    dest = root / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temp = dest.with_suffix(dest.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as src, temp.open("wb") as out:
            while chunk := src.read(2**20):
                out.write(chunk)
        temp.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return dict(path=path, url=url, revision=REVISION, bytes=dest.stat().st_size,
                sha256=digest, expected_sha256=expected, anonymous=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("data/raw/hot3d"))
    p.add_argument("--clip",type=int,default=0)
    p.add_argument("--object",type=int,default=4)
    args=p.parse_args()
    meta_url=f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/train_quest3?limit=1000"
    with urllib.request.urlopen(meta_url,timeout=30) as f: entries=json.load(f)
    path=f"train_quest3/clip-{args.clip:06d}.tar"
    entry=next(x for x in entries if x["path"]==path)
    records=[fetch(path,args.root,entry["lfs"]["oid"])]
    metadata_hashes={
        'clip_splits.json':'54067720be3645aff6e9d79dcba2bd8d1c6d3f91e1dfba2cfae42da9ec5f497c',
        'object_models_eval/models_info.json':'351e1b25a250606fa4b7500f5aeb800b55050bf1c638d05ceb464f8ddfb32e2b',
        'hot3d_dataset_license_agreement.pdf':'0e2142c2920d25128a15944078870e4c13f81cd3864c39c9d69c3cda21e01868'}
    for path,expected in metadata_hashes.items():
        records.append(fetch(path,args.root,expected))
    manifest=args.root/f"download_clip_{args.clip:06d}.json"
    manifest.write_text(json.dumps(records,indent=2)+"\n")
    meta_url=f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/object_models_eval?limit=1000"
    with urllib.request.urlopen(meta_url,timeout=30) as f: meshes=json.load(f)
    path=f"object_models_eval/obj_{args.object:06d}.glb"
    entry=next(x for x in meshes if x['path']==path)
    mesh=fetch(path,args.root,entry.get('lfs',{}).get('oid'))
    if 'lfs' not in entry:
        data=(args.root/path).read_bytes()
        blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        assert blob==entry['oid'],f'Git blob hash mismatch: {path}'
        mesh['verified_git_blob_sha1']=blob
    records.append(mesh)
    (args.root/f'download_mesh_{args.object:06d}.json').write_text(json.dumps([mesh],indent=2)+'\n')
    print(json.dumps(records,indent=2),flush=True)


if __name__=="__main__": main()
