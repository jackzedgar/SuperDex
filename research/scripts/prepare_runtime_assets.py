"""Create relocated runtime copies; never edit frozen trajectories, XML or logs."""
import hashlib,json,shutil,os
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

ROOT=Path(__file__).resolve().parents[1]
OLD='/lost+found/nvme/dexmimic'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'data/processed';dest=ROOT/'runtime/lambda/processed'
    assert not dest.exists(),'Runtime copies already exist; inspect their manifest instead of overwriting.'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,dest)
    mappings=[];missing=[];scenes=[]
    for p in dest.rglob('*.xml'):
        original=source/p.relative_to(dest)
        text=p.read_text();new=text.replace(OLD,str(ROOT.parent))
        # A copied scene may refer to another generated file under its original
        # processed directory. Redirect those to the same relocated runtime tree.
        new=new.replace(str(source),str(dest))
        p.write_text(new)
        mappings.append(dict(original=str(original.relative_to(ROOT)),runtime=str(p.relative_to(ROOT)),original_sha256=sha(original),runtime_sha256=sha(p),changed=text!=new))
        tree=ET.fromstring(new);compiler=tree.find('compiler')
        for e in tree.iter():
            name=e.get('file')
            if not name:continue
            asset=Path(name)
            if not asset.is_absolute():
                sub=compiler.get('meshdir','') if compiler is not None and e.tag=='mesh' else ''
                asset=p.parent/sub/asset
            if not asset.is_file():missing.append(dict(xml=str(p),asset=str(asset)))
        if p.name=='scene.xml':
            m=mujoco.MjModel.from_xml_path(str(p));scenes.append(dict(path=str(p.relative_to(ROOT)),nq=m.nq,nv=m.nv,nu=m.nu,ngeom=m.ngeom,nmesh=m.nmesh,timestep=m.opt.timestep))
    report=dict(source_prefix=OLD,destination_prefix=str(ROOT.parent),files=mappings,missing=missing,scenes=scenes,originals_modified=False)
    (dest.parent/'asset_mapping.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not missing,missing
    print(json.dumps(dict(xml_files=len(mappings),scenes=scenes,missing=missing),indent=2))
if __name__=='__main__':main()
