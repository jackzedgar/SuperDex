"""Wuji single/bimanual A1 XML assets, with explicit simulated actuator assumptions."""
from pathlib import Path
import copy,json,xml.etree.ElementTree as E
import numpy as np,mujoco,trimesh
from a1_data_sources import ROOT,write

ASSETS=ROOT/'data/a1_processed/assets'
def hand_tree(side):
    out=ASSETS/'hands'/side;out.mkdir(parents=True,exist_ok=True)
    source=ROOT.parent/f'assets/wuji-training/urdf/{side}.urdf'
    meshdir=(source.parent/f'../meshes/{side}').resolve()
    urdf=E.parse(source);urdf.getroot().find('mujoco/compiler').set('meshdir',str(meshdir))
    for mesh in urdf.getroot().iter('mesh'):mesh.set('filename',Path(mesh.get('filename')).name)
    fixed=out/'path_fixed.urdf';urdf.write(fixed)
    m=mujoco.MjModel.from_xml_path(str(fixed));imported=out/'imported.xml';mujoco.mj_saveLastXML(str(imported),m)
    root=E.parse(imported).getroot();root.find('compiler').set('meshdir','')
    for mesh in root.find('asset').findall('mesh'):mesh.set('file',str(meshdir/Path(mesh.get('file')).name))
    for i in range(1,6):
        parent=next(x for x in root.iter('body') if x.get('name')==f'{side}_finger{i}_link4')
        frame=next(x for x in parent.findall('frame') if any('tip_link' in g.get('mesh','') for g in x.findall('geom')))
        mesh=trimesh.load(meshdir/f'{side}_finger{i}_tip_link.STL',force='mesh')
        p=mesh.bounds.mean(0);p[2]=mesh.bounds[1,2]
        E.SubElement(frame,'site',name=f'{side}_tip_{i}',pos=' '.join(map(str,p)),size='.002')
        for label,number in [('mid',3),('mcp',2)]:
            b=next(x for x in root.iter('body') if x.get('name')==f'{side}_finger{i}_link{number}')
            E.SubElement(b,'site',name=f'{side}_{label}_{i}',pos='0 0 0',size='.001')
    world=root.find('worldbody');mount=E.Element('body',name=f'{side}_wrist')
    E.SubElement(mount,'inertial',pos='0 0 .04',mass='.25',diaginertia='.001 .001 .001')
    for j,axis in enumerate(['1 0 0','0 1 0','0 0 1']):
        E.SubElement(mount,'joint',name=f'{side}_wrist_p{j}',type='slide',axis=axis,range='-1 1',damping='5',armature='.01')
    for j,axis in enumerate(['1 0 0','0 1 0','0 0 1']):
        E.SubElement(mount,'joint',name=f'{side}_wrist_r{j}',type='hinge',axis=axis,range='-6.28 6.28',damping='.3',armature='.01')
    for child in list(world):world.remove(child);mount.append(child)
    world.append(mount)
    for b in mount.iter('body'):b.set('gravcomp','1')
    for i,g in enumerate(mount.iter('geom')):
        visual=g.get('group')=='1';g.set('contype','0' if visual else '1');g.set('conaffinity','0' if visual else '2')
        g.set('friction','.8 .005 .0001')
        g.set('name',f'{side}_geom_{i}')
    inertia=.0005;omega=2*np.pi*10;kp=inertia*omega**2;kd=2*inertia*2*omega
    for j in mount.iter('joint'):
        if '_finger' in j.get('name',''):j.set('armature',str(inertia));j.set('damping','0')
    act=root.find('actuator')
    if act is None:act=E.SubElement(root,'actuator')
    for i in range(3):
        name=f'{side}_wrist_p{i}';E.SubElement(act,'position',name=name,joint=name,kp='3000',kv='100',forcerange='-100 100',ctrlrange='-1 1')
    for i in range(3):
        name=f'{side}_wrist_r{i}';E.SubElement(act,'position',name=name,joint=name,kp='100',kv='5',forcerange='-15 15',ctrlrange='-6.28 6.28')
    for i in range(1,6):
        for j in range(1,5):
            name=f'{side}_finger{i}_joint{j}';joint=next(x for x in mount.iter('joint') if x.get('name')==name)
            E.SubElement(act,'position',name=name,joint=name,kp=str(kp),kv=str(kd),ctrlrange=joint.get('range'),forcerange=joint.get('actuatorfrcrange','-.4 .4'))
    opt=root.find('option')
    if opt is None:opt=E.SubElement(root,'option')
    opt.attrib.update(timestep='.002',integrator='implicitfast',solver='Newton',iterations='30',ls_iterations='50',gravity='0 0 -9.81')
    E.ElementTree(root).write(out/'hand.xml')
    m=mujoco.MjModel.from_xml_path(str(out/'hand.xml'))
    ids=np.array([[m.body(f'{side}_finger{i}_link{j}').id for j in range(1,5)] for i in range(1,6)])
    parents=m.body_parentid[ids[:,0]];assert len(set(parents.tolist()))==1
    parent=parents[0];wrist=m.body(f'{side}_wrist').id
    assert parent==wrist or m.body_parentid[parent]==wrist,('Unexpected palm hierarchy',side)
    root_pos=np.zeros(3) if parent==wrist else m.body_pos[parent]
    root_quat=np.array([1.,0,0,0]) if parent==wrist else m.body_quat[parent]
    ji=m.body_jntadr[ids]
    tipids=[m.site(f'{side}_tip_{i}').id for i in range(1,6)]
    data=dict(body_pos=m.body_pos[ids],body_quat=m.body_quat[ids],body_ipos=m.body_ipos[ids],joint_axis=m.jnt_axis[ji],joint_pos=m.jnt_pos[ji],joint_qpos0=m.qpos0[m.jnt_qposadr[ji]],tip_pos=m.site_pos[tipids],palm_point=m.body_ipos[wrist],root_pos=root_pos,root_quat=root_quat)
    np.savez(out/'kinematics.npz',**data)
    write(out/'actuator_assumptions.json',dict(reflected_inertia_kgm2=inertia,inertia_status='SIMULATED ASSUMPTION, not measured Wuji gear or rotor inertia',frequency_hz=10,omega_rad_s=float(omega),damping_ratio=2,kp=float(kp),kd=float(kd),finger_residual_scale='0.25*tau_max/kp',wrist_tau_s=.02,wrist_identified_delay_steps=1,identification_status='SIMULATED assumption, not xArm measurement',additional_delay_steps=[0,1],mass_inertia_source='Wuji URDF plus explicitly listed virtual wrist mount'))
    return root

def combined(sides):
    roots=[E.parse(ASSETS/'hands'/side/'hand.xml').getroot() if (ASSETS/'hands'/side/'hand.xml').exists() else hand_tree(side) for side in sides]
    root=copy.deepcopy(roots[0]);world=root.find('worldbody')
    for other in roots[1:]:
        for section in ['asset','actuator','worldbody']:
            for child in other.find(section):root.find(section).append(copy.deepcopy(child))
    E.SubElement(world,'geom',name='floor',type='plane',size='2 2 .05',contype='4',conaffinity='2',friction='.6 .005 .0001',rgba='.8 .8 .8 1')
    E.SubElement(world,'light',pos='0 -1 2',dir='0 0 -1')
    return root

def main():
    import torch
    from a1_geometry import HandKinematics
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(1);results={}
    for side in ['right','left']:
        hand_tree(side);out=ASSETS/'hands'/side
        m=mujoco.MjModel.from_xml_path(str(out/'hand.xml'));d=mujoco.MjData(m)
        fk=HandKinematics(dict(np.load(out/'kinematics.npz'))).double();maximum=0.
        rng=np.random.default_rng(280043)
        for _ in range(16):
            d.qpos[:3]=rng.normal(0,.05,3);d.qpos[3:6]=rng.normal(0,.2,3)
            d.qpos[6:]=rng.uniform(m.jnt_range[6:,0],m.jnt_range[6:,1]);mujoco.mj_forward(m,d)
            wr=d.xmat[m.body(f'{side}_wrist').id].reshape(3,3);wp=d.xpos[m.body(f'{side}_wrist').id]
            pts=fk(torch.tensor(d.qpos[None,6:])).detach().numpy()[0]@wr.T+wp
            target=np.array([d.site_xpos[m.site(f'{side}_tip_{i}').id] for i in range(1,6)]+[d.xipos[m.body(f'{side}_finger{i}_link{j}').id] for j in [1,2,3] for i in range(1,6)]+[d.xipos[m.body(f'{side}_wrist').id]])
            maximum=max(maximum,float(np.abs(pts-target).max()))
        assert maximum<2e-7,(side,maximum)
        results[side]=dict(max_query_error_m=maximum,random_poses=16,queries=21)
    write(ROOT/'runs/a1_full_20261028/hand_fk_check.json',results);print(json.dumps(results))
if __name__=='__main__':main()
