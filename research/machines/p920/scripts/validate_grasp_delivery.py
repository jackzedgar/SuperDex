"""Validate saved evidence and source-preservation claims, without resimulation."""
import json,hashlib,subprocess,re
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image
from grasp_lab import RUN,ROOT,DATA,write_json

def load(p):return json.loads(p.read_text())
initial=np.load(RUN/'fixed_initials.npz');model=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));checked=0
for base in [RUN/'cpu',RUN/'gpu',RUN/'gpu_original_graph']:
    for p in base.rglob('trial_*.npz'):
        a=np.load(p);rows=load(p.with_name(p.stem+'_metrics.json'));n=len(rows)
        assert a['qpos'].shape==(n+1,33) and a['qvel'].shape==(n+1,32) and a['ctrl'].shape==(n,26),p
        assert np.isfinite(a['qpos']).all() and np.isfinite(a['qvel']).all() and np.isfinite(a['ctrl']).all(),p
        assert np.all(a['ctrl']>=model.actuator_ctrlrange[:,0]-1e-6) and np.all(a['ctrl']<=model.actuator_ctrlrange[:,1]+1e-6),p
        trial=int(p.stem.split('_')[-1])
        if trial>=0:assert np.max(abs(a['qpos'][0]-initial['qpos'][trial]))<1e-7,p
        checked+=1
for n in [1,2,8]:
    s=load(RUN/f'batch_{n}/summary.json');assert s['status']=='completed' and s['max_total_gpu_contacts']<s['allocated_contact_capacity']
    a=np.load(RUN/f'batch_{n}/trajectory.npz');assert np.array_equal(a['ctrl'],np.broadcast_to(a['ctrl'][:,0:1],a['ctrl'].shape))
for n in [2,8]:assert all(x['passed'] for x in load(RUN/f'reset_markers_{n}.json')['tests'])
ev=load(RUN/'fixed_evaluation.json');assert all(x['successes']==0 for x in ev['runs'].values())
assert all(x['first_action_max_delta']==0 and x['identical_initial_state'] for x in ev['baseline_reproduction'])
for i in range(8):
    a=np.load(RUN/f'gpu_original_graph/baseline_checkpoint/trial_{i:02d}.npz')['action'][0]
    b=np.load(RUN/f'gpu_original_graph/grip_balance_checkpoint/trial_{i:02d}.npz')['action'][0]
    assert np.array_equal(a,b),'Candidate actor observation must match baseline exactly at the same initial state.'
assert load(RUN/'baseline_preservation.json')['all_unchanged']
assert not load(ROOT/'configs/grasp_repair_gate.json')['ready_for_bounded_ppo']
videos=[]
for task,expected in [('A',101),('B',52),('C',52)]:
    p=RUN/'visuals'/f'comparison_{task}.mp4';reader=imageio.get_reader(p);meta=reader.get_meta_data();frames=reader.count_frames();assert frames==expected and abs(meta['fps']-25)<1e-6
    for index in [0,frames//2,frames-1]:assert reader.get_data(index).shape==(544,1280,3)
    reader.close();videos.append(dict(file=p.name,frames=frames,fps=meta['fps'],dimensions=[1280,544]))
pngs=[]
for p in (RUN/'visuals').glob('*.png'):
    with Image.open(p) as im:im.verify()
    with Image.open(p) as im:pngs.append(dict(file=p.name,size=im.size,mode=im.mode))
report=load(RUN/'training_gate_test.json');assert report['passed']
subprocess.run(['bash','-n','scripts/reproduce_grasp_repair.sh'],cwd=ROOT,check=True)
write_json(RUN/'delivery_validation.json',dict(saved_trial_logs_checked=checked,all_finite=True,controls_within_limits=True,initial_states_preserved=True,batch_inputs_identical=True,reset_sentinels_pass=True,contact_capacity_ok=True,baseline_unchanged=True,no_new_training=True,video_metadata=videos,png_metadata=pngs,manual_visual_review='Review representative first/middle/final video frames and all figure PNGs in addition to these structural checks.'))
print('Validated',checked,'saved trials, 3 videos and',len(pngs),'PNGs; no resimulation.')
