"""CPU-only reference-shaped timing fixture. Never a scientific training/evaluation dataset."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np,torch,mujoco
from a1_data_sources import ROOT,write
from a1_task_runtime import TaskTensors
from a1_state import H,R,P,Q,matrix_euler
from world_model_mini import qmatrix,qlog
from a1_world_model import common_features,StreamingMoments
import a1_state as state


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);out=ROOT/'runs/a1_full_20261028/hardware_fixture_20261030_001';out.mkdir(exist_ok=False)
    rows=json.loads((ROOT/'configs/a1_probe40_v1.json').read_text())['tasks'];keys=sorted({json.loads((ROOT/r['task_path']/'qualification.json').read_text())['object']['object_key'] for r in rows})
    assert len(rows)==40 and len(keys)==19
    values={key:[] for key in ['start','commands','target','mask','context','object_id']};sources=[]
    start_time=time.monotonic();rng=np.random.default_rng(301102)
    for row in rows:
        task=TaskTensors(ROOT/row['task_path'],device='cpu');n=task.length
        assert n>36
        phases=np.sort(rng.choice(np.arange(1,n-32),size=min(12,n-33),replace=False))
        context=torch.zeros(276,dtype=torch.float32);m=task.m;ob=m.body('object').id
        for k,side in enumerate(task.sides):
            h=['right','left'].index(side);sl=slice(k*26+6,k*26+26)
            context[h*68+20:h*68+40]=torch.from_numpy(m.actuator_gainprm[sl,0].copy())
            context[h*68+40:h*68+60]=torch.from_numpy(-m.actuator_biasprm[sl,2].copy())
            context[h*68+60:h*68+62]=torch.tensor([1.,0.]);context[h*68+62:h*68+68]=.02
        context[136:139]=torch.tensor([m.body_mass[ob],.8,.6]);context[139:142]=torch.from_numpy(m.body_ipos[ob].copy())
        context[142:145]=torch.from_numpy(m.body_inertia[ob].copy());context[145:148]=qlog(torch.from_numpy(m.body_iquat[ob].copy()));context[148:]=task.bps
        for phase in phases:
            initial=task.ref[phase].clone();hands=initial[:274].reshape(2,H)
            for h in range(2):
                if task.mask[0,h]==0:continue
                command=torch.cat([hands[h,P],matrix_euler(qmatrix(hands[h,R])),hands[h,Q]])
                hands[h,53:131]=command.repeat(3);hands[h,131:137]=command[:6]
            values['start'].append(initial);values['commands'].append(task.base_command[phase:phase+32].clone())
            values['target'].append(task.ref[phase+1:phase+33].clone());values['mask'].append(task.mask[0].clone())
            values['context'].append(context.clone());values['object_id'].append(torch.tensor(keys.index(task.meta['object']['object_key'])))
            sources.append(dict(task=row['task_id'],stratum=task.meta['stratum'],phase=int(phase)))
        write(out/'progress.json',dict(tasks=len({s['task'] for s in sources}),windows=len(sources),seconds=time.monotonic()-start_time))
        del task
    tensors={k:torch.stack(v) for k,v in values.items()};assert all(torch.isfinite(v).all() for v in tensors.values())
    moments={k:StreamingMoments(w) for k,w in [('input',671),('delta',116),('loss',125)]}
    moments['input'].add(common_features(tensors['start'],tensors['commands'][:,0],tensors['mask'],tensors['context']))
    moments['delta'].add(state.delta_between(tensors['start'],tensors['target'][:,0],tensors['mask']))
    moments['loss'].add(state.encode_dynamics(tensors['target'][:,:8]))
    norm=dict(input_mean=moments['input'].mean.float(),input_std=moments['input'].std(),delta_std=moments['delta'].std(),loss_std=moments['loss'].std())
    filename=out/'fixture.npz';np.savez_compressed(filename,**{k:v.numpy() for k,v in tensors.items()},**{k:v.numpy() for k,v in norm.items()})
    write(out/'manifest.json',dict(status='complete',formal=False,source='Reference-shaped CPU fixture; not physical rollouts and not formal distinct windows',
        purpose='Hardware sizing, finite gradients and implementation checks only; every resulting weight is discarded',
        not_A1_success_or_G1_evidence=True,fixed_test_opened=False,object_keys=keys,tasks=40,windows=len(sources),horizon=32,
        fixture_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),sources=sources,seconds=time.monotonic()-start_time,
        normalization='Fixture-only; never copy these moments to formal model training'))
    print(json.dumps(dict(status='complete',tasks=40,objects=19,windows=len(sources),seconds=time.monotonic()-start_time)))


if __name__=='__main__':main()
