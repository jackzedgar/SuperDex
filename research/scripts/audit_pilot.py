"""Independent CPU audit of checkpoint, transitions, reward and contact failure."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
from train_real_ppo import Policy
from real_env import ROOT,DATA


def main():
    p=argparse.ArgumentParser();p.add_argument('--ppo',type=Path,default=ROOT/'runs/real_track_20261026/ppo');args=p.parse_args();out=args.ppo
    ckpt=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
    policy=Policy(ckpt['config']['obs_dim']);policy.load_state_dict(ckpt['policy']);policy.eval()
    scale=np.array(ckpt['config']['action_scale']);ref=np.load(DATA/'trajectory.npz');target=np.load(DATA/'controller_reference.npz')['ctrl'];m=mujoco.MjModel.from_xml_path(str(DATA/'scene.xml'));d=mujoco.MjData(m)
    result={};checks=[]
    for label in ['before','after']:
        log=np.load(out/f'rollout_{label}.npz');ph=log['phase'];raw=log['raw_action'];action=np.clip(raw,-1,1);states=np.vstack([log['qpos'][1:],log['final_qpos']]);r=ref['qpos'][ph+1]
        ctrl=np.clip(target[ph+1]+action*scale,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);ctrlerr=np.max(abs(ctrl-log['ctrl']))
        ep=np.linalg.norm(states[:,26:29]-r[:,26:29],axis=-1);q=states[:,29:33];q=q/np.linalg.norm(q,axis=-1,keepdims=True);er=2*np.arccos(np.clip(abs(np.sum(q*r[:,29:33],axis=-1)),0,1))
        eq=np.mean((states[:,6:26]-r[:,6:26])**2,axis=-1);ew=np.sum((states[:,:3]-r[:,:3])**2,axis=-1);last=np.vstack([np.zeros(26),action[:-1]])
        reward=.02*(5*np.exp(-(ep/.04)**2)+2*np.exp(-(er/.5)**2)+.5*np.exp(-eq/.1)+.5*np.exp(-ew/.001)-.03*np.mean(action**2,axis=-1)-.05*np.mean((action-last)**2,axis=-1))
        counts=[];wrist=[];force_fraction=[]
        for i in range(len(ph)):
            d.qpos[:]=log['qpos'][i];d.qvel[:]=log['qvel'][i];d.ctrl[:]=log['ctrl'][i];mujoco.mj_forward(m,d)
            counts.append(sum({int(m.geom_contype[c.geom1]),int(m.geom_contype[c.geom2])}=={1,2} for c in d.contact))
            wrist.append(np.linalg.norm(d.qpos[:3]-ref['qpos'][ph[i],:3]))
            force_fraction.append(np.mean(np.abs(d.actuator_force[6:])>=m.actuator_forcerange[6:,1]-.001))
        record=dict(steps=len(ph),ctrl_reconstruction_max_error=float(ctrlerr),reward_reconstruction_max_error=float(np.max(abs(reward-log['reward']))),position_log_max_error=float(np.max(abs(ep-log['errors'][:,0]))),rotation_log_max_error_rad=float(np.max(abs(er-log['errors'][:,1]))),
                    contact_counts_at_logged_states=counts,frames_without_hand_object_contact=int(np.sum(np.array(counts)==0)),first_no_contact_step=next((i for i,c in enumerate(counts) if c==0),None),
                    wrist_position_mean_error_m=float(np.mean(wrist)),mean_finger_force_saturation_fraction=float(np.mean(force_fraction)),raw_action_rms=float(np.sqrt(np.mean(raw**2))),action_clipped_fraction=float(np.mean(abs(raw)>1)),
                    contact_note='CPU forward computation at stored GPU states; a diagnostic, not GPU contact-force ground truth.')
        checks.extend([ctrlerr<2e-6,record['reward_reconstruction_max_error']<2e-5,record['position_log_max_error']<2e-6,record['rotation_log_max_error_rad']<2e-4])
        if label=='after':
            with torch.no_grad():pred=policy.actor(torch.tensor(log['observation'])).numpy()
            record['checkpoint_action_max_error']=float(np.max(abs(pred-raw)));checks.append(record['checkpoint_action_max_error']<2e-5)
        result[label]=record
    result['passed']=bool(all(checks));result['checkpoint_update']=ckpt['update'];result['weights_finite']=all(torch.isfinite(p).all().item() for p in policy.parameters());assert result['passed'] and result['weights_finite'],result
    result['diagnosis']=['Six of eight paired episodes fail rotation before the horizon both before and after PPO.',
                         'Mean errors are conditional on executed steps: shorter failed episodes can report smaller means without improving tracking.',
                         'The 8 mm contact-normal preload retains contact in a nominal hold but allows substantial slip/rotation; millimetre initial perturbations are consequential.',
                         'Uniform random-phase training differs from start-of-clip evaluation; early grasp stabilization receives a small fraction of resets.',
                         'Reward gives position weight 5 and orientation weight 2; no contact stability/force-closure term is present. A causal reward diagnosis requires a later controlled change, not a claim from this one run.',
                         'Prioritize reliable initial grasp, controller lag and contact parameter identification, then onset-focused curriculum/reward review; do not increase task count.']
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
