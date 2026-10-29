"""Fixed evaluation with original 10-substep graph and GPU actor; no learning.

Kept separate from the low-memory 9+1 substep contact-instrumented diagnostic.
The initial eight tensors are copied verbatim from the baseline evaluation.
"""
import argparse,json
import numpy as np,torch,warp as wp
from real_env import RealEnv
from train_real_ppo import Policy
from grasp_lab import RUN,OLD,DATA,fixed_initials,get_target,metrics,summarize,write_json
from grasp_batch_audit import gpu_contacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','grip_balance'],default='baseline');p.add_argument('--policy',choices=['zero','checkpoint'],default='checkpoint');args=p.parse_args()
    assert torch.cuda.mem_get_info()[0]>1024*2**20,'Require 1 GiB free for GPU actor evaluation; resource failure is not an algorithm result.'
    env=RealEnv(1);initial=fixed_initials();ref=np.load(DATA/'trajectory.npz');target=np.load(DATA/'controller_reference.npz')['ctrl']
    original_target=torch.tensor(target,device=env.device,dtype=torch.float32)
    if args.variant!='baseline':
        env.target[:]=torch.tensor(np.array([get_target(ref,target,k,args.variant) for k in range(len(target))]),device=env.device,dtype=torch.float32)
    ckpt=torch.load(OLD/'checkpoint.pt',map_location='cpu',weights_only=True);policy=Policy(118).to(env.device);policy.load_state_dict(ckpt['policy']);policy.eval()
    label=args.variant+'_'+args.policy;out=RUN/'gpu_original_graph'/label;out.mkdir(parents=True,exist_ok=True);results=[]
    for trial in range(8):
        env.reset(torch.tensor([0],device=env.device),False);env.qpos[0]=torch.tensor(initial['qpos'][trial],device=env.device);env.qvel[0]=torch.tensor(initial['qvel'][trial],device=env.device)
        qs=[env.qpos[0].cpu().numpy().copy()];vs=[env.qvel[0].cpu().numpy().copy()];actions=[];controls=[];rows=[];cslog=[];rewards=[];parts=[];reason='horizon'
        for k in range(1,52):
            # Preserve the policy observation of the ORIGINAL target. A controller
            # ablation must not simultaneously alter the learned observation.
            obs=env.observe()
            if args.variant!='baseline':obs[:,52:78]=original_target[k]-env.qpos[:,:26];obs.clamp_(-20,20)
            with torch.no_grad():action=policy.actor(obs) if args.policy=='checkpoint' else torch.zeros(1,26,device=env.device)
            last=env.last_action[0].cpu().numpy().copy()
            _,reward,done,info=env.step(action);q=env.qpos[0].cpu().numpy().copy();v=env.qvel[0].cpu().numpy().copy();a=action[0].cpu().numpy().copy();cs,_=gpu_contacts(env)
            r=metrics(env.m,q,v,ref['qpos'][k],cs,k*.02);rows.append(r);rewards.append(float(reward[0]));actions.append(a);controls.append(env.ctrl[0].cpu().numpy().copy());qs.append(q);vs.append(v)
            cslog.append(dict(contact_time_s=k*.02-.002,returned_state_time_s=k*.02,contacts=cs,note='Actual GPU contacts from the final substep. Pre-integration state is not copied inside this original graph; do not compare CPU contacts at the returned state.'))
            clipped=np.clip(a,-1,1);eq=np.mean((q[6:26]-ref['qpos'][k,6:26])**2);ew=np.sum((q[:3]-ref['qpos'][k,:3])**2)
            rp=np.array([5*np.exp(-(r['position_m']/.04)**2),2*np.exp(-(r['rotation_rad']/.5)**2),.5*np.exp(-eq/.1),.5*np.exp(-ew/.001),-.03*np.mean(clipped**2),-.05*np.mean((clipped-last)**2)])*.02
            parts.append(rp)
            assert abs(rp.sum()-float(reward[0]))<1e-4,'Reward component reconstruction does not match environment.'
            if bool(done[0]):
                reason={0:'horizon',1:'position_gt_15cm',2:'rotation_gt_1.4rad',3:'joint_limit',4:'nonfinite'}[int(info['reason'][0])];break
        stem=f'trial_{trial:02d}';np.savez_compressed(out/(stem+'.npz'),qpos=qs,qvel=vs,action=actions,ctrl=controls,reward=rewards,reward_components=parts)
        write_json(out/(stem+'_metrics.json'),rows);write_json(out/(stem+'_contacts.json'),cslog)
        r=summarize(rows,'C' if args.policy=='checkpoint' else 'B',reason);r.update(trial=trial,variant=args.variant,policy=args.policy,return_=sum(rewards),component_returns=dict(zip(['object_position','object_rotation','fingers','wrist','action','action_change'],np.array(parts).sum(0).tolist())),action_clipped_fraction=float(np.mean(abs(np.array(actions))>1)),gpu_actor=True,graph='original 10 substeps',success_from_original_environment=bool(info['success'][0]))
        assert r['dynamic_success']==r['success_from_original_environment'];results.append(r)
    write_json(out/'summary.json',results)
    print(json.dumps(dict(label=label,successes=sum(x['dynamic_success'] for x in results),mean_return=np.mean([x['return_'] for x in results]),mean_length=np.mean([x['steps'] for x in results]))),flush=True)


if __name__=='__main__':main()
