"""CPU-only post-hoc gradient decomposition; no policy or model updates."""
import argparse,json,time
from types import SimpleNamespace
import numpy as np,torch,mujoco
from train_real_ppo import Policy
from world_model_mini import WorldModel,Geometry
from bptt_common import ROOT,RUN,WMROOT,imagine,write
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();torch.set_num_threads(1);start=time.time();out=RUN/a.arm
 pc=torch.load(out/f'step_{a.step:06d}.pt',map_location='cpu',weights_only=False);mc=torch.load(out/f'models/version_{a.step:06d}.pt',map_location='cpu',weights_only=False)
 policy=Policy(118);policy.load_state_dict(pc['policy']);policy.critic.requires_grad_(False)
 model=WorldModel('W1',Geometry(WMROOT),mc['normalization']);model.load_state_dict(mc['model']);model.requires_grad_(False)
 episodes=torch.load(out/'replay_latest.pt',map_location='cpu',weights_only=False)
 train=[e for e in episodes if e['split']=='train'];selected=train[-32::2]
 q=np.load(ROOT/'runtime/lambda/processed/clip000000_obj000004_right/trajectory.npz')['qpos'];u=np.load(ROOT/'runtime/lambda/processed/clip000000_obj000004_right/controller_reference.npz')['ctrl']
 m=mujoco.MjModel.from_xml_path(str(ROOT/'runtime/lambda/processed/clip000000_obj000004_right/scene.xml'))
 env=SimpleNamespace(ref=torch.tensor(q,dtype=torch.float32),target=torch.tensor(u,dtype=torch.float32),length=len(q),lo=torch.tensor(m.actuator_ctrlrange[:,0],dtype=torch.float32),hi=torch.tensor(m.actuator_ctrlrange[:,1],dtype=torch.float32),scale=torch.tensor([.025]*3+[.18]*3+[.25]*20))
 states=torch.stack([e['states'][0] for e in selected]);ph=torch.tensor([e['phases'][0] for e in selected]);last=torch.zeros(len(states),26)
 loss,_,_,diag=imagine(policy,model,states,ph,last,env,32);full=torch.cat([x.flatten() for x in torch.autograd.grad(loss,list(policy.actor.parameters()))])
 with torch.no_grad():
  for x in policy.critic.parameters():x.zero_()
 rloss,_,_,_=imagine(policy,model,states,ph,last,env,32);reward=torch.cat([x.flatten() for x in torch.autograd.grad(rloss,list(policy.actor.parameters()))]);critic=full-reward
 result=dict(arm=a.arm,step=a.step,anchor_episode_ids=[e['id'] for e in selected],anchor_source='Deterministic recent training episodes from replay snapshot, first states; no fixed evaluation condition',cpu_seconds=time.time()-start,full_norm=float(full.norm()),reward_norm=float(reward.norm()),bootstrap_norm=float(critic.norm()),reward_bootstrap_cosine=float(torch.nn.functional.cosine_similarity(reward,critic,dim=0)),full_reward_cosine=float(torch.nn.functional.cosine_similarity(full,reward,dim=0)),imagined_return=diag['imagined_return'],imagined_reward_only_return=-float(rloss.detach()),scope='Exact additive gradient decomposition on same frozen imagined trajectories; no causal physical-validity or counterfactual-training claim',gpu_process_seconds=0)
 write(out/f'gradient_decomposition_{a.step:06d}.json',result);print(json.dumps(result))
if __name__=='__main__':main()

