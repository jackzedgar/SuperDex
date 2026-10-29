"""Execute a saved pilot checkpoint in new GPU physics and save fresh logs."""
import argparse
from pathlib import Path
import torch
from real_env import RealEnv
from train_real_ppo import Policy,evaluate

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
env=RealEnv(1);ckpt=torch.load(args.checkpoint,map_location=env.device,weights_only=True)
policy=Policy(env.obs_dim).to(env.device);policy.load_state_dict(ckpt['policy']);policy.eval()
evaluate(env,policy,args.output,'checkpoint')
