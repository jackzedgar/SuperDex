"""GAE on [time, independent-world] data; normalization stays in original PPO."""
import torch

def per_world_gae(reward,value,done,bootstrap,gamma=.99,lam=.95):
    assert reward.ndim==2 and reward.shape==value.shape==done.shape
    assert bootstrap.shape==(reward.shape[1],)
    advantage=torch.empty_like(reward);carry=torch.zeros_like(bootstrap);next_value=bootstrap
    for t in reversed(range(reward.shape[0])):
        alive=1-done[t].to(reward.dtype)
        delta=reward[t]+gamma*next_value*alive-value[t]
        carry=delta+gamma*lam*alive*carry
        advantage[t]=carry;next_value=value[t]
    return advantage,advantage+value
