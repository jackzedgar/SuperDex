"""S-mini frozen-model BPTT and episode-disjoint replay."""
from pathlib import Path
import json,collections
import numpy as np,torch
from world_model_mini import WorldModel,Geometry,qlog,qmul,qconj
from core_policy_common import observe_state
from probe_model_gradient import reward_from_state
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/bptt_iteration_20261027'
WMROOT=ROOT/'runs/learning_trend_20261027/wm'
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def load_wm():
    c=torch.load(WMROOT/'W1_final.pt',map_location='cpu',weights_only=False)
    m=WorldModel('W1',Geometry(WMROOT),c['normalization']).cuda();m.load_state_dict(c['model'])
    opt=torch.optim.Adam(m.parameters(),lr=2e-4)
    if 'optimizer' in c:opt.load_state_dict(c['optimizer'])
    for g in opt.param_groups:g['lr']=2e-4
    return m,opt,c['normalization']
def survival(s,ref,env):
    ep=(s[:,26:29]-ref[:,26:29]).norm(dim=-1)
    er=qlog(qmul(qconj(ref[:,29:33]),s[:,29:33])).norm(dim=-1)
    joint_margin=torch.minimum(s[:,:26]-(env.lo-.15),env.hi+.15-s[:,:26]).amin(-1)
    return torch.sigmoid((.15-ep)/.006)*torch.sigmoid((1.4-er)/.056)*torch.sigmoid(joint_margin/.006)
def imagine(policy,model,s,phase,last,env,H=32):
    # critic/model weights frozen by caller; input Jacobians stay live.
    obs=[];rewards=[];discounts=[];states=[];phases=[]
    alive=torch.ones(len(s),device=s.device);ret=torch.zeros_like(alive)
    for h in range(H):
        valid=phase<env.length-1
        o=observe_state(s,phase,last,env);a=policy.actor(o).clamp(-1,1)
        nxt=(phase+1).clamp(max=env.length-1)
        u=torch.maximum(env.lo,torch.minimum(env.hi,env.target[nxt]+a*env.scale))
        proposed=model(s,u);r,_=reward_from_state(proposed,env.ref[nxt],a,last)
        r=r*valid
        # Real reference end is absorbing; rollout sampling cutoff is not.
        d=.99*survival(proposed,env.ref[nxt],env)*(valid&(nxt<env.length-1))
        ret=ret+alive*r;alive=alive*d
        obs.append(o);rewards.append(r);discounts.append(d);states.append(s);phases.append(phase)
        s=torch.where(valid[:,None],proposed,s);last=torch.where(valid[:,None],a,last);phase=nxt
    tailobs=observe_state(s,phase,last,env);tail=policy.value(tailobs)
    ret=ret+alive*tail
    # Detached TD-lambda targets fixed once, not recomputed per critic epoch.
    with torch.no_grad():
        values=[policy.value(o.detach()) for o in obs]+[tail.detach()]
        target=values[-1];targets=[]
        for h in reversed(range(H)):
            target=rewards[h].detach()+discounts[h].detach()*(.05*values[h+1]+.95*target)
            targets.append(target)
        targets=torch.stack(targets[::-1]);valid=torch.stack(phases)<env.length-1
    return -ret.mean(),torch.stack(obs).detach()[valid],targets[valid].detach(),dict(imagined_return=float(ret.mean().detach()),survival_tail=float((alive/.99**H).mean().detach()),imagined_transitions=int(valid.sum()),computed_model_transitions=len(s)*H)
class Replay:
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed);self.episodes=collections.deque();self.counter=0
        archive=np.load(WMROOT/'dataset.npz');raw={k:archive[k] for k in ['states','controls']};archive.close();manifest=json.loads((WMROOT/'dataset_manifest.json').read_text())
        self.old=[]
        for e in manifest['episodes']:
            a=e['states_start'];b=e['controls_start'];n=e['steps']
            if e['split']=='train' and n>=8:self.old.append(dict(id='old_'+str(e['episode_id']),states=torch.tensor(raw['states'][a:a+n+1],device='cuda'),controls=torch.tensor(raw['controls'][b:b+n],device='cuda'),split='train'))
    def add(self,states,controls,phases,actions):
        n=len(controls);eid=self.counter;self.counter+=1
        if n<1:return
        self.episodes.append(dict(id=eid,states=torch.stack(states).detach(),controls=torch.stack(controls).detach(),phases=phases,actions=torch.stack(actions).detach(),split='validation' if eid%5==0 else 'train'))
        while sum(len(e['controls']) for e in self.episodes)>32768:self.episodes.popleft()
    def sample(self,batch=64,horizon=8,validation=False):
        new=[e for e in self.episodes if e['split']==('validation' if validation else 'train') and len(e['controls'])>=horizon]
        pools=[new] if validation else [self.old,new]
        pools=[p for p in pools if p]
        if not pools:return None
        picks=[];starts=[]
        for _ in range(batch):
            pool=pools[self.rng.integers(len(pools))];e=pool[self.rng.integers(len(pool))];k=self.rng.integers(len(e['controls'])-horizon+1)
            picks.append(e);starts.append(k)
        return torch.stack([e['states'][k:k+horizon+1] for e,k in zip(picks,starts)]),torch.stack([e['controls'][k:k+horizon] for e,k in zip(picks,starts)])
    def save(self,out):
        # Whole recent episodes; split fixed by episode ID before windowing.
        payload=[{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in e.items()} for e in self.episodes]
        torch.save(payload,out/'replay_latest.pt')
        write(out/'replay_manifest.json',dict(cap_controls=32768,split_rule='episode_id % 5 == 0 validation; incomplete episodes excluded',episodes=[dict(id=e['id'],split=e['split'],steps=len(e['controls'])) for e in self.episodes],old_source='runs/learning_trend_20261027/wm/dataset.npz',old_train_only=True))
def model_fit(model,opt,replay,updates=25):
    model.requires_grad_(True).train();losses=[];samples=0
    for j in range(updates):
        batch=replay.sample()
        if batch is None:break
        ss,uu=batch;s=ss[:,0];loss=0
        for h in range(8):s=model(s,uu[:,h]);loss=loss+model.loss(s,ss[:,h+1])/8
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(g);opt.step();losses.append(float(loss.detach()));samples+=len(s)*8
    opt.zero_grad(set_to_none=True);model.eval().requires_grad_(False)
    return dict(wm_loss=float(np.mean(losses)) if losses else None,wm_updates=len(losses),wm_training_samples=samples)
@torch.no_grad()
def model_valid(model,replay,H=8):
    batch=replay.sample(32,H,True)
    if batch is None:return {}
    ss,uu=batch;s=ss[:,0]
    for h in range(H):s=model(s,uu[:,h])
    return {f'validation_{H}_model_predictions':len(s)*H,f'validation_{H}_position_mm':float((s[:,26:29]-ss[:,-1,26:29]).norm(dim=-1).mean()*1000),f'validation_{H}_rotation_deg':float(qlog(qmul(qconj(ss[:,-1,29:33]),s[:,29:33])).norm(dim=-1).mean()*180/np.pi)}

