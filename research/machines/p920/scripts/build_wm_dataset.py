"""Whole-episode split first, windows second; preserve failures and source identity."""
import hashlib,json,collections
from pathlib import Path
import numpy as np
from exploration_common import ROOT,RUN,DATA,read_json,write_json

def classify(mask):
    if np.all(mask==0):return 0
    if mask[0]!=0 and np.all(mask==mask[0]):return 1
    return 2

def main():
    folder=RUN/'wm';folder.mkdir(exist_ok=True);sources=[]
    for arm in ['uniform_rsi','start50']:
        for p in sorted((RUN/arm/'episodes').glob('*.npz')):sources.append((p,'new_ppo_'+arm,None))
    for p in sorted((RUN/'collection/episodes').glob('*.npz')):sources.append((p,'collection',None))
    old=ROOT/'runs/grasp_repair_20261026'
    for base in ['gpu/baseline','gpu/grip_balance','gpu_original_graph/baseline_zero','gpu_original_graph/baseline_checkpoint','gpu_original_graph/grip_balance_checkpoint']:
        for p in sorted((old/base).rglob('trial_*.npz')):sources.append((p,'existing_gpu',int(p.stem.split('_')[-1])))
    states=[];actions=[];contacts=[];man=[];seen=set();state_offset=control_offset=0;duplicates=0
    for path,source,trial in sources:
        a=np.load(path);q=a['qpos'].astype(np.float32);v=a['qvel'].astype(np.float32);u=a['ctrl'].astype(np.float32);n=len(u)
        assert len(q)==len(v)==n+1 and np.isfinite(q).all() and np.isfinite(v).all()
        fingerprint=hashlib.sha256(q.tobytes()+v.tobytes()+u.tobytes()).hexdigest()
        if fingerprint in seen:duplicates+=1;continue
        seen.add(fingerprint)
        if 'hand_contact' in a:
            hand=a['hand_contact']>0;floor=a['floor_contact']>0
        else:
            logs=read_json(path.with_name(path.stem+'_contacts.json'));hand=[];floor=[]
            for row in logs:
                cs=row['contacts'];hand.append(any(c['kind']=='hand_object' and c['force_contact'][0]>.01 for c in cs));floor.append(any(c['kind']=='floor_object' and c['force_contact'][0]>.01 for c in cs))
            hand=np.array(hand);floor=np.array(floor)
        mask=hand.astype(np.uint8)+2*floor.astype(np.uint8);assert len(mask)==n
        initial=a['initial_ctrl'] if 'initial_ctrl' in a else u[0]
        history=np.tile(initial,(3,1));ss=[]
        for t in range(n+1):
            ss.append(np.r_[q[t],v[t],history.reshape(-1)])
            if t<n:history=np.vstack([history[1:],u[t]])
        group=f'existing_initial_condition_{trial}' if trial is not None else str(path.relative_to(RUN))
        split='validation' if (trial in [6,7] if trial is not None else int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0) else 'train'
        meta=read_json(path.with_suffix('.json')) if path.with_suffix('.json').exists() else {}
        man.append(dict(episode_id=len(man),source=source,path=str(path.relative_to(ROOT)),group=group,split=split,states_start=state_offset,controls_start=control_offset,steps=n,fingerprint=fingerprint,task_reason=meta.get('reason'),complete_episode=meta.get('complete_episode',True),initial_ctrl_source='logged reset control' if 'initial_ctrl' in a else 'first logged control repeated for unavailable prehistory; not future state data'))
        states.append(np.array(ss,dtype=np.float32));actions.append(u);contacts.append(mask);state_offset+=n+1;control_offset+=n
    s=np.concatenate(states);u=np.concatenate(actions);mask=np.concatenate(contacts);arrays=dict(states=s,controls=u,contact_mode=mask);windows={}
    for h in [8,32]:
        for split in ['train','validation']:
            ix=[]
            for ep in man:
                if ep['split']!=split:continue
                for t in range(ep['steps']-h+1):
                    c=ep['controls_start']+t;ix.append([ep['states_start']+t,c,ep['episode_id'],classify(mask[c:c+h])])
            x=np.array(ix,dtype=np.int64).reshape(-1,4);arrays[f'windows_{h}_{split}']=x;windows[f'{h}_{split}']={str(i):int((x[:,3]==i).sum()) for i in range(3)}
    train_groups={ep['group'] for ep in man if ep['split']=='train'};val_groups={ep['group'] for ep in man if ep['split']=='validation'};assert train_groups.isdisjoint(val_groups)
    np.savez_compressed(folder/'dataset.npz',**arrays)
    stats=[]
    for split in ['train','validation']:
        eps=[ep for ep in man if ep['split']==split];c=np.concatenate([mask[e['controls_start']:e['controls_start']+e['steps']] for e in eps]);stats.append(dict(split=split,episodes=len(eps),transitions=len(c),no_any_contact_fraction=float((c==0).mean()),hand_contact_fraction=float(((c&1)>0).mean()),floor_contact_fraction=float(((c&2)>0).mean()),floor_only_fraction=float((c==2).mean()),sources=dict(collections.Counter(e['source'] for e in eps))))
    rng=np.random.default_rng(29)
    for h in [8,32]:
        x=arrays[f'windows_{h}_validation'];chosen=[]
        for label in range(3):
            indices=np.flatnonzero(x[:,3]==label);chosen.extend(rng.choice(indices,min(256,len(indices)),replace=False).tolist())
        np.save(folder/f'validation_sample_{h}.npy',np.array(chosen,dtype=np.int64))
    write_json(folder/'dataset_manifest.json',dict(episodes=man,duplicate_episodes_removed=duplicates,statistics=stats,window_counts=windows,split_rule='Split whole episode/group before extracting windows. All old recordings of initial-condition families 6/7 are validation; 0..5 train. New independent episode IDs hash 20% to validation. Exact duplicate trajectories are removed. No fixed PPO evaluation trajectories are used for WM fitting.',contact_classes={'0':'no loaded hand OR floor contact throughout window','1':'same nonzero hand/floor contact mode throughout window','2':'mode changes at least once, including contact onset/loss or hand/floor transition'},contact_limit='Mode uses loaded contact >0.01N and distinguishes hand/floor, not individual fingertip swaps. Last physical substep sampled at t+18ms.',window_leakage_check=True,group_sets_disjoint=True,evaluation_sampling='Up to 256 windows per regime and horizon, fixed seed29. Report regime counts and natural-distribution weighted mean, not unweighted pooled success.',training_sampling='Equal probability of available 8-step contact regimes, then uniform within regime; identical sampled batches for W0 and W1.',full_task_success_required=False))
    print(json.dumps(dict(episodes=len(man),transitions=len(u),stats=stats,windows=windows),indent=2))


if __name__=='__main__':main()
