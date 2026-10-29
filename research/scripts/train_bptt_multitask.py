"""Two objects, ONE shared MLP and ONE shared actor/critic. Serial single worlds."""
import argparse,time,json,signal
import numpy as np,torch
from torch import nn
import real_env
from train_real_ppo import Policy
from train_core_policy import evaluate_policy
from exploration_common import Contacts,EpisodeLog,activate
from world_model_mini import WorldModel,Geometry,encode,compose
import bptt_common as bc
from bptt_common import ROOT,RUN,WMROOT,write,Replay,model_fit,model_valid
from core_policy_common import observe_state as original_observe,physical_state,model_diagnostics
from probe_model_gradient import reward_from_state
TASK_META=json.loads((RUN/'task_manifest.json').read_text())['tasks']
TASKS=[ROOT/t['path'] for t in TASK_META]
GEOMETRIES=[ROOT/t['geometry'] for t in TASK_META]
assert len(TASKS)==2, 'This registered pilot has exactly two tasks.'
def context(env,task):
    gi=np.flatnonzero(env.m.geom_contype==2)[0];bi=env.m.geom_bodyid[gi]
    physical=np.r_[env.m.body_mass[bi]*25,env.m.body_inertia[bi]*1e5,env.m.geom_friction[gi]]
    return torch.tensor(np.r_[np.eye(2)[task],physical],device='cuda',dtype=torch.float32)
def observe(s,ph,last,env):
    return torch.cat([original_observe(s,ph,last,env),env.context.expand(len(s),-1)],-1)
class TaskEnv(real_env.RealEnv):
    def __init__(self,task):
        old=real_env.DATA;real_env.DATA=TASKS[task]
        try:super().__init__(1)
        finally:real_env.DATA=old
        self.task=task;self.context=context(self,task)
    def observe(self):
        o=super().observe()
        return torch.cat([o,self.context.expand(len(o),-1)],-1) if hasattr(self,'context') else o
class SharedModel(WorldModel):
    def __init__(self,norm,contexts):
        super().__init__('W1',Geometry(GEOMETRIES[0]),norm)
        self.geometries=nn.ModuleList([self.geometry,Geometry(GEOMETRIES[1])])
        self.net[0]=nn.Linear(264,256);self.register_buffer('contexts',torch.stack(contexts));self.task=0
    def forward(self,s,u):
        f=[(encode(s)-self.input_mean)/self.input_std,(u-self.control_mean)/self.control_std,self.geometries[self.task](s),self.contexts[self.task].expand(len(s),-1)]
        d=self.net(torch.cat(f,-1).clamp(-20,20))*self.delta_std+self.delta_mean
        return compose(s,d,u)
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=131072);p.add_argument('--name',default='shared_two_objects');a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(71);start=time.time();out=RUN/a.name;out.mkdir(exist_ok=False)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    envs=[TaskEnv(i) for i in range(2)];contacts=[Contacts(e) for e in envs]
    src=torch.load(ROOT/'runs/explore_20261026/start50/checkpoint_final.pt',map_location='cpu',weights_only=False)
    policy=Policy(127).cuda()
    with torch.no_grad():
        for key,x in policy.state_dict().items():
            z=src['policy'][key]
            if key in ['actor.0.weight','critic.0.weight']:x.zero_();x[:,:118].copy_(z)
            else:x.copy_(z)
    base=torch.load(WMROOT/'W1_final.pt',map_location='cpu',weights_only=False)
    model=SharedModel(base['normalization'],[e.context for e in envs]).cuda()
    with torch.no_grad():
        ms=model.state_dict()
        for key,z in base['model'].items():
            if key=='net.0.weight':ms[key].zero_();ms[key][:,:255].copy_(z)
            elif key in ms:ms[key].copy_(z)
    # Original geometry buffers of task 0 are aliased intentionally; task 1 fixed.
    mopt=torch.optim.Adam(model.parameters(),lr=2e-4)
    aopt=torch.optim.Adam(policy.actor.parameters(),lr=3e-5,betas=(.9,.95));vopt=torch.optim.Adam(policy.critic.parameters(),lr=1e-4)
    model.requires_grad_(False);replays=[Replay(88+i) for i in range(2)];replays[1].old=[]
    bc.observe_state=observe
    cfg=dict(tasks=[dict(id=i,path=str(t.relative_to(ROOT)),object_id=TASK_META[i]['object_id']) for i,t in enumerate(TASKS)],actor_input='118 + two task indicators + mass/inertia/friction scaled context7',world_model='ONE3x256MLP,264 inputs; task geometry query buffers + physical context; weights shared and jointly updated',normalization='Frozen original W1 affine scales; DVD domain shift is reported; no validation fitting',seed=71,initialization='Original shared61440step pretrained spoon actor; new9 input weights zero; WM originalW1 plus9zero columns',controls=a.steps,warmup_controls=4096,horizon=32,actor_lr=3e-5,model_update='25updates per task every1024 realcontrols; completeepisodesonly',sampling='Alternate128-control task blocks; incomplete boundary episodes excluded fromWM fitting; no sampled cross-task window',single_world='Two one-world buffers run serially; only selected buffer stepped, no batching/DDP',shared_parameters=True)
    write(out/'config.json',cfg)
    initial=[dict(np.load(ROOT/t['development_initials'])) for t in TASK_META];curves=[];rows=open(out/'training.jsonl','w');version=[];ids=torch.tensor([0],device='cuda')
    counts=dict(real_controls=0,task_controls=[0,0],evaluation_controls=0,actor_updates=0,critic_updates=0,wm_updates=0,wm_training_samples=0,imagined_transitions=0,computed_model_transitions=0,diagnostic_model_predictions=0)
    def save(label):
        torch.save(dict(policy=policy.state_dict(),model=model.state_dict(),normalization=base['normalization'],actor_optimizer=aopt.state_dict(),critic_optimizer=vopt.state_dict(),model_optimizer=mopt.state_dict(),counts=counts,config=cfg),out/label)
    def evaluate(step):
        results=[]
        for i,env in enumerate(envs):
            activate(env);r=evaluate_policy(env,policy,out/f'development/step_{step:06d}/task_{i}',initial[i],contacts[i]);results.append(r);counts['evaluation_controls']+=r['physical_controls']
        curves.append(dict(steps=step,process_seconds=time.time()-start,tasks=results));write(out/'development_curve.json',curves);save(f'step_{step:06d}.pt')
        print('SHARED',step,[r['successes'] for r in results],flush=True)
    evaluate(0)
    for block in range(a.steps//128):
        if stop[0]:break
        task=block%2;env=envs[task];model.task=task;activate(env);replay=replays[task]
        def reset():
            env.reset(ids,bool(torch.rand((),device='cuda')>=.5));h=env.ctrl[:,None,:].expand(1,3,26).clone()
            return h,[physical_state(env,h)[0]],[],[],[]
        history,es,eu,ea,ep=reset();ss=[];ph=[];la=[];rs=[];err=[];nexts=[];ct=[];acts=[];donebuf=[]
        for t in range(128):
            with torch.no_grad():
                state=physical_state(env,history);last=env.last_action.clone();phase=env.phase.clone();dist=policy.distribution(env.observe());action=dist.sample()
                _,r,done,info=env.step(action);control=env.ctrl.clone();history=torch.cat([history[:,1:],control[:,None]],1);nxt=physical_state(env,history)
                ss.append(state[0]);ph.append(phase[0]);la.append(last[0]);rs.append(r[0]);err.append(torch.stack([info['pos_error'][0],info['rot_error'][0]]));nexts.append(nxt[0]);ct.append(control[0]);acts.append(action[0]);donebuf.append(done[0])
                es.append(nxt[0]);eu.append(control[0]);ea.append(action[0]);ep.append(int(phase[0]))
            if bool(done[0]):
                assert bool(info['finite'][0]);replay.add(es,eu,ep,ea);history,es,eu,ea,ep=reset()
            counts['real_controls']+=1;counts['task_controls'][task]+=1
        state=torch.stack(ss);phase=torch.stack(ph);last=torch.stack(la)
        row=dict(step=counts['real_controls'],task=task,reward=float(torch.stack(rs).mean()),position_mm=float(torch.stack(err)[:,0].mean()*1000),rotation_deg=float(torch.stack(err)[:,1].mean()*180/np.pi),process_seconds=time.time()-start)
        if block%8 in [0,1]:
            md=model_diagnostics(model,state,torch.stack(nexts),torch.stack(ct),torch.stack(acts),phase,last,torch.stack(rs),torch.stack(donebuf),env)
            counts['diagnostic_model_predictions']+=md.get('predictions',0)
            with (out/'prefit_model_diagnostics.jsonl').open('a') as f:f.write(json.dumps(dict(step=counts['real_controls'],task=task,before_fit=True,**md))+'\n')
        if counts['real_controls']%1024==0:
            for tid in [0,1]:
                model.task=tid;fit=model_fit(model,mopt,replays[tid],25)
                counts['wm_updates']+=fit['wm_updates'];counts['wm_training_samples']+=fit['wm_training_samples']
                valid={**model_valid(model,replays[tid],8),**model_valid(model,replays[tid],32)}
                counts['diagnostic_model_predictions']+=sum(v for k,v in valid.items() if k.endswith('model_predictions'))
                row[f'model_task{tid}']=dict(**fit,**valid)
            model.task=task
            version.append(dict(step=counts['real_controls'],training_ids=[[e['id'] for e in rp.episodes if e['split']=='train'] for rp in replays]))
            write(out/'model_versions.json',version)
            torch.save(dict(net=model.net.state_dict(),optimizer=mopt.state_dict(),normalization=base['normalization'],contexts=model.contexts,geometry_manifest='runs/bptt_iteration_20261027/task_manifest.json',counts=counts),out/f'model_stage_{counts["real_controls"]:06d}.pt')
        if counts['real_controls']>4096:
            policy.critic.requires_grad_(False);model.requires_grad_(False);ix=torch.arange(0,128,4,device='cuda')
            loss,o,targets,d=bc.imagine(policy,model,state[ix],phase[ix],last[ix],env,32)
            aopt.zero_grad(set_to_none=True);loss.backward();g=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);assert torch.isfinite(g)
            for x in policy.actor.parameters():x.grad.clamp_(-.1,.1)
            aopt.step();counts['actor_updates']+=1;counts['imagined_transitions']+=d['imagined_transitions'];counts['computed_model_transitions']+=d['computed_model_transitions'];row.update(d,grad_norm=float(g))
            policy.critic.requires_grad_(True)
            for _ in range(5):
                for sel in torch.randperm(len(targets),device='cuda').chunk(4):
                    if len(sel):
                        lv=.5*(policy.value(o[sel])-targets[sel]).square().mean();vopt.zero_grad(set_to_none=True);lv.backward();vg=torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.);assert torch.isfinite(vg);vopt.step();counts['critic_updates']+=1
            vopt.zero_grad(set_to_none=True)
        if block%8==0:rows.write(json.dumps(row)+'\n');rows.flush();write(out/'progress.json',dict(**counts,seconds=time.time()-start))
        if counts['real_controls']%8192==0:
            # Training-only anchors, diagnostic gradients; no optimizer step.
            gradients=[];gd=[];policy.critic.requires_grad_(False);model.requires_grad_(False)
            for tid in [0,1]:
                selected=[e for e in replays[tid].episodes if e['split']=='train'][-16:]
                if not selected:break
                model.task=tid
                anchors=torch.stack([e['states'][0] for e in selected])
                phases=torch.tensor([e['phases'][0] for e in selected],device='cuda')
                obj,_,_,diag=bc.imagine(policy,model,anchors,phases,torch.zeros(len(anchors),26,device='cuda'),envs[tid],32)
                gg=torch.autograd.grad(obj,list(policy.actor.parameters()))
                gradients.append(torch.cat([g.flatten() for g in gg]));gd.append(diag)
                counts['computed_model_transitions']+=diag['computed_model_transitions']
            if len(gradients)==2:
                cosine=float(torch.nn.functional.cosine_similarity(gradients[0],gradients[1],dim=0))
                with (out/'task_gradient_interference.jsonl').open('a') as f:f.write(json.dumps(dict(step=counts['real_controls'],cosine=cosine,norms=[float(g.norm()) for g in gradients],anchor_source='Most recent16 complete training episodes per task, first state; no evaluation data',interpretation='Gradient alignment diagnostic, not causal isolated-task interference proof'))+'\n')
            policy.critic.requires_grad_(True);model.task=task
            evaluate(counts['real_controls'])
    save('checkpoint_final.pt')
    for i,r in enumerate(replays):
        d=out/f'replay_task{i}';d.mkdir(exist_ok=True);r.save(d)
    if not stop[0] and curves[-1]['steps']!=counts['real_controls']:evaluate(counts['real_controls'])
    write(out/'summary.json',dict(**counts,seconds=time.time()-start,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,stopped=stop[0],shared_parameters=True));rows.close()
if __name__=='__main__':main()

