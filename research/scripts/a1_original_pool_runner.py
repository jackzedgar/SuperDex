"""Task-specific PPO with8independent unmodified nworld1 A1 environments."""
import copy
import numpy as np
import torch
import warp as wp
from a1_env import A1Env
from a1_policy import policy_inputs
from a1_state import encode
from a1_reset_memory import apply_velocity_reset_memory
from a1_pool_gae import per_world_gae

class OriginalPoolRunner:
    def __init__(self,task_path,policy,parent_worker_rng,validate=False):
        self.n=8;self.policy=policy;self.validate=validate
        self.envs=[A1Env(task_path,capture=True,determinism='constraint') for _ in range(self.n)]
        assert all(type(e) is A1Env and e.n==1 and e.determinism=='constraint' for e in self.envs)
        assert len({id(e.m) for e in self.envs})==self.n
        for key in ['qpos','qvel','ctrl','history','servo','context','phase','actions','last_action']:
            assert len({getattr(e,key).data_ptr() for e in self.envs})==self.n
        self.task=self.envs[0].task;self.stream=torch.cuda.Stream();torch.cuda.set_stream(self.stream)
        self.mask=self.task.mask.expand(self.n,-1).clone()
        self.state_history=torch.zeros(self.n,3,287,device='cuda');self.action_history=torch.zeros(self.n,3,52,device='cuda')
        self.context=torch.zeros(self.n,276,device='cuda');self.phase=torch.zeros(self.n,dtype=torch.long,device='cuda');self.last_action=torch.zeros(self.n,52,device='cuda')
        root=np.random.default_rng();root.bit_generator.state=copy.deepcopy(parent_worker_rng)
        self.lane_seeds=root.integers(0,2**63-1,size=self.n).tolist();self.rngs=[np.random.default_rng(s) for s in self.lane_seeds]
        self.root_rng_after_derivation=copy.deepcopy(root.bit_generator.state)
        self.real_controls=0;self.warmup_controls=sum(e.physical_controls for e in self.envs);self.episodes=[]
        self.ep_ids=[0]*self.n;self.ep_returns=[0.]*self.n;self.ep_controls=[0]*self.n;self.reset_records=[]
        self.audit=dict(input_checks=0,history_checks=0,reset_isolation_checks=0,max_obs_error=0.,max_goal_error=0.,max_value_input_error=0.)
        for i in range(self.n):self.reset_lane(i)
        with torch.no_grad():
            self.inference();self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=self.stream):self.inference()

    def inference(self):
        self.obs,self.goal=policy_inputs(self.task,self.state_history,self.action_history,self.phase)
        self.value_input=torch.cat([encode(self.state_history[:,-1],self.mask),self.goal,self.context,self.last_action],-1)
        self.action,self.logprob,self.value=self.policy.act(self.obs,self.goal,self.value_input,self.mask)

    @torch.no_grad()
    def reset_lane(self,i):
        e=self.envs[i];rng=self.rngs[i];phase=int(rng.integers(e.task.length-1));seed=int(rng.integers(2**31))
        self.stream.synchronize()
        others=[k for k in range(self.n) if k!=i]
        if self.validate:
            prior=[self.state_history[others].clone(),self.action_history[others].clone(),self.context[others].clone(),self.phase[others].clone()]
        with torch.cuda.stream(e.torch_stream):
            wp.set_stream(e.stream);e.reset(phase,seed=seed,randomize=True)
            apply_velocity_reset_memory(e)
        self.stream.wait_stream(e.torch_stream)
        with torch.cuda.stream(self.stream):
            self.state_history[i:i+1].copy_(e.state()[:,None].expand(-1,3,-1));self.action_history[i].zero_()
            self.context[i:i+1].copy_(e.context);self.phase[i:i+1].copy_(e.phase);self.last_action[i].zero_()
            if self.validate:
                actual=[self.state_history[others],self.action_history[others],self.context[others],self.phase[others]]
                assert all(torch.equal(a,b) for a,b in zip(prior,actual));self.audit['reset_isolation_checks']+=1
        self.ep_ids[i]+=1;self.ep_returns[i]=0.;self.ep_controls[i]=0
        self.reset_records.append(dict(lane=i,episode_id=self.ep_ids[i],phase=phase,seed=seed,reset_mode='uniform_reference',training_memory='velocity_consistent'))

    @torch.no_grad()
    def check_inputs(self):
        for i,e in enumerate(self.envs):
            obs,goal=policy_inputs(e.task,self.state_history[i:i+1],self.action_history[i:i+1],e.phase)
            value=torch.cat([encode(self.state_history[i:i+1,-1],e.task.mask),goal,e.context,e.last_action],-1)
            for key,a,b in [('obs',obs,self.obs[i:i+1]),('goal',goal,self.goal[i:i+1]),('value_input',value,self.value_input[i:i+1])]:
                delta=float((a-b).abs().max());self.audit['max_'+key+'_error']=max(self.audit['max_'+key+'_error'],delta)
                assert delta<=2e-5,(key,i,delta)
            assert torch.equal(self.state_history[i,-1],e.state()[0])
            assert torch.equal(self.last_action[i],e.last_action[0])
            self.audit['input_checks']+=1

    @torch.no_grad()
    def rollout(self,n=2048,should_stop=lambda:False):
        assert n==2048 and n%self.n==0
        torch.cuda.set_stream(self.stream)
        keys=['obs','goal','value_input','action','old_logprob','old_value','reward','done','mask','phase','episode_id']
        cols={k:[] for k in keys};first_real=self.real_controls
        for tick in range(n//self.n):
            if should_stop():break
            self.graph.replay()
            if self.validate and tick in [0,1,31,127,255]:self.check_inputs()
            for key,value in [('obs',self.obs),('goal',self.goal),('value_input',self.value_input),('action',self.action),('old_logprob',self.logprob),('old_value',self.value),('mask',self.mask),('phase',self.phase)]:cols[key].append(value.clone())
            cols['episode_id'].append(torch.tensor(self.ep_ids,device='cuda'))
            if self.validate:before_sh=self.state_history.clone();before_ah=self.action_history.clone()
            for i,e in enumerate(self.envs):
                e.torch_stream.wait_stream(self.stream)
                with torch.cuda.stream(e.torch_stream):wp.set_stream(e.stream);e.step(self.action[i:i+1])
            for e in self.envs:self.stream.wait_stream(e.torch_stream)
            with torch.cuda.stream(self.stream):
                states=torch.cat([e.current_state for e in self.envs]);last=torch.cat([e.last_action for e in self.envs]);phases=torch.cat([e.phase for e in self.envs])
                reward=torch.cat([e.reward for e in self.envs]);done=torch.cat([e.done for e in self.envs]);cols['reward'].append(reward.clone());cols['done'].append(done.clone())
                self.state_history.copy_(torch.cat([self.state_history[:,1:],states[:,None]],1));self.action_history.copy_(torch.cat([self.action_history[:,1:],last[:,None]],1));self.last_action.copy_(last);self.phase.copy_(phases)
                if self.validate:
                    assert torch.equal(self.state_history[:,:2],before_sh[:,1:]) and torch.equal(self.state_history[:,-1],states)
                    assert torch.equal(self.action_history[:,:2],before_ah[:,1:]) and torch.equal(self.action_history[:,-1],last)
                    assert all(torch.equal(self.action[i],e.actions[0]) for i,e in enumerate(self.envs));self.audit['history_checks']+=self.n
                host=torch.cat([torch.cat([e.reward[:,None],e.done.float()[:,None],e.info['reason'].float()[:,None],e.info['regular_success'].float()[:,None],e.info['strict_success'].float()[:,None],e.info['mean_errors'],e.count[:,None]],-1) for e in self.envs]).cpu().numpy()
            self.real_controls+=self.n
            for i,row in enumerate(host):
                self.ep_returns[i]+=float(row[0]);self.ep_controls[i]+=1
                assert int(row[-1])==self.ep_controls[i]
                if bool(row[1]):
                    e=self.envs[i]
                    self.episodes.append(dict(lane=i,episode_id=self.ep_ids[i],task=e.task.meta['task_id'],reset_mode='uniform_reference',reset_memory_mode='velocity_consistent',bootstrap=False,controls=self.ep_controls[i],start_phase=e.start_phase,seed=e.context_record['seed'],return_sum=self.ep_returns[i],failure_reason=int(row[2]),regular_success=bool(row[3]),strict_success=bool(row[4]),mean_errors=row[5:11].tolist()))
                    self.reset_lane(i)
        if not cols['reward']:return None
        self.graph.replay();bootstrap=self.value.clone()
        time_major={k:torch.stack(v) for k,v in cols.items()}
        advantage,returns=per_world_gae(time_major['reward'],time_major['old_value'],time_major['done'],bootstrap)
        time_major.update(advantage=advantage,**{'return':returns})
        d={k:v.flatten(0,1) for k,v in time_major.items()}
        assert len(d['reward'])==self.real_controls-first_real
        if self.validate:
            ids=time_major['episode_id'];ended=time_major['done']
            assert torch.equal(ids[1:]-ids[:-1],ended[:-1].to(ids.dtype))
        self.last_time_major={k:time_major[k].clone() for k in ['reward','old_value','done','advantage','return','episode_id']};self.last_bootstrap=bootstrap
        return d

    def rng_state(self):
        return dict(lane_seed_derivation=self.lane_seeds,root_after_derivation=self.root_rng_after_derivation,lanes=[copy.deepcopy(r.bit_generator.state) for r in self.rngs])
