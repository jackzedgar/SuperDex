"""Frozen mean policy plus bounded knots; independent original nworld1 environments."""
import numpy as np
import torch
import warp as wp
from a1_env import A1Env
from a1_policy import policy_inputs
from a1_control_witness_math import knot_bias

class WitnessPool:
    @torch.no_grad()
    def __init__(self,task_path,policy):
        self.n=8;self.policy=policy
        self.envs=[A1Env(task_path,capture=True,determinism='constraint') for _ in range(8)]
        assert all(type(e) is A1Env and e.n==1 and e.determinism=='constraint' for e in self.envs)
        assert len({id(e.m) for e in self.envs})==8
        for key in ['qpos','qvel','ctrl','history','servo','context','phase','actions','last_action']:
            assert len({getattr(e,key).data_ptr() for e in self.envs})==8
        self.task=self.envs[0].task;self.horizon=self.task.length-1
        assert self.horizon>=2
        self.stream=torch.cuda.Stream();torch.cuda.set_stream(self.stream)
        self.mask=self.task.mask.repeat_interleave(26,-1).expand(8,52).clone()
        self.state_history=torch.zeros(8,3,287,device='cuda');self.action_history=torch.zeros(8,3,52,device='cuda')
        self.phase=torch.zeros(8,dtype=torch.long,device='cuda');self.knots=torch.zeros(8,4,52,device='cuda')
        self.controls=0;self.warmup_controls=sum(e.physical_controls for e in self.envs)
        self.audit=dict(reset_checks=0,reset_isolation_checks=0,history_checks=0,input_checks=0,knot_checks=0,max_input_error=0.,max_mean_error=0.,max_knot_error=0.)
        for i in range(8):self.reset_lane(i,280043)
        self.inference();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):self.inference()

    @torch.no_grad()
    def inference(self):
        self.obs,self.goal=policy_inputs(self.task,self.state_history,self.action_history,self.phase)
        self.mean=self.policy.distribution(self.obs,self.goal).mean*self.mask
        x=(self.phase.float()/(self.horizon-1)).clamp(0,1)*3;left=x.long().clamp(max=2);w=(x-left.float())[:,None]
        a=self.knots.gather(1,left[:,None,None].expand(-1,1,52))[:,0]
        b=self.knots.gather(1,(left+1)[:,None,None].expand(-1,1,52))[:,0]
        self.bias=((1-w)*a+w*b)*self.mask;self.action=self.mean+self.bias

    @torch.no_grad()
    def reset_lane(self,i,seed):
        self.stream.synchronize();others=[j for j in range(8) if j!=i]
        prior=[self.state_history[others].clone(),self.action_history[others].clone(),self.phase[others].clone()]
        self.stream.synchronize();e=self.envs[i]
        with torch.cuda.stream(e.torch_stream):
            wp.set_stream(e.stream);e.reset(0,seed=int(seed),randomize=True)
            world=e._world(e.qpos[:,:e.task.nu])
            assert int(e.phase)==0 and not bool(e.last_action.any())
            assert torch.equal(e.ctrl,e.qpos[:,:e.task.nu]) and torch.equal(e.history,world[:,:,None].expand(-1,-1,3,-1)) and torch.equal(e.servo,world[:,:,:6])
            assert torch.equal(e.qpos,e.task.qpos[:1]) and torch.equal(e.qvel,e.task.qvel[:1])
        self.stream.wait_stream(e.torch_stream)
        with torch.cuda.stream(self.stream):
            self.state_history[i:i+1].copy_(e.state()[:,None].expand(-1,3,-1));self.action_history[i].zero_();self.phase[i]=0
            assert all(torch.equal(a,b) for a,b in zip(prior,[self.state_history[others],self.action_history[others],self.phase[others]]))
        self.audit['reset_checks']+=1;self.audit['reset_isolation_checks']+=1

    @torch.no_grad()
    def check_inputs(self,indices,knots):
        for i in indices:
            e=self.envs[i];obs,goal=policy_inputs(e.task,self.state_history[i:i+1],self.action_history[i:i+1],e.phase)
            error=max(float((obs-self.obs[i:i+1]).abs().max()),float((goal-self.goal[i:i+1]).abs().max()))
            mu=self.policy.distribution(obs,goal).mean*e.task.mask.repeat_interleave(26,-1)
            delta=float((mu-self.mean[i:i+1]).abs().max())
            assert error<=2e-5 and delta<=2e-5,(i,error,delta)
            expected=knot_bias(knots[i],int(e.phase),self.horizon,e.task.mask[0].cpu().numpy())
            kd=float(abs(expected-self.bias[i].cpu().numpy()).max());assert kd<=2e-6
            assert torch.equal(self.state_history[i,-1],e.state()[0])
            assert torch.equal(self.action_history[i,-1],e.last_action[0])
            self.audit['max_input_error']=max(self.audit['max_input_error'],error);self.audit['max_mean_error']=max(self.audit['max_mean_error'],delta)
            self.audit['max_knot_error']=max(self.audit['max_knot_error'],kd);self.audit['input_checks']+=1;self.audit['knot_checks']+=1

    @torch.no_grad()
    def run(self,knots,seeds,should_stop,validate=False):
        knots=np.asarray(knots,dtype=np.float32);assert 1<=len(knots)<=8 and knots.shape[1:]==(4,52) and len(seeds)==len(knots)
        assert np.isfinite(knots).all() and np.max(abs(knots))<=.25000001
        active_mask=self.mask[0].cpu().numpy();assert not np.any(knots*(1-active_mask))
        first=sum(e.physical_controls for e in self.envs);first_count=self.controls;rows=[]
        for i,seed in enumerate(seeds):
            if should_stop():return []
            self.reset_lane(i,seed);e=self.envs[i]
            with torch.cuda.stream(self.stream):
                initial=e.state()[0].cpu().numpy().copy();general=torch.cat([e.qpos,e.qvel],-1)[0].cpu().numpy().copy()
            rows.append(dict(seed=int(seed),controls=0,return_sum=0.,complete=False,regular_success=False,strict_success=False,
                failure_reason=None,phase_start=0,reset_memory='original',context=e.context_record,
                trace={'state':[initial],'generalized':[general],'raw_action':[],'last_action':[],'request':[],'applied':[],'bias':[],'reward':[],'position_rotation_error':[],'contact_summary':[],'phase':[0],'done':[]},knots=knots[i].tolist()))
        with torch.cuda.stream(self.stream):
            self.knots.zero_();self.knots[:len(knots)].copy_(torch.as_tensor(knots,device='cuda'))
        active=list(range(len(rows)));tick=0
        while active and not should_stop():
            torch.cuda.set_stream(self.stream);self.graph.replay()
            if tick==0 or (validate and tick in (1,7,31)):self.check_inputs(active,knots)
            if validate:old_sh=self.state_history.clone();old_ah=self.action_history.clone()
            for i in active:
                e=self.envs[i];e.torch_stream.wait_stream(self.stream)
                with torch.cuda.stream(e.torch_stream):wp.set_stream(e.stream);e.step(self.action[i:i+1])
            for i in active:self.stream.wait_stream(self.envs[i].torch_stream)
            with torch.cuda.stream(self.stream):
                fields=[]
                for i in active:
                    e=self.envs[i];q=torch.cat([e.qpos,e.qvel],-1);ci=torch.maximum(e.lo-e.ctrl,e.ctrl-e.hi).amax().clamp(min=0)
                    aux=torch.stack([e.phase[0],e.reward[0],e.done[0],e.info['reason'][0],e.info['regular_success'][0],e.info['strict_success'][0],e.info['pos_error'][0],e.info['rot_error'][0],e.info['finite'][0],e.contact_summary[9],ci,e.count[0]])[None]
                    fs=[e.current_state,q,e.actions,e.last_action,e.requested,e.applied,self.bias[i:i+1],e.contact_summary[None],e.info['mean_errors'],aux]
                    widths=[v.shape[-1] for v in fs];fields.append(torch.cat(fs,-1))
                    self.state_history[i:i+1].copy_(torch.cat([self.state_history[i:i+1,1:],e.current_state[:,None]],1))
                    self.action_history[i:i+1].copy_(torch.cat([self.action_history[i:i+1,1:],e.last_action[:,None]],1));self.phase[i:i+1].copy_(e.phase)
                    if validate:
                        assert torch.equal(self.state_history[i,:2],old_sh[i,1:]) and torch.equal(self.action_history[i,:2],old_ah[i,1:])
                        assert torch.equal(e.actions[0],self.action[i]) and torch.equal(e.last_action[0],self.action[i].clamp(-1,1)*self.mask[i])
                        self.audit['history_checks']+=1
                host=torch.cat(fields,0).cpu().numpy()
            still=[];self.controls+=len(active)
            for i,values in zip(active,host):
                assert np.isfinite(values).all(),'Nonfinite physical observation'
                parts=np.split(values,np.cumsum(widths)[:-1]);state,general,raw,last,request,applied,bias,contact,mean_errors,aux=parts
                phase,reward,done,reason,regular,strict,pos,rot,finite,overflow,ctrl_error,count=aux
                assert finite and not overflow and ctrl_error<=2e-5 and np.max(abs(bias))<=.250001
                row=rows[i];row['controls']+=1;row['return_sum']+=float(reward)
                assert int(count)==row['controls'] and int(phase)==row['controls']
                row.update(failure_reason=int(reason),regular_success=bool(regular),strict_success=bool(strict),mean_errors=mean_errors.tolist())
                for key,value in [('state',state),('generalized',general),('raw_action',raw),('last_action',last),('request',request),('applied',applied),('bias',bias),('reward',reward),('position_rotation_error',np.array([pos,rot])),('contact_summary',contact),('phase',phase),('done',done)]:row['trace'][key].append(np.array(value,copy=True))
                if done:
                    row['complete']=True
                    assert not regular or row['controls']==self.horizon
                else:still.append(i)
            active=still;tick+=1
        assert sum(e.physical_controls for e in self.envs)-first==self.controls-first_count==sum(r['controls'] for r in rows)
        for row in rows:
            if not row['complete']:row.update(status='budget_truncation',regular_success=False,strict_success=False)
            else:row['status']='reference_end' if row['controls']==self.horizon and row['failure_reason']==0 else 'original_termination'
        return rows
