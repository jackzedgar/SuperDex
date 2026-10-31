"""Complete-episode collection with eight independent unchanged original worlds."""
import copy,json,time
import numpy as np
import torch
import warp as wp
from a1_env import A1Env
from a1_policy import policy_inputs
from a1_episode_data import EpisodeWriter,pack_state
from a1_episode_contract import complete_controls,is_complete_episode
from a1_collection_pool_records import EpisodePlan,EpisodeBuffer,episode_noise

class CollectionPool:
    @torch.no_grad()
    def __init__(self,task_path,policy,output,seed,checkpoint_record,source_family,training_reset_mode,quality='hardware_smoke_only',formal=False,n=8,validate=False,should_stop=lambda:False):
        self.n=n;assert n==8;self.policy=policy;self.validate=validate;self.formal=formal;self.training_reset_mode=training_reset_mode
        self.envs=[];self.physical_at_close=None
        for _ in range(n):
            if should_stop():raise InterruptedError('Bound during environment construction')
            self.envs.append(A1Env(task_path,capture=True,determinism='constraint'))
        assert all(type(e) is A1Env and e.n==1 and e.determinism=='constraint' for e in self.envs)
        assert len({id(e.m) for e in self.envs})==n
        for key in ['qpos','qvel','ctrl','history','servo','context','phase','actions','last_action']:
            assert len({getattr(e,key).data_ptr() for e in self.envs})==n,key
        self.model_fields=['body_mass','body_inertia','body_ipos','body_iquat','geom_friction','actuator_gainprm','actuator_biasprm','body_subtreemass','body_invweight0','dof_invweight0']
        for key in self.model_fields:assert len({wp.to_torch(getattr(e.wm,key)).data_ptr() for e in self.envs})==n,key
        self.task=self.envs[0].task;self.stream=torch.cuda.Stream();torch.cuda.set_stream(self.stream)
        self.mask=self.task.mask.repeat_interleave(26,-1).expand(n,52).clone();self.phase=torch.zeros(n,dtype=torch.long,device='cuda')
        self.state_history=torch.zeros(n,3,287,device='cuda');self.action_history=torch.zeros(n,3,52,device='cuda');self.noise=torch.zeros(n,52,device='cuda')
        for i,e in enumerate(self.envs):
            self.stream.wait_stream(e.torch_stream);self.state_history[i].copy_(e.current_state.expand(3,-1))
        self.inference();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=self.stream):self.inference()
        self.writer=EpisodeWriter(output,self.task,quality,checkpoint_record,source_family)
        self.plan=EpisodePlan(self.writer.path/'pool_reservations.jsonl',seed,self.task.length-1);self.plan.verify_commits(self.writer.entries)
        self.buffers={};self.noise_rng={};self.real_controls=0;self.warmup_controls=sum(e.physical_controls for e in self.envs)
        self.audit=dict(reset_checks=0,reset_isolation_checks=0,history_checks=0,input_checks=0,packed_checks=0,max_input_error=0.,max_action_error=0.)
    def inference(self):
        self.obs,self.goal=policy_inputs(self.task,self.state_history,self.action_history,self.phase)
        dist=self.policy.distribution(self.obs,self.goal);self.action=(dist.mean+dist.stddev*self.noise)*self.mask
    def protected(self,excluding):
        values={}
        for i,e in enumerate(self.envs):
            if i==excluding:continue
            values[(i,'policy_state_history')]=self.state_history[i].clone();values[(i,'policy_action_history')]=self.action_history[i].clone();values[(i,'policy_phase')]=self.phase[i].clone()
            for key in ['qpos','qvel','ctrl','history','servo','context','phase','last_action','current_state']:values[(i,key)]=getattr(e,key).clone()
            for key in self.model_fields:values[(i,'model_'+key)]=wp.to_torch(getattr(e.wm,key)).clone()
        return values
    @torch.no_grad()
    def reset_lane(self,i):
        self.stream.synchronize();before=self.protected(i) if self.validate else None
        row=self.plan.reserve();e=self.envs[i]
        with torch.cuda.stream(e.torch_stream):
            wp.set_stream(e.stream);e.reset(row['phase'],seed=row['seed'],randomize=True)
            world=e._world(e.qpos[:,:e.task.nu])
            assert torch.equal(e.ctrl,e.qpos[:,:e.task.nu]) and torch.equal(e.history,world[:,:,None].expand(-1,-1,3,-1)) and torch.equal(e.servo,world[:,:,:6])
            assert not e.last_action.any() and torch.equal(e.qpos,e.task.qpos[row['phase']:row['phase']+1]) and torch.equal(e.qvel,e.task.qvel[row['phase']:row['phase']+1])
            contact=e.contacts.step();details=e.contacts.details
        self.stream.wait_stream(e.torch_stream)
        with torch.cuda.stream(self.stream):
            self.state_history[i].copy_(e.current_state.expand(3,-1));self.action_history[i].zero_();self.phase[i]=row['phase']
            self.buffers[i]=EpisodeBuffer(row,pack_state(e.current_state).cpu().numpy()[0],torch.cat([e.qpos,e.qvel],-1).cpu().numpy()[0],e.context.cpu().numpy()[0],e.history.cpu().numpy().reshape(156),contact.cpu().numpy(),details.cpu().numpy(),wp.to_torch(e.wd.qacc_warmstart).cpu().numpy()[0],copy.deepcopy(e.context_record))
        self.noise_rng[i]=episode_noise(row);self.audit['reset_checks']+=1
        if self.validate:
            after=self.protected(i);assert all(torch.equal(v,after[k]) for k,v in before.items());self.audit['reset_isolation_checks']+=1
    @torch.no_grad()
    def check_inputs(self,active):
        for i in active:
            e=self.envs[i];obs,goal=policy_inputs(e.task,self.state_history[i:i+1],self.action_history[i:i+1],e.phase)
            dist=self.policy.distribution(obs,goal);expected=(dist.mean+dist.stddev*self.noise[i:i+1])*self.mask[i:i+1]
            err=max(float(abs(obs-self.obs[i:i+1]).max()),float(abs(goal-self.goal[i:i+1]).max()));ae=float(abs(expected-self.action[i:i+1]).max())
            assert err<=2e-5 and ae<=2e-5,(err,ae)
            assert torch.equal(self.state_history[i,-1],e.current_state[0]) and torch.equal(self.action_history[i,-1],e.last_action[0])
            self.audit['input_checks']+=1;self.audit['max_input_error']=max(self.audit['max_input_error'],err);self.audit['max_action_error']=max(self.audit['max_action_error'],ae)
    @staticmethod
    def fields(e):
        return [pack_state(e.current_state),torch.cat([e.qpos,e.qvel],-1),e.requested,e.actions,e.applied,e.info['terms'],e.contact_summary[None],torch.stack([e.phase[0],e.done[0],e.info['finite'][0],e.info['reason'][0],e.info['regular_success'][0],e.info['strict_success'][0],e.count[0]])[None]]
    @torch.no_grad()
    def collect(self,quota,should_stop,progress=lambda _:None):
        started=time.monotonic();initial_complete=complete_controls(self.writer.entries);total=initial_complete;initial_stored=sum(e['controls'] for e in self.writer.entries);tick=0
        if total<quota:
            for i in range(self.n):
                if should_stop():break
                self.reset_lane(i)
        while self.buffers and not should_stop():
            active=sorted(self.buffers);noise=np.zeros((self.n,52),np.float32)
            for i in active:noise[i]=self.noise_rng[i].standard_normal(52).astype(np.float32)
            torch.cuda.set_stream(self.stream);self.noise.copy_(torch.from_numpy(noise));self.graph.replay()
            checked=self.validate or tick==0
            if checked:self.check_inputs(active);old_sh=self.state_history.clone();old_ah=self.action_history.clone()
            for i in active:
                e=self.envs[i];e.torch_stream.wait_stream(self.stream)
                with torch.cuda.stream(e.torch_stream):wp.set_stream(e.stream);e.step(self.action[i:i+1])
            for i in active:self.stream.wait_stream(self.envs[i].torch_stream)
            with torch.cuda.stream(self.stream):
                fields=[self.fields(self.envs[i]) for i in active];widths=[f.shape[-1] for f in fields[0]]
                host=torch.cat([torch.cat(f,-1) for f in fields]).cpu().numpy();details=torch.stack([self.envs[i].contact_details for i in active]).cpu().numpy()
                for i in active:
                    e=self.envs[i];self.state_history[i:i+1].copy_(torch.cat([self.state_history[i:i+1,1:],e.current_state[:,None]],1));self.action_history[i:i+1].copy_(torch.cat([self.action_history[i:i+1,1:],e.last_action[:,None]],1));self.phase[i].copy_(e.phase[0])
                    if checked:
                        assert torch.equal(self.state_history[i,:2],old_sh[i,1:]) and torch.equal(self.action_history[i,:2],old_ah[i,1:])
                        assert torch.equal(e.actions[0],self.action[i]) and torch.equal(e.last_action[0],self.action[i].clamp(-1,1)*self.mask[i]);self.audit['history_checks']+=1
                if checked:
                    direct=torch.cat([torch.cat(self.fields(self.envs[i]),-1) for i in active]).cpu().numpy()
                    assert np.array_equal(host,direct) and np.array_equal(details,np.stack([self.envs[i].contact_details.cpu().numpy() for i in active]),equal_nan=True);self.audit['packed_checks']+=len(active)
            assert np.isfinite(host).all()
            self.real_controls+=len(active)
            for i,packed,contact_details in zip(active,host,details):
                parts=np.split(packed,np.cumsum(widths)[:-1]);row=dict(zip(['state','generalized','request','action','applied','reward_terms','contact'],parts[:7]));aux=parts[7]
                ncontact=int(row['contact'][8]);assert 0<=ncontact<=len(contact_details) and np.isfinite(contact_details[:ncontact]).all()
                row.update(dict(zip(['phase','done','finite','reason','regular','strict','count'],aux)))
                self.buffers[i].step(row,contact_details)
                if row['done']:
                    entry=self.buffers.pop(i).finish(self.writer,i,self.formal,self.training_reset_mode);total+=entry['controls']
            # Commit every endpoint before drawing replacement episodes. Once the
            # quota is reached, drain active complete episodes without resetting.
            if total<quota:
                for i in active:
                    if i not in self.buffers and not should_stop():self.reset_lane(i)
            tick+=1
            if tick%32==0:progress(dict(controls=total,physical_controls=self.real_controls,episodes=len(self.writer.entries),active_lanes=len(self.buffers),seconds=time.monotonic()-started))
        zero=[]
        for i,buffer in list(self.buffers.items()):
            if buffer.controls:buffer.finish(self.writer,i,self.formal,self.training_reset_mode)
            else:zero.append(buffer.reservation['ordinal'])
        self.buffers.clear();self.plan.verify_commits(self.writer.entries)
        stored=sum(e['controls'] for e in self.writer.entries);complete=complete_controls(self.writer.entries);physical=sum(e.physical_controls for e in self.envs)
        assert stored-initial_stored==self.real_controls and physical==self.real_controls+self.warmup_controls
        elapsed=time.monotonic()-started
        return dict(status='quota_complete' if complete>=quota else 'budget_stop',formal=self.formal,quality=self.writer.quality,real_dataset_controls=complete,new_controls_this_process=complete-initial_complete,stored_controls=stored,new_stored_controls_this_process=stored-initial_stored,episodes=sum(is_complete_episode(e) for e in self.writer.entries),stored_episodes=len(self.writer.entries),excluded_incomplete_controls=stored-complete,all_physical_controls_this_process=physical,warmup_controls=self.warmup_controls,seconds=elapsed,controls_per_second=(complete-initial_complete)/elapsed,zero_control_reservations=zero,audit=self.audit,original_nworld_per_environment=1,independent_worlds=self.n,split_counts={s:sum(e['controls'] for e in self.writer.entries if e['split']==s and is_complete_episode(e)) for s in ['train','development','test']})
