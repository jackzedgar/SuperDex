"""Fourth-revision candidate authorized20261030; bounded by registered1GPUh."""
import torch
import warp as wp
from a1_original_pool_runner import OriginalPoolRunner

class FormalStartPoolCandidate(OriginalPoolRunner):
    @torch.no_grad()
    def reset_lane(self,i):
        e=self.envs[i];rng=self.rngs[i];seed=int(rng.integers(2**31))
        while seed in (280046,280047,280048,280049):seed=int(rng.integers(2**31))
        self.stream.synchronize();others=[k for k in range(self.n) if k!=i]
        if self.validate:prior=[self.state_history[others].clone(),self.action_history[others].clone(),self.context[others].clone(),self.phase[others].clone()]
        with torch.cuda.stream(e.torch_stream):
            wp.set_stream(e.stream);e.reset(0,seed=seed,randomize=True)
            world=e._world(e.qpos[:,:e.task.nu])
            assert int(e.phase)==0 and not bool(e.last_action.any())
            assert torch.equal(e.ctrl,e.qpos[:,:e.task.nu])
            assert torch.equal(e.history,world[:,:,None].expand(-1,-1,3,-1)) and torch.equal(e.servo,world[:,:,:6])
        self.stream.wait_stream(e.torch_stream)
        with torch.cuda.stream(self.stream):
            self.state_history[i:i+1].copy_(e.state()[:,None].expand(-1,3,-1));self.action_history[i].zero_()
            self.context[i:i+1].copy_(e.context);self.phase[i:i+1].copy_(e.phase);self.last_action[i].zero_()
            if self.validate:
                actual=[self.state_history[others],self.action_history[others],self.context[others],self.phase[others]]
                assert all(torch.equal(a,b) for a,b in zip(prior,actual));self.audit['reset_isolation_checks']+=1
        self.audit['formal_reset_checks']=self.audit.get('formal_reset_checks',0)+1
        self.ep_ids[i]+=1;self.ep_returns[i]=0.;self.ep_controls[i]=0
        self.reset_records.append(dict(lane=i,episode_id=self.ep_ids[i],phase=0,seed=seed,reset_mode='formal_episode_start',training_memory='original'))
