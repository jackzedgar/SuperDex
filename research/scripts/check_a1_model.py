"""Bounded full-sized implementation check, never a learned A1 result."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from a1_data_sources import ROOT,RUN,write
from a1_geometry import A1Geometry
from a1_world_model import A1WorldModel,CONTEXT,common_features
from a1_state import empty_state
def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=512);p.add_argument('--updates',type=int,default=3);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(280043);device='cuda'
    kin=[dict(np.load(ROOT/f'data/a1_processed/assets/hands/{side}/kinematics.npz')) for side in ['right','left']]
    gridpath=ROOT/'runs/explore_20261026/wm/sdf_grid.npz';grid=dict(np.load(gridpath))
    geom=A1Geometry(kin,[grid]).to(device)
    # Compare the SDF feature derivative with autograd through its exact scalar,
    # including an outside point with a nonzero tangential component.
    points=torch.tensor([[[.011,.013,.017],[1.1,.012,.016]]],device=device,requires_grad=True)
    oid=torch.zeros(1,dtype=torch.long,device=device)
    val,grad=geom.sdf(points,oid);actual=torch.autograd.grad(val.sum(),points)[0]
    err=float((actual-grad).abs().max());assert err<3e-5,err
    batch=a.batch;s=empty_state(batch,device=device);s[:,42]=.15;s[:,179]=.15;s[:,276]=.1
    mask=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],device=device).repeat((batch+2)//3,1)[:batch]
    c=torch.zeros(batch,CONTEXT,device=device)
    for h in range(2):
        c[:,h*68+20:h*68+40]=1.97392088;c[:,h*68+40:h*68+60]=.1256637
        c[:,h*68+60]=1;c[:,h*68+62:h*68+68]=.02
    c[:,136]=.04;c[:,137:139]=.8
    u=torch.zeros(batch,8,52,device=device);u[:,:,2]=.15;u[:,:,28]=.15
    ids=torch.zeros(batch,dtype=torch.long,device=device)
    target=s[:,None].repeat(1,8,1);target[:,:,274:277]+=torch.randn(batch,8,3,device=device)*.001
    results={};common_head=None;reference=None
    for kind in ['W0','W1']:
        torch.cuda.reset_peak_memory_stats()
        m=A1WorldModel(kind,geom if kind=='W1' else None).to(device)
        with torch.no_grad():
            if common_head is None:
                common_head=torch.randn_like(m.net[-1].weight)*1e-5
            m.net[-1].weight.copy_(common_head)
            pred=m(s[:4],u[:4,0],mask[:4],c[:4],ids[:4])
            if reference is None:reference=pred.clone()
            else:assert torch.allclose(reference,pred,atol=2e-7,rtol=2e-7),(reference-pred).abs().max()
        optimizer=torch.optim.Adam(m.parameters(),lr=2e-4)
        losses=[];times=[];gradnorms=[]
        for i in range(a.updates+1):
            torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            pred=m.rollout(s,u,mask,c,ids)
            loss=m.dynamics_loss(pred,target,mask)
            assert torch.isfinite(loss)
            loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(m.net.parameters(),1.)
            assert torch.isfinite(gn)
            optimizer.step();torch.cuda.synchronize()
            if i:times.append(time.monotonic()-start);losses.append(float(loss));gradnorms.append(float(gn))
        results[kind]=dict(config=m.config(),microbatch_windows=batch,horizon=8,measured_updates=len(times),median_micro_update_s=float(np.median(times)),estimated_40microbatches_adam_s=float(np.median(times)*40),losses=losses,gradient_norms=gradnorms,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del m,optimizer,pred,loss;torch.cuda.empty_cache()
    report=dict(scope='Synthetic numerical/performance fixture only; no research checkpoint or A1/G1 result',actual_spoon_geometry_fixture=str(gridpath.relative_to(ROOT)),sdf_autograd_max_discrepancy=err,common_nonzero_initial_prediction_matches=True,results=results,gpu_name=torch.cuda.get_device_name(0),physics_controls=0,fixture_model_transitions=2*(a.updates+1)*batch*8)
    write(RUN/'full_model_implementation_check.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
