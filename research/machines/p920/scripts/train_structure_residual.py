"""Train B only; reuse byte-identical A checkpoints, batches and predictions."""
import argparse,time,csv,hashlib
import numpy as np,torch
from exploration_common import RUN,ROOT,read_json,write_json,resource_check
from world_model_mini import Geometry,WorldModel
from world_model_residual import ResidualWorldModel,residual_normalization
from train_world_models_mini import evaluate

def main():
    start=time.perf_counter();p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,required=True);args=p.parse_args();cfg=read_json(RUN/'budget.json')['wm'];torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=RUN/'wm/B';out.mkdir(exist_ok=False);(out/'resources_start.txt').write_text(resource_check(2000))
    with np.load(RUN/'wm/dataset.npz') as f:raw={k:f[k] for k in f.files}
    manifest=read_json(RUN/'wm/dataset_manifest.json');base=dict(np.load(RUN/'wm/normalization.npz'));norm=residual_normalization(raw,manifest,base);np.savez(out/'normalization.npz',**norm)
    for k in ['input_mean','input_std','control_mean','control_std','state_std','loss_weight']:assert np.array_equal(base[k],norm[k])
    data={k:torch.tensor(raw[k],device='cuda') for k in ['states','controls']};geom=Geometry(RUN/'wm').to('cuda');torch.manual_seed(cfg['seed']);w0=WorldModel('W0',geom,base).to('cuda');a=WorldModel('W1',geom,base).to('cuda')
    with torch.no_grad():
        pp=dict(a.named_parameters())
        for name,v in w0.named_parameters():
            if name=='net.0.weight':pp[name].zero_();pp[name][:,:v.shape[1]].copy_(v)
            else:pp[name].copy_(v)
    model=ResidualWorldModel(geom,norm).to('cuda');model.net.load_state_dict(a.net.state_dict());assert all(torch.equal(v,a.net.state_dict()[k]) for k,v in model.net.state_dict().items())
    write_json(out/'initialization.json',dict(seed=23,network_parameter_count=sum(v.numel() for v in model.parameters()),shared_network_parameters_identical=True,all_outputs_active=True,normalization='Only delta_mean/std at slots 26:32 and 58:64 change. Train-only two-half impulse proxy means/std, floor 0.01 in m/s or rad/s. Inputs, hand outputs and loss normalization unchanged.'))
    torch.save(dict(net=model.net.state_dict(),normalization=norm),out/'initial.pt');del a,w0
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);windows=raw['windows_8_train'];pools=[np.flatnonzero(windows[:,3]==i) for i in range(3)];available=[i for i,p in enumerate(pools) if len(p)];rng=np.random.default_rng(23);digest=hashlib.sha256();history=[];schedule=cfg['validation_updates'];last=0;last_seconds=1.;stop='update_budget'
    vals={h:raw[f'windows_{h}_validation'][np.load(RUN/'wm'/f'validation_sample_{h}.npy')] for h in [8,32]}
    def validation(update):
        folder=out/'validation'/f'update_{update:04d}';folder.mkdir(parents=True,exist_ok=True)
        for h in [8,32]:
            stats,_=evaluate(model,'B',data,vals[h],h,folder);history.append(dict(update=update,horizon=h,stats=stats))
        write_json(out/'validation_history.json',history)
    def checkpoint(update,path):
        torch.save(dict(model=model.state_dict(),variant='B',normalization=norm,update=update,optimizer=opt.state_dict(),batch_rng=rng.bit_generator.state,batch_sequence_sha256=digest.hexdigest()),path)
    validation(0)
    with (out/'training.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['update','training_loss','preclip_gradient_norm','process_seconds']);writer.writeheader()
        for update in range(1,2001):
            if time.perf_counter()-start+last_seconds+75>args.seconds:stop='wall_budget';break
            t0=time.perf_counter();labels=rng.choice(available,64);ix=np.array([rng.choice(pools[int(k)]) for k in labels]);digest.update(ix.astype('<i8').tobytes());batch=windows[ix];si=torch.tensor(batch[:,0],device='cuda');ci=torch.tensor(batch[:,1],device='cuda');state=data['states'][si];loss=0.
            opt.zero_grad(set_to_none=True)
            for t in range(8):state=model(state,data['controls'][ci+t]);loss=loss+model.loss(state,data['states'][si+t+1])/8
            assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step();torch.cuda.synchronize();last=update;last_seconds=time.perf_counter()-t0
            row=dict(update=update,training_loss=float(loss.detach()),preclip_gradient_norm=float(grad),process_seconds=time.perf_counter()-start);writer.writerow(row);f.flush()
            if update in schedule:
                ref=torch.load(RUN/'wm/A'/f'W1_{update:04d}.pt',map_location='cpu',weights_only=False);assert ref['batch_sequence_sha256']==digest.hexdigest();checkpoint(update,out/f'B_{update:04d}.pt');validation(update);print(row,flush=True)
    checkpoint(last,out/'B_final.pt')
    if last not in schedule:validation(last)
    matched=max([u for u in schedule if u<=last and u>0],default=0)
    write_json(out/'summary.json',dict(updates=last,matched_comparison_update=matched,process_seconds=time.perf_counter()-start,stop_reason=stop,seed=23,batch_sequence_sha256=digest.hexdigest(),peak_memory_mb=torch.cuda.max_memory_allocated()/2**20,A_reused=True));print('B COMPLETE',last,time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
