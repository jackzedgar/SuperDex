"""Matched 8-step W0/W1 fitting and held-out 1/8/32-step prediction.

Wall time cap includes training and final prediction evaluation. No policy update
uses these models, and no validation state enters autoregressive rollouts.
"""
import time,csv,json,argparse,hashlib
import numpy as np,torch
from exploration_common import RUN,ROOT,GPU,read_json,write_json,resource_check,finish_timing
from world_model_mini import Geometry,WorldModel,encode,delta_state,qlog,qmul,qconj,simple_step


def normalization(data,manifest):
    sa=[];sb=[];uu=[]
    for ep in manifest['episodes']:
        if ep['split']!='train':continue
        s=ep['states_start'];c=ep['controls_start'];n=ep['steps'];sa.append(data['states'][s:s+n]);sb.append(data['states'][s+1:s+n+1]);uu.append(data['controls'][c:c+n])
    a=torch.tensor(np.concatenate(sa));b=torch.tensor(np.concatenate(sb));u=torch.tensor(np.concatenate(uu));x=encode(a);delta=delta_state(a,b)
    floors=torch.tensor([.01]*3+[.1]*23+[.01]*3+[.05]*6+[.1]*32+([.01]*3+[.1]*23)*3)
    scale=x.std(0,unbiased=False).maximum(floors);dscale=delta.std(0,unbiased=False).maximum(torch.tensor([.0005]*3+[.005]*23+[.0005]*3+[.005]*3+[.01]*32))
    weights=torch.ones(67);weights[26:35]=4;weights[61:67]=4
    return {k:v.numpy() for k,v in dict(input_mean=x.mean(0),input_std=scale,control_mean=u.mean(0),control_std=u.std(0,unbiased=False).maximum(torch.tensor([.01]*3+[.1]*23)),delta_mean=delta.mean(0),delta_std=dscale,state_std=scale[:67],loss_weight=weights).items()}


def errors(pred,truth):
    return torch.stack([torch.linalg.vector_norm(pred[:,26:29]-truth[:,26:29],dim=-1),torch.linalg.vector_norm(qlog(qmul(qconj(truth[:,29:33]),pred[:,29:33])),dim=-1),((pred[:,6:26]-truth[:,6:26]).square().mean(-1)).sqrt(),torch.linalg.vector_norm(pred[:,:3]-truth[:,:3],dim=-1),torch.linalg.vector_norm(pred[:,59:62]-truth[:,59:62],dim=-1),torch.linalg.vector_norm(pred[:,62:65]-truth[:,62:65],dim=-1)],-1)


@torch.no_grad()
def evaluate(model,name,data,indices,horizon,out=None):
    results=[];curves=[]
    for start in range(0,len(indices),128):
        ix=indices[start:start+128];si=torch.tensor(ix[:,0],device='cuda');ci=torch.tensor(ix[:,1],device='cuda');s=data['states'][si];curve=[s[:,:65].cpu().numpy()];errs=[]
        for t in range(horizon):
            u=data['controls'][ci+t];s=model(s,u) if model is not None else simple_step(s,u,name);truth=data['states'][si+t+1];errs.append(errors(s,truth).cpu().numpy());curve.append(s[:,:65].cpu().numpy())
        results.append(np.stack(errs,1));curves.append(np.stack(curve,1))
    e=np.concatenate(results);summary={}
    for label,labname in enumerate(['no_contact','sustained_contact','contact_switch']):
        sel=indices[:,3]==label;v=e[sel,-1];finite=np.isfinite(v).all(1)
        summary[labname]=dict(windows=int(sel.sum()),episodes=len(set(indices[sel,2].tolist())),nonfinite=int((~finite).sum()),position_median_mm=float(np.median(v[:,0])*1000) if len(v) else None,position_p90_mm=float(np.quantile(v[:,0],.9)*1000) if len(v) else None,position_mean_mm=float(np.mean(v[:,0])*1000) if len(v) else None,rotation_median_deg=float(np.rad2deg(np.median(v[:,1]))) if len(v) else None,rotation_p90_deg=float(np.rad2deg(np.quantile(v[:,1],.9))) if len(v) else None,finger_rmse_mean_rad=float(np.mean(v[:,2])) if len(v) else None,object_velocity_mean_m_s=float(np.mean(v[:,4])) if len(v) else None)
    if out is not None:np.savez_compressed(out/f'prediction_{name}_{horizon:02d}.npz',errors=e,predicted_states=np.concatenate(curves),windows=indices)
    return summary,e


def main():
    started=time.perf_counter();p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=600);p.add_argument('--updates',type=int,default=1000);p.add_argument('--trend',action='store_true');args=p.parse_args();assert 90<=args.seconds<=(875 if args.trend else 600) and 1<=args.updates<=(2000 if args.trend else 1000)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);out=RUN/'wm';(out/'resources_start.txt').write_text(resource_check(2000));assert str(torch.cuda.get_device_properties(0).uuid)==GPU[4:]
    cfg=read_json(ROOT/'configs'/('learning_trend_budget.json' if args.trend else 'exploration_budget.json'))['wm']
    if args.trend:
        assert not (out/'training.csv').exists(), 'Do not overwrite an earlier training attempt.'
        frozen=read_json(RUN/'input_manifest.json')
        for entry in frozen['frozen_wm_files']:
            assert hashlib.sha256((out/entry['name']).read_bytes()).hexdigest()==entry['sha256']
    with np.load(out/'dataset.npz') as archive:raw={k:archive[k] for k in archive.files}
    manifest=read_json(out/'dataset_manifest.json');norm=normalization(raw,manifest);np.savez(out/'normalization.npz',**norm)
    data={k:torch.tensor(raw[k],device='cuda') for k in ['states','controls']};geom=Geometry(out).to('cuda');torch.manual_seed(cfg['seed']);models={v:WorldModel(v,geom,norm).to('cuda') for v in ['W0','W1']}
    with torch.no_grad():
        params=dict(models['W1'].named_parameters())
        for name,p in models['W0'].named_parameters():
            if name=='net.0.weight':params[name].zero_();params[name][:,:p.shape[1]].copy_(p)
            else:params[name].copy_(p)
    batch0=data['states'][:8];u0=data['controls'][:8];assert torch.equal(models['W0'](batch0,u0),models['W1'](batch0,u0))
    opts={k:torch.optim.Adam(m.parameters(),lr=cfg['learning_rate']) for k,m in models.items()};windows=raw['windows_8_train'];pools=[np.flatnonzero(windows[:,3]==i) for i in range(3)];available=[i for i,p in enumerate(pools) if len(p)]
    val8=raw['windows_8_validation'][np.load(out/'validation_sample_8.npy')];val32=raw['windows_32_validation'][np.load(out/'validation_sample_32.npy')];rng=np.random.default_rng(cfg['seed']);records=[];component_time={v:0. for v in models}
    f=(out/'training.csv').open('w',newline='');writer=csv.DictWriter(f,fieldnames=['update','model','training_loss','preclip_gradient_norm','process_seconds','model_training_seconds','validation8_position_macro_mm']);writer.writeheader()
    history=[];batch_digest=hashlib.sha256();scheduled=set(cfg.get('validation_updates',[]))
    def validation(name,update):
        folder=out/'validation'/f'update_{update:04d}' if args.trend else None
        if folder is not None:folder.mkdir(parents=True,exist_ok=True)
        result=None
        for h,indices in ([(8,val8),(32,val32)] if args.trend else [(8,val8)]):
            stats,_=evaluate(models[name],name,data,indices,h,folder);history.append(dict(update=update,model=name,horizon=h,stats=stats))
            if h==8:result=stats
        write_json(out/'validation_history.json',history)
        return result
    def checkpoint(name,update,path):
        record=dict(model=models[name].state_dict(),variant=name,update=update,normalization=norm,architecture='3x256 SiLU')
        if args.trend:record.update(optimizer=opts[name].state_dict(),batch_rng=rng.bit_generator.state,batch_sequence_sha256=batch_digest.hexdigest())
        torch.save(record,path)
    for name,model in models.items():
        stats=validation(name,0);writer.writerow(dict(update=0,model=name,training_loss='',preclip_gradient_norm='',process_seconds=time.perf_counter()-started,model_training_seconds=0,validation8_position_macro_mm=np.mean([x['position_mean_mm'] for x in stats.values() if x['windows']])))
    last_pair=.5;last_update=0;stop='update_budget'
    for update in range(1,min(cfg['paired_update_cap'],args.updates)+1):
        # Reserve final 8/32 prediction evaluation within this process allocation.
        if time.perf_counter()-started+last_pair+60>args.seconds:stop='wall_budget';break
        t0=time.perf_counter();labels=rng.choice(available,64);ix=np.array([rng.choice(pools[int(k)]) for k in labels]);batch=windows[ix];batch_digest.update(ix.astype('<i8').tobytes());si=torch.tensor(batch[:,0],device='cuda');ci=torch.tensor(batch[:,1],device='cuda');logs=[]
        for name,model in models.items():
            tm=time.perf_counter();opts[name].zero_grad(set_to_none=True);state=data['states'][si];loss=0.
            for t in range(8):state=model(state,data['controls'][ci+t]);loss=loss+model.loss(state,data['states'][si+t+1])/8
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite {name} training loss at update {update}')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opts[name].step();torch.cuda.synchronize();component_time[name]+=time.perf_counter()-tm
            logs.append(dict(update=update,model=name,training_loss=float(loss.detach()),preclip_gradient_norm=float(grad),process_seconds=time.perf_counter()-started,model_training_seconds=component_time[name],validation8_position_macro_mm=''))
        last_pair=time.perf_counter()-t0;last_update=update
        if (update in scheduled if args.trend else update%100==0):
            for row in logs:
                name=row['model'];stats=validation(name,update);row['validation8_position_macro_mm']=np.mean([x['position_mean_mm'] for x in stats.values() if x['windows']]);checkpoint(name,update,out/f'{name}_{update:04d}.pt')
            print(json.dumps(logs),flush=True)
        writer.writerows(logs);f.flush()
    f.close();evaluation={}
    for name,model in list(models.items())+[(x,None) for x in ['persistence','constant_velocity','ballistic']]:
        if model is not None:checkpoint(name,last_update,out/f'{name}_final.pt')
        evaluation[name]={}
        for h,ix in [(8,val8),(32,val32)]:
            stats,e=evaluate(model,name,data,ix,h,out);evaluation[name][str(h)]=stats
            if args.trend and model is not None and not any(r['model']==name and r['update']==last_update and r.get('horizon',8)==h for r in history):history.append(dict(update=last_update,model=name,horizon=h,stats=stats))
            if h==8:
                first=e[:,0];evaluation[name]['1']=dict(position_median_mm=float(np.median(first[:,0])*1000),position_p90_mm=float(np.quantile(first[:,0],.9)*1000),rotation_median_deg=float(np.rad2deg(np.median(first[:,1]))),sampling='same regime-stratified sample selected for 8-step evaluation')
    write_json(out/'evaluation.json',evaluation);write_json(out/'validation_history.json',history)
    write_json(out/'config.json',dict(paired_updates=last_update,hidden=[256,256,256],batch=64,learning_rate=cfg['learning_rate'],rollout_train=8,rollout_eval=[8,32],seed=cfg['seed'],parameters={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},difference='Only W1 receives 84 scalar-SDF / analytic-gradient features; identical common weights initially and identical training windows per update.',normalization='Train episodes only; fixed mean/std floors; object pose/velocity weight 4, others 1.',sampler='Equal available contact regimes, shared sampled batch.',deviations_from_formal_W0_W1=['3x256 instead of 3x2048','Single fixed spoon and physical context; no varying xi or geometry code','Simulator generalized coordinates instead of yaw-canonical state; no equivariance claim','Zero actuator delay in current simulator; three executed-control history buffer updated analytically','No policy/model-gradient coupling or formal G1'],known_limit='Generalized qpos/qvel omits simulator warmstart internals; GPU contact sensitivity remains a possible irreducible target noise source.',stop_reason=stop))
    seconds=finish_timing('world_models',started,dict(paired_updates=last_update,component_training_seconds=component_time,stop_reason=stop,peak_memory_mb=torch.cuda.max_memory_allocated()/2**20));print('WM COMPLETE',last_update,seconds,flush=True)


if __name__=='__main__':main()
