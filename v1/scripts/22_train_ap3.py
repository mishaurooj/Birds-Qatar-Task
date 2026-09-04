from __future__ import annotations
import argparse,sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import load_config,project_path
from src.utils import seed_everything,ensure_dir,save_json,device_info,trainable_parameter_count
from src.backbone import build_backbone
from src.models_ap1 import AP1Model
from src.models_ap3 import DCGSModel,build_confusion_graph,graph_contrastive_loss
from src.datasets import ClipDataset
from src.training import class_weights_from_manifest,optimizer_for,cosine_warmup,save_history,evaluate_single,_make_scaler,_autocast

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--variant',default='C5_full_dcgs'); p.add_argument('--epochs',type=int); p.add_argument('--ap1-checkpoint'); a=p.parse_args(); cfg=load_config(a.config)
if a.epochs: cfg['training']['epochs']=a.epochs
seed_everything(int(cfg['project']['seed'])); labels=pd.read_csv(Path(cfg['_project_root'])/'data/qatar_labels.csv').sort_values('label_id'); names=labels.common_name.astype(str).tolist(); n=len(names); manifest=project_path(cfg,cfg['paths']['qatar_single_manifest'])
bb=build_backbone(cfg); ap1=AP1Model(bb,n,cfg,'A5_full_acpsar'); apck=Path(a.ap1_checkpoint or project_path(cfg,cfg['paths']['qatar_ap1_checkpoint'])); ck=torch.load(apck,map_location='cpu'); ap1.load_state_dict(ck['model_state_dict'],strict=False)
model=DCGSModel(ap1,n,cfg,a.variant); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
train=ClipDataset(manifest,cfg,'train',True); val=ClipDataset(manifest,cfg,'val',False); test=ClipDataset(manifest,cfg,'test',False); bs=int(cfg['training']['batch_size']); dl=lambda d,s:DataLoader(d,batch_size=bs,shuffle=s,num_workers=0)
tr,va,te=dl(train,True),dl(val,False),dl(test,False); counts,weights=class_weights_from_manifest(manifest,'train'); cw=torch.tensor(weights,dtype=torch.float32,device=device)
opt=optimizer_for(model,cfg); total=int(cfg['training']['epochs'])*max(1,len(tr)); sched=cosine_warmup(opt,total,int(cfg['training']['warmup_epochs'])*max(1,len(tr))); scaler=_make_scaler(device,cfg['training']['amp'])
out=ensure_dir(Path(cfg['_project_root'])/f'results/AP3/{a.variant}'); best=-1; stale=0; hist=[]; graph=np.eye(n,dtype=float)
for epoch in range(int(cfg['training']['epochs'])):
    model.train(); losses=[]
    for b in tr:
        opt.zero_grad(set_to_none=True); w=b['waveform'].to(device); c=b['condition'].to(device); y=b['label'].to(device)
        with _autocast(device,scaler.is_enabled()):
            o=model(w,c); loss=F.cross_entropy(o['logits'],y,weight=cw)
            if a.variant in ['C4_dynamic_contrastive','C5_full_dcgs']:
                loss=loss+float(cfg['ap3']['lambda_contrastive'])*graph_contrastive_loss(o['embedding'],y,model.graph)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip'])); scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach().cpu()))
    vm,vp,vz,vy=evaluate_single(model,va,device,names,counts)
    dynamic=a.variant in ['C3_dynamic_soft','C4_dynamic_contrastive','C5_full_dcgs']
    static=a.variant in ['C1_static_hard','C2_static_soft']
    rebuild = (static and epoch==0) or (dynamic and ((epoch+1)%int(cfg['ap3']['graph_rebuild_every'])==0 or epoch==0))
    if rebuild:
        graph=build_confusion_graph(vy,vp,vz,n,float(cfg['ap3']['graph_alpha_confusion']),float(cfg['ap3']['graph_beta_similarity']),float(cfg['ap3']['graph_gamma_uncertainty']))
        # keep all diagonal edges and threshold weak off-diagonal edges
        g2=graph.copy(); mask=(g2>=float(cfg['ap3']['graph_threshold'])); g2=g2*mask; np.fill_diagonal(g2,1); model.set_graph(g2); graph=g2; np.save(out/f'confusion_graph_epoch_{epoch+1:03d}.npy',graph)
    hist.append({'epoch':epoch+1,'train_loss':np.mean(losses),'val_macro_f1':vm['macro_f1'],'val_balanced_accuracy':vm['balanced_accuracy']})
    state={'model_state_dict':model.state_dict(),'epoch':epoch+1,'val_metrics':vm,'graph':graph,'cfg':cfg,'variant':a.variant}; torch.save(state,out/'last.pt')
    if vm['macro_f1']>best: best=vm['macro_f1']; stale=0; torch.save(state,out/'best.pt')
    else: stale+=1
    if stale>=int(cfg['training']['patience']): break
save_history(hist,out); bestck=torch.load(out/'best.pt',map_location=device); model.load_state_dict(bestck['model_state_dict'],strict=False); model.set_graph(bestck.get('graph',np.eye(n))); tm,tp,tz,ty=evaluate_single(model,te,device,names,counts,out/'test'); vm,_,_,_=evaluate_single(model,va,device,names,counts,out/'val')
np.save(out/'confusion_graph.npy',bestck.get('graph',np.eye(n))); pd.DataFrame(bestck.get('graph',np.eye(n)),index=names,columns=names).to_csv(out/'confusion_graph.csv'); save_json({'parameters':trainable_parameter_count(model),'device':device_info()},out/'run_info.json'); save_json(vm,out/'val_metrics.json'); save_json(tm,out/'test_metrics.json')
modeldir=ensure_dir(Path(cfg['_project_root'])/'models/qatar'); torch.save(bestck,modeldir/'ap3_best.pt'); print('AP3 test:',tm)
