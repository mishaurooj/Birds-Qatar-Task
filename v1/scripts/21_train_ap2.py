from __future__ import annotations
import argparse,sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset
from src.config import load_config,project_path
from src.utils import seed_everything,ensure_dir,save_json,device_info,trainable_parameter_count
from src.backbone import build_backbone,load_backbone_from_checkpoint
from src.models_ap1 import AP1Model
from src.models_ap2 import UAPSModel,hungarian_set_loss,query_probs_to_multilabel
from src.datasets import SyntheticPolyphonyDataset,RealMultiLabelDataset,poly_collate
from src.metrics import tune_thresholds,multilabel_metrics
from src.training import optimizer_for,cosine_warmup,save_history,evaluate_multilabel,_make_scaler,_autocast

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--variant',default='B5_full_uaps'); p.add_argument('--epochs',type=int); p.add_argument('--ap1-checkpoint'); a=p.parse_args(); cfg=load_config(a.config)
if a.epochs: cfg['training']['epochs']=a.epochs
seed_everything(int(cfg['project']['seed'])); labels=pd.read_csv(Path(cfg['_project_root'])/'data/qatar_labels.csv').sort_values('label_id'); n=len(labels)
# Recreate AP1 and load its full supervised checkpoint.
bb=build_backbone(cfg); ap1=AP1Model(bb,n,cfg,'A5_full_acpsar'); apck=Path(a.ap1_checkpoint or project_path(cfg,cfg['paths']['qatar_ap1_checkpoint']))
if not apck.exists(): raise SystemExit('AP1 checkpoint missing: '+str(apck))
ck=torch.load(apck,map_location='cpu'); ap1.load_state_dict(ck['model_state_dict'],strict=False)
model=UAPSModel(ap1,n,cfg,a.variant); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
single=project_path(cfg,cfg['paths']['qatar_single_manifest']); multi=project_path(cfg,cfg['paths']['qatar_multi_manifest'])
syn_train=SyntheticPolyphonyDataset(single,cfg,'train',size=max(1500,len(pd.read_csv(single))*2),train=True); real_train=RealMultiLabelDataset(multi,cfg,'train',True)
train_ds=ConcatDataset([syn_train,real_train]) if len(real_train)>0 else syn_train
real_val=RealMultiLabelDataset(multi,cfg,'val',False); real_test=RealMultiLabelDataset(multi,cfg,'test',False)
# If BirdNET pseudo-labelled local val/test is too small, create fixed synthetic validation/test and state this in run_info.
val_is_real=len(real_val)>=20; test_is_real=len(real_test)>=20
val_ds=real_val if val_is_real else SyntheticPolyphonyDataset(single,cfg,'val',size=400,train=False); test_ds=real_test if test_is_real else SyntheticPolyphonyDataset(single,cfg,'test',size=600,train=False)
bs=max(2,int(cfg['training']['batch_size'])//2); dl=lambda d,s:DataLoader(d,batch_size=bs,shuffle=s,num_workers=0,collate_fn=poly_collate)
train_loader=dl(train_ds,True); val_loader=dl(val_ds,False); test_loader=dl(test_ds,False)
opt=optimizer_for(model,cfg); total=int(cfg['training']['epochs'])*max(1,len(train_loader)); sched=cosine_warmup(opt,total,int(cfg['training']['warmup_epochs'])*max(1,len(train_loader))); scaler=_make_scaler(device,cfg['training']['amp'])
out=ensure_dir(Path(cfg['_project_root'])/f'results/AP2/{a.variant}'); best=-1; stale=0; hist=[]; difficulty=np.ones((n,n),float); np.fill_diagonal(difficulty,0)
for epoch in range(int(cfg['training']['epochs'])):
    syn_train.set_epoch(epoch); syn_train.set_difficulty(difficulty if a.variant=='B5_full_uaps' else None); model.train(); losses=[]
    for b in train_loader:
        opt.zero_grad(set_to_none=True); w5=b['wave5'].to(device); w10=b['wave10'].to(device); c=b['condition'].to(device); y=b['target'].to(device)
        with _autocast(device,scaler.is_enabled()):
            o=model(w5,w10,c); loss=F.binary_cross_entropy_with_logits(o['multi_logits'],y)
            if a.variant in ['B3_set_decoder','B4_set_intervals','B5_full_uaps']:
                cl,il=hungarian_set_loss(o,b['intervals'],n,device); loss=loss+cl+(il if a.variant!='B3_set_decoder' else 0)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip'])); scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach().cpu()))
    vm,vper,vpred,vprob,vy,th=evaluate_multilabel(model,val_loader,device,None,cfg['ap2']['threshold_grid'])
    # Difficulty: missed true class paired with high-probability wrong classes.
    d=np.zeros((n,n),float)
    for yy,pp,pr in zip(vy,vprob,vpred):
        true=np.where(yy>0)[0]; miss=[x for x in true if pr[x]==0]; wrong=np.argsort(-pp)[:4]
        for i in miss:
            for j in wrong:
                if i!=j: d[i,j]+=1; d[j,i]+=1
    if d.max()>0: d/=d.max(); difficulty=.2+d
    np.save(out/f'difficulty_epoch_{epoch+1:03d}.npy',difficulty)
    hist.append({'epoch':epoch+1,'train_loss':np.mean(losses),'val_macro_f1':vm['macro_f1'],'val_micro_f1':vm['micro_f1']})
    state={'model_state_dict':model.state_dict(),'epoch':epoch+1,'val_metrics':vm,'thresholds':th,'cfg':cfg,'variant':a.variant}; torch.save(state,out/'last.pt')
    if vm['macro_f1']>best: best=vm['macro_f1']; stale=0; torch.save(state,out/'best.pt')
    else: stale+=1
    if stale>=int(cfg['training']['patience']): break
save_history(hist,out); bestck=torch.load(out/'best.pt',map_location=device); model.load_state_dict(bestck['model_state_dict'],strict=False); thresholds=np.asarray(bestck['thresholds']); tm,tper,tpred,tprob,ty,_=evaluate_multilabel(model,test_loader,device,thresholds,cfg['ap2']['threshold_grid'])
save_json(bestck['val_metrics'],out/'val_metrics.json'); save_json(tm,out/'test_metrics.json'); save_json({'thresholds':thresholds},out/'thresholds.json'); tper.to_csv(out/'test_per_class.csv',index=False); np.save(out/'test_probabilities.npy',tprob); np.save(out/'test_targets.npy',ty); save_json({'validation_source':'BirdNET pseudo-labelled Qatar' if val_is_real else 'fixed synthetic fallback','test_source':'BirdNET pseudo-labelled Qatar' if test_is_real else 'fixed synthetic fallback','parameters':trainable_parameter_count(model),'device':device_info()},out/'run_info.json')
modeldir=ensure_dir(Path(cfg['_project_root'])/'models/qatar'); torch.save(bestck,modeldir/'ap2_best.pt'); print('AP2 test:',tm)
