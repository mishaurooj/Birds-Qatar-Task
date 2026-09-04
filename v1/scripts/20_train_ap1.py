from __future__ import annotations
import argparse,sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from src.config import load_config,project_path
from src.utils import seed_everything,ensure_dir,save_json
from src.backbone import build_backbone,load_backbone_from_checkpoint
from src.datasets import ClipDataset
from src.models_ap1 import AP1Model
from src.training import train_single,class_weights_from_manifest

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--variant',default='A5_full_acpsar'); p.add_argument('--epochs',type=int); a=p.parse_args(); cfg=load_config(a.config)
if a.epochs: cfg['training']['epochs']=a.epochs
seed_everything(int(cfg['project']['seed'])); manifest=project_path(cfg,cfg['paths']['qatar_single_manifest']); labels=pd.read_csv(Path(cfg['_project_root'])/'data/qatar_labels.csv').sort_values('label_id'); names=labels.common_name.astype(str).tolist(); n=len(names)
if not manifest.exists(): raise SystemExit('Run Qatar labeling + 13_build_qatar_manifests.py first.')
backbone=build_backbone(cfg); kck=project_path(cfg,cfg['paths']['kaggle_checkpoint'])
if not kck.exists(): raise SystemExit('Kaggle checkpoint missing. Train the full Kaggle phase first: '+str(kck))
load_report=load_backbone_from_checkpoint(backbone,str(kck)); print('Kaggle -> Qatar backbone transfer:',load_report['loaded_tensors'],'tensors')
model=AP1Model(backbone,n,cfg,a.variant)
# Optional exact scientific-name classifier-head initialization from Kaggle.
transfer=[]; kh=project_path(cfg,cfg['paths'].get('kaggle_classifier_head','models/kaggle_public/classifier_head.pt'))
if kh.exists():
    h=torch.load(kh,map_location='cpu'); klabels=pd.DataFrame(h['labels']);
    sci_lookup={str(x.scientific_name).strip().casefold():int(i) for i,x in klabels.iterrows() if 'scientific_name' in klabels.columns and str(x.scientific_name).strip() and str(x.scientific_name)!='nan'}
    common_lookup={str(x.common_name).strip().casefold():int(i) for i,x in klabels.iterrows() if 'common_name' in klabels.columns and str(x.common_name).strip()}
    with torch.no_grad():
        for _,r in labels.iterrows():
            skey=str(r.scientific_name).strip().casefold(); ckey=str(r.common_name).strip().casefold(); ki=sci_lookup.get(skey); method='scientific_name_exact'
            if ki is None: ki=common_lookup.get(ckey); method='common_name_exact' if ki is not None else 'none'
            if ki is not None and h['weight'].shape[1]==model.classifier.weight.shape[1]:
                model.classifier.weight[int(r.label_id)].copy_(h['weight'][ki]); model.classifier.bias[int(r.label_id)].copy_(h['bias'][ki]); transfer.append({'qatar_label':r.common_name,'scientific_name':r.scientific_name,'kaggle_row':ki,'match_method':method})
train=ClipDataset(manifest,cfg,'train',True); val=ClipDataset(manifest,cfg,'val',False); test=ClipDataset(manifest,cfg,'test',False); bs=int(cfg['training']['batch_size']); nw=int(cfg['training']['num_workers']); loader=lambda d,s:DataLoader(d,batch_size=bs,shuffle=s,num_workers=nw,pin_memory=torch.cuda.is_available())
counts,weights=class_weights_from_manifest(manifest,'train')
def aux(m,o,y,c,cfg):
    ls=m.auxiliary_losses(o,y,c,cfg); total=o['logits'].sum()*0
    total=total+float(cfg['ap1']['lambda_condition'])*ls.get('condition',total*0)+float(cfg['ap1']['lambda_prototype'])*ls.get('prototype',total*0)+float(cfg['ap1']['lambda_router_balance'])*ls.get('router_balance',total*0)
    return total
out=ensure_dir(Path(cfg['_project_root'])/f'results/AP1/{a.variant}'); pd.DataFrame(transfer).to_csv(out/'classifier_head_transfer.csv',index=False); save_json(load_report,out/'backbone_transfer.json')
model,tm,_,_,_=train_single(model,loader(train,True),loader(val,False),loader(test,False),cfg,names,counts,out,weights,aux)
modeldir=ensure_dir(Path(cfg['_project_root'])/'models/qatar'); shutil.copy2(out/'best.pt',modeldir/'ap1_best.pt'); print('AP1 test:',tm); print('Stable checkpoint:',modeldir/'ap1_best.pt')
