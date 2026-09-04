from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import load_config,project_path
from src.backbone import build_backbone
from src.datasets import ClipDataset
from src.training import evaluate_single,class_weights_from_manifest
class M(nn.Module):
 def __init__(self,b,n): super().__init__(); self.backbone=b; self.head=nn.Linear(b.embedding_dim,n)
 def forward(self,w,c): z=self.backbone(w); return {'logits':self.head(z),'embedding':z}
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--checkpoint'); a=p.parse_args(); cfg=load_config(a.config)
labels=pd.read_csv(project_path(cfg,cfg['paths']['kaggle_labels'])).sort_values('label_id'); names=labels.common_name.astype(str).tolist(); model=M(build_backbone(cfg),len(names)); ck=torch.load(a.checkpoint or project_path(cfg,cfg['paths']['kaggle_checkpoint']),map_location='cpu'); model.load_state_dict(ck['model_state_dict'],strict=False)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); ds=ClipDataset(project_path(cfg,cfg['paths']['kaggle_manifest']),cfg,'test',False); dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=False); counts,_=class_weights_from_manifest(project_path(cfg,cfg['paths']['kaggle_manifest']),'train')
m,_,_,_=evaluate_single(model,dl,device,names,counts,Path(cfg['_project_root'])/f'results/KAGGLE_PUBLIC_{len(names)}/re_evaluation'); print(m)
