import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.config import load_config
from src.backbone import build_backbone
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config)
model=build_backbone(cfg); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval()
w=torch.zeros(1,int(cfg['preprocessing']['sample_rate']*cfg['preprocessing']['clip_seconds']),device=device)
with torch.no_grad(): z=model(w)
print('BirdMAE output:',tuple(z.shape),'embedding_dim=',model.embedding_dim,'device=',device)
