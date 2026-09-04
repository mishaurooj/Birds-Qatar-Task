from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.config import load_config, project_path
from src.utils import seed_everything, ensure_dir
from src.backbone import build_backbone
from src.datasets import ClipDataset
from src.training import train_single, class_weights_from_manifest

class KaggleClassifier(nn.Module):
    def __init__(self,backbone,n): super().__init__(); self.backbone=backbone; self.head=nn.Linear(backbone.embedding_dim,n)
    def forward(self,w,c=None):
        z=self.backbone(w); return {'logits':self.head(z),'embedding':z}

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/default.yaml')
p.add_argument('--epochs',type=int)
p.add_argument('--batch-size',type=int,help='Override training.batch_size from YAML')
p.add_argument('--grad-accum',type=int,help='Override gradient_accumulation_steps from YAML')
p.add_argument('--num-workers',type=int,help='Override DataLoader workers from YAML')
p.add_argument('--backbone',choices=['birdmae','smoke_cnn'])
p.add_argument('--resume',default=None,help='Resume from last.pt/interrupted.pt checkpoint')
a=p.parse_args(); cfg=load_config(a.config)
if a.epochs: cfg['training']['epochs']=a.epochs
if a.batch_size: cfg['training']['batch_size']=a.batch_size
if a.grad_accum: cfg['training']['gradient_accumulation_steps']=a.grad_accum
if a.num_workers is not None: cfg['training']['num_workers']=a.num_workers
if a.backbone: cfg['backbone']['type']=a.backbone
seed_everything(int(cfg['project']['seed']))
manifest=project_path(cfg,cfg['paths']['kaggle_manifest'])
labels_path=project_path(cfg,cfg['paths']['kaggle_labels'])
if not manifest.exists(): raise SystemExit('Build Kaggle manifest first: python scripts\\02_build_kaggle_manifest.py --config configs\\default.yaml')
labels=pd.read_csv(labels_path).sort_values('label_id'); names=labels.common_name.astype(str).tolist(); n=len(names)
if n<2: raise SystemExit('Build Kaggle manifest first.')
mf=pd.read_csv(manifest,usecols=['audio_path','split','label_id'])
non_wav=~mf.audio_path.astype(str).str.lower().str.endswith('.wav')
if bool(cfg['kaggle'].get('require_cached_wav',True)) and non_wav.any():
    sample=mf.loc[non_wav,'audio_path'].iloc[0]
    raise SystemExit(
        f'Training manifest still contains {int(non_wav.sum())} compressed/non-WAV clip references.\n'
        f'Example: {sample}\n\n'
        'Run the one-time cache step first:\n'
        '  python scripts\\02b_cache_kaggle_audio.py --config configs\\default.yaml --workers 2\n'
        'This removes repeated mpg123 MP3 decoding from the training loop and logs unrecoverable source files.'
    )
for split in ['train','val','test']:
    if not (mf.split.astype(str)==split).any(): raise SystemExit(f'No {split} samples in manifest.')

# Report per-class coverage before spending hours training. Source-recording-disjoint
# splitting cannot put a class with fewer than 3 recordings into train+val+test.
coverage=[]
for lid,name in enumerate(names):
    g=mf[mf.label_id==lid]
    coverage.append({'label_id':lid,'common_name':name,
                     'train':int((g.split.astype(str)=='train').sum()),
                     'val':int((g.split.astype(str)=='val').sum()),
                     'test':int((g.split.astype(str)=='test').sum())})
cov=pd.DataFrame(coverage)
for split in ['val','test']:
    missing=cov[cov[split]==0]
    if len(missing):
        print(f'WARNING: {len(missing)} classes have no {split} clips under source-recording-disjoint splitting.')
        print('  ' + ', '.join(missing.common_name.astype(str).tolist()))
        print('Metrics will report supported-class macro scores plus evaluation_class_coverage; no sklearn undefined-class warnings should occur.')
if bool(cfg['kaggle'].get('require_all_splits',True)) and ((cov.train==0)|(cov.val==0)|(cov.test==0)).any():
    raise SystemExit('Some retained public classes are missing train/val/test support. Run 02_build_kaggle_manifest.py, 02b_cache_kaggle_audio.py, then 02c_check_split_coverage.py. Low-source classes should be removed before training.')
train=ClipDataset(manifest,cfg,'train',True); val=ClipDataset(manifest,cfg,'val',False); test=ClipDataset(manifest,cfg,'test',False)
bs=int(cfg['training']['batch_size']); nw=int(cfg['training']['num_workers'])
def loader(ds,shuffle):
    return DataLoader(ds,batch_size=bs,shuffle=shuffle,num_workers=nw,pin_memory=torch.cuda.is_available(),
                      persistent_workers=(nw>0),drop_last=False)
counts,weights=class_weights_from_manifest(manifest,'train')
backbone=build_backbone(cfg); model=KaggleClassifier(backbone,n)
out=ensure_dir(Path(cfg['_project_root'])/f'results/KAGGLE_PUBLIC_{n}/BirdMAE_full')
print(f'Kaggle training: classes={n}, train={len(train)}, val={len(val)}, test={len(test)}, batch={bs}, grad_accum={cfg["training"].get("gradient_accumulation_steps",1)}')
model,metrics,probs,z,y=train_single(model,loader(train,True),loader(val,False),loader(test,False),cfg,names,counts,out,weights,resume_path=a.resume)
model_dir=ensure_dir(project_path(cfg,cfg['paths']['kaggle_checkpoint']).parent); ck=torch.load(out/'best.pt',map_location='cpu'); torch.save(ck,model_dir/'best.pt')
model.eval(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device); sums=torch.zeros(n,backbone.embedding_dim,device=device); nums=torch.zeros(n,device=device)
with torch.no_grad():
    for b in loader(ClipDataset(manifest,cfg,'train',False),False):
        yy=b['label'].to(device); zz=model(b['waveform'].to(device),b['condition'].to(device))['embedding']
        sums.index_add_(0,yy,zz); nums.index_add_(0,yy,torch.ones_like(yy,dtype=torch.float32))
proto=(sums/nums.clamp_min(1).unsqueeze(1)).cpu(); torch.save({'prototypes':proto,'counts':nums.cpu(),'labels':labels.to_dict('records')},model_dir/'prototypes.pt')
torch.save({'weight':model.head.weight.detach().cpu(),'bias':model.head.bias.detach().cpu(),'labels':labels.to_dict('records')},project_path(cfg,cfg['paths'].get('kaggle_classifier_head','models/kaggle_public/classifier_head.pt')))
print('Saved Kaggle-trained encoder/model to',model_dir/'best.pt'); print('Test metrics:',metrics)
