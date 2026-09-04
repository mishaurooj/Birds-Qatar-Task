from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .audio import load_segment, report_sp_pipeline, condition_features, augment_waveform


class ClipDataset(Dataset):
    def __init__(self, csv_path, cfg, split=None, train=False):
        self.df = pd.read_csv(csv_path)
        if split is not None and 'split' in self.df.columns:
            self.df = self.df[self.df.split.astype(str)==str(split)].reset_index(drop=True)
        self.cfg = cfg; self.train = train
        self.sr = int(cfg['preprocessing']['sample_rate'])
        self.clip_s = float(cfg['preprocessing']['clip_seconds'])
        self.seed = int(cfg['project']['seed'])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        y, sr = load_segment(r.audio_path, float(r.get('start_sec',0)), float(r.get('duration_sec',self.clip_s)), self.sr)
        pre = str(r.get('preprocessed',False)).strip().lower() in {'1','true','yes','y'}
        if self.cfg['preprocessing'].get('use_report_sp_pipeline',False) and not pre:
            y = report_sp_pipeline(y,sr,self.cfg)
        if self.train:
            rng = np.random.default_rng(self.seed + idx + random.randint(0,10_000_000))
            y = augment_waveform(y,sr,self.cfg,rng)
        c = condition_features(y,sr)
        return {
            'waveform': torch.from_numpy(y),
            'condition': torch.from_numpy(c),
            'label': torch.tensor(int(r.label_id),dtype=torch.long),
            'row_index': int(r.name),
            'recording_id': str(r.get('recording_id','')),
        }


class SyntheticPolyphonyDataset(Dataset):
    """Creates 2-4 bird mixtures from single-label source clips.

    A selected 5-s call is placed at a random offset within a 10-s soundscape.
    This creates exact clip-relative intervals for AP2 without pretending the
    public folder-level labels provide event timestamps.
    """
    def __init__(self, csv_path, cfg, split='train', size=4000, train=True):
        df = pd.read_csv(csv_path)
        if 'split' in df: df = df[df.split.astype(str)==split]
        self.df = df.reset_index(drop=True)
        self.cfg=cfg; self.train=train; self.size=int(size)
        self.sr=int(cfg['preprocessing']['sample_rate']); self.seed=int(cfg['project']['seed'])
        labels_path = Path(cfg['_project_root']) / 'data' / 'qatar_labels.csv'
        self.num_classes = len(pd.read_csv(labels_path)) if labels_path.exists() else int(self.df.label_id.max())+1
        self.max_sources=int(cfg['ap2']['max_sources'])
        self.max_cardinality=2
        self.difficulty=None
        self.by_class={int(k):v.index.to_numpy() for k,v in self.df.groupby('label_id')}
        self.classes=np.array(sorted(self.by_class))
    def __len__(self): return self.size
    def set_epoch(self, epoch):
        e1,e2,e3 = self.cfg['ap2']['curriculum_epochs']
        self.max_cardinality = 2 if epoch<e1 else (3 if epoch<e2 else 4)
    def set_difficulty(self, mat): self.difficulty=np.asarray(mat) if mat is not None else None
    def _choose_classes(self, rng, k):
        first=int(rng.choice(self.classes)); chosen=[first]
        while len(chosen)<k:
            avail=np.array([c for c in self.classes if c not in chosen])
            if self.difficulty is not None and len(chosen)>0:
                score=np.max(self.difficulty[np.ix_(chosen,avail)],axis=0).astype(float)
                score=np.maximum(score,1e-4)**float(self.cfg['ap2'].get('hard_pair_gamma',2.0)); score/=score.sum()
                nxt=int(rng.choice(avail,p=score))
            else: nxt=int(rng.choice(avail))
            chosen.append(nxt)
        return chosen
    def __getitem__(self, idx):
        rng=np.random.default_rng(self.seed + idx + random.randint(0,9999999) if self.train else self.seed+idx)
        k=int(rng.integers(2,self.max_cardinality+1))
        classes=self._choose_classes(rng,k)
        out=np.zeros(int(10*self.sr),dtype=np.float32); intervals=[]; target=np.zeros(self.num_classes,dtype=np.float32)
        for cls in classes:
            ri=int(rng.choice(self.by_class[cls])); r=self.df.iloc[ri]
            y,_=load_segment(r.audio_path,float(r.get('start_sec',0)),5.0,self.sr)
            if self.cfg['preprocessing'].get('use_report_sp_pipeline',False): y=report_sp_pipeline(y,self.sr,self.cfg)
            onset=float(rng.uniform(0,5)); s=int(onset*self.sr); e=min(len(out),s+len(y)); amp=float(10**(rng.uniform(-6,3)/20))
            out[s:e]+=amp*y[:e-s]; intervals.append((cls,onset,(e/self.sr))); target[cls]=1
        out=np.clip(out,-1,1); wave5=out[:5*self.sr].copy()
        return {'wave5':torch.from_numpy(wave5),'wave10':torch.from_numpy(out),
                'target':torch.from_numpy(target),'intervals':intervals,'condition':torch.from_numpy(condition_features(wave5,self.sr))}


def poly_collate(batch):
    return {
        'wave5': torch.stack([b['wave5'] for b in batch]),
        'wave10': torch.stack([b['wave10'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'condition': torch.stack([b['condition'] for b in batch]),
        'intervals': [b['intervals'] for b in batch],
    }

class RealMultiLabelDataset(Dataset):
    def __init__(self,csv_path,cfg,split=None,train=False):
        self.df=pd.read_csv(csv_path)
        if split is not None and 'split' in self.df: self.df=self.df[self.df.split.astype(str)==split].reset_index(drop=True)
        self.cfg=cfg; self.train=train; self.sr=int(cfg['preprocessing']['sample_rate']); self.seed=int(cfg['project']['seed'])
        if len(self.df) and 'num_classes' in self.df.columns:
            self.num_classes = int(self.df['num_classes'].iloc[0])
        elif len(self.df) and 'label_ids_json' in self.df.columns:
            ids = [int(v) for x in self.df['label_ids_json'].dropna().astype(str) for v in json.loads(x)]
            self.num_classes = (max(ids) + 1) if ids else 0
        else:
            labels_path = Path(cfg['_project_root']) / 'data' / 'qatar_labels.csv'
            self.num_classes = len(pd.read_csv(labels_path)) if labels_path.exists() else (26 if cfg.get('qatar', {}).get('use_26_report_species', True) else 27)

    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]; y,_=load_segment(r.audio_path,0,5.0,self.sr)
        npath=str(r.get('next_audio_path',''))
        if npath and npath!='nan' and Path(npath).exists(): y2,_=load_segment(npath,0,5.0,self.sr)
        else: y2=np.zeros_like(y)
        w10=np.concatenate([y,y2]).astype(np.float32); ids=json.loads(r.label_ids_json) if isinstance(r.label_ids_json,str) else list(r.label_ids_json)
        target=np.zeros(self.num_classes,dtype=np.float32)
        for c in ids:
            if 0<=int(c)<self.num_classes: target[int(c)]=1
        intervals=[]
        for c in ids: intervals.append((int(c),0.0,5.0))
        return {'wave5':torch.from_numpy(y),'wave10':torch.from_numpy(w10),'target':torch.from_numpy(target),'intervals':intervals,'condition':torch.from_numpy(condition_features(y,self.sr))}
