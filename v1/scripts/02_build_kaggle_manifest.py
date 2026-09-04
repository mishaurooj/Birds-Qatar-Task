from __future__ import annotations
import argparse, sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from src.config import load_config, project_path
from src.audio import AUDIO_EXTS, audio_info
from src.utils import ensure_dir, save_json


def split_recordings(rec: pd.DataFrame, cfg) -> pd.DataFrame:
    """Class-aware source-recording split. Every retained class has train/val/test."""
    seed=int(cfg['project']['seed'])
    test=float(cfg['kaggle']['test_fraction']); val=float(cfg['kaggle']['val_fraction'])
    rec=rec.copy(); rec['split']='train'
    for lid,g in rec.groupby('label_id'):
        idx=g.index.to_numpy().copy(); rng=np.random.default_rng(seed+int(lid)); rng.shuffle(idx); n=len(idx)
        if n < 3:
            raise RuntimeError(f'label_id={lid} has only {n} recordings after filtering; minimum 3 is required.')
        n_test=max(1,int(round(n*test))); n_val=max(1,int(round(n*val)))
        while n_test+n_val >= n:
            if n_test>1: n_test-=1
            elif n_val>1: n_val-=1
            else: break
        rec.loc[idx[:n_test],'split']='test'
        rec.loc[idx[n_test:n_test+n_val],'split']='val'
    if rec.groupby('recording_id').split.nunique().max()!=1:
        raise RuntimeError('Recording split leakage detected.')
    return rec


def expand_clips(rec: pd.DataFrame, cfg) -> pd.DataFrame:
    clip=float(cfg['preprocessing']['clip_seconds']); maxclips=int(cfg['kaggle']['max_clips_per_recording']); rows=[]
    for _,r in rec.iterrows():
        dur=float(r.duration_total_sec)
        n=int(dur//clip)
        if n < 1: continue
        starts=np.linspace(0,max(0,dur-clip),maxclips) if maxclips>0 and n>maxclips else np.arange(n)*clip
        for j,s in enumerate(starts):
            d=r.to_dict(); d.update({'clip_id':f'{r.recording_id}__{j:04d}','start_sec':float(s),'duration_sec':clip,'preprocessed':False}); rows.append(d)
    return pd.DataFrame(rows)

p=argparse.ArgumentParser(description='Build leakage-safe public bird manifest and remove classes that cannot support independent train/val/test splits.')
p.add_argument('--config',default='configs/default.yaml')
p.add_argument('--min-source-recordings',type=int,help='Override kaggle.min_source_recordings. Minimum 3 is required for train/val/test.')
a=p.parse_args(); cfg=load_config(a.config)
if a.min_source_recordings is not None: cfg['kaggle']['min_source_recordings']=a.min_source_recordings
root=Path(cfg['paths']['kaggle_root'])
out_csv=project_path(cfg,cfg['paths']['kaggle_manifest']); labels_csv=project_path(cfg,cfg['paths']['kaggle_labels'])
recordings_csv=project_path(cfg,cfg['paths'].get('kaggle_recordings','data/kaggle_public_recordings.csv'))
dropped_csv=project_path(cfg,cfg['paths'].get('kaggle_dropped_classes','data/kaggle_dropped_low_source_classes.csv'))
ensure_dir(out_csv.parent)

# Optional metadata CSV supplied with the current Kaggle dataset.
csvs=list(root.rglob('*.csv')); meta=None
for c in csvs:
    try:
        t=pd.read_csv(c); cols=' '.join(map(str,t.columns)).lower()
        if ('scientific' in cols or 'common' in cols) and len(t)>20:
            meta=t; print('Using metadata:',c); break
    except Exception: pass

folders=sorted({f.parent.name for f in root.rglob('*') if f.is_file() and f.suffix.lower() in AUDIO_EXTS})
base_labels=[]
for original_id,folder in enumerate(folders):
    common=(folder[:-6] if folder.lower().endswith('_sound') else folder).replace('_',' ').strip(); sci=''
    if meta is not None:
        common_cols=[c for c in meta.columns if 'common' in c.lower() or c.lower() in ['name','bird name','bird']]
        sci_cols=[c for c in meta.columns if 'scientific' in c.lower()]
        for cc in common_cols:
            m=meta[meta[cc].astype(str).str.casefold()==common.casefold()]
            if len(m) and sci_cols: sci=str(m.iloc[0][sci_cols[0]]); break
    base_labels.append({'original_label_id':original_id,'common_name':common,'scientific_name':sci,'folder_name':folder})
base_labels=pd.DataFrame(base_labels); folder_to_original=dict(zip(base_labels.folder_name,base_labels.original_label_id))

rows=[]
for f in root.rglob('*'):
    if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS): continue
    inf=audio_info(f)
    if not inf.get('readable'): continue
    rid=f'{f.parent.name}/{f.stem}'
    rows.append({'recording_id':rid,'audio_path':str(f.resolve()),'folder_name':f.parent.name,'original_label_id':folder_to_original[f.parent.name],'duration_total_sec':float(inf['duration_sec'])})
rec=pd.DataFrame(rows)
if rec.empty: raise SystemExit(f'No readable audio found under {root}')

min_rec=max(3,int(cfg['kaggle'].get('min_source_recordings',3)))
counts=rec.groupby('original_label_id').recording_id.nunique()
base_labels['source_recordings']=base_labels.original_label_id.map(counts).fillna(0).astype(int)
if bool(cfg['kaggle'].get('drop_low_source_classes',True)):
    dropped=base_labels[base_labels.source_recordings<min_rec].copy()
    dropped['drop_reason']=f'fewer_than_{min_rec}_independent_source_recordings'
else:
    dropped=base_labels.iloc[0:0].copy()
dropped.to_csv(dropped_csv,index=False)
keep=base_labels[base_labels.source_recordings>=min_rec].copy() if bool(cfg['kaggle'].get('drop_low_source_classes',True)) else base_labels.copy()
keep=keep.reset_index(drop=True); keep['label_id']=np.arange(len(keep),dtype=int)
labels=keep[['label_id','original_label_id','common_name','scientific_name','folder_name','source_recordings']]
labels.to_csv(labels_csv,index=False)
map_id=dict(zip(labels.original_label_id,labels.label_id))
rec=rec[rec.original_label_id.isin(map_id)].copy(); rec['label_id']=rec.original_label_id.map(map_id).astype(int)
rec=split_recordings(rec,cfg)
clips=expand_clips(rec,cfg)
if clips.empty: raise SystemExit('No complete clips were produced.')
rec.to_csv(recordings_csv,index=False); clips.to_csv(out_csv,index=False)
summary={
    'raw_species_detected':int(len(base_labels)), 'retained_species':int(len(labels)), 'dropped_species':int(len(dropped)),
    'minimum_source_recordings':min_rec, 'retained_recordings':int(len(rec)), 'retained_clips_before_sp_cache':int(len(clips)),
    'split_recordings':rec.split.value_counts().to_dict(), 'split_clips':clips.split.value_counts().to_dict(),
    'dropped_names':dropped.common_name.astype(str).tolist(),
    'method':'source-recording-disjoint class-aware split; low-source classes removed before segmentation'
}
save_json(summary,out_csv.with_name('kaggle_public_selection_summary.json'))
print(json.dumps(summary,indent=2))
print('Labels:',labels_csv); print('Dropped classes:',dropped_csv); print('Recordings:',recordings_csv); print('Clips:',out_csv)
