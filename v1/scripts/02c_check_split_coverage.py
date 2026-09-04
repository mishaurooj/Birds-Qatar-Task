from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config, project_path
from src.utils import ensure_dir
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config)
manifest=project_path(cfg,cfg['paths']['kaggle_manifest']); labels_path=project_path(cfg,cfg['paths']['kaggle_labels'])
if not manifest.exists() or not labels_path.exists(): raise SystemExit('Build/cache the public manifest first.')
m=pd.read_csv(manifest); labels=pd.read_csv(labels_path).sort_values('label_id'); rows=[]
for _,r in labels.iterrows():
    lid=int(r.label_id); g=m[m.label_id==lid]
    row={'label_id':lid,'common_name':str(r.common_name)}
    for s in ['train','val','test']:
        sg=g[g.split.astype(str)==s]; row[f'{s}_clips']=len(sg); row[f'{s}_recordings']=sg.recording_id.nunique()
    rows.append(row)
out=pd.DataFrame(rows); out['total_recordings']=out[['train_recordings','val_recordings','test_recordings']].sum(axis=1)
for s in ['train','val','test']: out[f'has_{s}']=out[f'{s}_clips']>0
out['has_all_splits']=out.has_train & out.has_val & out.has_test
out_dir=ensure_dir(Path(cfg['_project_root'])/'results/kaggle_analysis'); out.to_csv(out_dir/'split_class_coverage.csv',index=False)
print(f'Classes: {len(out)}')
for s in ['train','val','test']:
    miss=out[~out[f'has_{s}']]; print(f'{s}: {len(out)-len(miss)}/{len(out)} classes represented; missing={len(miss)}')
    if len(miss): print('  '+', '.join(miss.common_name.tolist()))
print(f'All three splits: {int(out.has_all_splits.sum())}/{len(out)} classes')
print('Minimum source recordings in retained classes:',int(out.total_recordings.min()) if len(out) else 0)
print('Saved:',out_dir/'split_class_coverage.csv')
if bool(cfg['kaggle'].get('require_all_splits',True)) and not out.has_all_splits.all():
    raise SystemExit('ERROR: require_all_splits=true but some retained classes are absent from a split. Rebuild manifest/cache or increase min_source_recordings.')
