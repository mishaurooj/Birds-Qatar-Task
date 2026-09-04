from __future__ import annotations
import argparse, sys, os, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from src.config import load_config, project_path
from src.audio import AUDIO_EXTS, audio_info
from src.utils import ensure_dir, save_json, sha256_file

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--hash',action='store_true'); a=p.parse_args(); cfg=load_config(a.config)
root=Path(cfg['paths']['kaggle_root']); out=ensure_dir(Path(cfg['_project_root'])/'results/kaggle_analysis')
if not root.exists(): raise SystemExit(f'Kaggle root not found: {root}')
files=[x for x in root.rglob('*') if x.is_file() and x.suffix.lower() in AUDIO_EXTS]
rows=[]
for f in tqdm(files,desc='Auditing Kaggle audio'):
    info=audio_info(f); folder=f.parent.name; common=folder[:-6] if folder.lower().endswith('_sound') else folder
    common=common.replace('_',' ').strip()
    r={'audio_path':str(f.resolve()),'relative_path':str(f.relative_to(root)),'folder_name':folder,'common_name_folder':common,'file_name':f.name,'file_size_bytes':f.stat().st_size,**info}
    if a.hash and info.get('readable'): r['sha256']=sha256_file(f)
    rows.append(r)
df=pd.DataFrame(rows); df.to_csv(out/'recording_audit.csv',index=False)
if 'readable' in df.columns:
    df[df.readable!=True].to_csv(out/'corrupted_or_unreadable_files.csv',index=False)
else:
    pd.DataFrame().to_csv(out/'corrupted_or_unreadable_files.csv',index=False)
if a.hash and 'sha256' in df.columns:
    dup=df[df.sha256.notna() & df.sha256.duplicated(keep=False)].sort_values('sha256')
    dup.to_csv(out/'duplicate_files_sha256.csv',index=False)
read=df[df.readable==True].copy() if 'readable' in df else df.iloc[0:0]
species=read.groupby('common_name_folder').agg(recordings=('audio_path','count'),minutes=('duration_sec',lambda x:x.sum()/60),median_duration_sec=('duration_sec','median')).reset_index().sort_values('recordings',ascending=False)
species.to_csv(out/'species_statistics.csv',index=False)
if len(read):
    read.groupby(['samplerate','channels']).agg(files=('audio_path','count'),hours=('duration_sec',lambda x:x.sum()/3600)).reset_index().to_csv(out/'audio_format_statistics.csv',index=False)
    read.groupby('folder_name').agg(files=('audio_path','count'),hours=('duration_sec',lambda x:x.sum()/3600),min_duration_sec=('duration_sec','min'),median_duration_sec=('duration_sec','median'),max_duration_sec=('duration_sec','max')).reset_index().to_csv(out/'folder_audio_statistics.csv',index=False)
summary={'root':str(root),'audio_files':len(df),'readable':int(read.shape[0]),'unreadable':int((df.readable!=True).sum()) if 'readable' in df else 0,'species_folders':int(species.shape[0]),'total_hours':float(read.duration_sec.sum()/3600) if len(read) else 0,'sample_rates':read.samplerate.value_counts().to_dict() if len(read) else {},'duplicate_files_sha256':int(df[df.sha256.notna() & df.sha256.duplicated(keep=False)].shape[0]) if a.hash and 'sha256' in df.columns else None}
save_json(summary,out/'dataset_summary.json')
if len(species):
    fig=plt.figure(figsize=(14,6)); ax=fig.add_subplot(111); ax.bar(range(len(species)),species.recordings); ax.set_xlabel('Species sorted by recording count'); ax.set_ylabel('Recordings'); ax.set_title('Kaggle full dataset class distribution'); fig.tight_layout(); fig.savefig(out/'class_distribution.png',dpi=160); plt.close(fig)
if len(read):
    fig=plt.figure(figsize=(8,5)); ax=fig.add_subplot(111); ax.hist(read.duration_sec.clip(upper=read.duration_sec.quantile(.99)),bins=50); ax.set_xlabel('Duration (s)'); ax.set_ylabel('Files'); fig.tight_layout(); fig.savefig(out/'duration_distribution.png',dpi=160); plt.close(fig)
print(json.dumps(summary,indent=2))
