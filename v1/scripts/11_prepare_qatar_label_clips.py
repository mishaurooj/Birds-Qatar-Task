"""Replicates the SDP report's local-label preparation as closely as practical.

The report states that local recordings were silence-trimmed using a 40 dB threshold,
then partitioned into 5-second segments before BirdNET analysis. This script keeps the
raw files untouched, writes derived 5-s FLAC clips, and records source/session provenance.
"""
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from tqdm import tqdm
from src.config import load_config,project_path
from src.audio import load_audio,active_concat,write_wav
from src.utils import ensure_dir
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--limit-files',type=int); a=p.parse_args(); cfg=load_config(a.config)
audit=pd.read_csv(project_path(cfg,cfg['paths']['qatar_raw_audit'])); audit=audit[audit.readable==True].reset_index(drop=True)
if a.limit_files: audit=audit.head(a.limit_files)
outdir=ensure_dir(project_path(cfg,cfg['paths']['qatar_label_clips'])); rows=[]; sr=int(cfg['preprocessing']['sample_rate']); clip_n=int(sr*5); topdb=float(cfg['preprocessing']['silence_top_db'])
for _,r in tqdm(audit.iterrows(),total=len(audit),desc='Preparing 5-s Qatar label clips'):
    y,_=load_audio(r.audio_path,target_sr=sr,mono=True); active,intervals=active_concat(y,topdb)
    n=len(active)//clip_n
    for j in range(n):
        seg=active[j*clip_n:(j+1)*clip_n]
        sid=str(r.study_session).replace(' ','_').replace('[','').replace(']','').replace('+','p')
        stem=f'{sid}__{r.source_recording}__{j:05d}'; path=outdir/(stem+'.flac'); write_wav(path,seg,sr)
        rows.append({'clip_id':stem,'audio_path':str(path.resolve()),'study_session':r.study_session,'source_audio_path':r.audio_path,'source_recording':r.source_recording,'active_concat_index':j,'duration_sec':5.0,'active_intervals_json':json.dumps(intervals) if j==0 else ''})
manifest=pd.DataFrame(rows); dest=Path(cfg['_project_root'])/'data/qatar_label_clip_manifest.csv'; manifest.to_csv(dest,index=False); print('Wrote',len(manifest),'5-s clips to',outdir); print('Manifest:',dest)
