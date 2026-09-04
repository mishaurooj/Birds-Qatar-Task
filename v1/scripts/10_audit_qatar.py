from __future__ import annotations
import argparse,sys,re,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from tqdm import tqdm
from src.config import load_config,project_path
from src.audio import AUDIO_EXTS,audio_info
from src.utils import ensure_dir,save_json
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config)
root=Path(cfg['paths']['qatar_root']); out=project_path(cfg,cfg['paths']['qatar_raw_audit']); ensure_dir(out.parent)
if not root.exists(): raise SystemExit(f'Qatar root not found: {root}')
rows=[]
for f in tqdm([x for x in root.rglob('*') if x.is_file() and x.suffix.lower() in AUDIO_EXTS],desc='Auditing Qatar'):
    inf=audio_info(f); session=next((p.name for p in f.parents if p.parent==root),f.parent.name)
    rows.append({'audio_path':str(f.resolve()),'study_session':session,'source_recording':f.stem,'file_name':f.name,**inf})
df=pd.DataFrame(rows); df.to_csv(out,index=False)
summary={'sessions':int(df.study_session.nunique()) if len(df) else 0,'files':len(df),'readable':int(df.readable.sum()) if len(df) else 0,'total_hours':float(df.loc[df.readable==True,'duration_sec'].sum()/3600) if len(df) else 0,'by_session':df.groupby('study_session').duration_sec.sum().div(3600).to_dict() if len(df) else {}}
save_json(summary,out.with_suffix('.summary.json')); print(json.dumps(summary,indent=2))
