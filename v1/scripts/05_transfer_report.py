"""Creates a transparent mapping report between Kaggle labels and Qatar target labels.
It does not train anything. It is useful before Qatar transfer to verify that scientific-name
matches, not fuzzy common-name matches, are used for classifier-head initialization.
"""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from src.config import load_config,project_path
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config)
k=pd.read_csv(project_path(cfg,cfg['paths']['kaggle_labels'])); q=pd.read_csv(Path(cfg['_project_root'])/'configs/qatar_species_27.csv')
k['sci_key']=k.scientific_name.fillna('').astype(str).str.strip().str.casefold(); q['sci_key']=q.scientific_name.astype(str).str.strip().str.casefold()
k['common_key']=k.common_name.fillna('').astype(str).str.strip().str.casefold(); q['common_key']=q.common_name.astype(str).str.strip().str.casefold()
rows=[]
for _,r in q.iterrows():
    hit=k[(k.sci_key==r.sci_key) & (k.sci_key!='')]
    method='scientific_name_exact'
    if len(hit)==0:
        hit=k[k.common_key==r.common_key]; method='common_name_exact' if len(hit) else 'no_exact_match'
    h=hit.iloc[0] if len(hit) else None
    rows.append({'qatar_label_id':int(r.qatar_label_id),'common_name_qatar':r.common_name,'scientific_name_qatar':r.scientific_name,'kaggle_label_id':int(h.label_id) if h is not None else np.nan,'common_name_kaggle':h.common_name if h is not None else '','scientific_name_kaggle':h.scientific_name if h is not None else '','match_method':method})
m=pd.DataFrame(rows); out=Path(cfg['_project_root'])/'results_summary'; out.mkdir(parents=True,exist_ok=True); m.to_csv(out/'kaggle_to_qatar_species_mapping.csv',index=False); print(m.to_string(index=False))
