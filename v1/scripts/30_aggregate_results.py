import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config
from src.utils import ensure_dir
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config); root=Path(cfg['_project_root'])/'results'; rows=[]
for f in root.rglob('*metrics.json'):
    if f.name not in ['test_metrics.json','metrics.json']: continue
    try: d=json.load(open(f,encoding='utf-8'))
    except Exception: continue
    rel=f.relative_to(root); stage=rel.parts[0] if rel.parts else ''; variant=rel.parts[1] if len(rel.parts)>1 else rel.parent.name; rows.append({'stage':stage,'variant':variant,'metrics_file':str(f),**d})
out=ensure_dir(Path(cfg['_project_root'])/'results_summary'); df=pd.DataFrame(rows); df.to_csv(out/'all_test_metrics.csv',index=False)
if len(df):
    preferred=[c for c in ['stage','variant','macro_f1','micro_f1','balanced_accuracy','accuracy','rare_class_recall','macro_average_precision','exact_match'] if c in df.columns]
    df[preferred].to_csv(out/'headline_metrics.csv',index=False); print(df[preferred].to_string(index=False))
else: print('No completed test metric files found.')
