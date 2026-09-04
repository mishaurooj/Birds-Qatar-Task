from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import load_config, project_path
from src.utils import ensure_dir, save_json

p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config)
root=Path(cfg['_project_root']); out=ensure_dir(root/'results/qatar_analysis')
conf_path=project_path(cfg,cfg['paths']['qatar_confirmed_labels'])
if not conf_path.exists(): raise SystemExit('Run scripts/12_birdnet_label_qatar.py first.')
conf=pd.read_csv(conf_path)
q=pd.read_csv(root/'configs/qatar_species_27.csv')
if cfg['qatar_labeling'].get('use_26_report_species',True): q=q[q.report_training_status!='exclude_in_report'].copy()
qid_to_name=dict(zip(q.qatar_label_id.astype(int),q.common_name.astype(str)))
rows=[]; co=np.zeros((len(q),len(q)),dtype=int); qid_to_pos={int(v):i for i,v in enumerate(q.qatar_label_id)}
for _,r in conf.iterrows():
    try: cand=json.loads(r.labels_json) if isinstance(r.labels_json,str) else []
    except Exception: cand=[]
    ids=[int(x['qatar_label_id']) for x in cand if int(x['qatar_label_id']) in qid_to_pos]
    for x in cand:
        qi=int(x['qatar_label_id'])
        if qi not in qid_to_pos: continue
        rows.append({'clip_id':r.clip_id,'study_session':r.study_session,'source_recording':r.source_recording,'qatar_label_id':qi,'common_name':qid_to_name[qi],'mean_confidence':float(x.get('mean_confidence',np.nan)),'max_confidence':float(x.get('max_confidence',np.nan)),'count_views':int(x.get('count_views',0)),'clip_seconds':5.0})
    for i in ids:
        for j in ids: co[qid_to_pos[i],qid_to_pos[j]]+=1
long=pd.DataFrame(rows)
if len(long):
    stats=long.groupby(['qatar_label_id','common_name']).agg(confirmed_clips=('clip_id','nunique'),pseudo_label_minutes=('clip_seconds',lambda x:x.sum()/60),mean_confidence=('mean_confidence','mean'),median_confidence=('mean_confidence','median'),sessions_detected=('study_session','nunique'),source_recordings=('source_recording','nunique')).reset_index()
else:
    stats=pd.DataFrame(columns=['qatar_label_id','common_name','confirmed_clips','pseudo_label_minutes','mean_confidence','median_confidence','sessions_detected','source_recordings'])
base=q[['qatar_label_id','common_name','scientific_name','family','report_local_hms','report_local_seconds','report_local_note']].copy(); comp=base.merge(stats,on=['qatar_label_id','common_name'],how='left')
for c in ['confirmed_clips','pseudo_label_minutes','sessions_detected','source_recordings']: comp[c]=comp[c].fillna(0)
comp['report_local_minutes']=pd.to_numeric(comp.report_local_seconds,errors='coerce')/60
comp['pseudo_minus_report_minutes']=comp.pseudo_label_minutes-comp.report_local_minutes
comp.to_csv(out/'qatar_label_statistics_and_report_comparison.csv',index=False)
if len(long):
    long.to_csv(out/'confirmed_label_long.csv',index=False)
    sess=pd.pivot_table(long,index='study_session',columns='common_name',values='clip_id',aggfunc='nunique',fill_value=0); sess.to_csv(out/'session_species_clip_matrix.csv')
    confstat=long.groupby('common_name').mean_confidence.agg(['count','mean','median','min','max']).reset_index(); confstat.to_csv(out/'confidence_statistics.csv',index=False)
else: pd.DataFrame().to_csv(out/'session_species_clip_matrix.csv')
pd.DataFrame(co,index=q.common_name,columns=q.common_name).to_csv(out/'label_cooccurrence_matrix.csv')
card=conf.num_labels.value_counts().sort_index().rename_axis('num_labels').reset_index(name='clips') if 'num_labels' in conf else pd.DataFrame(); card.to_csv(out/'label_cardinality_distribution.csv',index=False)
summary={'prepared_5s_clips':int(len(conf)),'clips_with_any_target':int((conf.num_labels>0).sum()) if 'num_labels' in conf else 0,'clips_with_multiple_targets':int((conf.num_labels>1).sum()) if 'num_labels' in conf else 0,'confirmed_label_assignments':int(len(long)),'pseudo_label_assignment_hours':float(len(long)*5/3600),'unique_sessions':int(conf.study_session.nunique()) if len(conf) else 0,'threshold':float(conf.threshold.iloc[0]) if len(conf) and 'threshold' in conf else None,'sdp_report_stated_multilabel_total_hms':'52:34:50','note':'Pseudo-label assignment hours double-count overlap when several species are confirmed in one 5-s clip; compare species rows rather than treating it as unique recording time.'}
save_json(summary,out/'qatar_label_dataset_summary.json')
if len(comp):
    s=comp.sort_values('pseudo_label_minutes',ascending=False)
    fig=plt.figure(figsize=(13,6)); ax=fig.add_subplot(111); ax.bar(range(len(s)),s.pseudo_label_minutes); ax.set_xticks(range(len(s))); ax.set_xticklabels(s.common_name,rotation=80,ha='right',fontsize=7); ax.set_ylabel('Pseudo-labelled minutes'); ax.set_title('Qatar BirdNET pseudo-label duration by target'); fig.tight_layout(); fig.savefig(out/'qatar_pseudolabel_duration_by_species.png',dpi=170); plt.close(fig)
print(json.dumps(summary,indent=2)); print('Analysis:',out)
