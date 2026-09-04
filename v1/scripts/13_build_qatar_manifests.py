from __future__ import annotations
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from src.config import load_config,project_path
from src.utils import ensure_dir,save_json
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); a=p.parse_args(); cfg=load_config(a.config)
conf=pd.read_csv(project_path(cfg,cfg['paths']['qatar_confirmed_labels'])); q=pd.read_csv(Path(cfg['_project_root'])/'configs/qatar_species_27.csv')
if cfg['qatar_labeling'].get('use_26_report_species',True): q=q[q.report_training_status!='exclude_in_report'].copy()
q=q.reset_index(drop=True); q['label_id']=np.arange(len(q)); fams=sorted(q.family.unique()); fmap={f:i for i,f in enumerate(fams)}; q['family_id']=q.family.map(fmap); q.to_csv(Path(cfg['_project_root'])/'data/qatar_labels.csv',index=False)
mapid=dict(zip(q.qatar_label_id.astype(int),q.label_id.astype(int)))
# Session-disjoint split: with the 5 report sessions, use 3 train / 1 val / 1 test.
sessions=sorted(conf.study_session.dropna().astype(str).unique()); split_map={}
if len(sessions)>=3:
    for s in sessions: split_map[s]='train'
    split_map[sessions[-2]]='val'; split_map[sessions[-1]]='test'
else:
    # Fallback source-recording disjoint deterministic assignment.
    src=sorted(conf.source_recording.astype(str).unique())
    for i,s in enumerate(src): split_map[s]='test' if i%10<2 else ('val' if i%10==2 else 'train')

def split_for(r):
    return split_map.get(str(r.study_session),split_map.get(str(r.source_recording),'train'))
# Multi-label manifest.
mr=[]
for _,r in conf.iterrows():
    cand=json.loads(r.labels_json) if isinstance(r.labels_json,str) else []
    ids=[mapid[int(x['qatar_label_id'])] for x in cand if int(x['qatar_label_id']) in mapid]
    if not ids: continue
    mr.append({'clip_id':r.clip_id,'audio_path':r.audio_path,'study_session':r.study_session,'source_recording':r.source_recording,'label_ids_json':json.dumps(sorted(set(ids))),'qatar_label_ids_json':json.dumps([int(x['qatar_label_id']) for x in cand]),'num_labels':len(set(ids)),'num_classes':len(q),'split':split_for(r)})
multi_columns=['clip_id','audio_path','study_session','source_recording','label_ids_json','qatar_label_ids_json','num_labels','num_classes','split']
multi=pd.DataFrame(mr, columns=multi_columns)
# Attach the next 5-s clip from the same source to provide 10-s context without crossing sessions.
if len(multi):
    multi['next_audio_path']=''
    for src,g in multi.groupby('source_recording',sort=False):
        idx=list(g.index)
        for aidx,bidx in zip(idx[:-1],idx[1:]): multi.loc[aidx,'next_audio_path']=multi.loc[bidx,'audio_path']
multi.to_csv(project_path(cfg,cfg['paths']['qatar_multi_manifest']),index=False)
# Clean single-label transfer set: only clips with exactly one confirmed target.
sr=[]
for _,r in multi[multi.num_labels==1].iterrows():
    lid=int(json.loads(r.label_ids_json)[0]); rr=q[q.label_id==lid].iloc[0]
    sr.append({'clip_id':r.clip_id,'recording_id':r.source_recording,'audio_path':r.audio_path,'start_sec':0.0,'duration_sec':5.0,'label_id':lid,'qatar_label_id':int(rr.qatar_label_id),'common_name':rr.common_name,'scientific_name':rr.scientific_name,'family':rr.family,'family_id':int(rr.family_id),'study_session':r.study_session,'split':r.split})
single_columns=['clip_id','recording_id','audio_path','start_sec','duration_sec','label_id','qatar_label_id','common_name','scientific_name','family','family_id','study_session','split']
single=pd.DataFrame(sr, columns=single_columns); single.to_csv(project_path(cfg,cfg['paths']['qatar_single_manifest']),index=False)
# Dominant-label alternative retained for a report-replication ablation, but not used by default.
dom=[]
for _,r in conf[conf.num_labels>0].iterrows():
    qid=int(r.dominant_qatar_label_id)
    if qid not in mapid: continue
    rr=q[q.qatar_label_id==qid].iloc[0]; dom.append({'clip_id':r.clip_id,'recording_id':r.source_recording,'audio_path':r.audio_path,'start_sec':0,'duration_sec':5,'label_id':int(rr.label_id),'qatar_label_id':qid,'split':split_for(r),'dominant_confidence':r.dominant_confidence})
pd.DataFrame(dom).to_csv(Path(cfg['_project_root'])/'data/qatar_dominant_manifest.csv',index=False)
# Session fold table for stronger cross-session validation.
pd.DataFrame([{'study_session':s,'default_split':split_map[s],'leave_one_session_out_fold':i} for i,s in enumerate(sessions)]).to_csv(Path(cfg['_project_root'])/'data/qatar_session_splits.csv',index=False)
print('Qatar target classes:',len(q),'families:',len(fams)); print('Single-label clean clips:',len(single)); print('Multi-label/presence clips:',len(multi)); print('Split map:',split_map)
