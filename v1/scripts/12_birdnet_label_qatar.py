"""BirdNET pseudo-labelling using the SDP report rule.

For each derived 5-s clip, construct three 3-s views starting at 0, 1 and 2 seconds.
A target is confirmed in the 5-s clip only if it exceeds the configured threshold in
at least 2 of the 3 views. Predictions outside the Qatar target list are ignored, as in
report Test 1. The threshold is configurable because the report gives 0.6 for its
overall validation test but does not explicitly state one final production threshold.
"""
from __future__ import annotations
import argparse,sys,tempfile,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,soundfile as sf
from tqdm import tqdm
from src.config import load_config,project_path
from src.audio import load_audio
from src.utils import ensure_dir
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--threshold',type=float); p.add_argument('--batch-clips',type=int,default=50); p.add_argument('--limit-clips',type=int); a=p.parse_args(); cfg=load_config(a.config)
try:
    import birdnet
except Exception as e:
    raise SystemExit('Run this script in the qatarbird-birdnet environment. Import failed: '+repr(e))
clips=pd.read_csv(Path(cfg['_project_root'])/'data/qatar_label_clip_manifest.csv')
if a.limit_clips: clips=clips.head(a.limit_clips)
q=pd.read_csv(Path(cfg['_project_root'])/'configs/qatar_species_27.csv')
if cfg['qatar_labeling'].get('use_26_report_species',True): q=q[q.report_training_status!='exclude_in_report'].copy()
threshold=float(a.threshold if a.threshold is not None else cfg['qatar_labeling']['min_confidence']); min_windows=int(cfg['qatar_labeling']['confirmation_min_windows']); starts=list(map(float,cfg['qatar_labeling']['window_starts']))
# Scientific-name matching. Genus-level Anthus is treated as the report's Pipit spp. target.
q['sci_key']=q.scientific_name.astype(str).str.strip().str.casefold(); exact={r.sci_key:int(r.qatar_label_id) for _,r in q.iterrows() if ' ' in r.sci_key}; genus_targets={r.sci_key:int(r.qatar_label_id) for _,r in q.iterrows() if ' ' not in r.sci_key}
model=birdnet.load('acoustic',str(cfg['qatar_labeling']['birdnet_model_version']),str(cfg['qatar_labeling']['birdnet_backend']))
window_rows=[]; confirmed=[]; sr=int(cfg['preprocessing']['sample_rate'])

def to_df(pred,tmpcsv):
    if isinstance(pred,pd.DataFrame): return pred
    if hasattr(pred,'to_pandas'): return pred.to_pandas()
    try:
        pred.to_csv(tmpcsv); return pd.read_csv(tmpcsv)
    except Exception:
        return pd.DataFrame(pred)

def map_species(name):
    sci=str(name).split('_',1)[0].strip().casefold()
    genus=sci.split()[0] if sci else ''
    policy=str(cfg['qatar_labeling'].get('pipit_policy','separate_if_exact')).strip().lower()
    # SDP Table 4-2 merges all Anthus under one local row. The PhD default keeps
    # an exact Anthus similis identification separate; strict report replication can
    # instead map every Anthus detection to the dataset-level Pipit Spp. target.
    if policy == 'combine_anthus_report' and genus == 'anthus' and 'anthus' in genus_targets:
        return genus_targets['anthus']
    if sci in exact: return exact[sci]
    if genus in genus_targets: return genus_targets[genus]
    return None

for lo in tqdm(range(0,len(clips),a.batch_clips),desc='BirdNET labeling batches'):
    sub=clips.iloc[lo:lo+a.batch_clips]
    pack=[]; mapping=[]
    for _,r in sub.iterrows():
        y,_=load_audio(r.audio_path,target_sr=sr,mono=True)
        for vi,s in enumerate(starts):
            st=int(s*sr); en=st+int(3*sr); v=y[st:en]
            if len(v)<3*sr: v=np.pad(v,(0,3*sr-len(v)))
            pack.append(v); mapping.append((r.clip_id,vi))
    if not pack: continue
    packed=np.concatenate(pack).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav=Path(td)/'batch.wav'; sf.write(wav,packed,sr,subtype='PCM_16')
        pred=model.predict(str(wav)); pdf=to_df(pred,Path(td)/'pred.csv')
    # BirdNET emits one row per species prediction/window. Map its sequential 3-s window to our clip/view.
    if len(pdf)==0: pdf=pd.DataFrame(columns=['start_time','species_name','confidence'])
    # Normalize possible time formats by using row window order derived from seconds when possible.
    per={m:{} for m in mapping}
    for _,pr in pdf.iterrows():
        name=pr.get('species_name',pr.get('scientific_name',pr.get('label',''))); qid=map_species(name)
        if qid is None: continue
        conf=float(pr.get('confidence',pr.get('score',0.0)))
        st=pr.get('start_time',0)
        try:
            if isinstance(st,str) and ':' in st:
                parts=list(map(float,st.split(':'))); sec=parts[-1]+60*parts[-2]+(3600*parts[-3] if len(parts)>=3 else 0)
            else: sec=float(st)
        except Exception: sec=0
        wi=int(round(sec/3.0));
        if wi<0 or wi>=len(mapping): continue
        key=mapping[wi]; per.setdefault(key,{})[qid]=max(conf,per.setdefault(key,{}).get(qid,0))
    # Save target scores for all three views and apply 2-of-3 rule.
    clip_scores={str(r.clip_id):{} for _,r in sub.iterrows()}
    for wi,(cid,vi) in enumerate(mapping):
        scores=per.get((cid,vi),{})
        for qid,conf in scores.items():
            window_rows.append({'clip_id':cid,'view_index':vi,'qatar_label_id':qid,'confidence':conf,'threshold':threshold})
            clip_scores[cid].setdefault(qid,[]).append(conf)
    for _,r in sub.iterrows():
        cand=[]
        # Include zeros for missing views before counting threshold crossings.
        for qid in q.qatar_label_id.astype(int):
            vals=[]
            for vi in range(3): vals.append(per.get((r.clip_id,vi),{}).get(int(qid),0.0))
            count=sum(v>=threshold for v in vals)
            if count>=min_windows: cand.append({'qatar_label_id':int(qid),'count_views':int(count),'mean_confidence':float(np.mean(vals)),'max_confidence':float(np.max(vals))})
        cand=sorted(cand,key=lambda x:x['mean_confidence'],reverse=True)
        confirmed.append({'clip_id':r.clip_id,'audio_path':r.audio_path,'study_session':r.study_session,'source_audio_path':r.source_audio_path,'source_recording':r.source_recording,'labels_json':json.dumps(cand),'num_labels':len(cand),'dominant_qatar_label_id':cand[0]['qatar_label_id'] if cand else -1,'dominant_confidence':cand[0]['mean_confidence'] if cand else 0.0,'threshold':threshold})
win=pd.DataFrame(window_rows); conf=pd.DataFrame(confirmed); win.to_csv(project_path(cfg,cfg['paths']['qatar_birdnet_windows']),index=False); conf.to_csv(project_path(cfg,cfg['paths']['qatar_confirmed_labels']),index=False)
print('Confirmed clips:',int((conf.num_labels>0).sum()),'/',len(conf)); print('Multi-label clips:',int((conf.num_labels>1).sum())); print('Threshold:',threshold)
