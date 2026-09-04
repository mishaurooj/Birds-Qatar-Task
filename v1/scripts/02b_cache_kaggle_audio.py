from __future__ import annotations
import argparse, shutil, subprocess, sys, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, soundfile as sf
from tqdm import tqdm
from src.config import load_config, project_path
from src.audio import report_full_preprocess
from src.utils import ensure_dir, save_json


def safe_name(recording_id: str) -> str:
    parts=str(recording_id).replace('\\','/').split('/'); clean=[]
    for x in parts:
        x=''.join('_' if c in '<>:"|?*' else c for c in x).strip(' .'); clean.append(x or 'unnamed')
    return '/'.join(clean)


def verify_wav(path: Path, target_sr: int, min_duration: float=0.1):
    try:
        inf=sf.info(str(path)); ok=int(inf.samplerate)==target_sr and int(inf.channels)==1 and float(inf.duration)>=min_duration
        return ok,{'samplerate':int(inf.samplerate),'channels':int(inf.channels),'duration_sec':float(inf.duration)}
    except Exception as e: return False,{'error':repr(e)}


def transcode_one(row, cache_root: Path, cfg, overwrite: bool, ffmpeg: str):
    target_sr=int(cfg['preprocessing']['sample_rate']); clip_s=float(cfg['preprocessing']['clip_seconds'])
    apply_sp=bool(cfg['kaggle'].get('apply_report_sp_during_cache',True))
    src=Path(row.audio_path); dst=cache_root/Path(safe_name(row.recording_id)+'.wav'); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and not overwrite:
        ok,info=verify_wav(dst,target_sr,clip_s if cfg['kaggle'].get('drop_sources_shorter_than_clip_after_sp',True) else .1)
        if ok: return {'recording_id':row.recording_id,'original_audio_path':str(src),'cached_audio_path':str(dst.resolve()),'status':'existing_ok','preprocessed':apply_sp,**info}
    tmp=dst.with_suffix('.decode.tmp.wav')
    try:
        if tmp.exists(): tmp.unlink()
        cmd=[ffmpeg,'-hide_banner','-nostdin','-y','-loglevel','warning','-err_detect','ignore_err','-fflags','+discardcorrupt','-i',str(src),'-vn','-map_metadata','-1','-ac','1','-ar',str(target_sr),'-c:a','pcm_f32le',str(tmp)]
        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        if proc.returncode!=0 or not tmp.exists():
            return {'recording_id':row.recording_id,'original_audio_path':str(src),'cached_audio_path':str(dst),'status':'failed_decode','error':proc.stderr.decode('utf-8',errors='replace')[-3000:]}
        y,sr=sf.read(str(tmp),dtype='float32',always_2d=True); y=y.mean(axis=1)
        spstats={}
        if apply_sp:
            y,spstats=report_full_preprocess(y,int(sr),cfg)
        if len(y) < int(round(target_sr*clip_s)) and bool(cfg['kaggle'].get('drop_sources_shorter_than_clip_after_sp',True)):
            return {'recording_id':row.recording_id,'original_audio_path':str(src),'cached_audio_path':str(dst),'status':'too_short_after_preprocess','duration_sec':len(y)/target_sr,**spstats}
        sf.write(str(dst),y,target_sr,subtype='PCM_16')
        ok,info=verify_wav(dst,target_sr,.1)
        if not ok: return {'recording_id':row.recording_id,'original_audio_path':str(src),'cached_audio_path':str(dst),'status':'failed_verify','error':json.dumps(info)}
        return {'recording_id':row.recording_id,'original_audio_path':str(src),'cached_audio_path':str(dst.resolve()),'status':'converted','preprocessed':apply_sp,**info,**spstats,'ffmpeg_warning':proc.stderr.decode('utf-8',errors='replace')[-1500:]}
    except Exception as e:
        return {'recording_id':row.recording_id,'original_audio_path':str(src),'cached_audio_path':str(dst),'status':'failed_exception','error':repr(e)}
    finally:
        try:
            if tmp.exists(): tmp.unlink()
        except Exception: pass


def resplit(rec,cfg):
    seed=int(cfg['project']['seed']); test=float(cfg['kaggle']['test_fraction']); val=float(cfg['kaggle']['val_fraction']); rec=rec.copy(); rec['split']='train'
    for lid,g in rec.groupby('label_id'):
        idx=g.index.to_numpy().copy(); np.random.default_rng(seed+int(lid)).shuffle(idx); n=len(idx)
        if n<3: raise RuntimeError(f'Post-cache label {lid} has only {n} recordings.')
        nt=max(1,int(round(n*test))); nv=max(1,int(round(n*val)))
        while nt+nv>=n:
            if nt>1: nt-=1
            elif nv>1: nv-=1
            else: break
        rec.loc[idx[:nt],'split']='test'; rec.loc[idx[nt:nt+nv],'split']='val'
    return rec


def rebuild_clips(rec,cfg):
    clip=float(cfg['preprocessing']['clip_seconds']); maxclips=int(cfg['kaggle']['max_clips_per_recording']); rows=[]
    for _,r in rec.iterrows():
        dur=float(r.duration_total_sec); n=int(dur//clip)
        if n<1: continue
        starts=np.linspace(0,max(0,dur-clip),maxclips) if maxclips>0 and n>maxclips else np.arange(n)*clip
        for j,s in enumerate(starts):
            d=r.to_dict(); d.update({'clip_id':f'{r.recording_id}__{j:04d}','start_sec':float(s),'duration_sec':clip,'preprocessed':True}); rows.append(d)
    return pd.DataFrame(rows)

p=argparse.ArgumentParser(description='Create training-ready 32-kHz PCM cache. Optionally applies the student-report SP pipeline before 5-s segmentation.')
p.add_argument('--config',default='configs/default.yaml'); p.add_argument('--workers',type=int,default=2); p.add_argument('--overwrite',action='store_true')
a=p.parse_args(); cfg=load_config(a.config); ffmpeg=shutil.which('ffmpeg')
if not ffmpeg: raise SystemExit('ffmpeg not found. Install it in the conda environment.')
recordings_path=project_path(cfg,cfg['paths'].get('kaggle_recordings','data/kaggle_public_recordings.csv'))
clips_path=project_path(cfg,cfg['paths']['kaggle_manifest']); labels_path=project_path(cfg,cfg['paths']['kaggle_labels'])
dropped_path=project_path(cfg,cfg['paths'].get('kaggle_dropped_classes','data/kaggle_dropped_low_source_classes.csv'))
if not recordings_path.exists() or not clips_path.exists(): raise SystemExit('Run scripts\\02_build_kaggle_manifest.py first.')
rec=pd.read_csv(recordings_path); labels=pd.read_csv(labels_path); cache_root=project_path(cfg,cfg['paths']['kaggle_wav_cache']); ensure_dir(cache_root)
rows=[]
with ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
    futs=[ex.submit(transcode_one,r,cache_root,cfg,a.overwrite,ffmpeg) for r in rec.itertuples(index=False)]
    for fut in tqdm(as_completed(futs),total=len(futs),desc='Caching + report SP preprocessing'): rows.append(fut.result())
status=pd.DataFrame(rows); status_path=clips_path.with_name('kaggle_public_cache_status.csv'); status.to_csv(status_path,index=False)
ok=status[status.status.isin(['existing_ok','converted'])].copy(); fail=status[~status.status.isin(['existing_ok','converted'])].copy(); fail.to_csv(clips_path.with_name('kaggle_public_cache_failures.csv'),index=False)
rec=rec.merge(ok[['recording_id','cached_audio_path','duration_sec','preprocessed']],on='recording_id',how='inner'); rec['audio_path_original']=rec['audio_path']; rec['audio_path']=rec.cached_audio_path; rec['duration_total_sec']=rec.duration_sec
# Recheck class support after corrupt/short recordings were removed.
min_rec=max(3,int(cfg['kaggle'].get('min_source_recordings',3))); counts=rec.groupby('label_id').recording_id.nunique(); bad_ids=counts[counts<min_rec].index.tolist()
if bad_ids:
    add=labels[labels.label_id.isin(bad_ids)].copy(); add['drop_reason']=f'post_cache_fewer_than_{min_rec}_usable_recordings'
    old=pd.read_csv(dropped_path) if dropped_path.exists() else pd.DataFrame(); pd.concat([old,add],ignore_index=True).drop_duplicates(subset=['common_name','drop_reason']).to_csv(dropped_path,index=False)
    rec=rec[~rec.label_id.isin(bad_ids)].copy(); labels=labels[~labels.label_id.isin(bad_ids)].copy()
# Reindex after any post-cache drops.
old_to_new={int(old):new for new,old in enumerate(sorted(labels.label_id.astype(int).tolist()))}; labels['label_id']=labels.label_id.astype(int).map(old_to_new); rec['label_id']=rec.label_id.astype(int).map(old_to_new)
labels=labels.sort_values('label_id').reset_index(drop=True); labels['source_recordings']=labels.label_id.map(rec.groupby('label_id').recording_id.nunique()).fillna(0).astype(int); labels.to_csv(labels_path,index=False)
rec=resplit(rec.reset_index(drop=True),cfg); rec.to_csv(recordings_path,index=False); clips=rebuild_clips(rec,cfg); clips.to_csv(clips_path,index=False)
summary={'classes_after_cache':int(len(labels)),'usable_source_recordings':int(len(rec)),'failed_or_too_short_sources':int(len(fail)),'clips_after_report_sp':int(len(clips)),'split_recordings':rec.split.value_counts().to_dict(),'split_clips':clips.split.value_counts().to_dict(),'cache_root':str(cache_root),'report_sp_applied':bool(cfg['kaggle'].get('apply_report_sp_during_cache',True)),'all_paths_wav':bool(len(clips) and clips.audio_path.astype(str).str.lower().str.endswith('.wav').all())}
save_json(summary,clips_path.with_name('kaggle_public_cache_summary.json')); print(json.dumps(summary,indent=2)); print('Status:',status_path)
if len(fail): print('Dropped/failed source audio:',clips_path.with_name('kaggle_public_cache_failures.csv'))
