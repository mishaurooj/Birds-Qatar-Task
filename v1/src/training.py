from __future__ import annotations
from pathlib import Path
import math, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from .metrics import single_label_metrics, save_confusion, multilabel_metrics, tune_thresholds
from .utils import ensure_dir, save_json, device_info, trainable_parameter_count


def class_weights_from_manifest(csv_path, split='train'):
    all_df=pd.read_csv(csv_path)
    if len(all_df)==0: return np.zeros(0,dtype=float), np.zeros(0,dtype=float)
    n=int(all_df.label_id.max())+1
    df=all_df[all_df.split.astype(str)==split] if 'split' in all_df else all_df
    counts=df.groupby('label_id').size()
    arr=np.array([counts.get(i,0) for i in range(n)],float)
    w=np.zeros_like(arr); nz=arr>0
    if np.any(nz): w[nz]=arr[nz].sum()/(len(arr[nz])*arr[nz])
    return arr,w


def optimizer_for(model,cfg):
    lr=float(cfg['training']['lr']); wd=float(cfg['training']['weight_decay']); mult=float(cfg['training'].get('backbone_lr_multiplier',0.1))
    bb=[]; other=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (bb if 'backbone' in n or '.encoder.backbone' in n else other).append(p)
    groups=[]
    if other: groups.append({'params':other,'lr':lr})
    if bb: groups.append({'params':bb,'lr':lr*mult})
    if not groups: raise RuntimeError('No trainable parameters found.')
    return torch.optim.AdamW(groups,weight_decay=wd)


def cosine_warmup(optimizer,total_steps,warmup_steps):
    def f(step):
        if step<warmup_steps: return max(1e-3,(step+1)/max(1,warmup_steps))
        progress=(step-warmup_steps)/max(1,total_steps-warmup_steps); return .5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)


def save_history(history,out_dir):
    df=pd.DataFrame(history); df.to_csv(Path(out_dir)/'history.csv',index=False)
    if len(df):
        fig=plt.figure(figsize=(7,4)); ax=fig.add_subplot(111)
        for c in [x for x in df.columns if x not in ['epoch']]: ax.plot(df.epoch,df[c],label=c)
        ax.legend(fontsize=7); ax.set_xlabel('epoch'); fig.tight_layout(); fig.savefig(Path(out_dir)/'history.png',dpi=150); plt.close(fig)


def evaluate_single(model,loader,device,class_names,train_counts,out_dir=None):
    model.eval(); ys=[]; ps=[]; zs=[]
    with torch.no_grad():
        for b in tqdm(loader,desc='evaluate',leave=False):
            w=b['waveform'].to(device,non_blocking=True); c=b['condition'].to(device,non_blocking=True); y=b['label'].to(device,non_blocking=True)
            o=model(w,c); p=torch.softmax(o['logits'],-1)
            ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy()); zs.append(o['embedding'].cpu().numpy())
    if not ys: raise RuntimeError('Evaluation loader is empty. Check dataset split/manifests.')
    y=np.concatenate(ys); p=np.concatenate(ps); z=np.concatenate(zs)
    m,per,cm,pred=single_label_metrics(y,p,class_names,train_counts)
    if out_dir:
        out=ensure_dir(out_dir); save_json(m,out/'metrics.json'); per.to_csv(out/'per_class.csv',index=False)
        save_confusion(cm,class_names,out/'confusion.csv',out/'confusion.png')
        pd.DataFrame({'y_true':y,'y_pred':pred}).to_csv(out/'predictions.csv',index=False); np.save(out/'probabilities.npy',p); np.save(out/'embeddings.npy',z)
    return m,p,z,y


def _make_scaler(device, enabled):
    enabled=bool(enabled and device.type=='cuda')
    # PyTorch 2.4+ API. Fall back for older installations.
    try: return torch.amp.GradScaler('cuda',enabled=enabled)
    except Exception: return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device, enabled):
    enabled=bool(enabled and device.type=='cuda')
    try: return torch.amp.autocast(device_type='cuda',dtype=torch.float16,enabled=enabled)
    except Exception: return torch.cuda.amp.autocast(enabled=enabled)


def _checkpoint(model,opt,sched,scaler,epoch,best,stale,history,val_metrics,class_names,cfg):
    backbone=getattr(model,'backbone',getattr(getattr(model,'encoder',None),'backbone',model))
    return {
        'model_state_dict':model.state_dict(), 'backbone_state_dict':backbone.state_dict(),
        'optimizer_state_dict':opt.state_dict(), 'scheduler_state_dict':sched.state_dict(),
        'scaler_state_dict':scaler.state_dict(), 'epoch':int(epoch), 'best_val_macro_f1':float(best),
        'stale_epochs':int(stale), 'history':history, 'val_metrics':val_metrics,
        'class_names':class_names, 'cfg':cfg,
    }


def train_single(model,train_loader,val_loader,test_loader,cfg,class_names,train_counts,out_dir,loss_weights=None,aux_loss_fn=None,resume_path=None):
    out=ensure_dir(out_dir); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    opt=optimizer_for(model,cfg)
    accum=max(1,int(cfg['training'].get('gradient_accumulation_steps',1)))
    optimizer_steps_per_epoch=max(1,math.ceil(len(train_loader)/accum))
    total=int(cfg['training']['epochs'])*optimizer_steps_per_epoch
    warm=int(cfg['training']['warmup_epochs'])*optimizer_steps_per_epoch
    sched=cosine_warmup(opt,total,warm)
    scaler=_make_scaler(device,bool(cfg['training']['amp']))
    cw=torch.tensor(loss_weights,dtype=torch.float32,device=device) if loss_weights is not None else None
    patience=int(cfg['training']['patience']); best=-1.0; stale=0; history=[]; best_path=out/'best.pt'; start_epoch=0
    if resume_path:
        rp=Path(resume_path)
        if not rp.exists(): raise FileNotFoundError(f'Resume checkpoint not found: {rp}')
        ck=torch.load(rp,map_location=device)
        model.load_state_dict(ck['model_state_dict'],strict=False)
        if 'optimizer_state_dict' in ck: opt.load_state_dict(ck['optimizer_state_dict'])
        if 'scheduler_state_dict' in ck: sched.load_state_dict(ck['scheduler_state_dict'])
        if 'scaler_state_dict' in ck:
            try: scaler.load_state_dict(ck['scaler_state_dict'])
            except Exception: pass
        start_epoch=int(ck.get('epoch',0)); best=float(ck.get('best_val_macro_f1',ck.get('val_metrics',{}).get('macro_f1',-1)))
        stale=int(ck.get('stale_epochs',0)); history=list(ck.get('history',[]))
        print(f'Resuming from {rp}: next epoch={start_epoch+1}, best_val_macro_f1={best:.4f}')
    last_vm={}
    try:
        for epoch in range(start_epoch,int(cfg['training']['epochs'])):
            model.train(); losses=[]; opt.zero_grad(set_to_none=True)
            bar=tqdm(train_loader,desc=f'epoch {epoch+1}/{cfg["training"]["epochs"]}',dynamic_ncols=True)
            for step,b in enumerate(bar):
                w=b['waveform'].to(device,non_blocking=True); c=b['condition'].to(device,non_blocking=True); y=b['label'].to(device,non_blocking=True)
                with _autocast(device,scaler.is_enabled()):
                    o=model(w,c); loss=F.cross_entropy(o['logits'],y,weight=cw)
                    if aux_loss_fn: loss=loss+aux_loss_fn(model,o,y,c,cfg)
                    loss_for_backward=loss/accum
                scaler.scale(loss_for_backward).backward()
                do_step=((step+1)%accum==0) or (step+1==len(train_loader))
                if do_step:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']))
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                loss_value=float(loss.detach().cpu()); losses.append(loss_value)
                bar.set_postfix(loss=f'{np.mean(losses[-20:]):.4f}',lr=f'{opt.param_groups[0]["lr"]:.2e}')
            vm,_,_,_=evaluate_single(model,val_loader,device,class_names,train_counts); last_vm=vm
            history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_macro_f1':vm['macro_f1'],'val_balanced_accuracy':vm['balanced_accuracy']})
            improved=vm['macro_f1']>best
            if improved: best=float(vm['macro_f1']); stale=0
            else: stale+=1
            ck=_checkpoint(model,opt,sched,scaler,epoch+1,best,stale,history,vm,class_names,cfg)
            torch.save(ck,out/'last.pt')
            if improved: torch.save(ck,best_path)
            save_history(history,out)
            print(f'Epoch {epoch+1}: loss={np.mean(losses):.4f} val_macro_f1={vm["macro_f1"]:.4f} val_bal_acc={vm["balanced_accuracy"]:.4f}')
            if stale>=patience:
                print(f'Early stopping after {stale} stale epochs.')
                break
    except KeyboardInterrupt:
        ck=_checkpoint(model,opt,sched,scaler,max(start_epoch,len(history)),best,stale,history,last_vm,class_names,cfg)
        interrupted=out/'interrupted.pt'; torch.save(ck,interrupted); save_history(history,out)
        print(f'\nTraining interrupted by user. Recovery checkpoint saved to: {interrupted}')
        print(f'Resume with: --resume "{interrupted}"')
        raise
    if not best_path.exists():
        # This can happen only if a resumed checkpoint started with best state but no local best.pt.
        torch.save(_checkpoint(model,opt,sched,scaler,len(history),best,stale,history,last_vm,class_names,cfg),best_path)
    ck=torch.load(best_path,map_location=device); model.load_state_dict(ck['model_state_dict'],strict=False)
    evaluate_single(model,val_loader,device,class_names,train_counts,out/'val')
    tm,p,z,y=evaluate_single(model,test_loader,device,class_names,train_counts,out/'test')
    save_json({'parameters':trainable_parameter_count(model),'device':device_info(),'best_val_macro_f1':best,'gradient_accumulation_steps':accum},out/'run_info.json')
    return model,tm,p,z,y


def evaluate_multilabel(model,loader,device,thresholds=None,grid=None):
    model.eval(); ys=[]; ps=[]
    from .models_ap2 import query_probs_to_multilabel
    with torch.no_grad():
        for b in loader:
            o=model(b['wave5'].to(device),b['wave10'].to(device),b['condition'].to(device)); p=query_probs_to_multilabel(o,b['target'].shape[1])
            ys.append(b['target'].numpy()); ps.append(p.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    if thresholds is None: thresholds=tune_thresholds(y,p,grid)
    m,per,pred=multilabel_metrics(y,p,thresholds)
    return m,per,pred,p,y,thresholds
