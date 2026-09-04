from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, cohen_kappa_score, matthews_corrcoef, log_loss, roc_auc_score,
    average_precision_score, hamming_loss, jaccard_score)


def single_label_metrics(y_true, probs, class_names=None, train_counts=None):
    y_true=np.asarray(y_true,dtype=int); probs=np.asarray(probs); pred=probs.argmax(1); n=probs.shape[1]
    names=class_names or [str(i) for i in range(n)]
    present=np.unique(y_true).astype(int)
    missing=np.setdiff1d(np.arange(n),present)
    # For validation/test sets with source-recording-disjoint splits, some very rare
    # species can be absent because they have fewer than 3 source recordings.  We
    # therefore report both supported-class macro metrics and strict all-class metrics.
    m={
      'accuracy':accuracy_score(y_true,pred),
      'balanced_accuracy':recall_score(y_true,pred,labels=present,average='macro',zero_division=0),
      'macro_f1':f1_score(y_true,pred,labels=present,average='macro',zero_division=0),
      'macro_f1_all_classes':f1_score(y_true,pred,labels=np.arange(n),average='macro',zero_division=0),
      'weighted_f1':f1_score(y_true,pred,labels=present,average='weighted',zero_division=0),
      'macro_precision':precision_score(y_true,pred,labels=present,average='macro',zero_division=0),
      'macro_recall':recall_score(y_true,pred,labels=present,average='macro',zero_division=0),
      'evaluation_class_coverage':float(len(present)/max(1,n)),
      'evaluation_classes_present':int(len(present)),
      'evaluation_classes_total':int(n),
      'missing_class_ids':[int(x) for x in missing.tolist()],
      'missing_class_names':[names[int(x)] for x in missing.tolist()],
      'cohen_kappa':cohen_kappa_score(y_true,pred),'mcc':matthews_corrcoef(y_true,pred),
    }
    try: m['log_loss']=log_loss(y_true,probs,labels=np.arange(n))
    except Exception: pass
    # AUROC/AP are undefined for a class with no positives. Average only classes
    # that are actually represented in this split and record the coverage above.
    onehot=np.eye(n,dtype=np.int8)[y_true]
    aucs=[]; aps=[]
    for i in present:
        yi=onehot[:,i]
        if yi.min()==yi.max():
            continue
        try: aucs.append(float(roc_auc_score(yi,probs[:,i])))
        except Exception: pass
        try: aps.append(float(average_precision_score(yi,probs[:,i])))
        except Exception: pass
    if aucs: m['macro_auroc_ovr']=float(np.mean(aucs))
    if aps: m['macro_average_precision']=float(np.mean(aps))
    k=min(3,n); top=np.argsort(-probs,axis=1)[:,:k]; m[f'top{k}_accuracy']=float(np.mean([yt in row for yt,row in zip(y_true,top)]))
    if train_counts is not None:
        c=np.asarray(train_counts,float); cutoff=np.quantile(c[c>0],0.25) if np.any(c>0) else 0; rare=np.where(c<=cutoff)[0]
        rare_present=np.intersect1d(rare,present)
        mask=np.isin(y_true,rare_present)
        m['rare_class_recall']=recall_score(y_true[mask],pred[mask],labels=rare_present,average='macro',zero_division=0) if mask.any() and len(rare_present) else np.nan
        m['rare_classes_present']=int(len(rare_present)); m['rare_classes_total']=int(len(rare))
    rep=classification_report(y_true,pred,labels=np.arange(n),target_names=names,output_dict=True,zero_division=0)
    per=pd.DataFrame([{**{'class_id':i,'class_name':names[i],'present_in_split':bool(i in set(present.tolist()))},**rep.get(names[i],{})} for i in range(n)])
    cm=confusion_matrix(y_true,pred,labels=np.arange(n))
    return m,per,cm,pred


def tune_thresholds(y_true, probs, grid):
    y=np.asarray(y_true); p=np.asarray(probs); th=np.full(p.shape[1],0.5,float)
    for c in range(p.shape[1]):
        best=(-1,0.5)
        for t in grid:
            f=f1_score(y[:,c],p[:,c]>=t,zero_division=0)
            if f>best[0]: best=(f,t)
        th[c]=best[1]
    return th


def multilabel_metrics(y_true, probs, thresholds):
    y=np.asarray(y_true).astype(int); p=np.asarray(probs); pred=(p>=np.asarray(thresholds)[None,:]).astype(int)
    m={
      'micro_f1':f1_score(y,pred,average='micro',zero_division=0),'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
      'weighted_f1':f1_score(y,pred,average='weighted',zero_division=0),'micro_precision':precision_score(y,pred,average='micro',zero_division=0),
      'micro_recall':recall_score(y,pred,average='micro',zero_division=0),'macro_precision':precision_score(y,pred,average='macro',zero_division=0),
      'macro_recall':recall_score(y,pred,average='macro',zero_division=0),'exact_match':accuracy_score(y,pred),
      'hamming_loss':hamming_loss(y,pred),'sample_jaccard':jaccard_score(y,pred,average='samples',zero_division=0),
      'cardinality_mae':float(np.mean(np.abs(y.sum(1)-pred.sum(1))))
    }
    try: m['macro_average_precision']=average_precision_score(y,p,average='macro'); m['micro_average_precision']=average_precision_score(y,p,average='micro')
    except Exception: pass
    rows=[]
    for c in range(y.shape[1]):
        rows.append({'class_id':c,'precision':precision_score(y[:,c],pred[:,c],zero_division=0),'recall':recall_score(y[:,c],pred[:,c],zero_division=0),'f1':f1_score(y[:,c],pred[:,c],zero_division=0),'support':int(y[:,c].sum()),'threshold':float(thresholds[c])})
    return m,pd.DataFrame(rows),pred


def bootstrap_ci(metric_fn, y, pred_or_probs, n=500, seed=42):
    rng=np.random.default_rng(seed); vals=[]; y=np.asarray(y); z=np.asarray(pred_or_probs)
    for _ in range(n):
        idx=rng.integers(0,len(y),len(y))
        try: vals.append(float(metric_fn(y[idx],z[idx])))
        except Exception: pass
    if not vals: return [np.nan,np.nan]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def save_confusion(cm, names, csv_path, png_path, title='Confusion matrix'):
    pd.DataFrame(cm,index=names,columns=names).to_csv(csv_path)
    fig=plt.figure(figsize=(max(8,len(names)*.35),max(7,len(names)*.35)))
    ax=fig.add_subplot(111); im=ax.imshow(cm,aspect='auto'); ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    if len(names)<=35:
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=90,fontsize=6); ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=6)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(png_path,dpi=160); plt.close(fig)
