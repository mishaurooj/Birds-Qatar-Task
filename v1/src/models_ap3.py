from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGSModel(nn.Module):
    def __init__(self, ap1_model, num_classes, cfg, variant='C5_full_dcgs'):
        super().__init__(); self.encoder=ap1_model; self.dim=ap1_model.dim; self.num_classes=num_classes; self.variant=variant
        k=int(cfg['ap3']['specialists']); self.k=k; self.global_head=nn.Linear(self.dim,num_classes)
        self.router=nn.Sequential(nn.Linear(self.dim+2,128),nn.GELU(),nn.Linear(128,k))
        self.specialists=nn.ModuleList([nn.Sequential(nn.Linear(self.dim,128),nn.GELU(),nn.Linear(128,num_classes)) for _ in range(k)])
        self.correction_scale=float(cfg['ap3']['correction_scale'])
        self.register_buffer('graph',torch.eye(num_classes))
    def set_graph(self,g): self.graph.copy_(torch.as_tensor(g,dtype=self.graph.dtype,device=self.graph.device))
    def forward(self,waveform,condition):
        z=self.encoder.encode(waveform,condition)[0]; gl=self.global_head(z); gp=F.softmax(gl,-1); ent=-(gp*(gp+1e-8).log()).sum(-1,keepdim=True)/np.log(self.num_classes)
        margin=(gp.topk(2,dim=-1).values[:,0]-gp.topk(2,dim=-1).values[:,1]).unsqueeze(-1)
        rw=F.softmax(self.router(torch.cat([z,ent,margin],-1)),dim=-1)
        spec=torch.stack([s(z) for s in self.specialists],dim=1)
        if self.variant=='C0_global': return {'logits':gl,'embedding':z,'router_weights':rw}
        if self.variant=='C1_static_hard':
            idx=rw.argmax(-1); correction=spec[torch.arange(len(z),device=z.device),idx]
        else: correction=(spec*rw.unsqueeze(-1)).sum(1)
        top=gp.argmax(-1); mask=self.graph[top]
        if self.variant in ['C1_static_hard','C2_static_soft','C3_dynamic_soft','C4_dynamic_contrastive','C5_full_dcgs']:
            correction=correction*mask
        scale=self.correction_scale*(ent if self.variant=='C5_full_dcgs' else torch.ones_like(ent))
        return {'logits':gl+scale*correction,'global_logits':gl,'embedding':z,'router_weights':rw}


def build_confusion_graph(y_true, probs, embeddings, num_classes, alpha=.5,beta=.3,gamma=.2):
    y=np.asarray(y_true); p=np.asarray(probs); z=np.asarray(embeddings); pred=p.argmax(1)
    cm=np.zeros((num_classes,num_classes),float)
    for a,b in zip(y,pred):
        if a!=b: cm[a,b]+=1; cm[b,a]+=1
    if cm.max()>0: cm/=cm.max()
    prot=np.zeros((num_classes,z.shape[1]),float)
    for c in range(num_classes):
        zz=z[y==c]; prot[c]=zz.mean(0) if len(zz) else 0
    pn=prot/(np.linalg.norm(prot,axis=1,keepdims=True)+1e-8); sim=np.clip(pn@pn.T,0,1); np.fill_diagonal(sim,0)
    unc=np.zeros_like(cm)
    for i,(yt,pp) in enumerate(zip(y,p)):
        ent=-np.sum(pp*np.log(pp+1e-8))/np.log(num_classes)
        for j in np.argsort(-pp)[:3]:
            if j!=yt: unc[yt,j]+=ent; unc[j,yt]+=ent
    if unc.max()>0: unc/=unc.max()
    g=alpha*cm+beta*sim+gamma*unc; np.fill_diagonal(g,1.0)
    return np.clip(g,0,1)


def graph_contrastive_loss(z,labels,graph,margin=.5):
    if len(z)<2: return z.sum()*0
    zn=F.normalize(z,dim=-1); sim=zn@zn.T; lab=labels
    g=graph[lab][:,lab]; diff=(lab[:,None]!=lab[None,:]).float(); weights=g*diff
    loss=F.relu(sim-margin)*weights
    denom=(weights>0).float().sum().clamp_min(1)
    return loss.sum()/denom
