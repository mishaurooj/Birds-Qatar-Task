from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class UAPSModel(nn.Module):
    def __init__(self, ap1_model, num_classes, cfg, variant='B5_full_uaps'):
        super().__init__(); self.encoder=ap1_model; self.dim=ap1_model.dim; self.num_classes=num_classes; self.variant=variant
        self.context_gate=nn.Sequential(nn.Linear(self.dim*3,128),nn.GELU(),nn.Linear(128,3))
        self.sigmoid_head=nn.Linear(self.dim,num_classes)
        self.queries=int(cfg['ap2']['queries'])
        layer=nn.TransformerDecoderLayer(d_model=self.dim,nhead=8,dim_feedforward=self.dim*2,batch_first=True)
        self.decoder=nn.TransformerDecoder(layer,num_layers=int(cfg['ap2']['decoder_layers']))
        self.query_embed=nn.Parameter(torch.randn(self.queries,self.dim)*.02)
        self.class_head=nn.Linear(self.dim,num_classes+1)
        self.interval_head=nn.Sequential(nn.Linear(self.dim,128),nn.GELU(),nn.Linear(128,2),nn.Sigmoid())

    def _enc5(self,w,c): return self.encoder.encode(w,c)[0]
    def forward(self,wave5,wave10,condition):
        # 10-s context is represented by two shared 5-s encoder passes.
        n=wave10.shape[1]//2; a=wave10[:,:n]; b=wave10[:,n:]
        z5=self._enc5(wave5,condition); za=self._enc5(a,condition); zb=self._enc5(b,condition)
        if self.variant=='B0_5s_sigmoid':
            return {'multi_logits':self.sigmoid_head(z5),'context_weights':None}
        z10=(za+zb)/2
        if self.variant=='B1_10s_sigmoid':
            return {'multi_logits':self.sigmoid_head(z10),'context_weights':None}
        weights=F.softmax(self.context_gate(torch.cat([z5,za,zb],-1)),dim=-1)
        fused=weights[:,0:1]*z5+weights[:,1:2]*za+weights[:,2:3]*zb
        if self.variant=='B2_adaptive_multiscale':
            return {'multi_logits':self.sigmoid_head(fused),'context_weights':weights}
        memory=torch.stack([z5,za,zb],dim=1)
        q=self.query_embed.unsqueeze(0).expand(wave5.shape[0],-1,-1)
        h=self.decoder(q,memory)
        return {'query_class_logits':self.class_head(h),'query_intervals':self.interval_head(h),'context_weights':weights,'multi_logits':self.sigmoid_head(fused)}


def hungarian_set_loss(outputs, intervals_batch, num_classes, device):
    logits=outputs['query_class_logits']; pred_int=outputs['query_intervals']; B,Q,_=logits.shape
    total_cls=torch.tensor(0.,device=device); total_int=torch.tensor(0.,device=device)
    for b in range(B):
        tg=intervals_batch[b]
        target_cls=torch.full((Q,),num_classes,dtype=torch.long,device=device)
        if tg:
            probs=logits[b].softmax(-1)
            cost=[]
            for qi in range(Q):
                row=[]
                for cls,s,e in tg:
                    cls_cost=-torch.log(probs[qi,int(cls)]+1e-8)
                    t=torch.tensor([s/10.0,e/10.0],device=device,dtype=pred_int.dtype)
                    int_cost=torch.abs(pred_int[b,qi]-t).sum()
                    row.append((cls_cost+int_cost).detach().cpu().item())
                cost.append(row)
            rr,cc=linear_sum_assignment(cost)
            for qi,ti in zip(rr,cc):
                cls,s,e=tg[ti]; target_cls[qi]=int(cls)
                t=torch.tensor([s/10.0,e/10.0],device=device,dtype=pred_int.dtype)
                total_int=total_int+F.l1_loss(pred_int[b,qi],t,reduction='sum')
        total_cls=total_cls+F.cross_entropy(logits[b],target_cls)
    return total_cls/B, total_int/max(B,1)


def query_probs_to_multilabel(outputs,num_classes):
    if 'query_class_logits' not in outputs:
        return torch.sigmoid(outputs['multi_logits'])
    p=outputs['query_class_logits'].softmax(-1)[...,:num_classes]
    return p.max(dim=1).values
