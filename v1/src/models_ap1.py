from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lambd): ctx.lambd=lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.lambd*g, None


class ResidualAdapter(nn.Module):
    def __init__(self, dim, bottleneck=96):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,bottleneck),nn.GELU(),nn.Dropout(.1),nn.Linear(bottleneck,dim))
    def forward(self,x): return self.net(x)


class AP1Model(nn.Module):
    def __init__(self, backbone, num_classes, cfg, variant='A5_full_acpsar'):
        super().__init__(); self.backbone=backbone; self.dim=backbone.embedding_dim; self.num_classes=num_classes; self.variant=variant
        a=cfg['ap1']; self.num_experts=int(a['num_experts']); self.topk=int(a['top_k_experts'])
        self.single=ResidualAdapter(self.dim,int(a['adapter_bottleneck']))
        self.experts=nn.ModuleList([ResidualAdapter(self.dim,int(a['adapter_bottleneck'])) for _ in range(self.num_experts)])
        self.router=nn.Sequential(nn.Linear(7,32),nn.GELU(),nn.Linear(32,self.num_experts))
        self.classifier=nn.Linear(self.dim,num_classes)
        self.condition_regressor=nn.Sequential(nn.Linear(self.dim,128),nn.GELU(),nn.Linear(128,7))
        self.prototypes=nn.Parameter(torch.randn(num_classes,self.dim)*0.02)
        self.use_router=variant not in ['A0_frozen_linear','A1_single_adapter']
        self.use_condition=variant in ['A3_router_condition','A5_full_acpsar']
        self.use_proto=variant in ['A4_router_prototype','A5_full_acpsar']
        self.use_sparse=self.use_router
        if variant=='A0_frozen_linear':
            for p in self.backbone.parameters(): p.requires_grad=False

    def encode(self,waveform,condition):
        z0=self.backbone(waveform)
        if self.variant=='A0_frozen_linear': return z0, None
        if self.variant=='A1_single_adapter': return z0+self.single(z0), None
        w=F.softmax(self.router(condition),dim=-1)
        if self.use_sparse and self.topk<self.num_experts:
            vals,idx=torch.topk(w,self.topk,dim=-1); mask=torch.zeros_like(w).scatter(1,idx,1); w=w*mask; w=w/(w.sum(1,keepdim=True)+1e-8)
        adds=torch.stack([e(z0) for e in self.experts],dim=1)
        z=z0+(adds*w.unsqueeze(-1)).sum(1)
        return z,w

    def forward(self,waveform,condition):
        z,w=self.encode(waveform,condition); logits=self.classifier(z)
        return {'logits':logits,'embedding':z,'router_weights':w}

    def auxiliary_losses(self,out,labels,condition,cfg):
        losses={}; z=out['embedding']
        if self.use_condition:
            pred=self.condition_regressor(GradReverse.apply(z,1.0)); losses['condition']=F.mse_loss(pred,condition)
        if self.use_proto:
            zn=F.normalize(z,dim=-1); pn=F.normalize(self.prototypes[labels],dim=-1); losses['prototype']=(1-(zn*pn).sum(-1)).mean()
        if out['router_weights'] is not None:
            mean=out['router_weights'].mean(0); uniform=torch.full_like(mean,1/len(mean)); losses['router_balance']=F.kl_div((mean+1e-8).log(),uniform,reduction='batchmean')
        return losses
