from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torchaudio


class BirdMAEBackbone(nn.Module):
    def __init__(self, model_name='DBD-research-group/Bird-MAE-Base', trust_remote_code=True, freeze=False, unfreeze_last_blocks=0):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor
        self.model_name = model_name
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.embedding_dim = int(getattr(self.model.config, 'embed_dim', 768))
        if freeze:
            for p in self.model.parameters(): p.requires_grad = False
        if unfreeze_last_blocks and hasattr(self.model, 'blocks'):
            for p in self.model.parameters(): p.requires_grad = False
            for block in self.model.blocks[-int(unfreeze_last_blocks):]:
                for p in block.parameters(): p.requires_grad = True
            for name in ['norm','fc_norm']:
                m = getattr(self.model, name, None)
                if m is not None:
                    for p in m.parameters(): p.requires_grad = True

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # BirdMAE's public HF feature extractor expects 32 kHz, 5-s waveforms and
        # returns [B,1,512,128] fbank tensors. Keep feature extraction on CPU;
        # gradients are required only for the model parameters, not the DSP transform.
        x_cpu = waveforms.detach().float().cpu()
        feats = self.feature_extractor(x_cpu, return_tensors='pt')
        if isinstance(feats, dict):
            feats = feats.get('input_values', next(iter(feats.values())))
        device = next(self.model.parameters()).device
        feats = feats.to(device)
        out = self.model(input_values=feats)
        z = out.last_hidden_state
        if z.ndim == 3:
            z = z.mean(dim=1)
        return z


class SmokeBackbone(nn.Module):
    def __init__(self, sample_rate=32000, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=320, n_mels=128)
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(), nn.Linear(64,embedding_dim)
        )
    def forward(self, waveforms):
        x = self.mel(waveforms).clamp_min(1e-8).log().unsqueeze(1)
        return self.net(x)


def build_backbone(cfg: dict):
    b = cfg['backbone']
    typ = b.get('type','birdmae').lower()
    if typ == 'birdmae':
        return BirdMAEBackbone(
            model_name=b.get('hf_model','DBD-research-group/Bird-MAE-Base'),
            trust_remote_code=bool(b.get('trust_remote_code',True)),
            freeze=bool(b.get('freeze_backbone',False)),
            unfreeze_last_blocks=int(b.get('unfreeze_last_blocks',0)),
        )
    if typ == 'smoke_cnn':
        return SmokeBackbone(cfg['preprocessing']['sample_rate'], int(b.get('embedding_dim',128)))
    raise ValueError(f'Unknown backbone type: {typ}')


def load_backbone_from_checkpoint(backbone: nn.Module, checkpoint_path: str, strict=False):
    ck = torch.load(checkpoint_path, map_location='cpu')
    state = ck.get('backbone_state_dict') or ck.get('model_state_dict') or ck.get('state_dict') or ck
    # Accept keys from a wrapper model, e.g. backbone.model.blocks...
    own = backbone.state_dict()
    cleaned = {}
    for k,v in state.items():
        candidates = [k]
        for prefix in ['backbone.','encoder.','model.backbone.']:
            if k.startswith(prefix): candidates.append(k[len(prefix):])
        for kk in candidates:
            if kk in own and own[kk].shape == v.shape:
                cleaned[kk] = v; break
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    if len(cleaned)==0:
        warnings.warn(f'No compatible backbone tensors found in {checkpoint_path}')
    return {'loaded_tensors': len(cleaned), 'missing': list(missing), 'unexpected': list(unexpected)}
