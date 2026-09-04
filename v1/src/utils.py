from __future__ import annotations
import json, os, random, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def now_tag():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def device_info():
    d = {
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    if torch.cuda.is_available():
        d['gpu_name'] = torch.cuda.get_device_name(0)
        d['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return d


def trainable_parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'trainable_fraction': trainable/max(total,1)}
