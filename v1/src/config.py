from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_path'] = str(path.resolve())
    cfg['_project_root'] = str(path.resolve().parent.parent)
    return cfg


def project_path(cfg: dict, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return Path(cfg['_project_root']) / p
