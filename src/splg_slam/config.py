from pathlib import Path
from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)
