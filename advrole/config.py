from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def deep_merge(base: Dict, override: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _coerce(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v

def load_config(config_path: Optional[str] = None,
                local_path: Optional[str] = None,
                overrides: Optional[List[str]] = None) -> Dict:
    default_path = os.path.join(REPO_ROOT, "configs", "default.yaml")
    cfg: Dict = {}
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    if config_path and os.path.abspath(config_path) != os.path.abspath(default_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})

    local_path = local_path or os.path.join(REPO_ROOT, "configs", "local.yaml")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)

    return cfg

def resolve_path(p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
