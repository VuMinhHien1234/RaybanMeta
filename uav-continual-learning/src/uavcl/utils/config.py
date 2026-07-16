"""Đọc/ghi config yaml — mọi thí nghiệm chạy từ 1 file yaml + (tùy chọn) override."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a yaml mapping")
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """--set a.b=c  ->  cfg['a']['b'] = c (tự đoán kiểu int/float/bool)."""
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"Override '{pair}' must look like key.sub=value")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw.strip())
    return cfg


def _coerce(v: str):
    # Đoán kiểu bằng yaml: '0.01'->float, 'true'->bool, '[[4,1],[4,8]]'->list, còn lại giữ str.
    try:
        out = yaml.safe_load(v)
    except yaml.YAMLError:
        return v
    if out is None and v.strip() not in ("null", "~", ""):
        return v
    # yaml KHÔNG nhận '1e-4' là số (đòi '1.0e-4') -> thử float cho chuỗi trông-như-số
    if isinstance(out, str) and any(ch.isdigit() for ch in out):
        try:
            return float(out)
        except ValueError:
            return out
    return out
