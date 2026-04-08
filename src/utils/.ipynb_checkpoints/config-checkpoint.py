from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


Config = dict[str, Any]


def _deep_merge(base: Config, override: Mapping[str, Any]) -> Config:
    """Recursively merge override into base without mutating the input."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def load_config(path: str | Path) -> Config:
    """Load a YAML config and optionally merge it with its base_config.

    The child config can contain:
      base_config: configs/default.yaml

    Relative base paths are resolved from the child config directory first.
    """
    path = Path(path).expanduser().resolve()
    cfg = load_yaml(path)
    base_ref = cfg.pop("base_config", None)
    if not base_ref:
      return cfg

    base_path = Path(base_ref).expanduser()
    if not base_path.is_absolute():
        candidate = (path.parent / base_path).resolve()
        if candidate.exists():
            base_path = candidate
        else:
            base_path = (path.parent.parent / base_path).resolve()

    base_cfg = load_config(base_path)
    return _deep_merge(base_cfg, cfg)


def get_by_path(cfg: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def require_by_path(cfg: Mapping[str, Any], dotted_key: str) -> Any:
    value = get_by_path(cfg, dotted_key, default=None)
    if value is None:
        raise KeyError(f"Missing required config key: {dotted_key}")
    return value


def save_yaml(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, allow_unicode=False)
