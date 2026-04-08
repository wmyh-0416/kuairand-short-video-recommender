from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def project_root(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["project"]["root_dir"]).expanduser().resolve()


def raw_data_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["raw_data_dir"]).expanduser().resolve()


def processed_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["processed_dir"]).expanduser().resolve()


def artifacts_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["artifacts_dir"]).expanduser().resolve()


def logs_dir(cfg: Mapping[str, Any]) -> Path:
    return Path(cfg["paths"]["logs_dir"]).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs(cfg: Mapping[str, Any]) -> None:
    ensure_dir(processed_dir(cfg))
    ensure_dir(artifacts_dir(cfg))
    ensure_dir(logs_dir(cfg))


def raw_data_path(cfg: Mapping[str, Any], relative_path: str | Path) -> Path:
    return raw_data_dir(cfg) / relative_path


def raw_path(cfg: Mapping[str, Any], relative_path: str | Path) -> Path:
    """Backward-compatible alias for raw_data_path."""
    return raw_data_path(cfg, relative_path)


def processed_path(cfg: Mapping[str, Any], relative_path: str | Path) -> Path:
    return processed_dir(cfg) / relative_path


def artifact_path(cfg: Mapping[str, Any], relative_path: str | Path) -> Path:
    return artifacts_dir(cfg) / relative_path
