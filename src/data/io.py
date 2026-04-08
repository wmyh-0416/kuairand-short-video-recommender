from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.schema import (
    ITEM_BASIC_COLUMNS,
    ITEM_STAT_COLUMNS,
    LOG_COLUMNS,
    USER_FEATURE_COLUMNS,
    validate_columns,
)
from src.utils.paths import raw_data_path


def _read_csv(
    path: str | Path,
    nrows: int | None = None,
    dtype_backend: str | None = None,
) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    kwargs: dict[str, Any] = {"nrows": nrows}
    if dtype_backend:
        kwargs["dtype_backend"] = dtype_backend

    try:
        return pd.read_csv(path, **kwargs)
    except TypeError:
        kwargs.pop("dtype_backend", None)
        return pd.read_csv(path, **kwargs)


def read_log_tables(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    read_cfg = cfg.get("preprocess", {}).get("read", {})
    nrows = read_cfg.get("nrows")
    dtype_backend = read_cfg.get("dtype_backend")

    log_paths = list(cfg["data"]["log_files"]["standard_train"])
    if cfg["data"].get("include_random_log", False):
        log_paths.extend(cfg["data"]["log_files"].get("random", []))

    frames: list[pd.DataFrame] = []
    for rel_path in log_paths:
        path = raw_data_path(cfg, rel_path)
        if logger:
            logger.info("Reading log file: %s", path)
        df = _read_csv(path, nrows=nrows, dtype_backend=dtype_backend)
        validate_columns(df.columns, LOG_COLUMNS, table_name=str(path))
        df = df[LOG_COLUMNS].copy()
        df["source_file"] = Path(rel_path).name
        frames.append(df)

    if not frames:
        raise ValueError("No log files configured.")

    out = pd.concat(frames, ignore_index=True)
    if logger:
        logger.info("Loaded interactions: rows=%d, users=%d, items=%d", out.shape[0], out["user_id"].nunique(), out["video_id"].nunique())
    return out


def read_user_features(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    read_cfg = cfg.get("preprocess", {}).get("read", {})
    dtype_backend = read_cfg.get("dtype_backend")
    path = raw_data_path(cfg, cfg["data"]["user_features_file"])
    if logger:
        logger.info("Reading user features: %s", path)
    df = _read_csv(path, dtype_backend=dtype_backend)
    validate_columns(df.columns, USER_FEATURE_COLUMNS, table_name=str(path))
    return df[USER_FEATURE_COLUMNS].copy()


def read_item_features(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    read_cfg = cfg.get("preprocess", {}).get("read", {})
    dtype_backend = read_cfg.get("dtype_backend")

    basic_path = raw_data_path(cfg, cfg["data"]["item_basic_file"])
    stat_path = raw_data_path(cfg, cfg["data"]["item_stat_file"])

    if logger:
        logger.info("Reading item basic features: %s", basic_path)
    basic = _read_csv(basic_path, dtype_backend=dtype_backend)
    validate_columns(basic.columns, ITEM_BASIC_COLUMNS, table_name=str(basic_path))
    basic = basic[ITEM_BASIC_COLUMNS].copy()

    if logger:
        logger.info("Reading item statistic features: %s", stat_path)
    stat = _read_csv(stat_path, dtype_backend=dtype_backend)
    validate_columns(stat.columns, ITEM_STAT_COLUMNS, table_name=str(stat_path))
    stat = stat[ITEM_STAT_COLUMNS].copy()

    item_features = basic.merge(stat, on="video_id", how="left", validate="one_to_one")
    if logger:
        logger.info("Loaded item features: rows=%d", item_features.shape[0])
    return item_features


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    compression: str = "snappy",
    logger: logging.Logger | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info("Writing parquet: %s rows=%d cols=%d", path, df.shape[0], df.shape[1])
    df.to_parquet(path, index=False, compression=compression)
    return path
