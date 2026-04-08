from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_labels(df: pd.DataFrame, label_cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Add short-video feedback labels.

    Labels:
      like: original is_like column.
      watch_ratio: play_time_ms / duration_ms, clipped for robustness.
      finish: watch_ratio >= finish_ratio_threshold.
      long_watch: original long_view if configured and present, otherwise a
        threshold-based fallback using watch_ratio or watch time.
    """
    out = df.copy()

    play_time = _to_numeric(out["play_time_ms"]).fillna(0).clip(lower=0)
    duration = _to_numeric(out["duration_ms"]).fillna(0)
    safe_duration = duration.where(duration > 0, np.nan)
    watch_ratio = (play_time / safe_duration).replace([np.inf, -np.inf], np.nan).fillna(0)
    max_watch_ratio = float(label_cfg.get("max_watch_ratio", 5.0))
    out["watch_ratio"] = watch_ratio.clip(lower=0, upper=max_watch_ratio).astype("float32")

    finish_threshold = float(label_cfg.get("finish_ratio_threshold", 0.95))
    out["finish"] = (out["watch_ratio"] >= finish_threshold).astype("int8")

    out["like"] = _to_numeric(out["is_like"]).fillna(0).astype("int8")

    use_long_view = bool(label_cfg.get("use_long_view_column", True))
    if use_long_view and "long_view" in out.columns:
        long_watch = _to_numeric(out["long_view"]).fillna(0) > 0
    else:
        ratio_threshold = float(label_cfg.get("long_watch_ratio_threshold", 0.70))
        ms_threshold = float(label_cfg.get("long_watch_ms_threshold", 18000))
        long_watch = (out["watch_ratio"] >= ratio_threshold) | (play_time >= ms_threshold)
    out["long_watch"] = long_watch.astype("int8")

    return out


def add_positive_label(
    df: pd.DataFrame,
    output_col: str = "is_positive",
) -> pd.DataFrame:
    """Add a default positive label for recall/prerank sample construction."""
    out = df.copy()
    out[output_col] = (
        (out["long_watch"].fillna(0) > 0)
        | (out["finish"].fillna(0) > 0)
        | (out["like"].fillna(0) > 0)
    ).astype("int8")
    return out
