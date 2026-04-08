from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.io import read_item_features, read_log_tables, read_user_features, write_parquet
from src.data.labels import add_labels, add_positive_label
from src.data.sequence import build_user_sequences
from src.data.split import split_by_date
from src.utils.paths import ensure_dir, processed_dir


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string"), format="%Y%m%d", errors="coerce")


def _prepare_item_features(item_features: pd.DataFrame) -> pd.DataFrame:
    out = item_features.copy()
    out["upload_dt_parsed"] = pd.to_datetime(out["upload_dt"], errors="coerce")
    out["upload_date"] = out["upload_dt_parsed"].dt.strftime("%Y%m%d")
    out["upload_date"] = pd.to_numeric(out["upload_date"], errors="coerce").astype("Int64")
    out["tag"] = out["tag"].astype("string").fillna("unknown")
    out["author_id"] = pd.to_numeric(out["author_id"], errors="coerce").fillna(-1).astype("int64")
    return out


def _prepare_user_features(user_features: pd.DataFrame) -> pd.DataFrame:
    out = user_features.copy()
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce").astype("Int64")
    return out


def _prepare_interactions(
    logs: pd.DataFrame,
    item_features: pd.DataFrame,
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    label_cfg = cfg["preprocess"]["labels"]
    interactions = add_positive_label(add_labels(logs, label_cfg))

    interactions["user_id"] = pd.to_numeric(interactions["user_id"], errors="coerce").astype("int64")
    interactions["video_id"] = pd.to_numeric(interactions["video_id"], errors="coerce").astype("int64")
    interactions["date"] = pd.to_numeric(interactions["date"], errors="coerce").astype("int64")
    interactions["time_ms"] = pd.to_numeric(interactions["time_ms"], errors="coerce").fillna(0).astype("int64")
    interactions["hourmin"] = pd.to_numeric(interactions["hourmin"], errors="coerce").fillna(-1).astype("int32")

    item_cols = ["video_id", "author_id", "tag", "upload_dt", "upload_date", "video_duration"]
    item_side = item_features[item_cols].drop_duplicates("video_id")
    interactions = interactions.merge(item_side, on="video_id", how="left", validate="many_to_one")

    date_dt = _parse_yyyymmdd(interactions["date"])
    upload_dt = pd.to_datetime(interactions["upload_dt"], errors="coerce")
    interactions["freshness_days"] = (date_dt - upload_dt).dt.days
    interactions["freshness_days"] = interactions["freshness_days"].clip(lower=0)

    interactions["author_id"] = pd.to_numeric(interactions["author_id"], errors="coerce").fillna(-1).astype("int64")
    interactions["tag"] = interactions["tag"].astype("string").fillna("unknown")

    sort_cols = ["time_ms", "user_id", "video_id"]
    interactions = interactions.sort_values(sort_cols).reset_index(drop=True)

    if logger:
        logger.info(
            "Prepared interactions: rows=%d labels={like:%.4f, finish:%.4f, long_watch:%.4f}",
            interactions.shape[0],
            float(interactions["like"].mean()),
            float(interactions["finish"].mean()),
            float(interactions["long_watch"].mean()),
        )
    return interactions


def run_preprocess(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Run the full preprocessing job and return written artifact paths."""
    out_dir = ensure_dir(processed_dir(cfg))
    split_dir = ensure_dir(out_dir / cfg["preprocess"]["output"]["splits_dir"])
    compression = cfg["preprocess"]["output"].get("compression", "snappy")

    if logger:
        logger.info("Starting preprocessing. processed_dir=%s", out_dir)

    logs = read_log_tables(cfg, logger=logger)
    user_features = _prepare_user_features(read_user_features(cfg, logger=logger))
    item_features = _prepare_item_features(read_item_features(cfg, logger=logger))
    interactions = _prepare_interactions(logs, item_features, cfg, logger=logger)

    splits = split_by_date(interactions, cfg["preprocess"]["split"], logger=logger)
    user_sequences = build_user_sequences(
        splits["train"],
        cfg["preprocess"]["sequences"],
        logger=logger,
    )

    output_cfg = cfg["preprocess"]["output"]
    paths: dict[str, Path] = {}
    paths["interactions"] = write_parquet(
        interactions,
        out_dir / output_cfg["interactions_file"],
        compression=compression,
        logger=logger,
    )
    paths["user_features"] = write_parquet(
        user_features,
        out_dir / output_cfg["user_features_file"],
        compression=compression,
        logger=logger,
    )
    paths["item_features"] = write_parquet(
        item_features,
        out_dir / output_cfg["item_features_file"],
        compression=compression,
        logger=logger,
    )
    paths["user_sequences"] = write_parquet(
        user_sequences,
        out_dir / output_cfg["user_sequences_file"],
        compression=compression,
        logger=logger,
    )

    for split_name, split_df in splits.items():
        paths[f"split_{split_name}"] = write_parquet(
            split_df,
            split_dir / f"{split_name}.parquet",
            compression=compression,
            logger=logger,
        )

    if logger:
        logger.info("Preprocessing finished.")
    return paths
