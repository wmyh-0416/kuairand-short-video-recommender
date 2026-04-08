from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.prerank.features import (
    build_feature_frame,
    build_train_history_stats,
    fit_feature_spec,
    transform_features,
)
from src.utils.paths import artifacts_dir, processed_path


def recall_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["prerank"]["input"]["recall_dir"]


def prerank_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["prerank"]["output"]["prerank_dir"]


def metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["prerank"]["output"].get("metrics_dir", "metrics")


def load_recall_candidates(cfg: Mapping[str, Any], split: str, nrows: int | None = None) -> pd.DataFrame:
    filename = cfg["prerank"]["input"][f"{split}_candidates_file"]
    path = recall_dir(cfg) / filename
    if not path.exists():
        raise FileNotFoundError(f"Recall candidates not found: {path}")
    df = pd.read_parquet(path)
    if nrows is not None:
        df = df.head(nrows).copy()
    return df


def load_processed_tables(cfg: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    processed_cfg = cfg["prerank"]["processed"]
    splits_dir = Path(processed_cfg.get("splits_dir", "splits"))
    return {
        "user_features": pd.read_parquet(processed_path(cfg, processed_cfg["user_features_file"])),
        "item_features": pd.read_parquet(processed_path(cfg, processed_cfg["item_features_file"])),
        "user_sequences": pd.read_parquet(processed_path(cfg, processed_cfg["user_sequences_file"])),
        "train_split": pd.read_parquet(processed_path(cfg, splits_dir / "train.parquet")),
    }


def sample_negative_candidates(
    candidates: pd.DataFrame,
    cfg: Mapping[str, Any],
    split: str,
) -> pd.DataFrame:
    label_col = cfg["prerank"].get("label_col", "label")
    sampling_cfg = cfg["prerank"].get("sampling", {})
    if candidates.empty or label_col not in candidates.columns:
        return candidates.copy()
    if not bool(sampling_cfg.get("enabled", True)):
        return candidates.copy()

    if split == "train":
        negative_ratio = float(sampling_cfg.get("negative_sample_ratio", 5))
    elif split == "val":
        negative_ratio = float(sampling_cfg.get("val_negative_sample_ratio", 10))
    else:
        return candidates.copy()

    if negative_ratio <= 0:
        return candidates.loc[candidates[label_col] > 0].copy()

    max_negative_per_user = int(sampling_cfg.get("max_negative_per_user", 80))
    include_zero_pos = bool(sampling_cfg.get("include_users_without_positive", False))
    random_seed = int(sampling_cfg.get("random_seed", 2026))

    pos = candidates.loc[candidates[label_col] > 0].copy()
    neg = candidates.loc[candidates[label_col] <= 0].copy()
    if pos.empty or neg.empty:
        return candidates.copy()

    pos_count = pos.groupby("user_id").size().rename("_pos_count").reset_index()
    neg = neg.merge(pos_count, on="user_id", how="left")
    if include_zero_pos:
        neg["_pos_count"] = neg["_pos_count"].fillna(1)
    else:
        neg = neg.loc[neg["_pos_count"].notna()].copy()
    if neg.empty:
        return pos.reset_index(drop=True)

    neg["_quota"] = np.ceil(neg["_pos_count"].astype(float) * negative_ratio).clip(upper=max_negative_per_user).astype(int)
    neg["_rand"] = np.random.default_rng(random_seed).random(len(neg))
    neg = neg.sort_values(["user_id", "_rand"])
    neg["_rank"] = neg.groupby("user_id").cumcount() + 1
    neg = neg.loc[neg["_rank"] <= neg["_quota"]].drop(columns=["_pos_count", "_quota", "_rand", "_rank"])

    samples = pd.concat([pos, neg], ignore_index=True)
    return samples.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def build_prerank_samples(
    candidates: pd.DataFrame,
    cfg: Mapping[str, Any],
    processed_tables: Mapping[str, pd.DataFrame],
    train_stats: Mapping[str, pd.DataFrame],
    split: str,
    sample_negatives: bool = True,
) -> pd.DataFrame:
    if sample_negatives:
        candidates = sample_negative_candidates(candidates, cfg=cfg, split=split)
    return build_feature_frame(
        candidates=candidates,
        cfg=cfg,
        user_features=processed_tables["user_features"],
        item_features=processed_tables["item_features"],
        user_sequences=processed_tables["user_sequences"],
        train_stats=train_stats,
        split=split,
    )


def build_train_val_matrices(
    cfg: Mapping[str, Any],
    train_candidates: pd.DataFrame,
    val_candidates: pd.DataFrame,
    processed_tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any, pd.DataFrame, pd.DataFrame]:
    label_col = cfg["prerank"].get("label_col", "label")
    train_stats = build_train_history_stats(processed_tables["train_split"])

    train_samples = build_prerank_samples(
        train_candidates,
        cfg=cfg,
        processed_tables=processed_tables,
        train_stats=train_stats,
        split="train",
        sample_negatives=True,
    )
    val_samples = build_prerank_samples(
        val_candidates,
        cfg=cfg,
        processed_tables=processed_tables,
        train_stats=train_stats,
        split="val",
        sample_negatives=True,
    )
    spec = fit_feature_spec(train_samples, cfg)
    x_train = transform_features(train_samples, spec)
    x_val = transform_features(val_samples, spec)
    y_train = train_samples[label_col].astype("int8")
    y_val = val_samples[label_col].astype("int8")
    return x_train, y_train, x_val, y_val, spec, train_samples, val_samples
