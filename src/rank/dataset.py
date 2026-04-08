from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.rank.features import (
    RankFeatureSpec,
    RankFeatureStore,
    build_rank_frame,
    build_train_histories,
    fit_feature_spec,
    prepare_item_features,
    prepare_user_features,
    sample_rank_training_frame,
    transform_rank_frame,
)
from src.utils.paths import artifacts_dir, processed_path


def rank_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["rank"]["output"]["rank_dir"]


def metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["rank"]["output"].get("metrics_dir", "metrics")


def prerank_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["rank"]["input"]["prerank_dir"]


def load_prerank_topk(cfg: Mapping[str, Any], split: str, nrows: int | None = None) -> pd.DataFrame:
    path = prerank_dir(cfg) / cfg["rank"]["input"][f"{split}_topk_file"]
    if not path.exists():
        raise FileNotFoundError(f"Prerank topK not found: {path}")
    df = pd.read_parquet(path)
    if nrows is not None:
        df = df.head(nrows).copy()
    return df


def load_split(cfg: Mapping[str, Any], split: str) -> pd.DataFrame:
    splits_dir = Path(cfg["rank"]["processed"].get("splits_dir", "splits"))
    return pd.read_parquet(processed_path(cfg, splits_dir / f"{split}.parquet"))


def load_processed_tables(cfg: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    processed_cfg = cfg["rank"]["processed"]
    return {
        "user_features": pd.read_parquet(processed_path(cfg, processed_cfg["user_features_file"])),
        "item_features": pd.read_parquet(processed_path(cfg, processed_cfg["item_features_file"])),
        "train_split": load_split(cfg, "train"),
    }


def build_feature_store(
    cfg: Mapping[str, Any],
    processed_tables: Mapping[str, pd.DataFrame],
    item_encoder: Any | None = None,
) -> RankFeatureStore:
    item_features = prepare_item_features(processed_tables["item_features"], cfg)
    user_features = prepare_user_features(processed_tables["user_features"], cfg)
    raw_hist, enc_hist, author_counts, tag_counts = build_train_histories(
        processed_tables["train_split"],
        item_features=item_features,
        cfg=cfg,
        item_encoder=item_encoder,
    )
    item_meta = item_features[["video_id", "author_id", "tag"]].drop_duplicates("video_id")
    return RankFeatureStore(
        user_features=user_features,
        item_features=item_features,
        train_split=processed_tables["train_split"],
        history_raw=raw_hist,
        history_encoded=enc_hist,
        user_author_counts=author_counts,
        user_tag_counts=tag_counts,
        item_to_author=dict(zip(item_meta["video_id"].astype(int), item_meta["author_id"])),
        item_to_tag=dict(zip(item_meta["video_id"].astype(int), item_meta["tag"])),
    )


class RankTensorDataset:
    def __init__(self, arrays: Mapping[str, np.ndarray], spec: RankFeatureSpec, include_labels: bool = True) -> None:
        self.arrays = dict(arrays)
        self.spec = spec
        self.include_labels = include_labels
        self.length = int(self.arrays["video_id"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        item = {
            "user_id": self.arrays["user_id"][idx],
            "video_id": self.arrays["video_id"][idx],
            "author_id": self.arrays["author_id"][idx],
            "tag": self.arrays["tag"][idx],
            "hist_item_seq": self.arrays["hist_item_seq"][idx],
            "numeric": self.arrays["numeric"][idx],
        }
        other = {}
        for col in self.spec.other_categorical_columns:
            other[col] = self.arrays[col][idx]
        item["other_cats"] = np.asarray([other[col] for col in self.spec.other_categorical_columns], dtype=np.int64)
        if self.include_labels:
            item["labels"] = self.arrays["labels"][idx]
            item["rank_label"] = self.arrays["rank_label"][idx]
        return item


def prepare_train_val_datasets(
    cfg: Mapping[str, Any],
    train_rows: int | None = None,
    val_rows: int | None = None,
    logger: Any | None = None,
) -> tuple[RankTensorDataset, RankTensorDataset, RankFeatureSpec, RankFeatureStore, pd.DataFrame, pd.DataFrame]:
    processed_tables = load_processed_tables(cfg)
    empty_store = build_feature_store(cfg, processed_tables, item_encoder=None)

    train_candidates = load_prerank_topk(cfg, "train", nrows=train_rows)
    val_candidates = load_prerank_topk(cfg, "val", nrows=val_rows)
    train_frame = build_rank_frame(train_candidates, load_split(cfg, "train"), empty_store, cfg, split="train")
    val_frame = build_rank_frame(val_candidates, load_split(cfg, "val"), empty_store, cfg, split="val")
    train_frame = sample_rank_training_frame(train_frame, cfg, split="train")
    val_frame = sample_rank_training_frame(val_frame, cfg, split="val")

    spec = fit_feature_spec(train_frame, cfg)
    store = build_feature_store(cfg, processed_tables, item_encoder=spec.encoders["video_id"])
    train_arrays = transform_rank_frame(train_frame, spec, store, cfg, include_labels=True)
    val_arrays = transform_rank_frame(val_frame, spec, store, cfg, include_labels=True)
    if logger:
        logger.info(
            "Prepared rank datasets: train=%s val=%s features_numeric=%d other_cats=%d",
            train_frame.shape,
            val_frame.shape,
            len(spec.numeric_columns),
            len(spec.other_categorical_columns),
        )
    return (
        RankTensorDataset(train_arrays, spec, include_labels=True),
        RankTensorDataset(val_arrays, spec, include_labels=True),
        spec,
        store,
        train_frame,
        val_frame,
    )
