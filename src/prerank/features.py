from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _yyyymmdd_to_datetime(value: Any) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.to_datetime(str(int(value)), format="%Y%m%d", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(value, errors="coerce")


@dataclass
class CategoricalEncoder:
    name: str
    mapping: dict[Any, int] = field(default_factory=dict)
    as_string: bool = False

    @staticmethod
    def _normalize_value(value: Any, as_string: bool) -> Any:
        if pd.isna(value):
            return "__MISSING__" if as_string else -1
        if as_string:
            return str(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    def fit(self, series: pd.Series) -> "CategoricalEncoder":
        self.as_string = series.dtype == "object" or str(series.dtype).startswith("string") or str(series.dtype) == "category"
        values = series.map(lambda x: self._normalize_value(x, self.as_string)).drop_duplicates().tolist()
        self.mapping = {value: idx + 1 for idx, value in enumerate(values)}
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        values = series.map(lambda x: self._normalize_value(x, self.as_string))
        return values.map(self.mapping).fillna(0).astype("int32")


@dataclass
class FeatureSpec:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    encoders: dict[str, CategoricalEncoder]
    fill_values: dict[str, float]


def build_train_history_stats(train_split: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build leakage-safe user/item stats from train split only."""
    required = {"user_id", "video_id", "is_positive", "like", "finish", "long_watch"}
    missing = required - set(train_split.columns)
    if missing:
        raise KeyError(f"Train split missing required columns for stats: {sorted(missing)}")

    stat_cols = ["is_positive", "like", "finish", "long_watch"]
    train = train_split[["user_id", "video_id", *stat_cols]].copy()
    for col in stat_cols:
        train[col] = _to_numeric(train[col], default=0.0)

    user_stats = (
        train.groupby("user_id", as_index=False)
        .agg(
            user_train_exposure_count=("video_id", "size"),
            user_train_unique_items=("video_id", "nunique"),
            user_train_positive_count=("is_positive", "sum"),
            user_train_like_count=("like", "sum"),
            user_train_finish_count=("finish", "sum"),
            user_train_long_watch_count=("long_watch", "sum"),
        )
    )
    user_stats["user_train_positive_rate"] = (
        user_stats["user_train_positive_count"] / user_stats["user_train_exposure_count"].clip(lower=1)
    )

    item_stats = (
        train.groupby("video_id", as_index=False)
        .agg(
            item_train_exposure_count=("user_id", "size"),
            item_train_unique_users=("user_id", "nunique"),
            item_train_positive_count=("is_positive", "sum"),
            item_train_like_count=("like", "sum"),
            item_train_finish_count=("finish", "sum"),
            item_train_long_watch_count=("long_watch", "sum"),
        )
    )
    item_stats["item_train_positive_rate"] = (
        item_stats["item_train_positive_count"] / item_stats["item_train_exposure_count"].clip(lower=1)
    )
    return {"user_stats": user_stats, "item_stats": item_stats}


def _add_source_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    feature_cfg = cfg["prerank"]["features"]
    source_names = [str(x) for x in feature_cfg.get("source_names", [])]
    score_default = float(feature_cfg.get("source_score_default", 0.0))
    rank_default = float(feature_cfg.get("source_rank_default", 1_000_000))

    for source in source_names:
        score_col = f"{source}_score"
        rank_col = f"{source}_rank"
        flag_col = f"has_{source}"
        if score_col not in out.columns:
            out[score_col] = np.nan
        if rank_col not in out.columns:
            out[rank_col] = np.nan
        out[score_col] = _to_numeric(out[score_col], default=score_default).astype("float32")
        out[rank_col] = _to_numeric(out[rank_col], default=rank_default).astype("float32")
        out[flag_col] = (out[rank_col] < rank_default).astype("int8")

    if "source_count" not in out.columns:
        out["source_count"] = out[[f"has_{s}" for s in source_names]].sum(axis=1)
    out["source_count"] = _to_numeric(out["source_count"], default=0.0).astype("float32")
    return out


def _prepare_user_features(user_features: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    feature_cfg = cfg["prerank"]["features"]
    user_cols = ["user_id"]
    user_cols += _existing(feature_cfg.get("user_categorical_cols", []), user_features)
    user_cols += _existing(feature_cfg.get("user_numeric_cols", []), user_features)
    return user_features[user_cols].drop_duplicates("user_id")


def _prepare_item_features(item_features: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    feature_cfg = cfg["prerank"]["features"]
    item_cols = ["video_id"]
    item_cols += _existing(feature_cfg.get("item_categorical_cols", []), item_features)
    item_cols += _existing(feature_cfg.get("item_numeric_cols", []), item_features)
    item_cols += _existing(["upload_date"], item_features)
    if bool(feature_cfg.get("use_raw_item_stat_features", False)):
        item_cols += _existing(feature_cfg.get("raw_item_stat_cols", []), item_features)
    return item_features[item_cols].drop_duplicates("video_id")


def _reference_date_for_split(cfg: Mapping[str, Any], split: str) -> pd.Timestamp:
    ref_dates = cfg["prerank"]["features"].get("split_reference_dates", {})
    if split not in ref_dates:
        raise KeyError(f"Missing prerank.features.split_reference_dates.{split}")
    return _yyyymmdd_to_datetime(ref_dates[split])


def build_feature_frame(
    candidates: pd.DataFrame,
    cfg: Mapping[str, Any],
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_sequences: pd.DataFrame,
    train_stats: Mapping[str, pd.DataFrame],
    split: str,
) -> pd.DataFrame:
    """Join lightweight candidate, user, item, and train-only stats features."""
    if candidates.empty:
        return candidates.copy()
    required = {"user_id", "video_id"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"Candidates missing required columns: {sorted(missing)}")

    out = candidates.copy()
    out["user_id"] = out["user_id"].astype("int64")
    out["video_id"] = out["video_id"].astype("int64")

    for col in ["merged_score", "source_score", "source_rank", "merged_rank"]:
        if col not in out.columns:
            out[col] = 0.0
        default = 1_000_000.0 if "rank" in col else 0.0
        out[col] = _to_numeric(out[col], default=default).astype("float32")

    out = _add_source_features(out, cfg)
    out = out.merge(_prepare_user_features(user_features, cfg), on="user_id", how="left")
    out = out.merge(_prepare_item_features(item_features, cfg), on="video_id", how="left")
    if "history_len" in user_sequences.columns:
        out = out.merge(
            user_sequences[["user_id", "history_len"]].drop_duplicates("user_id"),
            on="user_id",
            how="left",
        )

    user_stats = train_stats.get("user_stats")
    item_stats = train_stats.get("item_stats")
    if user_stats is not None:
        out = out.merge(user_stats, on="user_id", how="left")
    if item_stats is not None:
        out = out.merge(item_stats, on="video_id", how="left")

    if "upload_date" in out.columns:
        ref_date = _reference_date_for_split(cfg, split)
        upload_dt = out["upload_date"].map(_yyyymmdd_to_datetime)
        out["freshness_days"] = (ref_date - upload_dt).dt.days.clip(lower=0).astype("float32")
    else:
        out["freshness_days"] = 0.0
    return out


def infer_feature_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    feature_cfg = cfg["prerank"]["features"]
    source_names = [str(x) for x in feature_cfg.get("source_names", [])]

    categorical_columns = [
        "user_id",
        "video_id",
        "recall_source",
        *_existing(feature_cfg.get("user_categorical_cols", []), df),
        *_existing(feature_cfg.get("item_categorical_cols", []), df),
    ]
    numeric_columns = [
        "source_score",
        "source_rank",
        "merged_score",
        "source_count",
        "merged_rank",
        "freshness_days",
        "history_len",
        *[f"{source}_score" for source in source_names if f"{source}_score" in df.columns],
        *[f"{source}_rank" for source in source_names if f"{source}_rank" in df.columns],
        *[f"has_{source}" for source in source_names if f"has_{source}" in df.columns],
        *_existing(feature_cfg.get("user_numeric_cols", []), df),
        *_existing(feature_cfg.get("item_numeric_cols", []), df),
        *_existing(
            [
                "user_train_exposure_count",
                "user_train_unique_items",
                "user_train_positive_count",
                "user_train_like_count",
                "user_train_finish_count",
                "user_train_long_watch_count",
                "user_train_positive_rate",
                "item_train_exposure_count",
                "item_train_unique_users",
                "item_train_positive_count",
                "item_train_like_count",
                "item_train_finish_count",
                "item_train_long_watch_count",
                "item_train_positive_rate",
            ],
            df,
        ),
    ]
    if bool(feature_cfg.get("use_raw_item_stat_features", False)):
        numeric_columns += _existing(feature_cfg.get("raw_item_stat_cols", []), df)

    categorical_columns = list(dict.fromkeys([col for col in categorical_columns if col in df.columns]))
    numeric_columns = list(dict.fromkeys([col for col in numeric_columns if col in df.columns and col not in categorical_columns]))
    feature_columns = [*categorical_columns, *numeric_columns]
    return feature_columns, categorical_columns, numeric_columns


def fit_feature_spec(df: pd.DataFrame, cfg: Mapping[str, Any]) -> FeatureSpec:
    feature_columns, categorical_columns, numeric_columns = infer_feature_columns(df, cfg)
    encoders = {col: CategoricalEncoder(name=col).fit(df[col]) for col in categorical_columns}
    fill_values = {col: 0.0 for col in numeric_columns}
    return FeatureSpec(
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        encoders=encoders,
        fill_values=fill_values,
    )


def transform_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in spec.categorical_columns:
        if col in df.columns:
            out[col] = spec.encoders[col].transform(df[col])
        else:
            out[col] = np.zeros(len(df), dtype=np.int32)
    for col in spec.numeric_columns:
        if col in df.columns:
            out[col] = _to_numeric(df[col], default=spec.fill_values.get(col, 0.0)).astype("float32")
        else:
            out[col] = np.zeros(len(df), dtype=np.float32)
    return out[spec.feature_columns]
