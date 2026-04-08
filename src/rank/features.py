from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def _existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _normalize_cat(value: Any, as_string: bool) -> Any:
    if pd.isna(value):
        return "__MISSING__" if as_string else -1
    if as_string:
        return str(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


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

    def fit(self, series: pd.Series) -> "CategoricalEncoder":
        self.as_string = series.dtype == "object" or str(series.dtype).startswith("string") or str(series.dtype) == "category"
        values = series.map(lambda x: _normalize_cat(x, self.as_string)).drop_duplicates().tolist()
        self.mapping = {value: idx + 1 for idx, value in enumerate(values)}
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        values = series.map(lambda x: _normalize_cat(x, self.as_string))
        return values.map(self.mapping).fillna(0).astype("int32")

    @property
    def vocab_size(self) -> int:
        return len(self.mapping) + 1


@dataclass
class RankFeatureSpec:
    categorical_columns: list[str]
    numeric_columns: list[str]
    other_categorical_columns: list[str]
    encoders: dict[str, CategoricalEncoder]
    fill_values: dict[str, float]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    max_seq_len: int
    tasks: list[str]

    def vocab_size(self, column: str) -> int:
        return self.encoders[column].vocab_size


@dataclass
class RankFeatureStore:
    user_features: pd.DataFrame
    item_features: pd.DataFrame
    train_split: pd.DataFrame
    history_raw: dict[int, np.ndarray]
    history_encoded: dict[int, np.ndarray]
    user_author_counts: dict[int, dict[Any, int]]
    user_tag_counts: dict[int, dict[Any, int]]
    item_to_author: dict[int, Any]
    item_to_tag: dict[int, Any]


def prepare_item_features(item_features: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    feature_cfg = cfg["rank"]["features"]
    cols = ["video_id"]
    cols += _existing(feature_cfg.get("item_categorical_cols", []), item_features)
    cols += _existing(feature_cfg.get("item_numeric_cols", []), item_features)
    cols += _existing(["upload_date"], item_features)
    return item_features[cols].drop_duplicates("video_id")


def prepare_user_features(user_features: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    feature_cfg = cfg["rank"]["features"]
    cols = ["user_id"]
    cols += _existing(feature_cfg.get("user_categorical_cols", []), user_features)
    cols += _existing(feature_cfg.get("user_numeric_cols", []), user_features)
    return user_features[cols].drop_duplicates("user_id")


def build_task_label_table(split_df: pd.DataFrame, tasks: list[str]) -> pd.DataFrame:
    required = {"user_id", "video_id", *tasks}
    missing = required - set(split_df.columns)
    if missing:
        raise KeyError(f"Split data missing task label columns: {sorted(missing)}")
    labels = split_df[["user_id", "video_id", *tasks]].copy()
    for task in tasks:
        labels[task] = _to_numeric(labels[task], default=0.0).clip(0, 1).astype("int8")
    return labels.groupby(["user_id", "video_id"], as_index=False)[tasks].max()


def build_train_histories(
    train_split: pd.DataFrame,
    item_features: pd.DataFrame,
    cfg: Mapping[str, Any],
    item_encoder: CategoricalEncoder | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict[Any, int]], dict[int, dict[Any, int]]]:
    """Build train-only user history maps for DIN to avoid val/test leakage."""
    seq_cfg = cfg["rank"]["sequence"]
    max_seq_len = int(seq_cfg.get("max_seq_len", 50))
    history = train_split[["user_id", "video_id", "time_ms", str(seq_cfg.get("history_label_col", "is_positive"))]].copy()
    label_col = str(seq_cfg.get("history_label_col", "is_positive"))
    if bool(seq_cfg.get("history_positive_only", True)):
        history = history.loc[pd.to_numeric(history[label_col], errors="coerce").fillna(0) > 0].copy()
    history = history.sort_values(["user_id", "time_ms", "video_id"])

    item_meta = item_features[["video_id", "author_id", "tag"]].drop_duplicates("video_id")
    history = history.merge(item_meta, on="video_id", how="left")

    raw: dict[int, np.ndarray] = {}
    encoded: dict[int, np.ndarray] = {}
    author_counts: dict[int, dict[Any, int]] = {}
    tag_counts: dict[int, dict[Any, int]] = {}

    for user_id, group in history.groupby("user_id", sort=False):
        item_ids = group["video_id"].tail(max_seq_len * 3).astype("int64").to_numpy()
        raw[int(user_id)] = item_ids
        if item_encoder is not None:
            encoded[int(user_id)] = item_encoder.transform(pd.Series(item_ids)).astype("int32").to_numpy()
        author_counts[int(user_id)] = group["author_id"].value_counts(dropna=True).to_dict()
        tag_counts[int(user_id)] = group["tag"].value_counts(dropna=True).to_dict()
    return raw, encoded, author_counts, tag_counts


def _reference_date_for_split(cfg: Mapping[str, Any], split: str) -> pd.Timestamp:
    ref_dates = cfg["rank"]["features"].get("split_reference_dates", {})
    if split not in ref_dates:
        raise KeyError(f"Missing rank.features.split_reference_dates.{split}")
    return _yyyymmdd_to_datetime(ref_dates[split])


def _add_source_features(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    feature_cfg = cfg["rank"]["features"]
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
    return out


def build_rank_frame(
    candidates: pd.DataFrame,
    split_df: pd.DataFrame,
    store: RankFeatureStore,
    cfg: Mapping[str, Any],
    split: str,
    precomputed_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    tasks = [str(x) for x in cfg["rank"]["tasks"]]
    labels = precomputed_labels if precomputed_labels is not None else build_task_label_table(split_df, tasks)
    out = candidates.copy()
    out["user_id"] = out["user_id"].astype("int64")
    out["video_id"] = out["video_id"].astype("int64")
    out = out.merge(labels, on=["user_id", "video_id"], how="left")
    for task in tasks:
        out[task] = out[task].fillna(0).astype("int8")
    out["rank_label"] = out[tasks].max(axis=1).astype("int8")

    out = _add_source_features(out, cfg)
    for col in ["source_score", "source_rank", "merged_score", "source_count", "merged_rank", "coarse_score", "coarse_rank"]:
        if col not in out.columns:
            out[col] = 0.0
        default = 1_000_000.0 if "rank" in col else 0.0
        out[col] = _to_numeric(out[col], default=default).astype("float32")

    out = out.merge(store.user_features, on="user_id", how="left")
    out = out.merge(store.item_features, on="video_id", how="left")
    out["history_len"] = out["user_id"].map(lambda u: len(store.history_raw.get(int(u), ())))
    out["history_len"] = _to_numeric(out["history_len"], default=0.0).astype("float32")

    authors = out["author_id"].tolist() if "author_id" in out.columns else [None] * len(out)
    tags = out["tag"].tolist() if "tag" in out.columns else [None] * len(out)
    users = out["user_id"].astype("int64").to_numpy()
    same_author = []
    same_tag = []
    for user_id, author_id, tag in zip(users, authors, tags):
        same_author.append(store.user_author_counts.get(int(user_id), {}).get(author_id, 0))
        same_tag.append(store.user_tag_counts.get(int(user_id), {}).get(tag, 0))
    out["hist_same_author_count"] = np.asarray(same_author, dtype=np.float32)
    out["hist_same_tag_count"] = np.asarray(same_tag, dtype=np.float32)
    out["hist_has_same_author"] = (out["hist_same_author_count"] > 0).astype("int8")
    out["hist_has_same_tag"] = (out["hist_same_tag_count"] > 0).astype("int8")

    if "upload_date" in out.columns:
        ref_date = _reference_date_for_split(cfg, split)
        upload_dt = out["upload_date"].map(_yyyymmdd_to_datetime)
        out["freshness_days"] = (ref_date - upload_dt).dt.days.clip(lower=0).fillna(0).astype("float32")
    else:
        out["freshness_days"] = 0.0
    return out


def infer_feature_columns(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    feature_cfg = cfg["rank"]["features"]
    source_names = [str(x) for x in feature_cfg.get("source_names", [])]
    categorical = [
        "user_id",
        "video_id",
        "author_id",
        "tag",
        "recall_source",
        *_existing(feature_cfg.get("user_categorical_cols", []), df),
        *_existing([c for c in feature_cfg.get("item_categorical_cols", []) if c not in {"author_id", "tag"}], df),
    ]
    numeric = [
        "source_score",
        "source_rank",
        "merged_score",
        "source_count",
        "merged_rank",
        "coarse_score",
        "coarse_rank",
        "history_len",
        "hist_same_author_count",
        "hist_same_tag_count",
        "hist_has_same_author",
        "hist_has_same_tag",
        "freshness_days",
        *[f"{source}_score" for source in source_names if f"{source}_score" in df.columns],
        *[f"{source}_rank" for source in source_names if f"{source}_rank" in df.columns],
        *[f"has_{source}" for source in source_names if f"has_{source}" in df.columns],
        *_existing(feature_cfg.get("user_numeric_cols", []), df),
        *_existing(feature_cfg.get("item_numeric_cols", []), df),
    ]
    categorical = list(dict.fromkeys([col for col in categorical if col in df.columns]))
    numeric = list(dict.fromkeys([col for col in numeric if col in df.columns and col not in categorical]))
    other_categorical = [col for col in categorical if col not in {"user_id", "video_id", "author_id", "tag"}]
    return categorical, numeric, other_categorical


def fit_feature_spec(df: pd.DataFrame, cfg: Mapping[str, Any]) -> RankFeatureSpec:
    categorical, numeric, other_categorical = infer_feature_columns(df, cfg)
    encoders = {col: CategoricalEncoder(name=col).fit(df[col]) for col in categorical}
    fill_values = {col: 0.0 for col in numeric}
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for col in numeric:
        values = _to_numeric(df[col], default=0.0).astype("float32")
        mean = float(values.mean())
        std = float(values.std())
        numeric_means[col] = mean if np.isfinite(mean) else 0.0
        numeric_stds[col] = std if np.isfinite(std) and std > 1e-6 else 1.0
    return RankFeatureSpec(
        categorical_columns=categorical,
        numeric_columns=numeric,
        other_categorical_columns=other_categorical,
        encoders=encoders,
        fill_values=fill_values,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        max_seq_len=int(cfg["rank"]["sequence"].get("max_seq_len", 50)),
        tasks=[str(x) for x in cfg["rank"]["tasks"]],
    )


def transform_rank_frame(
    df: pd.DataFrame,
    spec: RankFeatureSpec,
    store: RankFeatureStore,
    cfg: Mapping[str, Any],
    include_labels: bool = True,
) -> dict[str, np.ndarray]:
    n = len(df)
    arrays: dict[str, np.ndarray] = {}
    for col in spec.categorical_columns:
        if col in df.columns:
            arrays[col] = spec.encoders[col].transform(df[col]).to_numpy(dtype=np.int64)
        else:
            arrays[col] = np.zeros(n, dtype=np.int64)
    for col in spec.numeric_columns:
        if col in df.columns:
            values = _to_numeric(df[col], default=spec.fill_values.get(col, 0.0)).astype("float32")
        else:
            values = pd.Series(np.zeros(n, dtype=np.float32), index=df.index)
        mean = float(spec.numeric_means.get(col, 0.0))
        std = float(spec.numeric_stds.get(col, 1.0))
        arrays[f"num__{col}"] = ((values - mean) / max(std, 1e-6)).to_numpy(dtype=np.float32)

    max_seq_len = spec.max_seq_len
    hist_items = np.zeros((n, max_seq_len), dtype=np.int64)
    exclude_target = bool(cfg["rank"]["sequence"].get("exclude_target_from_history", True))
    users = df["user_id"].astype("int64").to_numpy()
    targets = df["video_id"].astype("int64").to_numpy()
    for i, (user_id, target_id) in enumerate(zip(users, targets)):
        raw_hist = store.history_raw.get(int(user_id))
        enc_hist = store.history_encoded.get(int(user_id))
        if raw_hist is None or enc_hist is None or len(enc_hist) == 0:
            continue
        if exclude_target:
            mask = raw_hist != int(target_id)
            seq = enc_hist[mask]
        else:
            seq = enc_hist
        if len(seq) == 0:
            continue
        seq = seq[-max_seq_len:]
        hist_items[i, -len(seq) :] = seq
    arrays["hist_item_seq"] = hist_items

    numeric_matrix = np.column_stack([arrays.pop(f"num__{col}") for col in spec.numeric_columns]).astype(np.float32)
    arrays["numeric"] = numeric_matrix
    if include_labels:
        labels = np.column_stack([df[task].fillna(0).astype("float32").to_numpy() for task in spec.tasks]).astype(np.float32)
        arrays["labels"] = labels
        arrays["rank_label"] = df["rank_label"].fillna(0).astype("int8").to_numpy()
    return arrays


def sample_rank_training_frame(df: pd.DataFrame, cfg: Mapping[str, Any], split: str) -> pd.DataFrame:
    sampling_cfg = cfg["rank"].get("sampling", {})
    if not bool(sampling_cfg.get("enabled", True)):
        return df.copy()
    tasks = [str(x) for x in cfg["rank"]["tasks"]]
    pos_mask = df[tasks].max(axis=1) > 0
    pos = df.loc[pos_mask].copy()
    neg = df.loc[~pos_mask].copy()
    if pos.empty or neg.empty:
        return df.copy()
    if split == "train":
        ratio = float(sampling_cfg.get("train_negative_sample_ratio", 4))
    else:
        ratio = float(sampling_cfg.get("val_negative_sample_ratio", 8))
    max_neg = int(sampling_cfg.get("max_negative_per_user", 80))
    seed = int(sampling_cfg.get("random_seed", 2026)) + (0 if split == "train" else 17)
    pos_count = pos.groupby("user_id").size().rename("_pos_count").reset_index()
    neg = neg.merge(pos_count, on="user_id", how="inner")
    if neg.empty:
        return pos.reset_index(drop=True)
    neg["_quota"] = np.ceil(neg["_pos_count"].astype(float) * ratio).clip(upper=max_neg).astype(int)
    neg["_rand"] = np.random.default_rng(seed).random(len(neg))
    neg = neg.sort_values(["user_id", "_rand"])
    neg["_rank"] = neg.groupby("user_id").cumcount() + 1
    neg = neg.loc[neg["_rank"] <= neg["_quota"]].drop(columns=["_pos_count", "_quota", "_rand", "_rank"])
    out = pd.concat([pos, neg], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
