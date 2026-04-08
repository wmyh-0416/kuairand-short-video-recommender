from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.prerank.dataset import (
    load_processed_tables,
    load_recall_candidates,
    metrics_dir,
    prerank_dir,
)
from src.prerank.features import build_feature_frame, build_train_history_stats, transform_features
from src.prerank.lightgbm_model import load_model_bundle, predict_lightgbm
from src.prerank.mlp_model import predict_mlp


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _predict(model_type: str, model_payload: Any, x: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    if model_type == "lightgbm":
        return predict_lightgbm(model_payload, x)
    if model_type == "mlp":
        return predict_mlp(model_payload, x, cfg)
    raise ValueError(f"Unsupported prerank model type: {model_type}")


def _iter_chunks(df: pd.DataFrame, batch_rows: int):
    if batch_rows <= 0:
        yield df
        return
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start : start + batch_rows].copy()


def score_candidates(
    candidates: pd.DataFrame,
    cfg: Mapping[str, Any],
    split: str,
    model_bundle: Mapping[str, Any],
    processed_tables: Mapping[str, pd.DataFrame],
    train_stats: Mapping[str, pd.DataFrame],
    logger: Any | None = None,
) -> pd.DataFrame:
    feature_spec = model_bundle["feature_spec"]
    model_type = str(model_bundle["model_type"])
    model_payload = model_bundle["model"]
    keep_columns = [col for col in cfg["prerank"]["inference"].get("keep_columns", []) if col in candidates.columns]
    if "label" in candidates.columns and "label" not in keep_columns:
        keep_columns.append("label")
    if "user_id" not in keep_columns:
        keep_columns.insert(0, "user_id")
    if "video_id" not in keep_columns:
        keep_columns.insert(1, "video_id")

    batch_rows = int(cfg["prerank"]["inference"].get("batch_rows", 2_000_000))
    parts: list[pd.DataFrame] = []
    for idx, chunk in enumerate(_iter_chunks(candidates, batch_rows=batch_rows)):
        feat_df = build_feature_frame(
            candidates=chunk,
            cfg=cfg,
            user_features=processed_tables["user_features"],
            item_features=processed_tables["item_features"],
            user_sequences=processed_tables["user_sequences"],
            train_stats=train_stats,
            split=split,
        )
        x = transform_features(feat_df, feature_spec)
        scored = chunk[keep_columns].copy()
        scored["coarse_score"] = _predict(model_type, model_payload, x, cfg)
        parts.append(scored)
        if logger:
            logger.info("Scored %s chunk %d rows=%d", split, idx + 1, len(chunk))
    return pd.concat(parts, ignore_index=True) if parts else candidates.iloc[0:0].copy()


def select_topk(scored: pd.DataFrame, topk: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    sort_cols = ["user_id", "coarse_score"]
    ascending = [True, False]
    if "merged_score" in scored.columns:
        sort_cols.append("merged_score")
        ascending.append(False)
    if "merged_rank" in scored.columns:
        sort_cols.append("merged_rank")
        ascending.append(True)
    sort_cols.append("video_id")
    ascending.append(True)
    out = scored.sort_values(sort_cols, ascending=ascending)
    out = out.groupby("user_id", as_index=False, group_keys=False).head(topk).reset_index(drop=True)
    out["coarse_rank"] = out.groupby("user_id").cumcount() + 1
    return out


def evaluate_prerank_topk(
    candidates: pd.DataFrame,
    topk_df: pd.DataFrame,
    eval_topks: list[int],
) -> dict[str, Any]:
    label_col = "label"
    total_positive = int(candidates[label_col].sum()) if label_col in candidates.columns else 0
    metrics: dict[str, Any] = {
        "input_candidates": int(candidates.shape[0]),
        "output_candidates": int(topk_df.shape[0]),
        "num_users": int(candidates["user_id"].nunique()) if not candidates.empty else 0,
        "input_positive": total_positive,
        "output_positive": int(topk_df[label_col].sum()) if label_col in topk_df.columns else 0,
        "candidate_compression_ratio": float(topk_df.shape[0] / max(candidates.shape[0], 1)),
        "recall_retained": float(topk_df[label_col].sum() / max(total_positive, 1)) if label_col in topk_df.columns else None,
    }
    if label_col in topk_df.columns:
        metrics["auc_on_scored_candidates"] = _safe_auc(topk_df[label_col].astype(int), topk_df["coarse_score"].astype(float))
    for k in eval_topks:
        k = int(k)
        top = topk_df.sort_values(["user_id", "coarse_rank"]).groupby("user_id", as_index=False, group_keys=False).head(k)
        if label_col in top.columns:
            metrics[f"precision@{k}"] = float(top[label_col].sum() / max(top.shape[0], 1))
            metrics[f"recall_retained@{k}"] = float(top[label_col].sum() / max(total_positive, 1))
    return metrics


def generate_prerank_topk(
    cfg: Mapping[str, Any],
    splits: list[str] | None = None,
    candidate_rows: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    splits = splits or ["train", "val", "test"]
    out_dir = prerank_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out_dir = metrics_dir(cfg)
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / cfg["prerank"]["output"]["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Prerank model not found: {model_path}")
    model_bundle = load_model_bundle(model_path)
    processed_tables = load_processed_tables(cfg)
    train_stats = build_train_history_stats(processed_tables["train_split"])
    topk = int(cfg["prerank"].get("topk", 100))
    eval_topks = [int(k) for k in cfg["prerank"].get("eval_topk", [50, 100])]

    all_metrics: dict[str, Any] = {}
    for split in splits:
        candidates = load_recall_candidates(cfg, split, nrows=candidate_rows)
        scored = score_candidates(
            candidates=candidates,
            cfg=cfg,
            split=split,
            model_bundle=model_bundle,
            processed_tables=processed_tables,
            train_stats=train_stats,
            logger=logger,
        )
        topk_df = select_topk(scored, topk=topk)
        out_file = cfg["prerank"]["output"][f"{split}_topk_file"]
        out_path = out_dir / out_file
        topk_df.to_parquet(out_path, index=False)
        all_metrics[split] = evaluate_prerank_topk(candidates, topk_df, eval_topks=eval_topks)
        if logger:
            logger.info("Saved %s prerank topK: %s rows=%d", split, out_path, topk_df.shape[0])

    metrics_path = metrics_out_dir / cfg["prerank"]["output"]["metrics_file"]
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing["inference"] = all_metrics
    metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved prerank inference metrics: %s", metrics_path)
    return all_metrics
