from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from src.rank.dataset import (
    RankTensorDataset,
    build_feature_store,
    load_prerank_topk,
    load_processed_tables,
    load_split,
    metrics_dir,
    rank_dir,
)
from src.rank.features import build_rank_frame, build_task_label_table, transform_rank_frame
from src.rank.model import build_model, compute_rank_score, resolve_device


def _move_batch(batch: Mapping[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6), labels=[0, 1]))


def _iter_chunks(df: pd.DataFrame, batch_rows: int):
    if batch_rows <= 0:
        yield df
        return
    for start in range(0, len(df), batch_rows):
        yield df.iloc[start : start + batch_rows].copy()


def _ndcg_at_k(group: pd.DataFrame, k: int) -> float:
    rel = group.head(k)["rank_label"].to_numpy(dtype=float)
    if rel.size == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discount).sum())
    ideal = np.sort(group["rank_label"].to_numpy(dtype=float))[::-1][:k]
    idcg = float((ideal * discount[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(group: pd.DataFrame, k: int) -> float:
    rel = group.head(k)["rank_label"].to_numpy(dtype=int)
    hits = np.where(rel > 0)[0]
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def evaluate_ranked(ranked: pd.DataFrame, cfg: Mapping[str, Any]) -> dict[str, Any]:
    tasks = [str(x) for x in cfg["rank"]["tasks"]]
    metrics: dict[str, Any] = {
        "num_candidates": int(ranked.shape[0]),
        "num_users": int(ranked["user_id"].nunique()) if not ranked.empty else 0,
        "positive_any": int(ranked["rank_label"].sum()) if "rank_label" in ranked.columns else 0,
    }
    for task in tasks:
        metrics[f"{task}_auc"] = _safe_auc(ranked[task].to_numpy(), ranked[f"{task}_score"].to_numpy())
        metrics[f"{task}_logloss"] = _safe_logloss(ranked[task].to_numpy(), ranked[f"{task}_score"].to_numpy())
    metrics["rank_auc"] = _safe_auc(ranked["rank_label"].to_numpy(), ranked["rank_score"].to_numpy())
    topks = [int(k) for k in cfg["rank"]["eval"].get("topk", [10, 20])]
    grouped = ranked.sort_values(["user_id", "rank_position"]).groupby("user_id", sort=False)
    for k in topks:
        metrics[f"ndcg@{k}"] = float(np.mean([_ndcg_at_k(group, k) for _, group in grouped])) if metrics["num_users"] else 0.0
    mrr_k = int(cfg["rank"]["eval"].get("mrr_k", 10))
    grouped = ranked.sort_values(["user_id", "rank_position"]).groupby("user_id", sort=False)
    metrics[f"mrr@{mrr_k}"] = float(np.mean([_mrr_at_k(group, mrr_k) for _, group in grouped])) if metrics["num_users"] else 0.0
    return metrics


def load_rank_checkpoint(cfg: Mapping[str, Any], device: str) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    ckpt_path = rank_dir(cfg) / cfg["rank"]["output"]["best_model_file"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Rank checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    spec = checkpoint["feature_spec"]
    merged_cfg = dict(cfg)
    merged_cfg["rank"] = checkpoint.get("cfg_rank", cfg["rank"])
    model = build_model(spec, merged_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, spec, merged_cfg


def predict_rank_scores(
    frame: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    spec: Any,
    model: torch.nn.Module,
    cfg: Mapping[str, Any],
    device: str,
) -> pd.DataFrame:
    ds = RankTensorDataset(arrays, spec, include_labels=True)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["rank"]["inference"].get("predict_batch_size", cfg["rank"]["model"].get("batch_size", 4096))),
        shuffle=False,
        num_workers=int(cfg["rank"]["model"].get("num_workers", 0)),
        pin_memory=device == "cuda",
    )
    tasks = list(spec.tasks)
    preds_by_task = {task: [] for task in tasks}
    rank_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            preds = model(batch)
            for task in tasks:
                preds_by_task[task].append(torch.sigmoid(preds[task]).detach().cpu().numpy())
            rank_scores.append(compute_rank_score(preds, cfg).detach().cpu().numpy())

    out = frame.copy()
    for task in tasks:
        out[f"{task}_score"] = np.concatenate(preds_by_task[task]).astype("float32")
    out["rank_score"] = np.concatenate(rank_scores).astype("float32")
    return out


def generate_ranked_candidates(
    cfg: Mapping[str, Any],
    splits: list[str] | None = None,
    candidate_rows: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    output_cfg = cfg["rank"]["output"]
    out_dir = rank_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out_dir = metrics_dir(cfg)
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(cfg["rank"]["model"].get("device", "auto")))
    model, spec, ckpt_cfg = load_rank_checkpoint(cfg, device=device)
    processed_tables = load_processed_tables(cfg)
    store = build_feature_store(ckpt_cfg, processed_tables, item_encoder=spec.encoders["video_id"])
    splits = splits or list(cfg["rank"]["inference"].get("splits", ["val", "test"]))
    batch_rows = int(cfg["rank"]["inference"].get("candidate_batch_rows", 300000))

    all_metrics: dict[str, Any] = {}
    for split in splits:
        candidates = load_prerank_topk(cfg, split, nrows=candidate_rows)
        split_df = load_split(cfg, split)
        split_labels = build_task_label_table(split_df, [str(x) for x in ckpt_cfg["rank"]["tasks"]])
        scored_parts: list[pd.DataFrame] = []
        for idx, chunk in enumerate(_iter_chunks(candidates, batch_rows=batch_rows)):
            frame = build_rank_frame(
                chunk,
                split_df,
                store,
                ckpt_cfg,
                split=split,
                precomputed_labels=split_labels,
            )
            arrays = transform_rank_frame(frame, spec, store, ckpt_cfg, include_labels=True)
            scored = predict_rank_scores(frame, arrays, spec, model, ckpt_cfg, device=device)
            scored_parts.append(scored)
            if logger:
                logger.info("Ranked %s chunk %d rows=%d device=%s", split, idx + 1, len(chunk), device)

        ranked = pd.concat(scored_parts, ignore_index=True) if scored_parts else candidates.iloc[0:0].copy()
        keep_cols = [col for col in cfg["rank"]["inference"].get("keep_columns", []) if col in ranked.columns]
        for col in ["rank_label", *spec.tasks, *[f"{task}_score" for task in spec.tasks], "rank_score"]:
            if col not in keep_cols and col in ranked.columns:
                keep_cols.append(col)
        ranked = ranked[keep_cols].sort_values(
            ["user_id", "rank_score", "coarse_score", "video_id"],
            ascending=[True, False, False, True],
        )
        ranked["rank_position"] = ranked.groupby("user_id").cumcount() + 1
        out_path = out_dir / output_cfg[f"{split}_ranked_file"]
        ranked.to_parquet(out_path, index=False)
        all_metrics[split] = evaluate_ranked(ranked, ckpt_cfg)
        if logger:
            logger.info("Saved %s ranked candidates: %s rows=%d", split, out_path, ranked.shape[0])

    metrics_path = metrics_out_dir / output_cfg["metrics_file"]
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing["inference"] = all_metrics
    metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved rank inference metrics: %s", metrics_path)
    return all_metrics
