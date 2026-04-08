from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.rerank.rules import rerank_candidates
from src.utils.paths import artifacts_dir, processed_path


def rerank_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["rerank"]["output"]["rerank_dir"]


def metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["rerank"]["output"].get("metrics_dir", "metrics")


def rank_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["rerank"]["input"]["rank_dir"]


def _reference_date(cfg: Mapping[str, Any], split: str) -> pd.Timestamp:
    value = cfg["rerank"]["split_reference_dates"][split]
    return pd.to_datetime(str(int(value)), format="%Y%m%d")


def load_ranked_split(cfg: Mapping[str, Any], split: str, nrows: int | None = None) -> pd.DataFrame:
    path = rank_dir(cfg) / cfg["rerank"]["input"][f"{split}_ranked_file"]
    if not path.exists():
        raise FileNotFoundError(f"Ranked file not found: {path}")
    df = pd.read_parquet(path)
    if nrows is not None:
        df = df.head(nrows).copy()
    return df


def _prepare_item_meta(cfg: Mapping[str, Any], split: str) -> pd.DataFrame:
    item_path = processed_path(cfg, cfg["rerank"]["processed"]["item_features_file"])
    item = pd.read_parquet(item_path, columns=["video_id", "author_id", "tag", "upload_date"])
    ref_date = _reference_date(cfg, split)
    upload = pd.to_datetime(item["upload_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    item["freshness_days"] = (ref_date - upload).dt.days.clip(lower=0).fillna(9999).astype("float32")
    return item.drop_duplicates("video_id")


def prepare_ranked_for_rerank(cfg: Mapping[str, Any], split: str, ranked: pd.DataFrame) -> pd.DataFrame:
    item_meta = _prepare_item_meta(cfg, split)
    out = ranked.merge(item_meta, on="video_id", how="left")
    out["author_id"] = out["author_id"].fillna(-1)
    out["tag"] = out["tag"].fillna(-1)
    out["freshness_days"] = pd.to_numeric(out["freshness_days"], errors="coerce").fillna(9999).astype("float32")
    out["rank_score"] = pd.to_numeric(out["rank_score"], errors="coerce").fillna(0.0).astype("float32")
    if "coarse_score" not in out.columns:
        out["coarse_score"] = 0.0
    if "merged_score" not in out.columns:
        out["merged_score"] = 0.0
    return out


def _ndcg_at_k(group: pd.DataFrame, rank_col: str, k: int) -> float:
    ordered = group.sort_values(rank_col).head(k)
    rel = ordered["rank_label"].to_numpy(dtype=float)
    if rel.size == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discount).sum())
    ideal = np.sort(group["rank_label"].to_numpy(dtype=float))[::-1][:k]
    idcg = float((ideal * discount[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(group: pd.DataFrame, rank_col: str, k: int) -> float:
    ordered = group.sort_values(rank_col).head(k)
    rel = ordered["rank_label"].to_numpy(dtype=int)
    hits = np.where(rel > 0)[0]
    return float(1.0 / (hits[0] + 1)) if hits.size else 0.0


def _adjacent_repeat_rate(df: pd.DataFrame, key: str, rank_col: str) -> float:
    grouped = df.sort_values(["user_id", rank_col]).groupby("user_id", sort=False)[key]
    values = [(series.shift() == series).fillna(False).mean() for _, series in grouped]
    return float(np.mean(values)) if values else 0.0


def compute_list_metrics(
    full_ranked_df: pd.DataFrame,
    list_df: pd.DataFrame,
    rank_col: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    topks = [int(k) for k in cfg["rerank"].get("eval_topk", [10, 20])]
    mrr_k = int(cfg["rerank"].get("mrr_k", 10))
    total_positives = int(full_ranked_df["rank_label"].sum())
    metrics: dict[str, Any] = {
        "num_rows": int(list_df.shape[0]),
        "num_users": int(list_df["user_id"].nunique()) if not list_df.empty else 0,
        "positive_hits": int(list_df["rank_label"].sum()) if "rank_label" in list_df.columns else 0,
        "author_coverage_all": float(list_df["author_id"].nunique() / max(full_ranked_df["author_id"].nunique(), 1)),
        "tag_coverage_all": float(list_df["tag"].nunique() / max(full_ranked_df["tag"].nunique(), 1)),
        "avg_unique_authors_per_user": float(list_df.groupby("user_id")["author_id"].nunique().mean()) if not list_df.empty else 0.0,
        "avg_unique_tags_per_user": float(list_df.groupby("user_id")["tag"].nunique().mean()) if not list_df.empty else 0.0,
        "adjacent_same_author_rate": _adjacent_repeat_rate(list_df, "author_id", rank_col) if not list_df.empty else 0.0,
        "adjacent_same_tag_rate": _adjacent_repeat_rate(list_df, "tag", rank_col) if not list_df.empty else 0.0,
        "avg_freshness_days": float(list_df["freshness_days"].mean()) if not list_df.empty else None,
        "median_freshness_days": float(list_df["freshness_days"].median()) if not list_df.empty else None,
        "p90_freshness_days": float(list_df["freshness_days"].quantile(0.9)) if not list_df.empty else None,
        "fresh_share_leq_threshold": float(
            (list_df["freshness_days"] <= int(cfg["rerank"]["freshness"].get("fresh_threshold_days", 20))).mean()
        )
        if not list_df.empty
        else None,
    }

    grouped_full = full_ranked_df.groupby("user_id", sort=False)
    grouped_list = list_df.groupby("user_id", sort=False)
    for k in topks:
        current = list_df.sort_values(["user_id", rank_col]).groupby("user_id", as_index=False, group_keys=False).head(k)
        metrics[f"recall@{k}"] = float(current["rank_label"].sum() / max(total_positives, 1))
        metrics[f"ndcg@{k}"] = float(
            np.mean([_ndcg_at_k(group, rank_col, k) for _, group in grouped_list])
        ) if metrics["num_users"] else 0.0
    metrics[f"mrr@{mrr_k}"] = float(
        np.mean([_mrr_at_k(group, rank_col, mrr_k) for _, group in grouped_list])
    ) if metrics["num_users"] else 0.0
    return metrics


def evaluate_before_after(
    full_ranked_df: pd.DataFrame,
    final_df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    topk = int(cfg["rerank"].get("topk", 20))
    before_df = (
        full_ranked_df.sort_values(["user_id", "rank_position"])
        .groupby("user_id", as_index=False, group_keys=False)
        .head(topk)
        .copy()
    )
    before_metrics = compute_list_metrics(full_ranked_df, before_df, rank_col="rank_position", cfg=cfg)
    after_metrics = compute_list_metrics(full_ranked_df, final_df, rank_col="final_rank", cfg=cfg)
    delta = {
        key: float(after_metrics[key] - before_metrics[key])
        for key in before_metrics
        if key in after_metrics and isinstance(before_metrics[key], (int, float)) and before_metrics[key] is not None and after_metrics[key] is not None
    }
    return {"before": before_metrics, "after": after_metrics, "delta": delta}


def run_rerank_for_splits(
    cfg: Mapping[str, Any],
    splits: list[str] | None = None,
    candidate_rows: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    splits = splits or list(cfg["rerank"]["inference"].get("splits", ["val", "test"]))
    out_dir = rerank_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = metrics_dir(cfg)
    metric_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, Any] = {}
    for split in splits:
        ranked = load_ranked_split(cfg, split, nrows=candidate_rows)
        prepared = prepare_ranked_for_rerank(cfg, split, ranked)
        final_df = rerank_candidates(prepared, cfg, logger=logger)
        final_df = final_df.sort_values(["user_id", "final_rank"]).reset_index(drop=True)

        keep_cols = [col for col in cfg["rerank"]["inference"].get("keep_columns", []) if col in final_df.columns]
        extra_cols = [
            "author_id",
            "tag",
            "freshness_days",
            "rerank_score",
            "score_multiplier",
            "author_penalty",
            "tag_penalty",
            "freshness_bonus",
            "new_author_bonus",
            "new_tag_bonus",
            "adjustment_reason",
            "final_rank",
        ]
        for col in extra_cols:
            if col not in keep_cols and col in final_df.columns:
                keep_cols.append(col)
        final_df = final_df[keep_cols]

        out_path = out_dir / cfg["rerank"]["output"][f"{split}_final_file"]
        final_df.to_parquet(out_path, index=False)
        all_metrics[split] = evaluate_before_after(prepared, final_df, cfg)
        if logger:
            logger.info("Saved %s final rerank file: %s rows=%d", split, out_path, final_df.shape[0])

    metrics_path = metric_dir / cfg["rerank"]["output"]["metrics_file"]
    metrics_path.write_text(json.dumps(all_metrics, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved rerank metrics: %s", metrics_path)
    return all_metrics


def build_pipeline_report(
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    metric_dir = metrics_dir(cfg)
    recall_path = metric_dir / "recall_metrics.json"
    prerank_path = metric_dir / "prerank_metrics.json"
    rank_path = metric_dir / "rank_metrics.json"
    rerank_path = metric_dir / cfg["rerank"]["output"]["metrics_file"]

    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    recall = _load(recall_path)
    prerank = _load(prerank_path)
    rank = _load(rank_path)
    rerank = _load(rerank_path)

    summary: dict[str, Any] = {
        "recall": {},
        "prerank": {},
        "rank": {},
        "rerank": rerank,
        "pipeline_summary": {},
    }
    for split in ["val", "test"]:
        summary["recall"][split] = {
            "num_candidates": recall.get(split, {}).get("num_candidates"),
            "recall@50": recall.get(split, {}).get("recall@50"),
            "recall@100": recall.get(split, {}).get("recall@100"),
            "recall@200": recall.get(split, {}).get("recall@200"),
            "coverage_all": recall.get(split, {}).get("coverage_all"),
        }
        summary["prerank"][split] = {
            "output_candidates": prerank.get("inference", {}).get(split, {}).get("output_candidates"),
            "candidate_compression_ratio": prerank.get("inference", {}).get(split, {}).get("candidate_compression_ratio"),
            "recall_retained@100": prerank.get("inference", {}).get(split, {}).get("recall_retained@100"),
        }
        summary["rank"][split] = rank.get("inference", {}).get(split, {})
        summary["pipeline_summary"][split] = {
            "recall_candidates": summary["recall"][split].get("num_candidates"),
            "prerank_top100": summary["prerank"][split].get("output_candidates"),
            "rank_candidates": rank.get("inference", {}).get(split, {}).get("num_candidates"),
            "recall@100": summary["recall"][split].get("recall@100"),
            "prerank_recall_retained@100": summary["prerank"][split].get("recall_retained@100"),
            "rank_ndcg@20": rank.get("inference", {}).get(split, {}).get("ndcg@20"),
            "rerank_before_ndcg@20": rerank.get(split, {}).get("before", {}).get("ndcg@20"),
            "rerank_after_ndcg@20": rerank.get(split, {}).get("after", {}).get("ndcg@20"),
            "rerank_delta_ndcg@20": rerank.get(split, {}).get("delta", {}).get("ndcg@20"),
            "rerank_before_avg_unique_tags_per_user": rerank.get(split, {}).get("before", {}).get("avg_unique_tags_per_user"),
            "rerank_after_avg_unique_tags_per_user": rerank.get(split, {}).get("after", {}).get("avg_unique_tags_per_user"),
            "rerank_before_avg_freshness_days": rerank.get(split, {}).get("before", {}).get("avg_freshness_days"),
            "rerank_after_avg_freshness_days": rerank.get(split, {}).get("after", {}).get("avg_freshness_days"),
        }

    out_path = metric_dir / cfg["rerank"]["output"]["pipeline_report_file"]
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved pipeline report: %s", out_path)
    return summary
