from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


def train_popular_recall(
    train_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build a global hot-video table from the train split.

    Each row in KuaiRand logs is an exposure/play event candidate. We use a
    weighted combination of play, like, long-watch, and finish counts to produce
    a robust popularity score for cold users and fallback recall.
    """
    popular_cfg = cfg["recall"]["popular"]
    weights = popular_cfg.get("weights", {})
    smoothing = float(popular_cfg.get("smoothing", 1.0))

    df = train_df.copy()
    df["play_event"] = (pd.to_numeric(df["play_time_ms"], errors="coerce").fillna(0) > 0).astype("int32")
    df["like_event"] = pd.to_numeric(df.get("like", df.get("is_like", 0)), errors="coerce").fillna(0).astype("int32")
    df["long_watch_event"] = pd.to_numeric(df.get("long_watch", 0), errors="coerce").fillna(0).astype("int32")
    df["finish_event"] = pd.to_numeric(df.get("finish", 0), errors="coerce").fillna(0).astype("int32")

    agg = (
        df.groupby("video_id", as_index=False)
        .agg(
            exposure_count=("video_id", "size"),
            play_count=("play_event", "sum"),
            like_count=("like_event", "sum"),
            long_watch_count=("long_watch_event", "sum"),
            finish_count=("finish_event", "sum"),
            unique_user_count=("user_id", "nunique"),
            last_time_ms=("time_ms", "max"),
        )
        .astype(
            {
                "video_id": "int64",
                "exposure_count": "int64",
                "play_count": "int64",
                "like_count": "int64",
                "long_watch_count": "int64",
                "finish_count": "int64",
                "unique_user_count": "int64",
                "last_time_ms": "int64",
            }
        )
    )

    score = (
        float(weights.get("play_count", 1.0)) * np.log1p(agg["play_count"] + smoothing)
        + float(weights.get("like_count", 6.0)) * np.log1p(agg["like_count"] + smoothing)
        + float(weights.get("long_watch_count", 3.0)) * np.log1p(agg["long_watch_count"] + smoothing)
        + float(weights.get("finish_count", 2.0)) * np.log1p(agg["finish_count"] + smoothing)
    )
    # A small exposure term keeps stable high-traffic videos above sparse noisy
    # events, without making pure exposure dominate engagement.
    score = score + 0.05 * np.log1p(agg["exposure_count"])

    agg["source_score"] = score.astype("float32")
    agg = agg.sort_values(
        ["source_score", "long_watch_count", "like_count", "play_count", "video_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    agg["popular_rank"] = np.arange(1, len(agg) + 1, dtype=np.int32)

    topk = int(popular_cfg.get("topk", 300))
    out = agg.head(topk).copy()
    if logger:
        logger.info(
            "Trained popular recall: items=%d topk=%d best_score=%.4f",
            agg.shape[0],
            out.shape[0],
            float(out["source_score"].iloc[0]) if not out.empty else 0.0,
        )
    return out


def save_popular_items(
    popular_items: pd.DataFrame,
    path: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    popular_items.to_parquet(path, index=False)
    if logger:
        logger.info("Saved popular recall table: %s rows=%d", path, popular_items.shape[0])
    return path


def load_popular_items(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path).expanduser().resolve())


def _build_seen_map(history_df: pd.DataFrame) -> dict[int, set[int]]:
    if history_df.empty:
        return {}
    seen = (
        history_df.groupby("user_id")["video_id"]
        .agg(lambda s: set(int(v) for v in s))
        .to_dict()
    )
    return {int(k): v for k, v in seen.items()}


def generate_popular_candidates(
    user_ids: Iterable[int],
    popular_items: pd.DataFrame,
    topk: int,
    history_df: pd.DataFrame | None = None,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """Generate the same hot item list for each user, optionally removing seen items."""
    top_items = popular_items.sort_values("popular_rank").head(max(topk * 3, topk)).copy()
    item_ids = top_items["video_id"].astype("int64").to_numpy()
    scores = top_items["source_score"].astype("float32").to_numpy()
    ranks = top_items["popular_rank"].astype("int32").to_numpy()
    seen_map = _build_seen_map(history_df) if exclude_seen and history_df is not None else {}

    rows: list[dict[str, object]] = []
    for user_id_raw in user_ids:
        user_id = int(user_id_raw)
        seen = seen_map.get(user_id, set())
        kept = 0
        for video_id, score, rank in zip(item_ids, scores, ranks):
            video_id_int = int(video_id)
            if video_id_int in seen:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "video_id": video_id_int,
                    "recall_source": "popular",
                    "source_score": float(score),
                    "source_rank": int(rank),
                }
            )
            kept += 1
            if kept >= topk:
                break

    return pd.DataFrame(rows)
