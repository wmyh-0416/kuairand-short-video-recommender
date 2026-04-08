from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["user_id", "video_id", "recall_source", "source_score", "source_rank"]


def _validate_candidate_frame(df: pd.DataFrame, name: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{name} candidates missing columns: {missing}")


def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source scores per recall channel to make merge weights usable."""
    out = df.copy()
    out["source_score"] = pd.to_numeric(out["source_score"], errors="coerce").fillna(0.0)

    normalized_parts: list[pd.DataFrame] = []
    for source, group in out.groupby("recall_source", sort=False):
        group = group.copy()
        min_score = float(group["source_score"].min())
        max_score = float(group["source_score"].max())
        if max_score > min_score:
            group["source_score_norm"] = (group["source_score"] - min_score) / (max_score - min_score)
        else:
            # If all scores are identical, rank is still informative enough.
            group["source_score_norm"] = 1.0 / np.log2(group["source_rank"].astype(float) + 1.0)
        normalized_parts.append(group)

    if not normalized_parts:
        out["source_score_norm"] = pd.Series(dtype="float32")
        return out
    out = pd.concat(normalized_parts, ignore_index=True)
    out["source_score_norm"] = out["source_score_norm"].astype("float32")
    return out


def apply_source_quota(
    candidates: pd.DataFrame,
    per_source_topk: Mapping[str, int],
) -> pd.DataFrame:
    """Keep at most K items per user and recall source before cross-source merge."""
    if candidates.empty:
        return candidates.copy()

    parts: list[pd.DataFrame] = []
    for source, group in candidates.groupby("recall_source", sort=False):
        quota = int(per_source_topk.get(str(source), per_source_topk.get("default", 10**9)))
        if quota <= 0:
            continue
        group = group.sort_values(
            ["user_id", "source_score", "source_rank", "video_id"],
            ascending=[True, False, True, True],
        )
        parts.append(group.groupby("user_id", as_index=False, group_keys=False).head(quota))

    if not parts:
        return candidates.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def merge_candidates(
    candidate_frames: list[pd.DataFrame],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Merge multi-source recall candidates into one candidate table.

    Duplicate (user, item) pairs from multiple sources are collapsed to one row.
    `recall_source` is preserved as a pipe-joined source list, while per-source
    original scores are also kept in `{source}_score` columns.
    """
    valid_frames: list[pd.DataFrame] = []
    for idx, frame in enumerate(candidate_frames):
        if frame is None or frame.empty:
            continue
        frame = frame.copy()
        _validate_candidate_frame(frame, name=f"candidate_frames[{idx}]")
        valid_frames.append(frame[REQUIRED_COLUMNS])

    if not valid_frames:
        return pd.DataFrame(
            columns=[
                "user_id",
                "video_id",
                "recall_source",
                "source_score",
                "source_rank",
                "merged_score",
                "source_count",
            ]
        )

    merge_cfg = cfg["recall"]["merge"]
    source_weights = deepcopy(merge_cfg.get("source_weights", {}))
    per_source_topk = merge_cfg.get("per_source_topk", {})
    final_topk = int(merge_cfg.get("final_topk", 500))

    raw = pd.concat(valid_frames, ignore_index=True)
    raw["user_id"] = raw["user_id"].astype("int64")
    raw["video_id"] = raw["video_id"].astype("int64")
    raw["recall_source"] = raw["recall_source"].astype("string")
    raw["source_rank"] = pd.to_numeric(raw["source_rank"], errors="coerce").fillna(10**9).astype("int64")
    raw = apply_source_quota(raw, per_source_topk=per_source_topk)
    raw = _normalize_scores(raw)
    raw["source_weight"] = raw["recall_source"].map(lambda s: float(source_weights.get(str(s), 1.0)))
    raw["weighted_score"] = raw["source_score_norm"] * raw["source_weight"]

    source_score_pivot = (
        raw.pivot_table(
            index=["user_id", "video_id"],
            columns="recall_source",
            values="source_score",
            aggfunc="max",
        )
        .rename(columns=lambda col: f"{col}_score")
        .reset_index()
    )
    source_rank_pivot = (
        raw.pivot_table(
            index=["user_id", "video_id"],
            columns="recall_source",
            values="source_rank",
            aggfunc="min",
        )
        .rename(columns=lambda col: f"{col}_rank")
        .reset_index()
    )

    merged = (
        raw.groupby(["user_id", "video_id"], as_index=False)
        .agg(
            recall_source=("recall_source", lambda s: "|".join(sorted(set(str(x) for x in s)))),
            source_score=("source_score", "max"),
            source_rank=("source_rank", "min"),
            merged_score=("weighted_score", "sum"),
            source_count=("recall_source", "nunique"),
        )
        .merge(source_score_pivot, on=["user_id", "video_id"], how="left")
        .merge(source_rank_pivot, on=["user_id", "video_id"], how="left")
    )
    merged["merged_score"] = merged["merged_score"].astype("float32")

    merged = merged.sort_values(
        ["user_id", "merged_score", "source_count", "source_rank", "video_id"],
        ascending=[True, False, False, True, True],
    )
    merged = merged.groupby("user_id", as_index=False, group_keys=False).head(final_topk)
    merged["merged_rank"] = merged.groupby("user_id").cumcount() + 1
    return merged.reset_index(drop=True)
