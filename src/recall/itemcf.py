from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


NeighborMap = dict[int, list[tuple[int, float]]]


def _dedupe_keep_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_positive_sequences(
    train_df: pd.DataFrame,
    label_col: str = "is_positive",
    max_user_history_len: int = 100,
) -> dict[int, list[int]]:
    """Build per-user positive item sequences ordered by event time."""
    if label_col not in train_df.columns:
        raise KeyError(f"Missing positive label column for ItemCF: {label_col}")

    pos = train_df.loc[train_df[label_col] > 0, ["user_id", "video_id", "time_ms"]].copy()
    pos = pos.sort_values(["user_id", "time_ms", "video_id"])
    sequences: dict[int, list[int]] = {}
    for user_id, group in pos.groupby("user_id", sort=False):
        seq = _dedupe_keep_order(group["video_id"].tail(max_user_history_len).tolist())
        if seq:
            sequences[int(user_id)] = seq
    return sequences


def train_itemcf(
    train_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Train normalized item-item co-occurrence recall.

    Similarity:
        sim(i, j) = cooc(i, j) / (sqrt(freq(i) * freq(j)) + shrinkage)

    Each user contributes 1 / log(2 + history_len) to each pair to avoid very
    long histories dominating the item graph.
    """
    itemcf_cfg = cfg["recall"]["itemcf"]
    label_col = cfg["recall"].get("positive_label_col", "is_positive")
    max_user_history_len = int(itemcf_cfg.get("max_user_history_len", 100))
    max_neighbors = int(itemcf_cfg.get("max_neighbors_per_item", 200))
    min_cooccurrence = float(itemcf_cfg.get("min_cooccurrence", 2.0))
    shrinkage = float(itemcf_cfg.get("shrinkage", 10.0))

    sequences = build_positive_sequences(
        train_df=train_df,
        label_col=label_col,
        max_user_history_len=max_user_history_len,
    )

    item_freq: Counter[int] = Counter()
    cooc: dict[int, Counter[int]] = defaultdict(Counter)

    for seq in sequences.values():
        if len(seq) < 2:
            for item in seq:
                item_freq[item] += 1
            continue

        unique_items = seq[-max_user_history_len:]
        weight = 1.0 / math.log2(2.0 + len(unique_items))
        for item_i in unique_items:
            item_freq[item_i] += 1
        for item_i in unique_items:
            row = cooc[item_i]
            for item_j in unique_items:
                if item_i == item_j:
                    continue
                row[item_j] += weight

    rows: list[dict[str, object]] = []
    for item_i, neighbors in cooc.items():
        scored: list[tuple[int, float, float]] = []
        freq_i = item_freq[item_i]
        for item_j, co_count in neighbors.items():
            if co_count < min_cooccurrence:
                continue
            denom = math.sqrt(freq_i * item_freq[item_j]) + shrinkage
            sim = float(co_count / denom) if denom > 0 else 0.0
            if sim > 0:
                scored.append((item_j, sim, float(co_count)))

        scored.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True)
        for rank, (item_j, sim, co_count) in enumerate(scored[:max_neighbors], start=1):
            rows.append(
                {
                    "video_id": int(item_i),
                    "neighbor_video_id": int(item_j),
                    "similarity": float(sim),
                    "cooccurrence": float(co_count),
                    "item_freq": int(freq_i),
                    "neighbor_freq": int(item_freq[item_j]),
                    "neighbor_rank": int(rank),
                }
            )

    neighbors_df = pd.DataFrame(rows)
    if logger:
        logger.info(
            "Trained ItemCF: users=%d source_items=%d edges=%d",
            len(sequences),
            neighbors_df["video_id"].nunique() if not neighbors_df.empty else 0,
            neighbors_df.shape[0],
        )
    return neighbors_df


def save_itemcf_neighbors(
    neighbors_df: pd.DataFrame,
    path: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    neighbors_df.to_parquet(path, index=False)
    if logger:
        logger.info("Saved ItemCF neighbors: %s rows=%d", path, neighbors_df.shape[0])
    return path


def load_itemcf_neighbors(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path).expanduser().resolve())


def neighbors_to_map(neighbors_df: pd.DataFrame) -> NeighborMap:
    if neighbors_df.empty:
        return {}
    neighbor_map: NeighborMap = {}
    sorted_df = neighbors_df.sort_values(["video_id", "neighbor_rank"])
    for item_id, group in sorted_df.groupby("video_id", sort=False):
        neighbor_map[int(item_id)] = [
            (int(row.neighbor_video_id), float(row.similarity))
            for row in group.itertuples(index=False)
        ]
    return neighbor_map


def _sequence_value_to_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    return []


def generate_itemcf_candidates(
    user_sequences: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    topk: int,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    itemcf_cfg = cfg["recall"]["itemcf"]
    sequence_col = itemcf_cfg.get("sequence_col", "watch_seq")
    max_user_history_len = int(itemcf_cfg.get("max_user_history_len", 100))
    recency_alpha = float(itemcf_cfg.get("recency_alpha", 0.92))
    exclude_seen = bool(itemcf_cfg.get("exclude_seen", True))

    if sequence_col not in user_sequences.columns:
        raise KeyError(f"Missing sequence column for ItemCF: {sequence_col}")

    neighbor_map = neighbors_to_map(neighbors_df)
    rows: list[dict[str, object]] = []

    for row in user_sequences[["user_id", sequence_col]].itertuples(index=False):
        user_id = int(row.user_id)
        seq = _sequence_value_to_list(getattr(row, sequence_col))[-max_user_history_len:]
        if not seq:
            continue

        seen = set(seq) if exclude_seen else set()
        scores: defaultdict[int, float] = defaultdict(float)
        ranks_from_history = list(reversed(seq))
        for hist_rank, hist_item in enumerate(ranks_from_history):
            recency_weight = recency_alpha ** hist_rank
            for neighbor_item, sim in neighbor_map.get(int(hist_item), []):
                if neighbor_item in seen:
                    continue
                scores[neighbor_item] += float(sim) * recency_weight

        ranked = sorted(scores.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)[:topk]
        for rank, (video_id, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "video_id": int(video_id),
                    "recall_source": "itemcf",
                    "source_score": float(score),
                    "source_rank": int(rank),
                }
            )

    return pd.DataFrame(rows)
