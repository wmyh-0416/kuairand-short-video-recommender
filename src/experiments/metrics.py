from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from src.experiments.policy import PolicyItem


@dataclass
class GroundTruth:
    positive_items: dict[int, set[int]]
    long_view_items: dict[int, set[int]]
    like_items: dict[int, set[int]]
    click_items: dict[int, set[int]]
    impression_counts: dict[int, int]
    catalog_size: int
    item_tag_map: dict[int, str]


def _dcg(binary_relevance: np.ndarray) -> float:
    if binary_relevance.size == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, binary_relevance.size + 2))
    return float((binary_relevance * discount).sum())


def _ndcg_at_k(recommended_ids: list[int], positives: set[int], k: int) -> float:
    if not recommended_ids or not positives or k <= 0:
        return 0.0
    top_items = recommended_ids[:k]
    rel = np.asarray([1.0 if video_id in positives else 0.0 for video_id in top_items], dtype=np.float32)
    dcg = _dcg(rel)
    ideal_hits = min(len(positives), k)
    ideal = np.ones(ideal_hits, dtype=np.float32)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _rate_at_k(recommended_ids: list[int], positives: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_items = recommended_ids[:k]
    if not top_items:
        return 0.0
    hits = sum(1 for video_id in top_items if video_id in positives)
    return float(hits / k)


def _hit_rate_at_k(recommended_ids: list[int], positives: set[int], k: int) -> float:
    top_items = recommended_ids[:k]
    if not top_items:
        return 0.0
    return float(any(video_id in positives for video_id in top_items))


def _recall_at_k(recommended_ids: list[int], positives: set[int], k: int) -> float:
    if not positives:
        return 0.0
    top_items = recommended_ids[:k]
    hits = sum(1 for video_id in top_items if video_id in positives)
    return float(hits / max(len(positives), 1))


def _precision_at_k(recommended_ids: list[int], positives: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_items = recommended_ids[:k]
    if not top_items:
        return 0.0
    hits = sum(1 for video_id in top_items if video_id in positives)
    return float(hits / k)


def _category_diversity_at_k(recommended_ids: list[int], item_tag_map: Mapping[int, str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_items = recommended_ids[:k]
    if not top_items:
        return 0.0
    tags = {item_tag_map.get(int(video_id)) for video_id in top_items if item_tag_map.get(int(video_id)) is not None}
    return float(len(tags) / max(len(top_items), 1))


def evaluate_group(
    recommendations_by_user: Mapping[int, list[PolicyItem]],
    users: list[int],
    ground_truth: GroundTruth,
    topks: list[int],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "users": int(len(users)),
        "impressions": int(sum(int(ground_truth.impression_counts.get(int(user_id), 0)) for user_id in users)),
    }
    per_user_metrics: dict[str, list[float]] = {}

    max_top_k = max(topks) if topks else 0
    coverage_sets: dict[int, set[int]] = {int(k): set() for k in topks}
    score_values: list[float] = []

    for user_id in users:
        recs = recommendations_by_user.get(int(user_id), [])
        recommended_ids = [int(item.video_id) for item in recs]
        positive_set = ground_truth.positive_items.get(int(user_id), set())
        long_view_set = ground_truth.long_view_items.get(int(user_id), set())
        like_set = ground_truth.like_items.get(int(user_id), set())

        for item in recs[:max_top_k]:
            if item.score is not None:
                score_values.append(float(item.score))

        for k in topks:
            top_items = recommended_ids[: int(k)]
            coverage_sets[int(k)].update(top_items)

            metric_values = {
                f"hit_rate@{k}": _hit_rate_at_k(recommended_ids, positive_set, int(k)),
                f"recall@{k}": _recall_at_k(recommended_ids, positive_set, int(k)),
                f"ndcg@{k}": _ndcg_at_k(recommended_ids, positive_set, int(k)),
                f"precision@{k}": _precision_at_k(recommended_ids, positive_set, int(k)),
                f"long_view_rate@{k}": _rate_at_k(recommended_ids, long_view_set, int(k)),
                f"like_rate@{k}": _rate_at_k(recommended_ids, like_set, int(k)),
                f"category_diversity@{k}": _category_diversity_at_k(recommended_ids, ground_truth.item_tag_map, int(k)),
            }
            for metric_name, metric_value in metric_values.items():
                per_user_metrics.setdefault(metric_name, []).append(float(metric_value))

    for metric_name, values in per_user_metrics.items():
        metrics[metric_name] = _mean(values)

    for k in topks:
        metrics[f"coverage@{k}"] = float(len(coverage_sets[int(k)]) / max(ground_truth.catalog_size, 1))

    metrics["average_score"] = _mean(score_values) if score_values else None
    return {"metrics": metrics, "per_user_metrics": per_user_metrics}
