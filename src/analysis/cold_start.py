from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.metrics import GroundTruth, evaluate_group
from src.experiments.policy import PolicyContext, PolicyItem, build_policy
from src.utils.paths import artifact_path, artifacts_dir, ensure_dir, processed_dir


USER_SEGMENT_ORDER = [
    "new_user",
    "low_active_user",
    "medium_active_user",
    "high_active_user",
]

ITEM_SEGMENT_ORDER = [
    "new_item",
    "low_exposure_item",
    "medium_exposure_item",
    "popular_item",
]


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except OSError:
        return False


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    return value


def _relative_lift(base: Any, new: Any) -> float | None:
    if base is None or new is None:
        return None
    try:
        base_value = float(base)
        new_value = float(new)
    except (TypeError, ValueError):
        return None
    if abs(base_value) < 1e-12:
        return None
    return float((new_value - base_value) / base_value)


def _recommendation_source_label(policy_metadata: Mapping[str, Any]) -> dict[str, Any]:
    available = [entry for entry in policy_metadata.get("artifacts", []) if entry.get("available")]
    primary = available[0]["label"] if available else None
    backfill = [entry["label"] for entry in available[1:]]
    return {
        "primary_artifact": primary,
        "backfill_artifacts": backfill,
        "artifacts": available,
        "missing_artifacts": list(policy_metadata.get("missing_artifacts", [])),
    }


def _build_ground_truth(test_df: pd.DataFrame, item_tag_map: Mapping[int, str], positive_label: str) -> GroundTruth:
    if positive_label in test_df.columns:
        positive_mask = test_df[positive_label].fillna(0).astype(int) > 0
    elif "is_positive" in test_df.columns:
        positive_mask = test_df["is_positive"].fillna(0).astype(int) > 0
    else:
        positive_mask = pd.Series(False, index=test_df.index)
        for col in ["long_watch", "finish", "like", "is_click"]:
            if col in test_df.columns:
                positive_mask = positive_mask | (test_df[col].fillna(0).astype(int) > 0)

    if "long_watch" in test_df.columns:
        long_view_mask = test_df["long_watch"].fillna(0).astype(int) > 0
    elif "long_view" in test_df.columns:
        long_view_mask = test_df["long_view"].fillna(0).astype(int) > 0
    elif {"play_time_ms", "duration_ms"} <= set(test_df.columns):
        ratio = test_df["play_time_ms"] / test_df["duration_ms"].replace(0, np.nan)
        long_view_mask = ratio.fillna(0.0) >= 0.7
    else:
        long_view_mask = pd.Series(False, index=test_df.index)

    if "like" in test_df.columns:
        like_mask = test_df["like"].fillna(0).astype(int) > 0
    elif "is_like" in test_df.columns:
        like_mask = test_df["is_like"].fillna(0).astype(int) > 0
    else:
        like_mask = pd.Series(False, index=test_df.index)

    if "is_click" in test_df.columns:
        click_mask = test_df["is_click"].fillna(0).astype(int) > 0
    else:
        click_mask = pd.Series(False, index=test_df.index)

    def _to_map(mask: pd.Series) -> dict[int, set[int]]:
        filtered = test_df.loc[mask, ["user_id", "video_id"]]
        output: dict[int, set[int]] = {}
        for user_id, group in filtered.groupby("user_id", sort=False):
            output[int(user_id)] = {int(video_id) for video_id in group["video_id"].tolist()}
        return output

    impression_counts = {
        int(user_id): int(count)
        for user_id, count in test_df.groupby("user_id", sort=False).size().items()
    }
    return GroundTruth(
        positive_items=_to_map(positive_mask),
        long_view_items=_to_map(long_view_mask),
        like_items=_to_map(like_mask),
        click_items=_to_map(click_mask),
        impression_counts=impression_counts,
        catalog_size=int(len(item_tag_map) or test_df["video_id"].nunique()),
        item_tag_map={int(video_id): str(tag) for video_id, tag in item_tag_map.items()},
    )


@dataclass
class ColdStartStats:
    user_segment_map: dict[int, str]
    item_segment_map: dict[int, str]
    user_positive_count: dict[int, int]
    user_exposure_count: dict[int, int]
    user_recent_tags: dict[int, list[str]]
    item_exposure_count: dict[int, int]
    item_positive_count: dict[int, int]
    user_segment_distribution: dict[str, dict[str, int]]
    item_segment_distribution: dict[str, dict[str, int]]
    category_popular_map: dict[str, list[PolicyItem]]
    category_popular_details: dict[str, Any]


def _build_user_recent_tags(
    train_df: pd.DataFrame,
    user_sequences: pd.DataFrame,
    item_tag_map: Mapping[int, str],
    positive_label: str,
    tag_limit: int,
) -> dict[int, list[str]]:
    positive_col = positive_label if positive_label in train_df.columns else "is_positive"
    if positive_col in train_df.columns:
        positive_history = train_df.loc[train_df[positive_col].fillna(0).astype(int) > 0, ["user_id", "tag", "time_ms"]].copy()
    else:
        positive_history = train_df[["user_id", "tag", "time_ms"]].copy()
    positive_history["tag"] = positive_history["tag"].astype(str)
    positive_history = positive_history.sort_values(["user_id", "time_ms"], ascending=[True, False])

    output: dict[int, list[str]] = {}
    for user_id, group in positive_history.groupby("user_id", sort=False):
        tags: list[str] = []
        seen: set[str] = set()
        for tag in group["tag"].tolist():
            if tag in seen:
                continue
            seen.add(tag)
            tags.append(str(tag))
            if len(tags) >= int(tag_limit):
                break
        output[int(user_id)] = tags

    for row in user_sequences.itertuples(index=False):
        user_id = int(getattr(row, "user_id"))
        if output.get(user_id):
            continue
        watch_seq = np.asarray(getattr(row, "watch_seq", []), dtype=np.int64).tolist()
        tags: list[str] = []
        seen: set[str] = set()
        for video_id in reversed(watch_seq):
            tag = item_tag_map.get(int(video_id))
            if tag is None or tag in seen:
                continue
            seen.add(str(tag))
            tags.append(str(tag))
            if len(tags) >= int(tag_limit):
                break
        output[user_id] = tags
    return output


def _build_category_popular_map(
    train_df: pd.DataFrame,
    tag_limit: int,
) -> tuple[dict[str, list[PolicyItem]], dict[str, Any]]:
    stats = (
        train_df.groupby(["tag", "video_id"], sort=False)
        .agg(
            exposure_count=("video_id", "size"),
            play_count=("is_click", "sum"),
            like_count=("like", "sum"),
            long_watch_count=("long_watch", "sum"),
            finish_count=("finish", "sum"),
        )
        .reset_index()
    )
    stats["tag"] = stats["tag"].astype(str)
    stats["score"] = (
        stats["play_count"].astype(float)
        + 6.0 * stats["like_count"].astype(float)
        + 3.0 * stats["long_watch_count"].astype(float)
        + 2.0 * stats["finish_count"].astype(float)
    )
    stats = stats.sort_values(["tag", "score", "video_id"], ascending=[True, False, True])

    category_map: dict[str, list[PolicyItem]] = {}
    for tag, group in stats.groupby("tag", sort=False):
        category_map[str(tag)] = [
            PolicyItem(
                video_id=int(row.video_id),
                score=float(row.score),
                source="category_popular",
                reason=f"category_popular:{tag}",
            )
            for row in group.head(int(tag_limit)).itertuples(index=False)
        ]

    details = {
        "num_categories": int(len(category_map)),
        "topn_per_category": int(tag_limit),
    }
    return category_map, details


def _classify_user_segments(
    train_df: pd.DataFrame,
    user_sequences: pd.DataFrame,
    test_users: list[int],
    low_active_max: int,
    medium_active_max: int,
    positive_label: str,
) -> tuple[dict[int, str], dict[int, int], dict[int, int], dict[str, dict[str, int]]]:
    positive_col = positive_label if positive_label in train_df.columns else "is_positive"
    user_positive_count = (
        train_df.groupby("user_id", sort=False)[positive_col].sum().astype(int).to_dict()
        if positive_col in train_df.columns
        else {}
    )
    user_exposure_count = train_df.groupby("user_id", sort=False).size().astype(int).to_dict()
    history_len_map = {
        int(row.user_id): int(getattr(row, "history_len", 0))
        for row in user_sequences.itertuples(index=False)
    }

    segment_map: dict[int, str] = {}
    distribution = {
        segment: {"user_count": 0, "impression_count": 0}
        for segment in USER_SEGMENT_ORDER
    }
    for user_id in test_users:
        positives = int(user_positive_count.get(int(user_id), 0))
        history_len = int(history_len_map.get(int(user_id), 0))
        if positives <= 0 or history_len <= 0:
            segment = "new_user"
        elif positives <= int(low_active_max):
            segment = "low_active_user"
        elif positives <= int(medium_active_max):
            segment = "medium_active_user"
        else:
            segment = "high_active_user"
        segment_map[int(user_id)] = segment
        distribution[segment]["user_count"] += 1
    return segment_map, {int(k): int(v) for k, v in user_positive_count.items()}, {int(k): int(v) for k, v in user_exposure_count.items()}, distribution


def _classify_item_segments(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    low_exposure_max: int,
    medium_exposure_max: int,
    positive_label: str,
) -> tuple[dict[int, str], dict[int, int], dict[int, int], dict[str, dict[str, int]]]:
    positive_col = positive_label if positive_label in train_df.columns else "is_positive"
    item_exposure_count = train_df.groupby("video_id", sort=False).size().astype(int).to_dict()
    item_positive_count = (
        train_df.groupby("video_id", sort=False)[positive_col].sum().astype(int).to_dict()
        if positive_col in train_df.columns
        else {}
    )

    segment_map: dict[int, str] = {}
    distribution = {
        segment: {"item_count": 0, "train_exposure_count": 0, "train_positive_count": 0}
        for segment in ITEM_SEGMENT_ORDER
    }
    all_video_ids = item_features["video_id"].astype(int).tolist()
    for video_id in all_video_ids:
        exposure = int(item_exposure_count.get(int(video_id), 0))
        positives = int(item_positive_count.get(int(video_id), 0))
        if exposure <= 0 or positives <= 0:
            segment = "new_item"
        elif exposure <= int(low_exposure_max):
            segment = "low_exposure_item"
        elif exposure <= int(medium_exposure_max):
            segment = "medium_exposure_item"
        else:
            segment = "popular_item"
        segment_map[int(video_id)] = segment
        distribution[segment]["item_count"] += 1
        distribution[segment]["train_exposure_count"] += exposure
        distribution[segment]["train_positive_count"] += positives
    return segment_map, {int(k): int(v) for k, v in item_exposure_count.items()}, {int(k): int(v) for k, v in item_positive_count.items()}, distribution


def _collect_recommendations(policy: Any, users: list[int], top_k: int, logger: Any | None, label: str) -> dict[int, list[PolicyItem]]:
    output: dict[int, list[PolicyItem]] = {}
    for idx, user_id in enumerate(users, start=1):
        output[int(user_id)] = policy.recommend_with_details(int(user_id), int(top_k))
        if logger and idx % 5000 == 0:
            logger.info("%s replay progress: %d/%d users", label, idx, len(users))
    return output


def _category_candidates_for_user(
    user_id: int,
    recent_tags: Mapping[int, list[str]],
    category_popular_map: Mapping[str, list[PolicyItem]],
    seen: set[int],
    per_tag_limit: int,
) -> list[PolicyItem]:
    tags = recent_tags.get(int(user_id), [])
    aggregated: dict[int, dict[str, Any]] = {}
    for tag_rank, tag in enumerate(tags):
        tag_items = category_popular_map.get(str(tag), [])
        tag_weight = 1.0 / math.log2(tag_rank + 2)
        for item_rank, item in enumerate(tag_items[: int(per_tag_limit)]):
            video_id = int(item.video_id)
            if video_id in seen:
                continue
            score = tag_weight / math.log2(item_rank + 2)
            entry = aggregated.setdefault(
                video_id,
                {
                    "video_id": video_id,
                    "score": 0.0,
                    "reason": [],
                },
            )
            entry["score"] += float(score)
            entry["reason"].append(f"tag:{tag}")
    ranked = sorted(
        aggregated.values(),
        key=lambda row: (-float(row["score"]), int(row["video_id"])),
    )
    return [
        PolicyItem(
            video_id=int(row["video_id"]),
            score=float(row["score"]),
            source="category_popular",
            reason="|".join(row["reason"]),
        )
        for row in ranked
    ]


def _enhance_user_recommendations(
    user_id: int,
    user_segment: str,
    baseline_items: list[PolicyItem],
    global_popular_items: list[PolicyItem],
    category_items: list[PolicyItem],
    item_segment_map: Mapping[int, str],
    cfg: Mapping[str, Any],
    top_k: int,
) -> tuple[list[PolicyItem], dict[str, Any]]:
    strategy_cfg = cfg["cold_start"]["strategies"]
    baseline_weights = {
        "new_user": 0.45,
        "low_active_user": 0.70,
        "medium_active_user": 1.0,
        "high_active_user": 1.0,
    }
    category_weights = {
        "new_user": 1.15,
        "low_active_user": 0.70,
        "medium_active_user": 0.15,
        "high_active_user": 0.05,
    }
    global_weights = {
        "new_user": 0.95,
        "low_active_user": 0.45,
        "medium_active_user": 0.05,
        "high_active_user": 0.0,
    }
    freshness_weight = float(strategy_cfg.get("freshness_boost_weight", 0.05))
    enable_global = bool(strategy_cfg.get("enable_global_popular", True))
    enable_category = bool(strategy_cfg.get("enable_category_popular", True))
    enable_freshness = bool(strategy_cfg.get("enable_freshness_boost", True))

    aggregated: dict[int, dict[str, Any]] = {}

    def _register(item: PolicyItem, origin: str, rank: int, weight: float) -> None:
        video_id = int(item.video_id)
        entry = aggregated.setdefault(
            video_id,
            {
                "video_id": video_id,
                "score": 0.0,
                "baseline_rank": 10**9,
                "reasons": [],
                "base_score": item.score,
            },
        )
        entry["score"] += float(weight / math.log2(rank + 2))
        if origin == "baseline":
            entry["baseline_rank"] = min(int(entry["baseline_rank"]), int(rank))
            entry["base_score"] = item.score
        entry["reasons"].append(origin)

    for rank, item in enumerate(baseline_items):
        _register(item, "baseline", rank, baseline_weights[user_segment])
        if enable_freshness:
            item_segment = item_segment_map.get(int(item.video_id), "new_item")
            if item_segment == "new_item":
                aggregated[int(item.video_id)]["score"] += freshness_weight
                aggregated[int(item.video_id)]["reasons"].append("freshness:new_item")
            elif item_segment == "low_exposure_item":
                aggregated[int(item.video_id)]["score"] += freshness_weight * 0.6
                aggregated[int(item.video_id)]["reasons"].append("freshness:low_exposure")

    if enable_category:
        for rank, item in enumerate(category_items):
            _register(item, "category_popular", rank, category_weights[user_segment])

    if enable_global:
        for rank, item in enumerate(global_popular_items):
            _register(item, "global_popular", rank, global_weights[user_segment])

    ranked = sorted(
        aggregated.values(),
        key=lambda row: (-float(row["score"]), int(row["baseline_rank"]), int(row["video_id"])),
    )

    final_items = [
        PolicyItem(
            video_id=int(row["video_id"]),
            score=float(row["score"]),
            source="cold_start_enhanced",
            reason="|".join(row["reasons"]),
        )
        for row in ranked[: int(top_k)]
    ]
    details = {
        "used_global_popular": bool(enable_global and user_segment in {"new_user", "low_active_user"}),
        "used_category_popular": bool(enable_category and len(category_items) > 0),
        "used_freshness_boost": bool(enable_freshness),
    }
    return final_items, details


def _evaluate_by_user_segment(
    recommendations_by_user: Mapping[int, list[PolicyItem]],
    user_segment_map: Mapping[int, str],
    ground_truth: GroundTruth,
    topks: list[int],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for segment in USER_SEGMENT_ORDER:
        users = [int(user_id) for user_id, value in user_segment_map.items() if value == segment]
        result = evaluate_group(recommendations_by_user, users, ground_truth, topks)
        output[segment] = result["metrics"]
    return output


def _evaluate_by_item_segment(
    recommendations_by_user: Mapping[int, list[PolicyItem]],
    item_segment_map: Mapping[int, str],
    item_distribution: Mapping[str, Mapping[str, int]],
    item_exposure_count: Mapping[int, int],
    ground_truth: GroundTruth,
    topks: list[int],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for k in topks:
        recommendation_count = 0
        total_hits = 0
        segment_stats = {
            segment: {
                "item_count": int(item_distribution[segment]["item_count"]),
                "exposure_count": int(item_distribution[segment]["train_exposure_count"]),
                "recommendation_count": 0,
                "hit_count": 0,
            }
            for segment in ITEM_SEGMENT_ORDER
        }
        for user_id, items in recommendations_by_user.items():
            positives = ground_truth.positive_items.get(int(user_id), set())
            for item in items[: int(k)]:
                segment = item_segment_map.get(int(item.video_id), "new_item")
                segment_stats[segment]["recommendation_count"] += 1
                recommendation_count += 1
                if int(item.video_id) in positives:
                    segment_stats[segment]["hit_count"] += 1
                    total_hits += 1
        current: dict[str, dict[str, Any]] = {}
        for segment in ITEM_SEGMENT_ORDER:
            stats = segment_stats[segment]
            rec_count = int(stats["recommendation_count"])
            hit_count = int(stats["hit_count"])
            current[segment] = {
                "item_count": int(stats["item_count"]),
                "exposure_count": int(stats["exposure_count"]),
                "recommendation_count": rec_count,
                "hit_count": hit_count,
                "hit_rate": float(hit_count / rec_count) if rec_count else 0.0,
                "share_of_recommendations": float(rec_count / recommendation_count) if recommendation_count else 0.0,
                "share_of_hits": float(hit_count / total_hits) if total_hits else 0.0,
            }
        output[f"top_{int(k)}"] = current
    return output


def _build_summary_markdown(report: Mapping[str, Any]) -> str:
    cold_cfg = report["config"]["cold_start"]
    topks = cold_cfg["top_k"]
    lines = [
        "# Cold-start Analysis",
        "",
        "This section is a **heuristic cold-start simulation**, not a real online cold-start model.",
        "",
        "## Definitions",
        "",
        f"- new_user: no positive train history or no train history",
        f"- low_active_user: positive_count <= `{report['thresholds']['user_segments']['low_active_max']}`",
        f"- medium_active_user: positive_count <= `{report['thresholds']['user_segments']['medium_active_max']}`",
        f"- high_active_user: above medium threshold",
        f"- new_item: no train exposure or no train positive feedback",
        f"- low_exposure_item: exposure <= `{report['thresholds']['item_segments']['low_exposure_max']}`",
        f"- medium_exposure_item: exposure <= `{report['thresholds']['item_segments']['medium_exposure_max']}`",
        "",
        "## User Segment Distribution",
        "",
        "| segment | user_count | impression_count |",
        "|---|---:|---:|",
    ]
    for segment in USER_SEGMENT_ORDER:
        values = report["user_segment_distribution"][segment]
        lines.append(f"| {segment} | {values['user_count']} | {values['impression_count']} |")

    lines.extend(
        [
            "",
            "## Item Segment Distribution",
            "",
            "| segment | item_count | train_exposure_count | train_positive_count |",
            "|---|---:|---:|---:|",
        ]
    )
    for segment in ITEM_SEGMENT_ORDER:
        values = report["item_segment_distribution"][segment]
        lines.append(
            f"| {segment} | {values['item_count']} | {values['train_exposure_count']} | {values['train_positive_count']} |"
        )

    lines.extend(
        [
            "",
            "## User Segment Metrics",
            "",
            "| segment | baseline hit@10 | enhanced hit@10 | hit@10 lift | baseline recall@50 | enhanced recall@50 | recall@50 lift | baseline long_view@10 | enhanced long_view@10 | long_view@10 lift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for segment in USER_SEGMENT_ORDER:
        base = report["baseline_metrics_by_user_segment"][segment]
        enhanced = report["enhanced_metrics_by_user_segment"][segment]
        lift = report["lift_by_user_segment"][segment]
        lines.append(
            "| {segment} | {b1:.6f} | {e1:.6f} | {l1} | {b2:.6f} | {e2:.6f} | {l2} | {b3:.6f} | {e3:.6f} | {l3} |".format(
                segment=segment,
                b1=float(base.get("hit_rate@10", 0.0)),
                e1=float(enhanced.get("hit_rate@10", 0.0)),
                l1="n/a" if lift.get("hit_rate@10 lift") is None else f"{float(lift['hit_rate@10 lift']):.4%}",
                b2=float(base.get("recall@50", 0.0)),
                e2=float(enhanced.get("recall@50", 0.0)),
                l2="n/a" if lift.get("recall@50 lift") is None else f"{float(lift['recall@50 lift']):.4%}",
                b3=float(base.get("long_view_rate@10", 0.0)),
                e3=float(enhanced.get("long_view_rate@10", 0.0)),
                l3="n/a" if lift.get("long_view_rate@10 lift") is None else f"{float(lift['long_view_rate@10 lift']):.4%}",
            )
        )

    lines.extend(
        [
            "",
            "## Strategy Notes",
            "",
            f"- global_popular enabled: `{report['strategy_details']['enable_global_popular']}`",
            f"- category_popular enabled: `{report['strategy_details']['enable_category_popular']}`",
            f"- freshness_boost enabled: `{report['strategy_details']['enable_freshness_boost']}`",
            f"- freshness boost uses low-exposure / zero-positive items as a lightweight freshness proxy.",
            "",
            "## Limitations",
            "",
            "- This is an offline analysis over logged data and offline recommendation artifacts.",
            "- The enhanced cold-start pipeline is heuristic; it is not a learned cold-start model and does not update training artifacts.",
            "- If some lower-stage artifacts are missing or stored as Git LFS pointers, the analysis falls back and records warnings in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_cold_start_analysis(
    cfg: Mapping[str, Any],
    component_cfgs: Mapping[str, Mapping[str, Any]],
    logger: Any | None = None,
) -> dict[str, Any]:
    cold_cfg = cfg["cold_start"]
    split = str(cold_cfg.get("split", "test"))
    topks = sorted({int(k) for k in cold_cfg.get("top_k", [10, 50])})
    positive_label = str(cold_cfg.get("positive_label", "is_positive"))
    candidate_pool_size = int(cold_cfg.get("candidate_pool_size", max(max(topks), 50) * 2))

    recall_cfg = component_cfgs["recall"]
    train_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["splits_dir"] / "train.parquet"
    test_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["splits_dir"] / f"{split}.parquet"
    seq_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["user_sequences_file"]
    item_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["item_features_file"]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    user_sequences = pd.read_parquet(seq_path)
    item_features = pd.read_parquet(item_path)

    item_tag_map = {
        int(row.video_id): str(row.tag)
        for row in item_features[["video_id", "tag"]].drop_duplicates("video_id").itertuples(index=False)
    }
    test_users = sorted(int(user_id) for user_id in test_df["user_id"].unique().tolist())

    user_segment_map, user_positive_count, user_exposure_count, user_segment_distribution = _classify_user_segments(
        train_df=train_df,
        user_sequences=user_sequences,
        test_users=test_users,
        low_active_max=int(cold_cfg["user_segments"]["low_active_max"]),
        medium_active_max=int(cold_cfg["user_segments"]["medium_active_max"]),
        positive_label=positive_label,
    )
    for segment in USER_SEGMENT_ORDER:
        users = [user_id for user_id, value in user_segment_map.items() if value == segment]
        user_segment_distribution[segment]["impression_count"] = int(
            test_df.loc[test_df["user_id"].isin(users)].shape[0]
        )

    item_segment_map, item_exposure_count, item_positive_count, item_segment_distribution = _classify_item_segments(
        train_df=train_df,
        item_features=item_features,
        low_exposure_max=int(cold_cfg["item_segments"]["low_exposure_max"]),
        medium_exposure_max=int(cold_cfg["item_segments"]["medium_exposure_max"]),
        positive_label=positive_label,
    )

    user_recent_tags = _build_user_recent_tags(
        train_df=train_df,
        user_sequences=user_sequences,
        item_tag_map=item_tag_map,
        positive_label=positive_label,
        tag_limit=int(cold_cfg["strategies"].get("category_tag_limit", 3)),
    )
    category_popular_map, category_popular_details = _build_category_popular_map(
        train_df=train_df,
        tag_limit=int(cold_cfg["strategies"].get("category_popular_topn_per_tag", 100)),
    )

    context = PolicyContext(
        experiment_cfg={"faiss_index_type": "hnsw"},
        component_cfgs=component_cfgs,
        split=split,
        max_top_k=int(candidate_pool_size),
        logger=logger,
        user_sequences=user_sequences,
        item_features=item_features,
    )
    baseline_policy = build_policy(
        {"name": "baseline_full_pipeline", "type": "full_pipeline"},
        context=context,
        logger=logger,
    )
    popular_policy = build_policy(
        {"name": "global_popular", "type": "popular"},
        context=context,
        logger=logger,
    )

    baseline_recommendations = _collect_recommendations(
        baseline_policy,
        users=test_users,
        top_k=int(candidate_pool_size),
        logger=logger,
        label="baseline_full_pipeline",
    )
    global_popular_recommendations = _collect_recommendations(
        popular_policy,
        users=test_users,
        top_k=int(candidate_pool_size),
        logger=logger,
        label="global_popular",
    )

    enhanced_recommendations: dict[int, list[PolicyItem]] = {}
    enhancement_usage = {
        "global_popular_users": 0,
        "category_popular_users": 0,
        "freshness_boost_users": 0,
    }
    per_tag_limit = int(cold_cfg["strategies"].get("category_popular_topn_per_tag", 100))
    for idx, user_id in enumerate(test_users, start=1):
        seen = context.user_seen_map.get(int(user_id), set())
        category_candidates = _category_candidates_for_user(
            user_id=int(user_id),
            recent_tags=user_recent_tags,
            category_popular_map=category_popular_map,
            seen=seen,
            per_tag_limit=per_tag_limit,
        )
        enhanced_items, details = _enhance_user_recommendations(
            user_id=int(user_id),
            user_segment=user_segment_map[int(user_id)],
            baseline_items=baseline_recommendations[int(user_id)],
            global_popular_items=global_popular_recommendations[int(user_id)],
            category_items=category_candidates,
            item_segment_map=item_segment_map,
            cfg=cfg,
            top_k=int(candidate_pool_size),
        )
        enhanced_recommendations[int(user_id)] = enhanced_items
        if details["used_global_popular"]:
            enhancement_usage["global_popular_users"] += 1
        if details["used_category_popular"]:
            enhancement_usage["category_popular_users"] += 1
        if details["used_freshness_boost"]:
            enhancement_usage["freshness_boost_users"] += 1
        if logger and idx % 5000 == 0:
            logger.info("cold_start_enhanced replay progress: %d/%d users", idx, len(test_users))

    ground_truth = _build_ground_truth(
        test_df=test_df,
        item_tag_map=item_tag_map,
        positive_label=positive_label,
    )

    baseline_metrics_by_user_segment = _evaluate_by_user_segment(
        baseline_recommendations,
        user_segment_map=user_segment_map,
        ground_truth=ground_truth,
        topks=topks,
    )
    enhanced_metrics_by_user_segment = _evaluate_by_user_segment(
        enhanced_recommendations,
        user_segment_map=user_segment_map,
        ground_truth=ground_truth,
        topks=topks,
    )
    baseline_item_metrics = _evaluate_by_item_segment(
        baseline_recommendations,
        item_segment_map=item_segment_map,
        item_distribution=item_segment_distribution,
        item_exposure_count=item_exposure_count,
        ground_truth=ground_truth,
        topks=topks,
    )
    enhanced_item_metrics = _evaluate_by_item_segment(
        enhanced_recommendations,
        item_segment_map=item_segment_map,
        item_distribution=item_segment_distribution,
        item_exposure_count=item_exposure_count,
        ground_truth=ground_truth,
        topks=topks,
    )

    lift_by_user_segment: dict[str, dict[str, float | None]] = {}
    for segment in USER_SEGMENT_ORDER:
        baseline_metrics = baseline_metrics_by_user_segment[segment]
        enhanced_metrics = enhanced_metrics_by_user_segment[segment]
        lift_by_user_segment[segment] = {
            "hit_rate@10 lift": _relative_lift(baseline_metrics.get("hit_rate@10"), enhanced_metrics.get("hit_rate@10")),
            "recall@50 lift": _relative_lift(baseline_metrics.get("recall@50"), enhanced_metrics.get("recall@50")),
            "long_view_rate@10 lift": _relative_lift(
                baseline_metrics.get("long_view_rate@10"),
                enhanced_metrics.get("long_view_rate@10"),
            ),
            "coverage@10 lift": _relative_lift(baseline_metrics.get("coverage@10"), enhanced_metrics.get("coverage@10")),
        }

    warnings: list[str] = []
    warnings.extend(baseline_policy.metadata().get("warnings", []))
    warnings.extend(popular_policy.metadata().get("warnings", []))
    missing_artifacts = baseline_policy.metadata().get("missing_artifacts", []) + popular_policy.metadata().get("missing_artifacts", [])
    if missing_artifacts:
        warnings.append("One or more recommendation artifacts were missing or unreadable; see missing_artifacts for details.")

    report = {
        "config": cfg,
        "thresholds": {
            "user_segments": dict(cold_cfg["user_segments"]),
            "item_segments": dict(cold_cfg["item_segments"]),
        },
        "actual_artifact_used": {
            "baseline": _recommendation_source_label(baseline_policy.metadata()),
            "global_popular": _recommendation_source_label(popular_policy.metadata()),
            "category_popular": {
                "built_from": str(train_path),
                "details": category_popular_details,
            },
        },
        "warnings": warnings,
        "missing_artifacts": missing_artifacts,
        "user_segment_distribution": user_segment_distribution,
        "item_segment_distribution": item_segment_distribution,
        "baseline_metrics_by_user_segment": baseline_metrics_by_user_segment,
        "enhanced_metrics_by_user_segment": enhanced_metrics_by_user_segment,
        "baseline_item_segment_metrics": baseline_item_metrics,
        "enhanced_item_segment_metrics": enhanced_item_metrics,
        "lift_by_user_segment": lift_by_user_segment,
        "strategy_details": {
            **dict(cold_cfg["strategies"]),
            "enhancement_usage": enhancement_usage,
            "candidate_pool_size": int(candidate_pool_size),
            "freshness_proxy": "new_item_or_low_exposure_item_based_on_train_exposure",
        },
        "ground_truth": {
            "positive_label": positive_label,
            "positive_users": int(len(ground_truth.positive_items)),
            "long_view_users": int(len(ground_truth.long_view_items)),
            "like_users": int(len(ground_truth.like_items)),
            "click_users": int(len(ground_truth.click_items)),
        },
        "limitations": [
            "Cold-start enhancement is heuristic and offline-only.",
            "No models are retrained; the analysis only reuses existing recommendation artifacts and train/test splits.",
            "Offline replay cannot estimate causal online lift or fully correct for exposure bias.",
        ],
    }

    report_path = artifact_path(cfg, cold_cfg["output"]["report_file"])
    summary_path = artifact_path(cfg, cold_cfg["output"]["summary_file"])
    ensure_dir(report_path.parent)
    ensure_dir(summary_path.parent)
    report_path.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=False), encoding="utf-8")
    summary_path.write_text(_build_summary_markdown(report), encoding="utf-8")
    if logger:
        logger.info("Saved cold-start report: %s", report_path)
        logger.info("Saved cold-start summary: %s", summary_path)
    return report
