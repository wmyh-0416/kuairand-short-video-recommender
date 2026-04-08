from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _safe_value(value: Any, missing_token: str) -> Any:
    if pd.isna(value):
        return missing_token
    return value


def _freshness_bonus(freshness_days: float, cfg: Mapping[str, Any]) -> float:
    fresh_cfg = cfg["rerank"].get("freshness", {})
    if not fresh_cfg.get("enabled", True):
        return 0.0
    weight = float(fresh_cfg.get("bonus_weight", 0.08))
    half_life = float(fresh_cfg.get("half_life_days", 10.0))
    freshness_days = max(float(freshness_days), 0.0)
    return weight * math.exp(-freshness_days / max(half_life, 1e-6))


def _candidate_adjustment(
    row: Mapping[str, Any],
    state: dict[str, Any],
    cfg: Mapping[str, Any],
    enforce_hard: bool = True,
) -> dict[str, Any] | None:
    base_score = float(row.get("rank_score", 0.0))
    author = _safe_value(row.get("author_id"), "__AUTHOR_MISSING__")
    tag = _safe_value(row.get("tag"), "__TAG_MISSING__")
    freshness_days = float(row.get("freshness_days", 0.0))

    author_cfg = cfg["rerank"].get("author", {})
    tag_cfg = cfg["rerank"].get("tag", {})
    scoring_cfg = cfg["rerank"].get("scoring", {})
    reasons: list[str] = []

    multiplier = 1.0
    author_penalty = 0.0
    tag_penalty = 0.0
    freshness_bonus = _freshness_bonus(freshness_days, cfg)
    new_author_bonus = 0.0
    new_tag_bonus = 0.0

    author_count = int(state["author_counts"].get(author, 0))
    tag_count = int(state["tag_counts"].get(tag, 0))
    if author_cfg.get("enabled", True):
        max_count = int(author_cfg.get("max_count_in_topk", 10))
        if author_count >= max_count:
            if enforce_hard and bool(author_cfg.get("hard_max_count", False)):
                return None
            overflow = author_count - max_count + 1
            author_penalty += float(author_cfg.get("repeat_count_penalty", 0.05)) * overflow
            reasons.append("author_cap_penalty")
        elif author_count == 0:
            new_author_bonus = float(author_cfg.get("new_author_bonus", 0.0))
            if new_author_bonus > 0:
                reasons.append("new_author_bonus")
        if state["last_author"] == author:
            streak = int(state["author_streak"])
            max_consecutive = int(author_cfg.get("max_consecutive", 99))
            if streak >= max_consecutive:
                if enforce_hard and bool(author_cfg.get("hard_max_consecutive", False)):
                    return None
                author_penalty += float(author_cfg.get("consecutive_penalty", 0.2)) * streak
                reasons.append("author_consecutive_penalty")
        elif author_count > 0:
            author_penalty += float(author_cfg.get("repeat_count_penalty", 0.05)) * author_count
            reasons.append("author_repeat_penalty")

    if tag_cfg.get("enabled", True):
        max_count = int(tag_cfg.get("max_count_in_topk", 10))
        if tag_count >= max_count:
            if enforce_hard and bool(tag_cfg.get("hard_max_count", False)):
                return None
            overflow = tag_count - max_count + 1
            tag_penalty += float(tag_cfg.get("repeat_count_penalty", 0.03)) * overflow
            reasons.append("tag_cap_penalty")
        elif tag_count == 0:
            new_tag_bonus = float(tag_cfg.get("new_tag_bonus", 0.0))
            if new_tag_bonus > 0:
                reasons.append("new_tag_bonus")
        if state["last_tag"] == tag:
            streak = int(state["tag_streak"])
            max_consecutive = int(tag_cfg.get("max_consecutive", 99))
            if streak >= max_consecutive:
                if enforce_hard and bool(tag_cfg.get("hard_max_consecutive", False)):
                    return None
                tag_penalty += float(tag_cfg.get("consecutive_penalty", 0.10)) * streak
                reasons.append("tag_consecutive_penalty")
        elif tag_count > 0:
            tag_penalty += float(tag_cfg.get("repeat_count_penalty", 0.03)) * tag_count
            reasons.append("tag_repeat_penalty")

    if freshness_bonus > 0:
        reasons.append("freshness_bonus")
    multiplier += freshness_bonus + new_author_bonus + new_tag_bonus - author_penalty - tag_penalty
    multiplier = float(np.clip(multiplier, float(scoring_cfg.get("min_multiplier", 0.55)), float(scoring_cfg.get("max_multiplier", 1.35))))
    rerank_score = base_score * multiplier

    return {
        "rerank_score": float(rerank_score),
        "score_multiplier": float(multiplier),
        "author_penalty": float(author_penalty),
        "tag_penalty": float(tag_penalty),
        "freshness_bonus": float(freshness_bonus),
        "new_author_bonus": float(new_author_bonus),
        "new_tag_bonus": float(new_tag_bonus),
        "adjustment_reason": "|".join(reasons) if reasons else "base",
    }


def greedy_rerank_user(user_df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    topk = int(cfg["rerank"].get("topk", 20))
    if user_df.empty:
        return user_df.copy()

    ordered = user_df.sort_values(
        ["rank_score", "coarse_score", "merged_score", "video_id"],
        ascending=[False, False, False, True],
    )
    remaining: list[dict[str, Any]] = ordered.to_dict("records")
    state = {
        "author_counts": Counter(),
        "tag_counts": Counter(),
        "last_author": None,
        "last_tag": None,
        "author_streak": 0,
        "tag_streak": 0,
    }

    selected_rows: list[dict[str, Any]] = []
    while len(selected_rows) < min(topk, len(user_df)) and remaining:
        best_idx = None
        best_adjustment = None
        for idx, row in enumerate(remaining):
            adjustment = _candidate_adjustment(row, state, cfg, enforce_hard=True)
            if adjustment is None:
                continue
            if best_adjustment is None or adjustment["rerank_score"] > best_adjustment["rerank_score"]:
                best_idx = idx
                best_adjustment = adjustment

        if best_idx is None:
            for idx, row in enumerate(remaining):
                adjustment = _candidate_adjustment(row, state, cfg, enforce_hard=False)
                if adjustment is None:
                    continue
                adjustment["adjustment_reason"] = f'{adjustment["adjustment_reason"]}|relaxed_constraints'
                if best_adjustment is None or adjustment["rerank_score"] > best_adjustment["rerank_score"]:
                    best_idx = idx
                    best_adjustment = adjustment

        if best_idx is None or best_adjustment is None:
            break

        chosen = dict(remaining[int(best_idx)])
        chosen.update(best_adjustment)
        chosen["final_rank"] = len(selected_rows) + 1
        selected_rows.append(chosen)

        author = _safe_value(chosen.get("author_id"), "__AUTHOR_MISSING__")
        tag = _safe_value(chosen.get("tag"), "__TAG_MISSING__")
        state["author_counts"][author] += 1
        state["tag_counts"][tag] += 1
        if state["last_author"] == author:
            state["author_streak"] += 1
        else:
            state["last_author"] = author
            state["author_streak"] = 1
        if state["last_tag"] == tag:
            state["tag_streak"] += 1
        else:
            state["last_tag"] = tag
            state["tag_streak"] = 1
        remaining.pop(int(best_idx))

    return pd.DataFrame(selected_rows)


def rerank_candidates(ranked_df: pd.DataFrame, cfg: Mapping[str, Any], logger: Any | None = None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for idx, (_user_id, group) in enumerate(ranked_df.groupby("user_id", sort=False), start=1):
        parts.append(greedy_rerank_user(group, cfg))
        if logger is not None and idx % 5000 == 0:
            logger.info("Reranked %d users", idx)
    if not parts:
        return ranked_df.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["user_id", "final_rank"]).reset_index(drop=True)
    return out
