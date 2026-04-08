from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd


def _tail_list(values: pd.Series, max_seq_len: int) -> list[int]:
    return [int(v) for v in values.tail(max_seq_len).tolist()]


def build_user_sequences(
    interactions: pd.DataFrame,
    seq_cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build per-user recent behavior sequences from historical interactions.

    The expected input is the training split only, so validation/test examples
    do not leak future behavior into user history.
    """
    max_seq_len = int(seq_cfg.get("max_seq_len", 50))
    min_history_len = int(seq_cfg.get("min_history_len", 1))
    build_watch = bool(seq_cfg.get("build_watch_sequence", True))
    build_like = bool(seq_cfg.get("build_like_sequence", True))
    build_long_watch = bool(seq_cfg.get("build_long_watch_sequence", True))

    if interactions.empty:
        return pd.DataFrame(
            columns=[
                "user_id",
                "watch_seq",
                "like_seq",
                "long_watch_seq",
                "history_len",
                "last_time_ms",
            ]
        )

    sort_cols = ["user_id", "time_ms", "video_id"]
    df = interactions.sort_values(sort_cols).copy()

    rows: list[dict[str, object]] = []
    for user_id, group in df.groupby("user_id", sort=False):
        history_len = int(group.shape[0])
        if history_len < min_history_len:
            continue

        row: dict[str, object] = {
            "user_id": int(user_id),
            "history_len": history_len,
            "last_time_ms": int(pd.to_numeric(group["time_ms"], errors="coerce").max()),
        }
        row["watch_seq"] = _tail_list(group["video_id"], max_seq_len) if build_watch else []

        if build_like:
            row["like_seq"] = _tail_list(group.loc[group["like"] > 0, "video_id"], max_seq_len)
        else:
            row["like_seq"] = []

        if build_long_watch:
            row["long_watch_seq"] = _tail_list(
                group.loc[group["long_watch"] > 0, "video_id"],
                max_seq_len,
            )
        else:
            row["long_watch_seq"] = []

        rows.append(row)

    out = pd.DataFrame(rows)
    if logger:
        logger.info(
            "Built user sequences: users=%d max_seq_len=%d",
            out.shape[0],
            max_seq_len,
        )
    return out
