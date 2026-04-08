from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd


def split_by_date(
    interactions: pd.DataFrame,
    split_cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, pd.DataFrame]:
    """Split interactions by date using inclusive integer YYYYMMDD boundaries."""
    strategy = split_cfg.get("strategy", "date")
    if strategy != "date":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    date = pd.to_numeric(interactions["date"], errors="coerce").astype("Int64")
    train_end = int(split_cfg["train_end_date"])
    val_start = int(split_cfg["val_start_date"])
    val_end = int(split_cfg["val_end_date"])
    test_start = int(split_cfg["test_start_date"])

    train = interactions.loc[date <= train_end].copy()
    val = interactions.loc[(date >= val_start) & (date <= val_end)].copy()
    test = interactions.loc[date >= test_start].copy()

    splits = {"train": train, "val": val, "test": test}
    if logger:
        for name, df in splits.items():
            if df.empty:
                logger.warning("Split %s is empty.", name)
            else:
                logger.info(
                    "Split %-5s rows=%d users=%d items=%d date_range=[%s, %s]",
                    name,
                    df.shape[0],
                    df["user_id"].nunique(),
                    df["video_id"].nunique(),
                    df["date"].min(),
                    df["date"].max(),
                )

    return splits
