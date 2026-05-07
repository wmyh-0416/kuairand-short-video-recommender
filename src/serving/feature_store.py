from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.prerank.features import build_train_history_stats
from src.rank.features import RankFeatureStore
from src.recall.popular import generate_popular_candidates, load_popular_items
from src.serving.user_state import UserState
from src.utils.paths import artifacts_dir, processed_path


def _safe_seq_list(seq: Any) -> list[int]:
    if seq is None:
        return []
    if isinstance(seq, np.ndarray):
        values = seq.tolist()
    elif isinstance(seq, (list, tuple)):
        values = list(seq)
    else:
        values = []
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


class ServingFeatureStore:
    def __init__(
        self,
        serving_cfg: Mapping[str, Any],
        component_cfgs: Mapping[str, Mapping[str, Any]],
        logger: Optional[Any] = None,
    ) -> None:
        self.serving_cfg = serving_cfg
        self.component_cfgs = component_cfgs
        self.logger = logger

        recall_cfg = component_cfgs["recall"]
        prerank_cfg = component_cfgs["prerank"]
        rank_cfg = component_cfgs["rank"]

        self.user_features = pd.read_parquet(processed_path(prerank_cfg, prerank_cfg["prerank"]["processed"]["user_features_file"]))
        self.item_features = pd.read_parquet(processed_path(prerank_cfg, prerank_cfg["prerank"]["processed"]["item_features_file"]))
        self.user_sequences = pd.read_parquet(processed_path(recall_cfg, recall_cfg["recall"]["processed"]["user_sequences_file"]))
        self.train_split = pd.read_parquet(
            processed_path(rank_cfg, Path(rank_cfg["rank"]["processed"].get("splits_dir", "splits")) / "train.parquet")
        )
        self.train_stats = build_train_history_stats(self.train_split)

        self.rank_processed_tables = {
            "user_features": self.user_features,
            "item_features": self.item_features,
            "train_split": self.train_split,
        }
        self.prerank_processed_tables = {
            "user_features": self.user_features,
            "item_features": self.item_features,
            "user_sequences": self.user_sequences,
            "train_split": self.train_split,
        }

        self._base_user_state_map = self._build_base_user_state_map(self.user_sequences)
        self._item_meta = self.item_features[
            ["video_id", "author_id", "tag", "upload_date"]
        ].drop_duplicates("video_id").copy()
        self._item_author_map = {
            int(video_id): author_id for video_id, author_id in zip(self._item_meta["video_id"], self._item_meta["author_id"])
        }
        self._item_tag_map = {
            int(video_id): tag for video_id, tag in zip(self._item_meta["video_id"], self._item_meta["tag"])
        }
        self._item_meta_map = {
            int(row.video_id): {
                "video_id": int(row.video_id),
                "author_id": row.author_id,
                "tag": row.tag,
                "category": row.tag,
                "upload_date": getattr(row, "upload_date", None),
            }
            for row in self._item_meta.itertuples(index=False)
        }
        self.popular_items = self._load_popular_items(component_cfgs["recall"])

    @staticmethod
    def _build_base_user_state_map(user_sequences: pd.DataFrame) -> dict[int, dict[str, Any]]:
        if user_sequences.empty:
            return {}
        out: dict[int, dict[str, Any]] = {}
        cols = ["user_id", "history_len", "last_time_ms", "watch_seq", "like_seq", "long_watch_seq"]
        available = [col for col in cols if col in user_sequences.columns]
        for row in user_sequences[available].itertuples(index=False):
            payload = row._asdict()
            user_id = int(payload["user_id"])
            out[user_id] = {
                "user_id": user_id,
                "history_len": int(payload.get("history_len", 0) or 0),
                "last_time_ms": int(payload.get("last_time_ms", 0) or 0),
                "watch_seq": _safe_seq_list(payload.get("watch_seq")),
                "like_seq": _safe_seq_list(payload.get("like_seq")),
                "long_watch_seq": _safe_seq_list(payload.get("long_watch_seq")),
            }
        return out

    def _load_popular_items(self, recall_cfg: Mapping[str, Any]) -> Optional[pd.DataFrame]:
        popular_path = (
            artifacts_dir(recall_cfg)
            / recall_cfg["recall"]["output"]["recall_dir"]
            / recall_cfg["recall"]["output"]["popular_items_file"]
        )
        if not popular_path.exists():
            return None
        try:
            return load_popular_items(popular_path)
        except Exception as exc:
            if self.logger:
                self.logger.warning("Failed to load popular recall fallback table %s: %s", popular_path, exc)
            return None

    def get_base_user_state(self, user_id: int) -> Optional[UserState]:
        payload = self._base_user_state_map.get(int(user_id))
        if payload is None:
            return None
        return UserState.from_dict(
            deepcopy(payload),
            max_history_len=int(self.serving_cfg["serving"]["feedback"].get("max_history_len", 50)),
            long_watch_ratio=float(self.serving_cfg["serving"]["feedback"].get("long_watch_ratio", 0.7)),
        )

    def get_item_features(self, video_id: int) -> dict[str, Any]:
        return dict(self._item_meta_map.get(int(video_id), {"video_id": int(video_id)}))

    def get_effective_user_state(self, user_id: int, runtime_state: Optional[Any] = None) -> UserState:
        max_history_len = int(self.serving_cfg["serving"]["feedback"].get("max_history_len", 50))
        long_watch_ratio = float(self.serving_cfg["serving"]["feedback"].get("long_watch_ratio", 0.7))
        if isinstance(runtime_state, UserState):
            return runtime_state
        if runtime_state is not None:
            coerced = UserState.from_dict(
                runtime_state,
                max_history_len=max_history_len,
                long_watch_ratio=long_watch_ratio,
            )
            if coerced is not None:
                return coerced
        base = self.get_base_user_state(int(user_id))
        if base is not None:
            return base
        return UserState.empty(
            int(user_id),
            max_history_len=max_history_len,
            long_watch_ratio=long_watch_ratio,
        )

    def build_user_sequences_frame(
        self,
        user_id: int,
        runtime_state: Optional[Any] = None,
    ) -> pd.DataFrame:
        state = self.get_effective_user_state(int(user_id), runtime_state=runtime_state)
        return pd.DataFrame(
            [
                {
                    "user_id": int(user_id),
                    "history_len": int(len(state.recent_viewed_video_ids)),
                    "last_time_ms": int(state.last_active_time),
                    "watch_seq": list(state.recent_viewed_video_ids),
                    "like_seq": list(state.recent_liked_video_ids),
                    "long_watch_seq": list(state.recent_long_view_video_ids),
                }
            ]
        )

    def build_seen_history_df(
        self,
        user_id: int,
        runtime_state: Optional[Any] = None,
    ) -> pd.DataFrame:
        state = self.get_effective_user_state(int(user_id), runtime_state=runtime_state)
        seq = list(state.recent_viewed_video_ids)
        if not seq:
            return pd.DataFrame(columns=["user_id", "video_id"])
        return pd.DataFrame({"user_id": [int(user_id)] * len(seq), "video_id": seq})

    def get_popular_candidates(
        self,
        user_id: int,
        topk: int,
        runtime_state: Optional[Any] = None,
    ) -> pd.DataFrame:
        if self.popular_items is None or self.popular_items.empty:
            return pd.DataFrame(columns=["user_id", "video_id", "recall_source", "source_score", "source_rank"])
        history_df = self.build_seen_history_df(user_id, runtime_state=runtime_state)
        return generate_popular_candidates(
            user_ids=[int(user_id)],
            popular_items=self.popular_items,
            topk=int(topk),
            history_df=history_df,
            exclude_seen=True,
        )

    def build_single_user_rank_store(
        self,
        item_encoder: Any,
        user_id: int,
        runtime_state: Optional[Any],
        max_seq_len: int,
    ) -> RankFeatureStore:
        state = self.get_effective_user_state(int(user_id), runtime_state=runtime_state)
        raw_history = np.asarray(list(state.recent_viewed_video_ids)[-max(max_seq_len * 3, max_seq_len) :], dtype=np.int64)
        encoded_history = item_encoder.transform(pd.Series(raw_history)).astype("int32").to_numpy() if len(raw_history) else np.array([], dtype=np.int32)
        author_counts: dict[Any, int] = {}
        tag_counts: dict[Any, int] = {}
        for video_id in raw_history.tolist():
            author = self._item_author_map.get(int(video_id))
            tag = self._item_tag_map.get(int(video_id))
            if author is not None:
                author_counts[author] = author_counts.get(author, 0) + 1
            if tag is not None:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return RankFeatureStore(
            user_features=self.user_features,
            item_features=self.item_features,
            train_split=self.train_split,
            history_raw={int(user_id): raw_history},
            history_encoded={int(user_id): encoded_history},
            user_author_counts={int(user_id): author_counts},
            user_tag_counts={int(user_id): tag_counts},
            item_to_author=dict(self._item_author_map),
            item_to_tag=dict(self._item_tag_map),
        )

    def prepare_rerank_frame(
        self,
        frame: pd.DataFrame,
        rerank_cfg: Mapping[str, Any],
        reference_split: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        out = frame.copy()
        missing_meta_cols = [col for col in ["author_id", "tag", "upload_date"] if col not in out.columns]
        if missing_meta_cols:
            out = out.merge(self._item_meta, on="video_id", how="left")
        ref_value = rerank_cfg["rerank"]["split_reference_dates"][reference_split]
        ref_date = pd.to_datetime(str(int(ref_value)), format="%Y%m%d")
        if "upload_date" in out.columns:
            upload_dt = pd.to_datetime(out["upload_date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
            out["freshness_days"] = (ref_date - upload_dt).dt.days.clip(lower=0).fillna(9999).astype("float32")
        else:
            out["freshness_days"] = 9999.0

        for col in ["rank_score", "coarse_score", "merged_score"]:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype("float32")
        out["author_id"] = out["author_id"].fillna(-1)
        out["tag"] = out["tag"].fillna(-1)
        return out
