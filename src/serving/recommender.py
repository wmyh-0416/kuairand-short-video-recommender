from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from hashlib import md5
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.prerank.features import build_feature_frame, transform_features
from src.prerank.infer import select_topk
from src.rank.features import build_rank_frame
from src.rerank.rules import rerank_candidates
from src.serving.health import build_health_response
from src.serving.monitoring import MetricsRegistry
from src.serving.schemas import FeedbackRequest, RecommendationRequest
from src.serving.user_state import UserState


def _stable_int(namespace: str, value: str) -> int:
    digest = md5(f"{namespace}:{value}".encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) % 1_000_000_000 + 1_000_000_000


def normalize_user_id(raw_user_id: str) -> tuple[str, int]:
    try:
        numeric = int(raw_user_id)
        return str(numeric), numeric
    except (TypeError, ValueError):
        return f"anon:{raw_user_id}", _stable_int("user", str(raw_user_id))


def normalize_video_id(raw_video_id: str) -> int:
    try:
        return int(raw_video_id)
    except (TypeError, ValueError):
        return _stable_int("video", str(raw_video_id))


class OnlineRecommender:
    def __init__(
        self,
        serving_cfg: Mapping[str, Any],
        model_loader: Any,
        feature_store: Any,
        cache_manager: Any,
        metrics_registry: MetricsRegistry,
        logger: Optional[Any] = None,
    ) -> None:
        self.serving_cfg = serving_cfg
        self.model_loader = model_loader
        self.feature_store = feature_store
        self.cache_manager = cache_manager
        self.metrics_registry = metrics_registry
        self.logger = logger

        feedback_cfg = serving_cfg["serving"]["feedback"]
        monitoring_cfg = serving_cfg["serving"].get("monitoring", {})
        serving_dir = Path(serving_cfg["paths"]["artifacts_dir"]).expanduser().resolve() / feedback_cfg.get("serving_dir", "serving")
        serving_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_log_path = serving_dir / feedback_cfg.get("feedback_log_file", "feedback_log.jsonl")
        self.request_log_path = serving_dir / monitoring_cfg.get("request_log_file", feedback_cfg.get("request_log_file", "request_log.jsonl"))

    def health(self) -> dict[str, Any]:
        self.metrics_registry.incr("request_count")
        self.metrics_registry.incr("health_request_count")
        return build_health_response(self.model_loader, self.cache_manager)

    def metrics_snapshot(self) -> dict[str, Any]:
        self.metrics_registry.incr("request_count")
        cache_stats = self.cache_manager.get_cache_stats()
        monitor_snapshot = self.metrics_registry.snapshot()
        counters = monitor_snapshot.get("counters", {})
        latency_summary = monitor_snapshot.get("latency_ms", {})
        total_latency = latency_summary.get("total_latency_ms", {"mean": 0.0, "p95": 0.0})
        out: dict[str, Any] = {}
        out.update(
            {
                "request_count": int(counters.get("request_count", 0)),
                "recommend_request_count": int(counters.get("recommend_request_count", 0)),
                "feedback_request_count": int(counters.get("feedback_request_count", 0)),
                "feedback_count": int(counters.get("feedback_request_count", 0)),
                "health_request_count": int(counters.get("health_request_count", 0)),
                "error_count": int(counters.get("error_count", 0)),
                "average_latency_ms": float(total_latency.get("mean", 0.0)),
                "p95_latency_ms": float(total_latency.get("p95", 0.0)),
                "cache_hit_rate": float(cache_stats.get("cache_hit_rate", 0.0)),
                "degraded_mode_count": int(counters.get("degraded_mode_count", 0)),
                "cache_hit_count": int(cache_stats.get("cache_hit_count", 0)),
                "cache_miss_count": int(cache_stats.get("cache_miss_count", 0)),
                "cache_invalidation_count": int(cache_stats.get("cache_invalidation_count", 0)),
                "user_state_read_count": int(cache_stats.get("user_state_read_count", 0)),
                "user_state_write_count": int(cache_stats.get("user_state_write_count", 0)),
                "recommended_items_count": int(counters.get("recommended_items_count", 0)),
                "latency_ms": {
                    "total": self._latency_block(latency_summary, "total_latency_ms"),
                    "recall": self._latency_block(latency_summary, "recall_latency_ms"),
                    "prerank": self._latency_block(latency_summary, "prerank_latency_ms"),
                    "rank": self._latency_block(latency_summary, "rank_latency_ms"),
                    "rerank": self._latency_block(latency_summary, "rerank_latency_ms"),
                    "feedback": self._latency_block(latency_summary, "feedback_latency_ms"),
                },
                "cache": {
                    "hit": int(cache_stats.get("cache_hit_count", 0)),
                    "miss": int(cache_stats.get("cache_miss_count", 0)),
                    "hit_rate": float(cache_stats.get("cache_hit_rate", 0.0)),
                    "invalidation": int(cache_stats.get("cache_invalidation_count", 0)),
                },
                "user_state": {
                    "read": int(cache_stats.get("user_state_read_count", 0)),
                    "write": int(cache_stats.get("user_state_write_count", 0)),
                },
            }
        )
        return out

    def recommend(self, request: RecommendationRequest) -> dict[str, Any]:
        self.metrics_registry.incr("request_count")
        self.metrics_registry.incr("recommend_request_count")
        request_id = request.request_id or uuid.uuid4().hex
        user_key, user_id = normalize_user_id(request.user_id)
        top_k = max(1, min(int(request.top_k), int(self.serving_cfg["serving"]["api"].get("max_top_k", 50))))
        context_hash = self._recommend_context_hash(top_k=top_k, context=request.context)
        cached = self.cache_manager.get_recommendation_cache(user_key, context_hash)
        if cached is not None:
            started = time.perf_counter()
            items = cached.get("items", [])
            response = {
                "request_id": request_id,
                "user_id": request.user_id,
                "items": items,
                "degraded_mode": bool(cached.get("degraded_mode", False)),
                "latency_ms": {
                    "total": float((time.perf_counter() - started) * 1000.0),
                    "recall": 0.0,
                    "prerank": 0.0,
                    "rank": 0.0,
                    "rerank": 0.0,
                },
            }
            self.metrics_registry.observe_latency("total_latency_ms", response["latency_ms"]["total"])
            if response["degraded_mode"]:
                self.metrics_registry.incr("degraded_mode_count")
            self.metrics_registry.incr("recommended_items_count", len(items))
            self._write_request_log(
                {
                    "event": "recommend_cache_hit",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "request_id": request_id,
                    "user_id": request.user_id,
                    "top_k": top_k,
                    "returned_items": len(items),
                    "cache_hit": True,
                    "context_hash": context_hash,
                    "degraded_mode": response["degraded_mode"],
                    "total_latency_ms": response["latency_ms"]["total"],
                    "recall_latency_ms": 0.0,
                    "prerank_latency_ms": 0.0,
                    "rank_latency_ms": 0.0,
                    "rerank_latency_ms": 0.0,
                    "context": request.context,
                }
            )
            return response

        started = time.perf_counter()
        latencies_ms = {"recall": 0.0, "prerank": 0.0, "rank": 0.0, "rerank": 0.0}
        degraded_reasons: list[str] = []

        runtime_state = self._load_runtime_state(user_key=user_key, user_id=user_id)
        candidates, recall_reason = self._recall_candidates(user_id=user_id, runtime_state=runtime_state)
        if recall_reason is not None:
            degraded_reasons.append(recall_reason)
        candidates = self._filter_recent_viewed_candidates(candidates, runtime_state=runtime_state)
        candidates = self._supplement_with_popular(
            user_id=user_id,
            runtime_state=runtime_state,
            candidates=candidates,
            min_candidates=max(top_k, int(self.serving_cfg["serving"]["pipeline"].get("prerank_top_k", 100))),
        )
        latencies_ms["recall"] = (time.perf_counter() - started) * 1000.0
        if candidates.empty:
            raise RuntimeError("No recall candidates available. FAISS and fallback recall are both unavailable.")

        prerank_start = time.perf_counter()
        preranked = self._run_prerank(candidates=candidates, user_id=user_id, runtime_state=runtime_state, top_k=top_k)
        latencies_ms["prerank"] = (time.perf_counter() - prerank_start) * 1000.0
        if not self.model_loader.loaded_components.get("prerank", False):
            degraded_reasons.append("prerank_unavailable")

        rank_start = time.perf_counter()
        ranked = self._run_rank(preranked=preranked, user_id=user_id, runtime_state=runtime_state)
        latencies_ms["rank"] = (time.perf_counter() - rank_start) * 1000.0
        if not self.model_loader.loaded_components.get("rank", False):
            degraded_reasons.append("rank_unavailable")

        rerank_start = time.perf_counter()
        final_df = self._run_rerank(ranked=ranked, top_k=top_k)
        latencies_ms["rerank"] = (time.perf_counter() - rerank_start) * 1000.0
        if not self.model_loader.loaded_components.get("rerank", False):
            degraded_reasons.append("rerank_unavailable")

        degraded_mode = len(degraded_reasons) > 0
        items = self._format_items(final_df, default_reason="|".join(degraded_reasons) if degraded_reasons else None)
        total_ms = float((time.perf_counter() - started) * 1000.0)

        response = {
            "request_id": request_id,
            "user_id": request.user_id,
            "items": items,
            "degraded_mode": degraded_mode,
            "latency_ms": {
                "total": total_ms,
                "recall": float(latencies_ms["recall"]),
                "prerank": float(latencies_ms["prerank"]),
                "rank": float(latencies_ms["rank"]),
                "rerank": float(latencies_ms["rerank"]),
            },
        }
        self.cache_manager.set_recommendation_cache(
            user_id=user_key,
            context_hash=context_hash,
            result={"items": items, "degraded_mode": degraded_mode},
            ttl_seconds=int(self.serving_cfg["serving"]["cache"].get("recommendation_ttl_seconds", 300)),
        )
        self.metrics_registry.observe_latency("total_latency_ms", total_ms)
        self.metrics_registry.observe_latency("recall_latency_ms", float(latencies_ms["recall"]))
        self.metrics_registry.observe_latency("prerank_latency_ms", float(latencies_ms["prerank"]))
        self.metrics_registry.observe_latency("rank_latency_ms", float(latencies_ms["rank"]))
        self.metrics_registry.observe_latency("rerank_latency_ms", float(latencies_ms["rerank"]))
        if degraded_mode:
            self.metrics_registry.incr("degraded_mode_count")
        self.metrics_registry.incr("recommended_items_count", len(items))
        self._write_request_log(
            {
                "event": "recommend",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
                "user_id": request.user_id,
                "internal_user_id": user_id,
                "top_k": top_k,
                "returned_items": len(items),
                "cache_hit": False,
                "context": request.context,
                "degraded_mode": degraded_mode,
                "degraded_reasons": degraded_reasons,
                "total_latency_ms": total_ms,
                "recall_latency_ms": float(latencies_ms["recall"]),
                "prerank_latency_ms": float(latencies_ms["prerank"]),
                "rank_latency_ms": float(latencies_ms["rank"]),
                "rerank_latency_ms": float(latencies_ms["rerank"]),
            }
        )
        return response

    def feedback(self, request: FeedbackRequest) -> dict[str, Any]:
        self.metrics_registry.incr("request_count")
        self.metrics_registry.incr("feedback_request_count")
        started = time.perf_counter()
        user_key, user_id = normalize_user_id(request.user_id)
        video_id = normalize_video_id(request.video_id)
        state = self._load_runtime_state(user_key=user_key, user_id=user_id)
        item_features = self.feature_store.get_item_features(video_id)
        state.update(
            {
                "video_id": int(video_id),
                "watch_time": float(request.watch_time),
                "duration": float(request.duration),
                "click": int(request.click),
                "like": int(request.like),
                "timestamp": request.timestamp,
            },
            item_features=item_features,
        )
        persisted_to_redis = self.cache_manager.set_user_state(
            user_id=user_key,
            state=state,
            ttl_seconds=int(self.serving_cfg["serving"]["cache"].get("user_state_ttl_seconds", 86400)),
        )
        invalidated_count = self.cache_manager.invalidate_recommendation_cache(user_key)
        long_view = UserState.is_long_view(
            watch_time=float(request.watch_time),
            duration=float(request.duration),
            ratio_threshold=float(self.serving_cfg["serving"]["feedback"].get("long_watch_ratio", 0.7)),
        )
        self._write_jsonl(
            self.feedback_log_path,
            {
                "event": "feedback",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_id": request.user_id,
                "internal_user_id": user_id,
                "video_id": request.video_id,
                "internal_video_id": video_id,
                "watch_time": float(request.watch_time),
                "duration": float(request.duration),
                "click": int(request.click),
                "like": int(request.like),
                "long_view": bool(long_view),
                "state_summary": state.get_recent_features(),
                "history_len": int(len(state.recent_viewed_video_ids)),
                "cache_invalidated": bool(invalidated_count > 0),
                "persisted_to_redis": bool(persisted_to_redis),
            },
        )
        feedback_latency_ms = float((time.perf_counter() - started) * 1000.0)
        self.metrics_registry.observe_latency("feedback_latency_ms", feedback_latency_ms)
        return {
            "status": "ok",
            "user_id": request.user_id,
            "video_id": request.video_id,
            "state_updated": True,
            "persisted_to_file": True,
            "persisted_to_redis": bool(persisted_to_redis),
            "history_len": int(len(state.recent_viewed_video_ids)),
            "recent_viewed_count": int(len(state.recent_viewed_video_ids)),
            "like_count": int(state.like_count),
            "long_view_count": int(state.long_view_count),
            "cache_invalidated": bool(invalidated_count > 0),
            "recent_viewed_contains_video": bool(int(video_id) in state.get_recent_viewed_set()),
        }

    def record_recommend_error(self, request: RecommendationRequest, exc: Exception) -> None:
        self.metrics_registry.incr("error_count")
        request_id = request.request_id or ""
        self._write_request_log(
            {
                "event": "recommend_error",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "request_id": request_id,
                "user_id": request.user_id,
                "top_k": int(request.top_k),
                "returned_items": 0,
                "degraded_mode": False,
                "cache_hit": False,
                "total_latency_ms": 0.0,
                "recall_latency_ms": 0.0,
                "prerank_latency_ms": 0.0,
                "rank_latency_ms": 0.0,
                "rerank_latency_ms": 0.0,
                "error": str(exc),
            }
        )

    def record_feedback_error(self, request: FeedbackRequest, exc: Exception) -> None:
        self.metrics_registry.incr("error_count")
        self._write_jsonl(
            self.feedback_log_path,
            {
                "event": "feedback_error",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_id": request.user_id,
                "video_id": request.video_id,
                "watch_time": float(request.watch_time),
                "duration": float(request.duration),
                "click": int(request.click),
                "like": int(request.like),
                "error": str(exc),
            },
        )

    @staticmethod
    def _latency_block(latency_summary: Mapping[str, Any], key: str) -> dict[str, float]:
        block = latency_summary.get(key, {})
        return {
            "mean": float(block.get("mean", 0.0)),
            "p50": float(block.get("p50", 0.0)),
            "p95": float(block.get("p95", 0.0)),
            "p99": float(block.get("p99", 0.0)),
        }

    @staticmethod
    def _recommend_context_hash(top_k: int, context: Mapping[str, Any]) -> str:
        return CacheHashHelper.hash({"top_k": int(top_k), "context": dict(context)})

    def _load_runtime_state(self, user_key: str, user_id: int) -> UserState:
        cached_state = self.cache_manager.get_user_state(user_key)
        if cached_state is not None:
            return cached_state
        base_state = self.feature_store.get_base_user_state(user_id)
        if base_state is not None:
            return base_state
        feedback_cfg = self.serving_cfg["serving"]["feedback"]
        return UserState.empty(
            user_id=user_id,
            max_history_len=int(feedback_cfg.get("max_history_len", 50)),
            long_watch_ratio=float(feedback_cfg.get("long_watch_ratio", 0.7)),
        )

    def _recall_candidates(
        self,
        user_id: int,
        runtime_state: Optional[UserState],
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        pipeline_cfg = self.serving_cfg["serving"]["pipeline"]
        fallback_cfg = self.serving_cfg["serving"]["fallback"]
        recall_top_k = int(pipeline_cfg.get("recall_top_k", 500))
        user_sequences = self.feature_store.build_user_sequences_frame(user_id=user_id, runtime_state=runtime_state)

        user_embedding = None
        reason: Optional[str] = None
        prefer_realtime_encode = runtime_state is not None and int(runtime_state.feedback_count) > 0
        if not prefer_realtime_encode:
            user_embedding = self.model_loader.get_user_embedding(user_id)
        if user_embedding is None:
            user_embedding = self.model_loader.encode_user_embedding(user_id=user_id, user_sequences=user_sequences)
        if user_embedding is None:
            user_embedding = self.model_loader.get_user_embedding(user_id)
        if user_embedding is None and bool(fallback_cfg.get("allow_synthetic_mean_user", True)):
            user_embedding = self.model_loader.mean_user_embedding
            if user_embedding is not None:
                reason = "synthetic_mean_user_embedding"

        if user_embedding is not None and self.model_loader.faiss_service is not None:
            results = self.model_loader.recall_faiss(user_embedding=user_embedding, top_k=recall_top_k)
            return self._build_candidate_frame(user_id=user_id, recall_results=results, recall_source="twotower_faiss"), reason

        if bool(fallback_cfg.get("allow_popular", True)):
            cache_key = f"popular:{recall_top_k}"
            cached = self.cache_manager.get_popular_candidates(cache_key)
            if cached is not None:
                popular_df = pd.DataFrame(cached)
            else:
                popular_df = self.feature_store.get_popular_candidates(user_id=user_id, topk=int(fallback_cfg.get("popular_top_k", recall_top_k)), runtime_state=runtime_state)
                self.cache_manager.set_popular_candidates(
                    cache_key=cache_key,
                    items=popular_df.to_dict("records"),
                    ttl_seconds=int(self.serving_cfg["serving"]["cache"].get("popular_ttl_seconds", 3600)),
                )
            return self._build_candidate_frame_from_df(popular_df), "cold_start_popular"

        return pd.DataFrame(), "recall_unavailable"

    def _build_candidate_frame(
        self,
        user_id: int,
        recall_results: list[dict[str, Any]],
        recall_source: str,
    ) -> pd.DataFrame:
        rows = []
        for rank, row in enumerate(recall_results, start=1):
            rows.append(
                {
                    "user_id": int(user_id),
                    "video_id": int(row["video_id"]),
                    "recall_source": recall_source,
                    "source_score": float(row.get("recall_score", row.get("source_score", 0.0))),
                    "source_rank": int(rank),
                }
            )
        return self._build_candidate_frame_from_df(pd.DataFrame(rows))

    @staticmethod
    def _build_candidate_frame_from_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "video_id",
                    "recall_source",
                    "source_score",
                    "source_rank",
                    "merged_score",
                    "source_count",
                    "merged_rank",
                ]
            )
        out = df.copy()
        out["user_id"] = out["user_id"].astype("int64")
        out["video_id"] = out["video_id"].astype("int64")
        out["recall_source"] = out["recall_source"].astype("string")
        out["source_score"] = pd.to_numeric(out["source_score"], errors="coerce").fillna(0.0).astype("float32")
        out["source_rank"] = pd.to_numeric(out["source_rank"], errors="coerce").fillna(1_000_000).astype("int64")
        out = out.sort_values(["user_id", "source_score", "source_rank", "video_id"], ascending=[True, False, True, True]).reset_index(drop=True)
        out["merged_score"] = out["source_score"].astype("float32")
        out["source_count"] = 1
        out["merged_rank"] = out.groupby("user_id").cumcount() + 1
        unique_sources = out["recall_source"].dropna().astype(str).unique().tolist()
        for source in unique_sources:
            source_mask = out["recall_source"].astype(str) == str(source)
            out[f"{source}_score"] = np.where(source_mask, out["source_score"], np.nan).astype("float32")
            out[f"{source}_rank"] = np.where(source_mask, out["source_rank"], np.nan).astype("float32")
        return out

    def _filter_recent_viewed_candidates(self, candidates: pd.DataFrame, runtime_state: Optional[UserState]) -> pd.DataFrame:
        if candidates.empty or runtime_state is None:
            return candidates
        realtime_cfg = self.serving_cfg["serving"].get("realtime_features", {})
        if not bool(realtime_cfg.get("enabled", True)) or not bool(realtime_cfg.get("filter_recent_viewed", True)):
            return candidates
        recent_viewed = runtime_state.get_recent_viewed_set(size=int(realtime_cfg.get("recent_viewed_filter_size", 50)))
        if not recent_viewed:
            return candidates
        filtered = candidates.loc[~candidates["video_id"].astype("int64").isin(recent_viewed)].reset_index(drop=True)
        return filtered

    def _supplement_with_popular(
        self,
        user_id: int,
        runtime_state: Optional[UserState],
        candidates: pd.DataFrame,
        min_candidates: int,
    ) -> pd.DataFrame:
        if len(candidates) >= int(min_candidates):
            return candidates
        if not bool(self.serving_cfg["serving"]["fallback"].get("allow_popular", True)):
            return candidates
        popular_df = self.feature_store.get_popular_candidates(
            user_id=user_id,
            topk=max(int(min_candidates), int(self.serving_cfg["serving"]["fallback"].get("popular_top_k", min_candidates))),
            runtime_state=runtime_state,
        )
        if popular_df.empty:
            return candidates
        combined = pd.concat([candidates, popular_df], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=["user_id", "video_id"], keep="first").reset_index(drop=True)
        return self._build_candidate_frame_from_df(combined)

    @staticmethod
    def _inject_realtime_features(frame: pd.DataFrame, runtime_state: Optional[UserState]) -> pd.DataFrame:
        if frame.empty or runtime_state is None:
            return frame
        out = frame.copy()
        for col, value in runtime_state.get_recent_features().items():
            out[col] = value
        return out

    def _run_prerank(
        self,
        candidates: pd.DataFrame,
        user_id: int,
        runtime_state: Optional[UserState],
        top_k: int,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates.copy()
        if self.model_loader.prerank_bundle is None:
            out = self._inject_realtime_features(candidates.copy(), runtime_state)
            out["coarse_score"] = out["merged_score"].astype("float32")
            out["coarse_rank"] = out.groupby("user_id").cumcount() + 1
            out["prerank_score"] = out["coarse_score"]
            return out

        prerank_cfg = self.model_loader.component_cfgs["prerank"]
        feature_spec = self.model_loader.prerank_bundle["feature_spec"]
        split = str(self.serving_cfg["serving"]["pipeline"].get("reference_split", "test"))
        user_sequences = self.feature_store.build_user_sequences_frame(user_id=user_id, runtime_state=runtime_state)
        feat_df = build_feature_frame(
            candidates=candidates,
            cfg=prerank_cfg,
            user_features=self.feature_store.user_features,
            item_features=self.feature_store.item_features,
            user_sequences=user_sequences,
            train_stats=self.feature_store.train_stats,
            split=split,
        )
        feat_df = self._inject_realtime_features(feat_df, runtime_state)
        x = transform_features(feat_df, feature_spec)
        out = self._inject_realtime_features(candidates.copy(), runtime_state)
        out["coarse_score"] = self.model_loader.predict_prerank_scores(x)
        out["prerank_score"] = out["coarse_score"].astype("float32")
        selected = select_topk(out, topk=max(top_k, int(self.serving_cfg["serving"]["pipeline"].get("prerank_top_k", 100))))
        return selected

    def _run_rank(
        self,
        preranked: pd.DataFrame,
        user_id: int,
        runtime_state: Optional[UserState],
    ) -> pd.DataFrame:
        if preranked.empty:
            return preranked.copy()
        if self.model_loader.rank_model is None or self.model_loader.rank_spec is None:
            out = self._inject_realtime_features(preranked.copy(), runtime_state)
            if "coarse_score" not in out.columns:
                out["coarse_score"] = out["merged_score"].astype("float32")
            out["rank_score"] = out["coarse_score"].astype("float32")
            out = out.sort_values(["user_id", "rank_score", "video_id"], ascending=[True, False, True]).reset_index(drop=True)
            out["rank_position"] = out.groupby("user_id").cumcount() + 1
            return out

        runtime_cfg = self.model_loader.rank_runtime_cfg
        if runtime_cfg is None:
            raise RuntimeError("Rank runtime config is unavailable.")
        split = str(self.serving_cfg["serving"]["pipeline"].get("reference_split", "test"))
        empty_labels = pd.DataFrame(
            {
                "user_id": pd.Series(dtype="int64"),
                "video_id": pd.Series(dtype="int64"),
                **{task: pd.Series(dtype="int8") for task in self.model_loader.rank_spec.tasks},
            }
        )
        request_store = self.feature_store.build_single_user_rank_store(
            item_encoder=self.model_loader.rank_spec.encoders["video_id"],
            user_id=user_id,
            runtime_state=runtime_state,
            max_seq_len=int(self.model_loader.rank_spec.max_seq_len),
        )
        frame = build_rank_frame(
            preranked,
            split_df=empty_labels,
            store=request_store,
            cfg=runtime_cfg,
            split=split,
            precomputed_labels=empty_labels,
        )
        frame = self._inject_realtime_features(frame, runtime_state)
        ranked = self.model_loader.predict_rank_frame(frame, request_store=request_store)
        ranked = self._inject_realtime_features(ranked, runtime_state)
        ranked = ranked.sort_values(
            ["user_id", "rank_score", "coarse_score", "video_id"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
        ranked["rank_position"] = ranked.groupby("user_id").cumcount() + 1
        return ranked

    def _run_rerank(self, ranked: pd.DataFrame, top_k: int) -> pd.DataFrame:
        if ranked.empty:
            return ranked.copy()
        if self.model_loader.rerank_cfg is None:
            score_col = "rank_score" if "rank_score" in ranked.columns else "coarse_score"
            return ranked.sort_values(["user_id", score_col, "video_id"], ascending=[True, False, True]).groupby("user_id", as_index=False, group_keys=False).head(top_k).reset_index(drop=True)

        rerank_cfg = deepcopy(self.model_loader.rerank_cfg)
        rerank_cfg["rerank"]["topk"] = int(top_k)
        prepared = self.feature_store.prepare_rerank_frame(
            frame=ranked,
            rerank_cfg=rerank_cfg,
            reference_split=str(self.serving_cfg["serving"]["pipeline"].get("rerank_reference_split", "test")),
        )
        final_df = rerank_candidates(prepared, rerank_cfg, logger=None)
        if final_df.empty:
            return ranked.sort_values(["user_id", "rank_score", "video_id"], ascending=[True, False, True]).groupby("user_id", as_index=False, group_keys=False).head(top_k).reset_index(drop=True)
        return final_df.sort_values(["user_id", "final_rank"]).reset_index(drop=True)

    @staticmethod
    def _format_items(frame: pd.DataFrame, default_reason: Optional[str] = None) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        score_col = "rerank_score" if "rerank_score" in frame.columns else "rank_score" if "rank_score" in frame.columns else "coarse_score"
        items: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            reason = getattr(row, "adjustment_reason", None) or default_reason
            if reason == "base":
                reason = None
            items.append(
                {
                    "video_id": str(int(getattr(row, "video_id"))),
                    "score": float(getattr(row, score_col, 0.0)),
                    "recall_score": float(getattr(row, "source_score", 0.0)) if hasattr(row, "source_score") else None,
                    "recall_source": str(getattr(row, "recall_source", "")) if hasattr(row, "recall_source") else None,
                    "prerank_score": float(getattr(row, "coarse_score", 0.0)) if hasattr(row, "coarse_score") else None,
                    "rank_score": float(getattr(row, "rank_score", getattr(row, "coarse_score", 0.0))) if hasattr(row, "coarse_score") or hasattr(row, "rank_score") else None,
                    "reason": reason,
                }
            )
        return items

    def _write_request_log(self, payload: Mapping[str, Any]) -> None:
        self._write_jsonl(self.request_log_path, payload)
        if self.logger:
            self.logger.info(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


class CacheHashHelper:
    @staticmethod
    def hash(payload: dict[str, Any]) -> str:
        return md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
