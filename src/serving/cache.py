from __future__ import annotations

import json
import time
from dataclasses import dataclass
from hashlib import md5
from threading import Lock
from typing import Any, Optional, Tuple

from src.serving.monitoring import MetricsRegistry
from src.serving.user_state import UserState


@dataclass
class CacheStats:
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_invalidation_count: int = 0
    user_state_read_count: int = 0
    user_state_write_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        total = self.cache_hit_count + self.cache_miss_count
        return {
            "cache_hit_count": int(self.cache_hit_count),
            "cache_miss_count": int(self.cache_miss_count),
            "cache_invalidation_count": int(self.cache_invalidation_count),
            "user_state_read_count": int(self.user_state_read_count),
            "user_state_write_count": int(self.user_state_write_count),
            "cache_hit_rate": float(self.cache_hit_count / max(total, 1)),
        }


class CacheManager:
    def __init__(self, cfg: dict[str, Any], logger: Optional[Any] = None, metrics_registry: Optional[MetricsRegistry] = None) -> None:
        cache_cfg = cfg["serving"]["cache"]
        feedback_cfg = cfg["serving"]["feedback"]
        self.logger = logger
        self.metrics_registry = metrics_registry
        self.namespace = str(cache_cfg.get("namespace", "kuairand_recsys"))
        self.use_redis = bool(cache_cfg.get("use_redis", False))
        self.recommendation_ttl_seconds = int(cache_cfg.get("recommendation_ttl_seconds", 30))
        self.user_state_ttl_seconds = int(cache_cfg.get("user_state_ttl_seconds", 604800))
        self.max_history_len = int(feedback_cfg.get("max_history_len", 50))
        self.long_watch_ratio = float(feedback_cfg.get("long_watch_ratio", 0.7))
        self.redis_client: Optional[Any] = None
        self.redis_connected = False
        self._memory: dict[str, Tuple[Optional[float], Any]] = {}
        self._recommend_index: dict[str, set[str]] = {}
        self._stats = CacheStats()
        self._stats_lock = Lock()
        if self.use_redis:
            self._connect_redis(str(cache_cfg.get("redis_url", "redis://localhost:6379/0")))

    def _connect_redis(self, redis_url: str) -> None:
        try:
            import redis
        except ImportError as exc:
            if self.logger:
                self.logger.warning("Redis client is not installed; falling back to in-memory cache: %s", exc)
            return
        try:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            self.redis_client = client
            self.redis_connected = True
            if self.logger:
                self.logger.info("Connected to Redis cache: %s", redis_url)
        except Exception as exc:  # pragma: no cover - depends on runtime service.
            if self.logger:
                self.logger.warning("Redis is unavailable; falling back to in-memory cache: %s", exc)
            self.redis_client = None
            self.redis_connected = False

    @staticmethod
    def _ttl_deadline(ttl_seconds: Optional[int]) -> Optional[float]:
        if ttl_seconds is None or ttl_seconds <= 0:
            return None
        return time.time() + float(ttl_seconds)

    def _memory_get(self, cache_key: str) -> Optional[Any]:
        entry = self._memory.get(cache_key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and time.time() > expires_at:
            self._memory.pop(cache_key, None)
            return None
        return value

    def _memory_set(self, cache_key: str, payload: Any, ttl_seconds: Optional[int]) -> None:
        self._memory[cache_key] = (self._ttl_deadline(ttl_seconds), payload)

    def _memory_delete(self, cache_key: str) -> bool:
        return self._memory.pop(cache_key, None) is not None

    def _user_state_key(self, user_id: str) -> str:
        return f"user_state:{self.namespace}:{user_id}"

    def _rec_cache_key(self, user_id: str, context_hash: str) -> str:
        return f"rec_cache:{self.namespace}:{user_id}:{context_hash}"

    def _rec_index_key(self, user_id: str) -> str:
        return f"rec_cache_index:{self.namespace}:{user_id}"

    def _popular_key(self, cache_key: str) -> str:
        return f"popular:{self.namespace}:{cache_key}"

    def _record_recommend_cache_result(self, hit: bool) -> None:
        with self._stats_lock:
            if hit:
                self._stats.cache_hit_count += 1
            else:
                self._stats.cache_miss_count += 1
        if self.metrics_registry is not None:
            self.metrics_registry.incr("cache_hit_count" if hit else "cache_miss_count")

    def _record_user_state_read(self) -> None:
        with self._stats_lock:
            self._stats.user_state_read_count += 1
        if self.metrics_registry is not None:
            self.metrics_registry.incr("user_state_read_count")

    def _record_user_state_write(self) -> None:
        with self._stats_lock:
            self._stats.user_state_write_count += 1
        if self.metrics_registry is not None:
            self.metrics_registry.incr("user_state_write_count")

    def _record_invalidation(self, count: int) -> None:
        with self._stats_lock:
            self._stats.cache_invalidation_count += int(max(count, 0))
        if self.metrics_registry is not None and count > 0:
            self.metrics_registry.incr("cache_invalidation_count", int(count))

    def _json_get_by_key(self, storage_key: str) -> Optional[Any]:
        if self.redis_client is not None:
            try:
                payload = self.redis_client.get(storage_key)
            except Exception as exc:  # pragma: no cover - runtime dependent.
                if self.logger:
                    self.logger.warning("Redis get failed for %s: %s", storage_key, exc)
                payload = None
            if payload is None:
                return None
            return json.loads(payload)
        return self._memory_get(storage_key)

    def _json_set_by_key(self, storage_key: str, payload: Any, ttl_seconds: Optional[int]) -> None:
        if self.redis_client is not None:
            try:
                self.redis_client.set(storage_key, json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)
                return
            except Exception as exc:  # pragma: no cover - runtime dependent.
                if self.logger:
                    self.logger.warning("Redis set failed for %s: %s", storage_key, exc)
        self._memory_set(storage_key, payload, ttl_seconds=ttl_seconds)

    def _delete_by_key(self, storage_key: str) -> bool:
        deleted = False
        if self.redis_client is not None:
            try:
                deleted = bool(self.redis_client.delete(storage_key))
            except Exception as exc:  # pragma: no cover - runtime dependent.
                if self.logger:
                    self.logger.warning("Redis delete failed for %s: %s", storage_key, exc)
        if self._memory_delete(storage_key):
            deleted = True
        return deleted

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        self._record_user_state_read()
        payload = self._json_get_by_key(self._user_state_key(user_id))
        return UserState.from_dict(
            payload,
            max_history_len=self.max_history_len,
            long_watch_ratio=self.long_watch_ratio,
        )

    def set_user_state(self, user_id: str, state: UserState, ttl_seconds: Optional[int] = None) -> bool:
        self._record_user_state_write()
        self._json_set_by_key(
            self._user_state_key(user_id),
            state.to_dict(),
            ttl_seconds=int(ttl_seconds or self.user_state_ttl_seconds),
        )
        return self.redis_connected

    def delete_user_state(self, user_id: str) -> bool:
        return self._delete_by_key(self._user_state_key(user_id))

    def get_recommendation_cache(self, user_id: str, context_hash: str) -> Optional[dict[str, Any]]:
        payload = self._json_get_by_key(self._rec_cache_key(user_id, context_hash))
        self._record_recommend_cache_result(hit=payload is not None)
        return dict(payload) if isinstance(payload, dict) else None

    def set_recommendation_cache(
        self,
        user_id: str,
        context_hash: str,
        result: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        ttl = int(ttl_seconds or self.recommendation_ttl_seconds)
        storage_key = self._rec_cache_key(user_id, context_hash)
        self._json_set_by_key(storage_key, result, ttl_seconds=ttl)
        if self.redis_client is not None:
            index_key = self._rec_index_key(user_id)
            try:
                self.redis_client.sadd(index_key, context_hash)
                self.redis_client.expire(index_key, ttl)
            except Exception as exc:  # pragma: no cover - runtime dependent.
                if self.logger:
                    self.logger.warning("Redis recommendation index update failed for %s: %s", user_id, exc)
        else:
            self._recommend_index.setdefault(user_id, set()).add(context_hash)
        return self.redis_connected

    def invalidate_recommendation_cache(self, user_id: str) -> int:
        removed = 0
        if self.redis_client is not None:
            index_key = self._rec_index_key(user_id)
            try:
                hashes = self.redis_client.smembers(index_key)
                for context_hash in hashes or []:
                    if self._delete_by_key(self._rec_cache_key(user_id, str(context_hash))):
                        removed += 1
                self.redis_client.delete(index_key)
            except Exception as exc:  # pragma: no cover - runtime dependent.
                if self.logger:
                    self.logger.warning("Redis recommendation invalidation failed for %s: %s", user_id, exc)
        hashes = self._recommend_index.pop(user_id, set())
        for context_hash in hashes:
            if self._delete_by_key(self._rec_cache_key(user_id, context_hash)):
                removed += 1
        self._record_invalidation(removed)
        return int(removed)

    def get_popular_candidates(self, cache_key: str) -> Optional[list[dict[str, Any]]]:
        payload = self._json_get_by_key(self._popular_key(cache_key))
        return list(payload) if isinstance(payload, list) else None

    def set_popular_candidates(self, cache_key: str, items: list[dict[str, Any]], ttl_seconds: int) -> None:
        self._json_set_by_key(self._popular_key(cache_key), items, ttl_seconds=ttl_seconds)

    def get_cache_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            payload = self._stats.to_dict()
        payload["redis_connected"] = bool(self.redis_connected)
        return payload

    @property
    def cache_hit_rate(self) -> float:
        return float(self.get_cache_stats()["cache_hit_rate"])

    @staticmethod
    def hash_context(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return md5(encoded).hexdigest()
