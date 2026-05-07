from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Any, Deque, Dict

import numpy as np


class MetricsRegistry:
    def __init__(self, *, enabled: bool = True, latency_window_size: int = 5000) -> None:
        self.enabled = bool(enabled)
        self.latency_window_size = int(max(latency_window_size, 1))
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, Deque[float]] = {}

    def incr(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[str(name)] += int(value)

    def observe_latency(self, name: str, value_ms: float) -> None:
        if not self.enabled:
            return
        series_name = str(name)
        value = float(value_ms)
        with self._lock:
            if series_name not in self._latencies:
                self._latencies[series_name] = deque(maxlen=self.latency_window_size)
            self._latencies[series_name].append(value)

    def reset(self) -> None:
        with self._lock:
            self._counters = defaultdict(int)
            self._latencies = {}

    @staticmethod
    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float64), float(p)))

    def _latency_summary(self, series_name: str) -> dict[str, float]:
        values = list(self._latencies.get(series_name, []))
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0.0, "sum": 0.0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p50": self.percentile(values, 50),
            "p95": self.percentile(values, 95),
            "p99": self.percentile(values, 99),
            "count": float(arr.size),
            "sum": float(arr.sum()),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            latency_names = list(self._latencies.keys())
            latencies = {name: self._latency_summary(name) for name in latency_names}
        return {
            "enabled": bool(self.enabled),
            "counters": counters,
            "latency_ms": latencies,
        }

    def to_prometheus_text(self) -> str:
        snap = self.snapshot()
        counters = snap["counters"]
        latencies = snap["latency_ms"]

        lines = [
            "# HELP recommender_requests_total Total HTTP requests handled by the serving API",
            "# TYPE recommender_requests_total counter",
            f"recommender_requests_total {int(counters.get('request_count', 0))}",
            "# HELP recommender_recommend_requests_total Total recommend requests",
            "# TYPE recommender_recommend_requests_total counter",
            f"recommender_recommend_requests_total {int(counters.get('recommend_request_count', 0))}",
            "# HELP recommender_feedback_requests_total Total feedback requests",
            "# TYPE recommender_feedback_requests_total counter",
            f"recommender_feedback_requests_total {int(counters.get('feedback_request_count', 0))}",
            "# HELP recommender_health_requests_total Total health requests",
            "# TYPE recommender_health_requests_total counter",
            f"recommender_health_requests_total {int(counters.get('health_request_count', 0))}",
            "# HELP recommender_errors_total Total serving errors",
            "# TYPE recommender_errors_total counter",
            f"recommender_errors_total {int(counters.get('error_count', 0))}",
            "# HELP recommender_cache_hits_total Total recommendation cache hits",
            "# TYPE recommender_cache_hits_total counter",
            f"recommender_cache_hits_total {int(counters.get('cache_hit_count', 0))}",
            "# HELP recommender_cache_misses_total Total recommendation cache misses",
            "# TYPE recommender_cache_misses_total counter",
            f"recommender_cache_misses_total {int(counters.get('cache_miss_count', 0))}",
            "# HELP recommender_cache_invalidations_total Total recommendation cache invalidations",
            "# TYPE recommender_cache_invalidations_total counter",
            f"recommender_cache_invalidations_total {int(counters.get('cache_invalidation_count', 0))}",
            "# HELP recommender_user_state_reads_total Total user state reads",
            "# TYPE recommender_user_state_reads_total counter",
            f"recommender_user_state_reads_total {int(counters.get('user_state_read_count', 0))}",
            "# HELP recommender_user_state_writes_total Total user state writes",
            "# TYPE recommender_user_state_writes_total counter",
            f"recommender_user_state_writes_total {int(counters.get('user_state_write_count', 0))}",
            "# HELP recommender_recommended_items_total Total items returned by /recommend",
            "# TYPE recommender_recommended_items_total counter",
            f"recommender_recommended_items_total {int(counters.get('recommended_items_count', 0))}",
        ]

        for stage_name, summary in latencies.items():
            count = float(summary.get("count", 0.0))
            total_sum = float(summary.get("sum", 0.0))
            p50 = float(summary.get("p50", 0.0))
            p95 = float(summary.get("p95", 0.0))
            p99 = float(summary.get("p99", 0.0))
            lines.extend(
                [
                    "# TYPE recommender_latency_ms summary",
                    f'recommender_latency_ms{{stage="{stage_name}",quantile="0.5"}} {p50}',
                    f'recommender_latency_ms{{stage="{stage_name}",quantile="0.95"}} {p95}',
                    f'recommender_latency_ms{{stage="{stage_name}",quantile="0.99"}} {p99}',
                    f'recommender_latency_ms_sum{{stage="{stage_name}"}} {total_sum}',
                    f'recommender_latency_ms_count{{stage="{stage_name}"}} {count}',
                ]
            )
        return "\n".join(lines) + "\n"
