from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    faiss_loaded: bool
    redis_connected: bool
    degraded_mode: bool
    loaded_components: dict[str, bool]
    component_errors: dict[str, str] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = Field(default=10, ge=1, le=200)
    request_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    video_id: str
    score: float
    recall_score: Optional[float] = None
    recall_source: Optional[str] = None
    prerank_score: Optional[float] = None
    rank_score: Optional[float] = None
    reason: Optional[str] = None


class LatencyBreakdown(BaseModel):
    total: float
    recall: float
    prerank: float
    rank: float
    rerank: float


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    items: List[RecommendationItem]
    degraded_mode: bool
    latency_ms: LatencyBreakdown


class FeedbackRequest(BaseModel):
    user_id: str
    video_id: str
    watch_time: float = 0.0
    duration: float = 0.0
    click: int = 0
    like: int = 0
    timestamp: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    user_id: str
    video_id: str
    state_updated: bool
    persisted_to_file: bool
    persisted_to_redis: bool
    history_len: int
    recent_viewed_count: int = 0
    like_count: int = 0
    long_view_count: int = 0
    cache_invalidated: bool = False
    recent_viewed_contains_video: bool = False


class LatencySummaryStats(BaseModel):
    mean: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class MetricsCacheBlock(BaseModel):
    hit: int = 0
    miss: int = 0
    hit_rate: float = 0.0
    invalidation: int = 0


class MetricsUserStateBlock(BaseModel):
    read: int = 0
    write: int = 0


class MetricsResponse(BaseModel):
    request_count: int
    recommend_request_count: int
    feedback_count: int
    feedback_request_count: int = 0
    health_request_count: int = 0
    error_count: int = 0
    average_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    degraded_mode_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_invalidation_count: int
    user_state_read_count: int
    user_state_write_count: int
    recommended_items_count: int = 0
    latency_ms: Dict[str, LatencySummaryStats] = Field(default_factory=dict)
    cache: MetricsCacheBlock = Field(default_factory=MetricsCacheBlock)
    user_state: MetricsUserStateBlock = Field(default_factory=MetricsUserStateBlock)
