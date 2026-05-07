from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional


def _safe_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = []
    out: list[int] = []
    for value in seq:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _safe_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = []
    out: list[float] = []
    for value in seq:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _safe_str_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = []
    out: list[str] = []
    for value in seq:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


@dataclass
class UserState:
    user_id: int
    recent_viewed_video_ids: list[int] = field(default_factory=list)
    recent_positive_video_ids: list[int] = field(default_factory=list)
    recent_liked_video_ids: list[int] = field(default_factory=list)
    recent_long_view_video_ids: list[int] = field(default_factory=list)
    recent_categories: list[str] = field(default_factory=list)
    recent_watch_times: list[float] = field(default_factory=list)
    skip_count: int = 0
    like_count: int = 0
    long_view_count: int = 0
    feedback_count: int = 0
    last_active_time: int = 0
    max_history_len: int = 50
    long_watch_ratio: float = 0.7

    @staticmethod
    def is_long_view(watch_time: float, duration: float, ratio_threshold: float) -> bool:
        if float(watch_time) <= 0 or float(duration) <= 0:
            return False
        return float(watch_time) / max(float(duration), 1e-6) >= float(ratio_threshold)

    @staticmethod
    def _parse_timestamp(raw: Any) -> int:
        if raw is None:
            return int(time.time() * 1000)
        if isinstance(raw, (int, float)):
            value = float(raw)
            return int(value * 1000) if value < 10_000_000_000 else int(value)
        text = str(raw).strip()
        if not text:
            return int(time.time() * 1000)
        try:
            if text.endswith("Z"):
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return int(time.time() * 1000)

    @staticmethod
    def _trim(seq: list[Any], max_history_len: int) -> list[Any]:
        if max_history_len <= 0:
            return list(seq)
        return list(seq[-max_history_len:])

    @classmethod
    def empty(
        cls,
        user_id: int,
        *,
        max_history_len: int,
        long_watch_ratio: float,
    ) -> "UserState":
        return cls(
            user_id=int(user_id),
            max_history_len=int(max_history_len),
            long_watch_ratio=float(long_watch_ratio),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Optional[Mapping[str, Any]],
        *,
        max_history_len: int = 50,
        long_watch_ratio: float = 0.7,
    ) -> Optional["UserState"]:
        if payload is None:
            return None
        user_id = int(payload.get("user_id", 0) or 0)
        if "recent_viewed_video_ids" in payload:
            recent_viewed = _safe_int_list(payload.get("recent_viewed_video_ids"))
            recent_positive = _safe_int_list(payload.get("recent_positive_video_ids"))
            recent_liked = _safe_int_list(payload.get("recent_liked_video_ids"))
            recent_long_view = _safe_int_list(payload.get("recent_long_view_video_ids"))
            recent_categories = _safe_str_list(payload.get("recent_categories"))
            recent_watch_times = _safe_float_list(payload.get("recent_watch_times"))
        else:
            recent_viewed = _safe_int_list(payload.get("watch_seq"))
            recent_positive = _safe_int_list(payload.get("long_watch_seq"))
            recent_liked = _safe_int_list(payload.get("like_seq"))
            recent_long_view = _safe_int_list(payload.get("long_watch_seq"))
            recent_categories = _safe_str_list(payload.get("recent_categories"))
            recent_watch_times = _safe_float_list(payload.get("recent_watch_times"))
        state = cls(
            user_id=user_id,
            recent_viewed_video_ids=cls._trim(recent_viewed, int(max_history_len)),
            recent_positive_video_ids=cls._trim(recent_positive, int(max_history_len)),
            recent_liked_video_ids=cls._trim(recent_liked, int(max_history_len)),
            recent_long_view_video_ids=cls._trim(recent_long_view, int(max_history_len)),
            recent_categories=cls._trim(recent_categories, int(max_history_len)),
            recent_watch_times=cls._trim(recent_watch_times, int(max_history_len)),
            skip_count=int(payload.get("skip_count", 0) or 0),
            like_count=int(payload.get("like_count", len(recent_liked)) or 0),
            long_view_count=int(payload.get("long_view_count", len(recent_long_view)) or 0),
            feedback_count=int(payload.get("feedback_count", 0) or 0),
            last_active_time=int(payload.get("last_active_time", payload.get("last_time_ms", 0)) or 0),
            max_history_len=int(payload.get("max_history_len", max_history_len) or max_history_len),
            long_watch_ratio=float(payload.get("long_watch_ratio", long_watch_ratio) or long_watch_ratio),
        )
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": int(self.user_id),
            "recent_viewed_video_ids": list(self.recent_viewed_video_ids),
            "recent_positive_video_ids": list(self.recent_positive_video_ids),
            "recent_liked_video_ids": list(self.recent_liked_video_ids),
            "recent_long_view_video_ids": list(self.recent_long_view_video_ids),
            "recent_categories": list(self.recent_categories),
            "recent_watch_times": list(self.recent_watch_times),
            "skip_count": int(self.skip_count),
            "like_count": int(self.like_count),
            "long_view_count": int(self.long_view_count),
            "feedback_count": int(self.feedback_count),
            "last_active_time": int(self.last_active_time),
            "max_history_len": int(self.max_history_len),
            "long_watch_ratio": float(self.long_watch_ratio),
            "history_len": int(len(self.recent_viewed_video_ids)),
            "watch_seq": list(self.recent_viewed_video_ids),
            "like_seq": list(self.recent_liked_video_ids),
            "long_watch_seq": list(self.recent_long_view_video_ids),
            "last_time_ms": int(self.last_active_time),
        }

    def get_recent_features(self) -> dict[str, Any]:
        avg_watch_time = sum(self.recent_watch_times) / len(self.recent_watch_times) if self.recent_watch_times else 0.0
        return {
            "realtime_recent_view_count": int(len(self.recent_viewed_video_ids)),
            "realtime_recent_positive_count": int(len(self.recent_positive_video_ids)),
            "realtime_recent_like_count": int(len(self.recent_liked_video_ids)),
            "realtime_recent_long_view_count": int(len(self.recent_long_view_video_ids)),
            "realtime_avg_watch_time": float(avg_watch_time),
            "realtime_skip_count": int(self.skip_count),
        }

    def get_recent_viewed_set(self, size: Optional[int] = None) -> set[int]:
        if size is None or int(size) <= 0:
            seq = self.recent_viewed_video_ids
        else:
            seq = self.recent_viewed_video_ids[-int(size) :]
        return {int(video_id) for video_id in seq}

    def update(self, feedback_event: Any, item_features: Optional[Mapping[str, Any]] = None) -> None:
        if isinstance(feedback_event, Mapping):
            payload = dict(feedback_event)
        else:
            payload = dict(getattr(feedback_event, "dict", lambda: {})())
        video_id = int(payload.get("video_id"))
        watch_time = float(payload.get("watch_time", 0.0) or 0.0)
        duration = float(payload.get("duration", 0.0) or 0.0)
        click = int(payload.get("click", 0) or 0)
        like = int(payload.get("like", 0) or 0)
        timestamp = self._parse_timestamp(payload.get("timestamp"))
        long_view = self.is_long_view(watch_time, duration, self.long_watch_ratio)
        watch_ratio = watch_time / max(duration, 1e-6) if duration > 0 else 0.0
        is_skip = (click <= 0 and watch_time <= 0) or (duration > 0 and watch_ratio < min(0.2, self.long_watch_ratio))

        if click > 0 or watch_time > 0:
            self.recent_viewed_video_ids.append(video_id)
            self.recent_watch_times.append(watch_time)
        if click > 0 or like > 0 or long_view:
            self.recent_positive_video_ids.append(video_id)
        if like > 0:
            self.recent_liked_video_ids.append(video_id)
            self.like_count += 1
        if long_view:
            self.recent_long_view_video_ids.append(video_id)
            self.long_view_count += 1
        if is_skip:
            self.skip_count += 1
        category_value = None
        if item_features is not None:
            category_value = item_features.get("category")
            if category_value is None:
                category_value = item_features.get("tag")
            if category_value is None:
                category_value = item_features.get("author_id")
        if category_value is not None:
            self.recent_categories.append(str(category_value))

        self.feedback_count += 1
        self.last_active_time = int(timestamp)
        self.recent_viewed_video_ids = self._trim(self.recent_viewed_video_ids, self.max_history_len)
        self.recent_positive_video_ids = self._trim(self.recent_positive_video_ids, self.max_history_len)
        self.recent_liked_video_ids = self._trim(self.recent_liked_video_ids, self.max_history_len)
        self.recent_long_view_video_ids = self._trim(self.recent_long_view_video_ids, self.max_history_len)
        self.recent_categories = self._trim(self.recent_categories, self.max_history_len)
        self.recent_watch_times = self._trim(self.recent_watch_times, self.max_history_len)

