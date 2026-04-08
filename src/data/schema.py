from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


USER_ID = "user_id"
ITEM_ID = "video_id"
AUTHOR_ID = "author_id"
DATE = "date"
HOURMIN = "hourmin"
TIME_MS = "time_ms"


LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
]

USER_FEATURE_COLUMNS = [
    "user_id",
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num",
    "follow_user_num_range",
    "fans_user_num",
    "fans_user_num_range",
    "friend_user_num",
    "friend_user_num_range",
    "register_days",
    "register_days_range",
    "onehot_feat0",
    "onehot_feat1",
    "onehot_feat2",
    "onehot_feat3",
    "onehot_feat4",
    "onehot_feat5",
    "onehot_feat6",
    "onehot_feat7",
    "onehot_feat8",
    "onehot_feat9",
    "onehot_feat10",
    "onehot_feat11",
    "onehot_feat12",
    "onehot_feat13",
    "onehot_feat14",
    "onehot_feat15",
    "onehot_feat16",
    "onehot_feat17",
]

ITEM_BASIC_COLUMNS = [
    "video_id",
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "video_duration",
    "server_width",
    "server_height",
    "music_id",
    "music_type",
    "tag",
]

ITEM_STAT_COLUMNS = [
    "video_id",
    "counts",
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
    "direct_comment_user_num",
    "reply_comment_user_num",
    "share_all_cnt",
    "share_all_user_num",
    "outsite_share_all_cnt",
]

LABEL_COLUMNS = ["like", "finish", "long_watch", "watch_ratio"]

SPLIT_NAMES = ["train", "val", "test"]


@dataclass(frozen=True)
class RawFileSpec:
    name: str
    required_columns: list[str]


RAW_FILE_SPECS = {
    "log": RawFileSpec(name="log", required_columns=LOG_COLUMNS),
    "user_features": RawFileSpec(
        name="user_features",
        required_columns=USER_FEATURE_COLUMNS,
    ),
    "item_basic": RawFileSpec(
        name="item_basic",
        required_columns=ITEM_BASIC_COLUMNS,
    ),
    "item_stat": RawFileSpec(
        name="item_stat",
        required_columns=ITEM_STAT_COLUMNS,
    ),
}


def missing_columns(columns: Iterable[str], required_columns: Iterable[str]) -> list[str]:
    existing = set(columns)
    return [col for col in required_columns if col not in existing]


def validate_columns(
    columns: Iterable[str],
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    missing = missing_columns(columns, required_columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")
