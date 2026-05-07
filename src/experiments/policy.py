from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.recall.faiss_service import FaissRecallService
from src.utils.paths import artifacts_dir, processed_dir


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
        return first_line == "version https://git-lfs.github.com/spec/v1"
    except OSError:
        return False


def _safe_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


@dataclass
class PolicyItem:
    video_id: int
    score: float | None = None
    source: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": int(self.video_id),
            "score": None if self.score is None else float(self.score),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass
class PolicyContext:
    experiment_cfg: Mapping[str, Any]
    component_cfgs: Mapping[str, Mapping[str, Any]]
    split: str
    max_top_k: int
    logger: Any | None
    user_sequences: pd.DataFrame
    item_features: pd.DataFrame
    user_history_map: dict[int, list[int]] = field(init=False)
    user_seen_map: dict[int, set[int]] = field(init=False)
    item_tag_map: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.user_history_map = {}
        self.user_seen_map = {}
        for row in self.user_sequences.itertuples(index=False):
            history_raw = getattr(row, "watch_seq", None)
            if history_raw is None:
                history: list[int] = []
            else:
                history = [int(x) for x in np.asarray(history_raw).tolist()]
            user_id = int(getattr(row, "user_id"))
            self.user_history_map[user_id] = history
            self.user_seen_map[user_id] = set(history)

        item_meta = self.item_features[["video_id", "tag"]].drop_duplicates("video_id")
        self.item_tag_map = {
            int(row.video_id): str(row.tag)
            for row in item_meta.itertuples(index=False)
        }


@dataclass
class ArtifactStatus:
    label: str
    path: str
    available: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "available": bool(self.available),
            "detail": self.detail,
        }


class RecommendationPolicy:
    def __init__(
        self,
        name: str,
        policy_type: str,
        context: PolicyContext,
        logger: Any | None = None,
    ) -> None:
        self.name = str(name)
        self.policy_type = str(policy_type)
        self.context = context
        self.logger = logger
        self.artifact_statuses: list[ArtifactStatus] = []
        self.missing_artifacts: list[dict[str, str]] = []
        self.warnings: list[str] = []

    def recommend(self, user_id: int, top_k: int) -> list[int]:
        return [item.video_id for item in self.recommend_with_details(user_id=user_id, top_k=top_k)]

    def recommend_with_details(self, user_id: int, top_k: int) -> list[PolicyItem]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.policy_type,
            "artifacts": [status.to_dict() for status in self.artifact_statuses],
            "missing_artifacts": list(self.missing_artifacts),
            "warnings": list(self.warnings),
        }

    def _register_artifact(self, label: str, path: Path, available: bool, detail: str | None = None) -> None:
        status = ArtifactStatus(label=label, path=str(path), available=available, detail=detail)
        self.artifact_statuses.append(status)
        if not available:
            self.missing_artifacts.append(
                {
                    "policy": self.name,
                    "label": label,
                    "path": str(path),
                    "detail": detail or "artifact unavailable",
                }
            )

    def _log(self, message: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(message, *args)


class RankingArtifactSource:
    def __init__(
        self,
        label: str,
        path: Path,
        rank_col: str,
        score_cols: list[str],
        max_top_k: int,
        logger: Any | None = None,
    ) -> None:
        self.label = label
        self.path = path
        self.rank_col = rank_col
        self.score_cols = list(score_cols)
        self.max_top_k = int(max_top_k)
        self.logger = logger
        self.available = False
        self.detail: str | None = None
        self.score_col: str | None = None
        self.items_by_user: dict[int, list[PolicyItem]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.detail = "file not found"
            return
        if _is_lfs_pointer(self.path):
            self.detail = "git-lfs pointer found instead of materialized parquet"
            return
        try:
            df = pd.read_parquet(self.path)
        except Exception as exc:  # pragma: no cover - depends on environment/filesystem state.
            self.detail = f"failed to read parquet: {exc}"
            return

        if "user_id" not in df.columns or "video_id" not in df.columns or self.rank_col not in df.columns:
            self.detail = f"required columns missing: user_id/video_id/{self.rank_col}"
            return

        score_col = next((col for col in self.score_cols if col in df.columns), None)
        cols = ["user_id", "video_id", self.rank_col]
        if score_col:
            cols.append(score_col)
        out = df[cols].copy()
        out[self.rank_col] = pd.to_numeric(out[self.rank_col], errors="coerce")
        out = out.dropna(subset=[self.rank_col])
        out = out.sort_values(["user_id", self.rank_col, "video_id"])
        out = out.groupby("user_id", sort=False, as_index=False, group_keys=False).head(self.max_top_k)

        items_by_user: dict[int, list[PolicyItem]] = {}
        for user_id, group in out.groupby("user_id", sort=False):
            rows: list[PolicyItem] = []
            for row in group.itertuples(index=False):
                score = _safe_score(getattr(row, score_col)) if score_col else None
                rows.append(
                    PolicyItem(
                        video_id=int(getattr(row, "video_id")),
                        score=score,
                        source=self.label,
                        reason=self.label,
                    )
                )
            items_by_user[int(user_id)] = rows

        self.available = True
        self.score_col = score_col
        self.items_by_user = items_by_user
        if self.logger:
            self.logger.info(
                "Loaded ranking artifact source %s from %s users=%d score_col=%s",
                self.label,
                self.path,
                len(self.items_by_user),
                self.score_col,
            )

    def get(self, user_id: int) -> list[PolicyItem]:
        return self.items_by_user.get(int(user_id), [])


class PopularPolicy(RecommendationPolicy):
    def __init__(self, name: str, context: PolicyContext, logger: Any | None = None) -> None:
        super().__init__(name=name, policy_type="popular", context=context, logger=logger)
        self.popular_items: list[PolicyItem] = []
        self._load()

    def _popular_path(self) -> Path:
        recall_cfg = self.context.component_cfgs["recall"]
        return (
            artifacts_dir(recall_cfg)
            / recall_cfg["recall"]["output"]["recall_dir"]
            / recall_cfg["recall"]["output"]["popular_items_file"]
        )

    def _load(self) -> None:
        path = self._popular_path()
        if path.exists() and not _is_lfs_pointer(path):
            try:
                popular_df = pd.read_parquet(path)
                score_col = "source_score" if "source_score" in popular_df.columns else None
                rank_col = "popular_rank" if "popular_rank" in popular_df.columns else score_col
                if rank_col is None:
                    raise KeyError("popular_rank/source_score not found")
                popular_df = popular_df.sort_values([rank_col, "video_id"])
                self.popular_items = [
                    PolicyItem(
                        video_id=int(row.video_id),
                        score=_safe_score(getattr(row, score_col)) if score_col else None,
                        source="popular",
                        reason="popular",
                    )
                    for row in popular_df.itertuples(index=False)
                ]
                self._register_artifact("popular_items", path, True, "loaded popular artifact")
                return
            except Exception as exc:  # pragma: no cover - file-specific.
                self._register_artifact("popular_items", path, False, f"failed to read popular artifact: {exc}")
        else:
            detail = "file not found" if not path.exists() else "git-lfs pointer found instead of parquet"
            self._register_artifact("popular_items", path, False, detail)

        train_path = processed_dir(self.context.component_cfgs["recall"]) / "splits" / "train.parquet"
        try:
            train_df = pd.read_parquet(
                train_path,
                columns=["video_id", "is_click", "like", "long_watch", "finish", "time_ms"],
            )
            stats = train_df.groupby("video_id").agg(
                play_count=("is_click", "sum"),
                like_count=("like", "sum"),
                long_watch_count=("long_watch", "sum"),
                finish_count=("finish", "sum"),
                last_time_ms=("time_ms", "max"),
            )
            weights = {"play_count": 1.0, "like_count": 6.0, "long_watch_count": 3.0, "finish_count": 2.0}
            score = sum(stats[col] * weight for col, weight in weights.items())
            stats = stats.assign(source_score=score).reset_index().sort_values(["source_score", "video_id"], ascending=[False, True])
            self.popular_items = [
                PolicyItem(video_id=int(row.video_id), score=float(row.source_score), source="popular", reason="popular")
                for row in stats.itertuples(index=False)
            ]
            self._register_artifact("popular_train_fallback", train_path, True, "computed from train split")
            self.warnings.append("popular artifact missing; recomputed popular ranking from train split")
        except Exception as exc:
            self._register_artifact("popular_train_fallback", train_path, False, f"failed to build popular fallback: {exc}")
            self.warnings.append("popular policy has no available artifact and no fallback output")

    def recommend_with_details(self, user_id: int, top_k: int) -> list[PolicyItem]:
        seen = self.context.user_seen_map.get(int(user_id), set())
        out: list[PolicyItem] = []
        for item in self.popular_items:
            if item.video_id in seen:
                continue
            out.append(item)
            if len(out) >= int(top_k):
                break
        return out


class ItemCFPolicy(RecommendationPolicy):
    def __init__(self, name: str, context: PolicyContext, logger: Any | None = None) -> None:
        super().__init__(name=name, policy_type="itemcf", context=context, logger=logger)
        self.neighbor_map: dict[int, list[tuple[int, float]]] = {}
        self.max_history_len = int(context.component_cfgs["recall"]["recall"]["itemcf"].get("max_user_history_len", 100))
        self.recency_alpha = float(context.component_cfgs["recall"]["recall"]["itemcf"].get("recency_alpha", 0.92))
        self._load()

    def _neighbor_path(self) -> Path:
        recall_cfg = self.context.component_cfgs["recall"]
        return (
            artifacts_dir(recall_cfg)
            / recall_cfg["recall"]["output"]["recall_dir"]
            / recall_cfg["recall"]["output"]["itemcf_neighbors_file"]
        )

    def _load(self) -> None:
        path = self._neighbor_path()
        if not path.exists():
            self._register_artifact("itemcf_neighbors", path, False, "file not found")
            self.warnings.append("itemcf neighbors artifact is missing")
            return
        if _is_lfs_pointer(path):
            self._register_artifact("itemcf_neighbors", path, False, "git-lfs pointer found instead of parquet")
            self.warnings.append("itemcf neighbors artifact is a git-lfs pointer")
            return
        try:
            neighbors = pd.read_parquet(path, columns=["video_id", "neighbor_video_id", "similarity", "neighbor_rank"])
            neighbors = neighbors.sort_values(["video_id", "neighbor_rank", "neighbor_video_id"])
            neighbor_map: dict[int, list[tuple[int, float]]] = {}
            for video_id, group in neighbors.groupby("video_id", sort=False):
                neighbor_map[int(video_id)] = [
                    (int(row.neighbor_video_id), float(row.similarity))
                    for row in group.itertuples(index=False)
                ]
            self.neighbor_map = neighbor_map
            self._register_artifact("itemcf_neighbors", path, True, "loaded itemcf neighbors")
        except Exception as exc:
            self._register_artifact("itemcf_neighbors", path, False, f"failed to read itemcf neighbors: {exc}")
            self.warnings.append("itemcf policy could not load neighbors")

    def recommend_with_details(self, user_id: int, top_k: int) -> list[PolicyItem]:
        history = self.context.user_history_map.get(int(user_id), [])
        seen = self.context.user_seen_map.get(int(user_id), set())
        if not history or not self.neighbor_map:
            return []
        scores: dict[int, float] = {}
        recent_history = history[-self.max_history_len :]
        reversed_history = list(reversed(recent_history))
        for idx, video_id in enumerate(reversed_history):
            weight = self.recency_alpha ** idx
            for neighbor_id, similarity in self.neighbor_map.get(int(video_id), []):
                if neighbor_id in seen:
                    continue
                scores[neighbor_id] = scores.get(neighbor_id, 0.0) + float(similarity) * weight

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: int(top_k)]
        return [
            PolicyItem(video_id=int(video_id), score=float(score), source="itemcf", reason="itemcf")
            for video_id, score in ranked
        ]


class TwotowerFaissPolicy(RecommendationPolicy):
    def __init__(self, name: str, context: PolicyContext, logger: Any | None = None) -> None:
        super().__init__(name=name, policy_type="twotower_faiss", context=context, logger=logger)
        self.faiss_service: Optional[FaissRecallService] = None
        self.user_embedding_index: dict[int, int] = {}
        self.user_vectors: np.ndarray | None = None
        self.query_cache: dict[int, list[PolicyItem]] = {}
        self._load()

    def _faiss_index_path(self) -> Path:
        index_type = str(self.context.experiment_cfg.get("faiss_index_type", "hnsw")).lower()
        return artifacts_dir(self.context.component_cfgs["recall"]) / "faiss" / f"faiss_{index_type}.index"

    def _faiss_id_map_path(self) -> Path:
        return artifacts_dir(self.context.component_cfgs["recall"]) / "faiss" / "video_id_map.pkl"

    def _user_embedding_path(self) -> Path:
        recall_cfg = self.context.component_cfgs["recall"]
        return (
            artifacts_dir(recall_cfg)
            / recall_cfg["recall"]["output"]["recall_dir"]
            / recall_cfg["recall"]["output"]["twotower_user_embeddings_file"]
        )

    def _load(self) -> None:
        index_path = self._faiss_index_path()
        id_map_path = self._faiss_id_map_path()
        try:
            self.faiss_service = FaissRecallService(index_path=index_path, id_map_path=id_map_path, normalize=True)
            self._register_artifact("faiss_index", index_path, True, "loaded faiss index")
            self._register_artifact("faiss_id_map", id_map_path, True, "loaded faiss id map")
        except Exception as exc:
            self._register_artifact("faiss_index", index_path, False, f"failed to load faiss index: {exc}")
            self._register_artifact("faiss_id_map", id_map_path, False, f"failed to load faiss id map: {exc}")
            self.warnings.append("twotower_faiss policy is unavailable because FAISS assets failed to load")
            return

        user_emb_path = self._user_embedding_path()
        try:
            payload = np.load(user_emb_path)
            user_ids = np.asarray(payload["user_ids"], dtype=np.int64)
            user_vectors = np.asarray(payload["user_vectors"], dtype=np.float32)
            self.user_embedding_index = {int(user_id): int(idx) for idx, user_id in enumerate(user_ids)}
            self.user_vectors = user_vectors
            self._register_artifact("twotower_user_embeddings", user_emb_path, True, "loaded two-tower user embeddings")
        except Exception as exc:
            self._register_artifact(
                "twotower_user_embeddings",
                user_emb_path,
                False,
                f"failed to load two-tower user embeddings: {exc}",
            )
            self.warnings.append("twotower_faiss policy is unavailable because user embeddings are missing")

    def recommend_with_details(self, user_id: int, top_k: int) -> list[PolicyItem]:
        cache_key = int(user_id)
        cached = self.query_cache.get(cache_key)
        if cached is not None and len(cached) >= int(top_k):
            return cached[: int(top_k)]

        if self.faiss_service is None or self.user_vectors is None:
            return []
        idx = self.user_embedding_index.get(int(user_id))
        if idx is None:
            return []
        user_embedding = np.asarray(self.user_vectors[idx], dtype=np.float32)
        seen = self.context.user_seen_map.get(int(user_id), set())
        search_top_k = max(int(top_k) + len(seen) + 20, int(top_k))
        results = self.faiss_service.recall(user_embedding=user_embedding, top_k=search_top_k)
        out: list[PolicyItem] = []
        for row in results:
            video_id = int(row["video_id"])
            if video_id in seen:
                continue
            out.append(
                PolicyItem(
                    video_id=video_id,
                    score=float(row["recall_score"]),
                    source="twotower_faiss",
                    reason="twotower_faiss",
                )
            )
            if len(out) >= int(top_k):
                break
        self.query_cache[cache_key] = out
        return out


class CompositeArtifactPolicy(RecommendationPolicy):
    def __init__(
        self,
        name: str,
        policy_type: str,
        context: PolicyContext,
        sources: list[RankingArtifactSource],
        logger: Any | None = None,
    ) -> None:
        super().__init__(name=name, policy_type=policy_type, context=context, logger=logger)
        self.sources = list(sources)
        for source in self.sources:
            self._register_artifact(source.label, source.path, source.available, source.detail or "loaded")
        if not any(source.available for source in self.sources):
            self.warnings.append(f"{policy_type} has no readable ranking artifacts")

    def recommend_with_details(self, user_id: int, top_k: int) -> list[PolicyItem]:
        selected: list[PolicyItem] = []
        seen_ids: set[int] = set()
        for source in self.sources:
            if not source.available:
                continue
            for item in source.get(int(user_id)):
                if item.video_id in seen_ids:
                    continue
                selected.append(item)
                seen_ids.add(item.video_id)
                if len(selected) >= int(top_k):
                    return selected
        return selected


def _load_ranking_source(
    label: str,
    path: Path,
    rank_col: str,
    score_cols: list[str],
    max_top_k: int,
    logger: Any | None,
) -> RankingArtifactSource:
    return RankingArtifactSource(
        label=label,
        path=path,
        rank_col=rank_col,
        score_cols=score_cols,
        max_top_k=max_top_k,
        logger=logger,
    )


def build_policy(policy_cfg: Mapping[str, Any], context: PolicyContext, logger: Any | None = None) -> RecommendationPolicy:
    policy_type = str(policy_cfg.get("type", "")).lower()
    policy_name = str(policy_cfg.get("name", policy_type or "policy"))

    if policy_type == "popular":
        return PopularPolicy(name=policy_name, context=context, logger=logger)

    if policy_type == "itemcf":
        return ItemCFPolicy(name=policy_name, context=context, logger=logger)

    if policy_type == "twotower_faiss":
        return TwotowerFaissPolicy(name=policy_name, context=context, logger=logger)

    recall_cfg = context.component_cfgs["recall"]
    prerank_cfg = context.component_cfgs["prerank"]
    rank_cfg = context.component_cfgs["rank"]
    rerank_cfg = context.component_cfgs["rerank"]

    recall_dir = artifacts_dir(recall_cfg) / recall_cfg["recall"]["output"]["recall_dir"]
    prerank_dir = artifacts_dir(prerank_cfg) / prerank_cfg["prerank"]["output"]["prerank_dir"]
    rank_dir = artifacts_dir(rank_cfg) / rank_cfg["rank"]["output"]["rank_dir"]
    rerank_dir = artifacts_dir(rerank_cfg) / rerank_cfg["rerank"]["output"]["rerank_dir"]

    recall_source = _load_ranking_source(
        label="recall_candidates",
        path=recall_dir / f"{context.split}_candidates.parquet",
        rank_col="merged_rank",
        score_cols=["merged_score", "source_score"],
        max_top_k=context.max_top_k,
        logger=logger,
    )
    prerank_source = _load_ranking_source(
        label="prerank",
        path=prerank_dir / prerank_cfg["prerank"]["output"][f"{context.split}_topk_file"],
        rank_col="coarse_rank",
        score_cols=["coarse_score", "merged_score", "source_score"],
        max_top_k=context.max_top_k,
        logger=logger,
    )
    rank_source = _load_ranking_source(
        label="rank",
        path=rank_dir / rank_cfg["rank"]["output"][f"{context.split}_ranked_file"],
        rank_col="rank_position",
        score_cols=["rank_score", "long_watch_score", "finish_score", "like_score"],
        max_top_k=context.max_top_k,
        logger=logger,
    )
    rerank_source = _load_ranking_source(
        label="rerank",
        path=rerank_dir / rerank_cfg["rerank"]["output"][f"{context.split}_final_file"],
        rank_col="final_rank",
        score_cols=["rerank_score", "rank_score", "long_watch_score"],
        max_top_k=context.max_top_k,
        logger=logger,
    )

    if policy_type == "full_pipeline":
        return CompositeArtifactPolicy(
            name=policy_name,
            policy_type=policy_type,
            context=context,
            sources=[rerank_source, rank_source, prerank_source, recall_source],
            logger=logger,
        )

    if policy_type == "full_pipeline_without_rerank":
        return CompositeArtifactPolicy(
            name=policy_name,
            policy_type=policy_type,
            context=context,
            sources=[rank_source, prerank_source, recall_source],
            logger=logger,
        )

    raise ValueError(f"Unsupported policy type: {policy_type}")
