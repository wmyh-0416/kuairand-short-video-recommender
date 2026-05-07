from __future__ import annotations

import importlib.util
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.recall.candidate_merge import merge_candidates
from src.recall.graph_emb import generate_graph_emb_candidates, load_graph_item_embeddings
from src.recall.itemcf import generate_itemcf_candidates, load_itemcf_neighbors
from src.recall.popular import generate_popular_candidates, load_popular_items
from src.utils.paths import artifacts_dir, processed_path


def _recall_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["recall"]["output"]["recall_dir"]


def _metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["recall"]["output"].get("metrics_dir", "metrics")


def _load_split(cfg: Mapping[str, Any], split: str) -> pd.DataFrame:
    splits_dir = cfg["recall"]["processed"].get("splits_dir", "splits")
    return pd.read_parquet(processed_path(cfg, Path(splits_dir) / f"{split}.parquet"))


def _positive_pairs(split_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in split_df.columns:
        raise KeyError(f"Split data missing label column: {label_col}")
    return (
        split_df.loc[split_df[label_col] > 0, ["user_id", "video_id"]]
        .drop_duplicates()
        .assign(label=1)
    )


def attach_labels(candidates: pd.DataFrame, split_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if candidates.empty:
        out = candidates.copy()
        out["label"] = pd.Series(dtype="int8")
        return out
    labels = _positive_pairs(split_df, label_col)
    out = candidates.merge(labels, on=["user_id", "video_id"], how="left")
    out["label"] = out["label"].fillna(0).astype("int8")
    return out


def _history_for_split(split: str, train_df: pd.DataFrame) -> pd.DataFrame | None:
    # For training candidates, we keep positives in the candidate pool. For
    # validation/test, exclude items already consumed in train history.
    if split == "train":
        return None
    return train_df[["user_id", "video_id"]].drop_duplicates()


def _split_generation_cfg(cfg: Mapping[str, Any], split: str) -> dict[str, Any]:
    split_cfg = deepcopy(dict(cfg))
    if split == "train":
        split_cfg["recall"]["popular"]["exclude_seen"] = False
        split_cfg["recall"]["itemcf"]["exclude_seen"] = False
        split_cfg["recall"]["twotower"]["exclude_seen"] = False
        split_cfg["recall"]["graph_emb"]["exclude_seen"] = False
    return split_cfg


def _filter_seen_and_topk(
    candidates: pd.DataFrame,
    history_df: pd.DataFrame | None,
    topk: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    out = candidates.copy()
    if history_df is not None and not history_df.empty:
        seen = history_df[["user_id", "video_id"]].drop_duplicates()
        seen["_seen"] = 1
        out = out.merge(seen, on=["user_id", "video_id"], how="left")
        out = out.loc[out["_seen"].isna()].drop(columns=["_seen"])
    out = out.sort_values(
        ["user_id", "source_score", "source_rank", "video_id"],
        ascending=[True, False, True, True],
    )
    return out.groupby("user_id", as_index=False, group_keys=False).head(topk).reset_index(drop=True)


def _twotower_source_name(cfg: Mapping[str, Any]) -> str:
    return "twotower_faiss" if bool(cfg["recall"]["twotower"].get("use_faiss", False)) else "twotower"


def _resolve_optional_artifact_path(cfg: Mapping[str, Any], raw_path: Any) -> Path | None:
    if raw_path is None:
        return None
    value = str(raw_path).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = artifacts_dir(cfg) / path
    return path.resolve()


def _resolve_faiss_index_path(cfg: Mapping[str, Any]) -> Path:
    tower_cfg = cfg["recall"]["twotower"]
    explicit_path = _resolve_optional_artifact_path(cfg, tower_cfg.get("faiss_index_path"))
    if explicit_path is not None:
        return explicit_path

    index_type = str(tower_cfg.get("faiss_index_type", "hnsw")).lower()
    if index_type == "flat":
        file_name = "faiss_flat.index"
    elif index_type == "ivf":
        file_name = "faiss_ivf.index"
    elif index_type == "hnsw":
        file_name = "faiss_hnsw.index"
    else:
        raise ValueError(f"Unsupported recall.twotower.faiss_index_type: {index_type}")
    return (artifacts_dir(cfg) / "faiss" / file_name).resolve()


def _resolve_faiss_id_map_path(cfg: Mapping[str, Any]) -> Path:
    tower_cfg = cfg["recall"]["twotower"]
    explicit_path = _resolve_optional_artifact_path(cfg, tower_cfg.get("faiss_id_map_path"))
    if explicit_path is not None:
        return explicit_path
    return (artifacts_dir(cfg) / "faiss" / "video_id_map.pkl").resolve()


def _retrieve_topk_faiss(
    cfg: Mapping[str, Any],
    user_ids: np.ndarray,
    user_vectors: np.ndarray,
    topk: int,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    from src.recall.faiss_service import FaissRecallService

    tower_cfg = cfg["recall"]["twotower"]
    source_name = _twotower_source_name(cfg)
    index_path = _resolve_faiss_index_path(cfg)
    id_map_path = _resolve_faiss_id_map_path(cfg)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Build it first with scripts/09_build_faiss_index.py "
            "or disable recall.twotower.use_faiss."
        )
    if not id_map_path.exists():
        raise FileNotFoundError(
            f"FAISS id map not found: {id_map_path}. Build it first with scripts/09_build_faiss_index.py "
            "or disable recall.twotower.use_faiss."
        )

    service = FaissRecallService(
        index_path=index_path,
        id_map_path=id_map_path,
        normalize=bool(tower_cfg.get("faiss_normalize", True)),
    )
    rows: list[dict[str, object]] = []
    for idx, (user_id, user_vector) in enumerate(zip(user_ids, user_vectors)):
        results = service.recall(user_vector, top_k=topk)
        for rank, result in enumerate(results, start=1):
            rows.append(
                {
                    "user_id": int(user_id),
                    "video_id": int(result["video_id"]),
                    "recall_source": source_name,
                    "source_score": float(result["recall_score"]),
                    "source_rank": int(rank),
                }
            )
        if logger and (idx + 1) % 1000 == 0:
            logger.info("FAISS two-tower recall progress: users=%d/%d", idx + 1, len(user_ids))
    return pd.DataFrame(rows, columns=["user_id", "video_id", "recall_source", "source_score", "source_rank"])


def generate_twotower_candidates_if_available(
    cfg: Mapping[str, Any],
    split: str,
    user_ids: np.ndarray,
    user_sequences: pd.DataFrame,
    item_features: pd.DataFrame,
    history_df: pd.DataFrame | None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    tower_cfg = cfg["recall"]["twotower"]
    topk = int(tower_cfg.get("candidate_topk", 300))
    if not tower_cfg.get("enabled", True):
        return pd.DataFrame()
    if importlib.util.find_spec("torch") is None:
        if logger:
            logger.warning("Torch is not installed; skip two-tower candidate generation.")
        return pd.DataFrame()

    use_faiss = bool(tower_cfg.get("use_faiss", False))
    if use_faiss:
        from src.recall.twotower import (
            encode_users,
            load_twotower_checkpoint,
            resolve_device,
        )
    else:
        from src.recall.twotower import (
            encode_items,
            encode_users,
            load_twotower_checkpoint,
            resolve_device,
            retrieve_topk_numpy,
        )

    recall_dir = _recall_dir(cfg)
    ckpt_path = recall_dir / cfg["recall"]["output"]["twotower_checkpoint_file"]
    item_emb_path = recall_dir / cfg["recall"]["output"]["twotower_item_embeddings_file"]
    if not ckpt_path.exists():
        if logger:
            logger.warning("Two-tower checkpoint not found: %s; skip.", ckpt_path)
        return pd.DataFrame()

    device = resolve_device(str(tower_cfg.get("device", "auto")))
    model, encoders, saved_tower_cfg = load_twotower_checkpoint(ckpt_path, device=device)
    max_seq_len = int(saved_tower_cfg.get("max_seq_len", tower_cfg.get("max_seq_len", 50)))
    sequence_col = str(saved_tower_cfg.get("sequence_col", tower_cfg.get("sequence_col", "watch_seq")))

    user_ids, user_vectors = encode_users(
        model,
        user_ids=np.asarray(user_ids, dtype=np.int64),
        user_sequences=user_sequences,
        encoders=encoders,
        max_seq_len=max_seq_len,
        sequence_col=sequence_col,
        device=device,
    )
    fetch_topk = topk * 3 if tower_cfg.get("exclude_seen", True) and history_df is not None else topk
    if use_faiss:
        fetch_topk = max(fetch_topk, int(tower_cfg.get("faiss_top_k", fetch_topk)))
        candidates = _retrieve_topk_faiss(
            cfg=cfg,
            user_ids=user_ids,
            user_vectors=user_vectors,
            topk=fetch_topk,
            logger=logger,
        )
    else:
        if item_emb_path.exists():
            item_payload = np.load(item_emb_path)
            item_ids = item_payload["item_ids"]
            item_vectors = item_payload["item_vectors"]
        else:
            item_ids, item_vectors = encode_items(
                model,
                item_features=item_features,
                encoders=encoders,
                device=device,
            )
        candidates = retrieve_topk_numpy(
            user_ids=user_ids,
            user_vectors=user_vectors,
            item_ids=item_ids,
            item_vectors=item_vectors,
            topk=fetch_topk,
        )
    if tower_cfg.get("exclude_seen", True):
        candidates = _filter_seen_and_topk(candidates, history_df=history_df, topk=topk)
    if logger:
        logger.info(
            "Generated %s candidates for %s: rows=%d",
            _twotower_source_name(cfg),
            split,
            candidates.shape[0],
        )
    return candidates


def generate_graph_emb_candidates_if_available(
    cfg: Mapping[str, Any],
    split: str,
    user_ids: np.ndarray,
    train_df: pd.DataFrame,
    history_df: pd.DataFrame | None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    graph_cfg = cfg["recall"]["graph_emb"]
    if not graph_cfg.get("enabled", True):
        return pd.DataFrame()

    recall_dir = _recall_dir(cfg)
    emb_path = recall_dir / cfg["recall"]["output"]["graph_emb_item_embeddings_file"]
    if not emb_path.exists():
        if logger:
            logger.warning("graph_emb item embeddings not found: %s; skip.", emb_path)
        return pd.DataFrame()

    item_ids, item_vectors = load_graph_item_embeddings(emb_path)
    candidates = generate_graph_emb_candidates(
        user_ids=user_ids,
        train_df=train_df,
        item_ids=item_ids,
        item_vectors=item_vectors,
        cfg=cfg,
        history_df=history_df,
    )
    if logger:
        logger.info("Generated graph_emb candidates for %s: rows=%d", split, candidates.shape[0])
    return candidates


def generate_candidates_for_split(
    cfg: Mapping[str, Any],
    split: str,
    train_df: pd.DataFrame,
    split_df: pd.DataFrame,
    user_sequences: pd.DataFrame,
    item_features: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    split_cfg = _split_generation_cfg(cfg, split)
    recall_dir = _recall_dir(cfg)
    label_col = cfg["recall"].get("positive_label_col", "is_positive")
    user_ids = np.sort(split_df["user_id"].dropna().astype("int64").unique())
    history_df = _history_for_split(split, train_df)

    frames: dict[str, pd.DataFrame] = {}
    if cfg["recall"]["popular"].get("enabled", True):
        popular_path = recall_dir / cfg["recall"]["output"]["popular_items_file"]
        popular_items = load_popular_items(popular_path)
        frames["popular"] = generate_popular_candidates(
            user_ids=user_ids,
            popular_items=popular_items,
            topk=int(cfg["recall"]["popular"].get("candidate_topk", cfg["recall"]["popular"].get("topk", 300))),
            history_df=history_df,
            exclude_seen=bool(split_cfg["recall"]["popular"].get("exclude_seen", True)),
        )
        if logger:
            logger.info("Generated popular candidates for %s: rows=%d", split, frames["popular"].shape[0])

    if cfg["recall"]["itemcf"].get("enabled", True):
        itemcf_path = recall_dir / cfg["recall"]["output"]["itemcf_neighbors_file"]
        neighbors = load_itemcf_neighbors(itemcf_path)
        frames["itemcf"] = generate_itemcf_candidates(
            user_sequences=user_sequences[user_sequences["user_id"].isin(user_ids)],
            neighbors_df=neighbors,
            topk=int(cfg["recall"]["itemcf"].get("topk", 300)),
            cfg=split_cfg,
        )
        if logger:
            logger.info("Generated ItemCF candidates for %s: rows=%d", split, frames["itemcf"].shape[0])

    twotower_source = _twotower_source_name(split_cfg)
    frames[twotower_source] = generate_twotower_candidates_if_available(
        cfg=split_cfg,
        split=split,
        user_ids=user_ids,
        user_sequences=user_sequences,
        item_features=item_features,
        history_df=history_df,
        logger=logger,
    )
    frames["graph_emb"] = generate_graph_emb_candidates_if_available(
        cfg=split_cfg,
        split=split,
        user_ids=user_ids,
        train_df=train_df,
        history_df=history_df,
        logger=logger,
    )

    merged = merge_candidates(list(frames.values()), cfg=cfg)
    merged = attach_labels(merged, split_df=split_df, label_col=label_col)
    if logger:
        logger.info(
            "Merged recall candidates for %s: rows=%d users=%d positives=%d",
            split,
            merged.shape[0],
            merged["user_id"].nunique() if not merged.empty else 0,
            int(merged["label"].sum()) if "label" in merged.columns else 0,
        )
    return merged, frames


def evaluate_recall(
    candidates: pd.DataFrame,
    split_df: pd.DataFrame,
    raw_source_frames: Mapping[str, pd.DataFrame],
    label_col: str,
    topks: list[int],
    catalog_size: int,
) -> dict[str, Any]:
    positives = _positive_pairs(split_df, label_col)
    positives_flag = positives.rename(columns={"label": "_positive"})
    total_positive = int(positives.shape[0])
    positive_counts = positives.groupby("user_id").size().astype("int64")

    def _compute_ndcg_at_k(ranked_frame: pd.DataFrame, k: int) -> float:
        if ranked_frame.empty or positive_counts.empty:
            return 0.0
        topk_df = ranked_frame.groupby("user_id", as_index=False, group_keys=False).head(int(k)).copy()
        if topk_df.empty:
            return 0.0
        topk_df["_row_rank"] = topk_df.groupby("user_id").cumcount() + 1
        hits = topk_df.merge(positives_flag, on=["user_id", "video_id"], how="left")
        hits["_positive"] = hits["_positive"].fillna(0).astype("float32")
        hits["dcg_gain"] = hits["_positive"] / np.log2(hits["_row_rank"].astype(float) + 1.0)
        dcg = hits.groupby("user_id")["dcg_gain"].sum()

        max_pos = int(min(positive_counts.max(), int(k)))
        if max_pos <= 0:
            return 0.0
        discount_cumsum = np.concatenate(
            [
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(1.0 / np.log2(np.arange(2, max_pos + 2, dtype=np.float64))),
            ]
        )
        ideal = positive_counts.clip(upper=int(k)).map(lambda count: float(discount_cumsum[int(count)]))
        ndcg = dcg.reindex(positive_counts.index, fill_value=0.0) / ideal.replace(0.0, np.nan)
        return float(ndcg.fillna(0.0).mean())

    def _evaluate_ranked_frame(frame: pd.DataFrame, sort_cols: list[str]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "candidate_count": int(frame.shape[0]),
            "num_users": int(frame["user_id"].nunique()) if not frame.empty else 0,
            "coverage_all": float(frame["video_id"].nunique() / max(catalog_size, 1)) if not frame.empty else 0.0,
        }
        if frame.empty:
            for k in topks:
                metrics[f"recall@{k}"] = 0.0
                metrics[f"coverage@{k}"] = 0.0
                metrics[f"ndcg@{k}"] = 0.0
            return metrics

        ranked_frame = frame.sort_values(sort_cols, ascending=[True, False, True])
        for k in topks:
            topk_df = ranked_frame.groupby("user_id", as_index=False, group_keys=False).head(int(k))
            hits = topk_df.merge(positives, on=["user_id", "video_id"], how="inner")
            metrics[f"recall@{k}"] = float(hits.shape[0] / max(total_positive, 1))
            metrics[f"coverage@{k}"] = (
                float(topk_df["video_id"].nunique() / max(catalog_size, 1)) if not topk_df.empty else 0.0
            )
            metrics[f"ndcg@{k}"] = _compute_ndcg_at_k(ranked_frame, int(k))
        return metrics

    metrics: dict[str, Any] = {
        "num_candidates": int(candidates.shape[0]),
        "num_users": int(candidates["user_id"].nunique()) if not candidates.empty else 0,
        "num_positive_pairs": total_positive,
        "coverage_all": float(candidates["video_id"].nunique() / max(catalog_size, 1)) if not candidates.empty else 0.0,
    }
    overall_metrics = _evaluate_ranked_frame(candidates, sort_cols=["user_id", "merged_score", "merged_rank"])
    for key, value in overall_metrics.items():
        if key == "candidate_count":
            continue
        metrics[key] = value

    source_sets = {
        source: set(zip(df["user_id"].astype(int), df["video_id"].astype(int)))
        for source, df in raw_source_frames.items()
        if df is not None and not df.empty
    }
    overlap: dict[str, Any] = {}
    sources = sorted(source_sets)
    for i, left in enumerate(sources):
        for right in sources[i + 1 :]:
            inter = len(source_sets[left] & source_sets[right])
            union = len(source_sets[left] | source_sets[right])
            overlap[f"{left}__{right}"] = {
                "intersection": inter,
                "union": union,
                "jaccard": float(inter / union) if union else 0.0,
            }
    metrics["source_candidate_counts"] = {source: int(len(values)) for source, values in source_sets.items()}
    metrics["source_metrics"] = {
        source: _evaluate_ranked_frame(df, sort_cols=["user_id", "source_score", "source_rank"])
        for source, df in raw_source_frames.items()
        if df is not None
    }
    metrics["source_overlap"] = overlap
    return metrics


def generate_all_splits(
    cfg: Mapping[str, Any],
    splits: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    splits = splits or ["train", "val", "test"]
    recall_dir = _recall_dir(cfg)
    recall_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = _metrics_dir(cfg)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(cfg, "train")
    user_sequences = pd.read_parquet(processed_path(cfg, cfg["recall"]["processed"]["user_sequences_file"]))
    item_features = pd.read_parquet(processed_path(cfg, cfg["recall"]["processed"]["item_features_file"]))
    catalog_size = int(item_features["video_id"].nunique())
    label_col = cfg["recall"].get("positive_label_col", "is_positive")
    topks = [int(k) for k in cfg["recall"]["eval"].get("topk", [50, 100, 200])]

    all_metrics: dict[str, Any] = {}
    for split in splits:
        split_df = _load_split(cfg, split)
        candidates, raw_frames = generate_candidates_for_split(
            cfg=cfg,
            split=split,
            train_df=train_df,
            split_df=split_df,
            user_sequences=user_sequences,
            item_features=item_features,
            logger=logger,
        )
        out_path = recall_dir / f"{split}_candidates.parquet"
        candidates.to_parquet(out_path, index=False)
        if logger:
            logger.info("Saved %s recall candidates: %s", split, out_path)

        all_metrics[split] = evaluate_recall(
            candidates=candidates,
            split_df=split_df,
            raw_source_frames=raw_frames,
            label_col=label_col,
            topks=topks,
            catalog_size=catalog_size,
        )

    metrics_path = metrics_dir / cfg["recall"]["output"]["recall_metrics_file"]
    metrics_path.write_text(json.dumps(all_metrics, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved recall metrics: %s", metrics_path)
    return all_metrics
