from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:  # pragma: no cover - depends on runtime env.
    faiss = None

from src.utils.paths import artifacts_dir, processed_path


def require_faiss() -> Any:
    if faiss is None:
        raise ImportError(
            "FAISS is required for ANN retrieval. "
            "Install faiss-cpu or faiss-gpu before running scripts/09_build_faiss_index.py "
            "or scripts/10_test_faiss_recall.py."
        )
    return faiss


def faiss_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["faiss"]["output"]["faiss_dir"]


def metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["faiss"]["output"].get("metrics_dir", "metrics")


def recall_dir(cfg: Mapping[str, Any]) -> Path:
    return artifacts_dir(cfg) / cfg["faiss"]["input"]["recall_dir"]


def _resolve_similarity(cfg: Mapping[str, Any]) -> tuple[str, bool]:
    similarity = str(cfg["faiss"].get("similarity", "cosine")).lower()
    if similarity == "cosine":
        return similarity, True
    if similarity in {"ip", "inner_product"}:
        return "inner_product", False
    raise ValueError(f"Unsupported faiss similarity: {similarity}")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (values / norms).astype(np.float32, copy=False)


def load_video_id_map(path: str | Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FAISS video id map not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "video_ids" in payload:
        video_ids = payload["video_ids"]
    else:
        video_ids = payload
    return np.asarray(video_ids, dtype=np.int64)


def _load_twotower_item_embeddings_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Two-tower item embedding file not found: {path}")
    payload = np.load(path)
    required = {"item_ids", "item_vectors"}
    missing = required - set(payload.files)
    if missing:
        raise KeyError(f"Two-tower item embeddings missing arrays: {sorted(missing)}")
    item_ids = np.asarray(payload["item_ids"], dtype=np.int64)
    item_vectors = np.asarray(payload["item_vectors"], dtype=np.float32)
    if item_ids.ndim != 1:
        raise ValueError(f"item_ids must be 1D, got shape={item_ids.shape}")
    if item_vectors.ndim != 2:
        raise ValueError(f"item_vectors must be 2D, got shape={item_vectors.shape}")
    if item_ids.shape[0] != item_vectors.shape[0]:
        raise ValueError(
            "two-tower item embedding rows do not match item ids: "
            f"{item_ids.shape[0]} vs {item_vectors.shape[0]}"
        )
    return item_ids, item_vectors


def _load_twotower_user_embeddings_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Two-tower user embedding file not found: {path}")
    payload = np.load(path)
    required = {"user_ids", "user_vectors"}
    missing = required - set(payload.files)
    if missing:
        raise KeyError(f"Two-tower user embeddings missing arrays: {sorted(missing)}")
    user_ids = np.asarray(payload["user_ids"], dtype=np.int64)
    user_vectors = np.asarray(payload["user_vectors"], dtype=np.float32)
    if user_ids.ndim != 1:
        raise ValueError(f"user_ids must be 1D, got shape={user_ids.shape}")
    if user_vectors.ndim != 2:
        raise ValueError(f"user_vectors must be 2D, got shape={user_vectors.shape}")
    if user_ids.shape[0] != user_vectors.shape[0]:
        raise ValueError(
            "two-tower user embedding rows do not match user ids: "
            f"{user_ids.shape[0]} vs {user_vectors.shape[0]}"
        )
    return user_ids, user_vectors


def _rebuild_item_embeddings_from_checkpoint(
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    from src.recall.twotower import encode_items, load_twotower_checkpoint, resolve_device

    recall_cfg = cfg["faiss"]["input"]
    ckpt_path = recall_dir(cfg) / recall_cfg["twotower_checkpoint_file"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Two-tower checkpoint not found for FAISS rebuild: {ckpt_path}")

    item_features_path = processed_path(cfg, cfg["faiss"]["processed"]["item_features_file"])
    if not item_features_path.exists():
        raise FileNotFoundError(f"Item features not found for FAISS rebuild: {item_features_path}")

    item_features = pd.read_parquet(item_features_path)
    device = resolve_device("auto")
    model, encoders, _ = load_twotower_checkpoint(ckpt_path, device=device)
    item_ids, item_vectors = encode_items(
        model,
        item_features=item_features,
        encoders=encoders,
        device=device,
    )
    if logger:
        logger.info(
            "Rebuilt item embeddings from checkpoint: checkpoint=%s rows=%d dim=%d",
            ckpt_path,
            item_vectors.shape[0],
            item_vectors.shape[1] if item_vectors.ndim == 2 else 0,
        )
    return item_ids, item_vectors, str(ckpt_path)


def load_or_build_item_embeddings(
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    recall_cfg = cfg["faiss"]["input"]
    npz_path = recall_dir(cfg) / recall_cfg["twotower_item_embeddings_file"]
    if npz_path.exists():
        item_ids, item_vectors = _load_twotower_item_embeddings_npz(npz_path)
        if logger:
            logger.info(
                "Loaded two-tower item embeddings: path=%s rows=%d dim=%d",
                npz_path,
                item_vectors.shape[0],
                item_vectors.shape[1] if item_vectors.ndim == 2 else 0,
            )
        return item_ids, item_vectors, str(npz_path)
    return _rebuild_item_embeddings_from_checkpoint(cfg, logger=logger)


def load_or_build_user_embeddings(
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    recall_cfg = cfg["faiss"]["input"]
    user_npz_path = recall_dir(cfg) / recall_cfg["twotower_user_embeddings_file"]
    if user_npz_path.exists():
        user_ids, user_vectors = _load_twotower_user_embeddings_npz(user_npz_path)
        if logger:
            logger.info(
                "Loaded two-tower user embeddings: path=%s rows=%d dim=%d",
                user_npz_path,
                user_vectors.shape[0],
                user_vectors.shape[1] if user_vectors.ndim == 2 else 0,
            )
        return user_ids, user_vectors, str(user_npz_path)

    from src.recall.twotower import encode_users, load_twotower_checkpoint, resolve_device

    ckpt_path = recall_dir(cfg) / recall_cfg["twotower_checkpoint_file"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Two-tower checkpoint not found for user embedding rebuild: {ckpt_path}")

    user_sequences_path = processed_path(cfg, cfg["faiss"]["processed"]["user_sequences_file"])
    if not user_sequences_path.exists():
        raise FileNotFoundError(f"User sequences not found for user embedding rebuild: {user_sequences_path}")
    user_sequences = pd.read_parquet(user_sequences_path)
    if user_sequences.empty or "user_id" not in user_sequences.columns:
        raise ValueError(f"User sequences are empty or missing user_id: {user_sequences_path}")

    device = resolve_device("auto")
    model, encoders, saved_cfg = load_twotower_checkpoint(ckpt_path, device=device)
    max_seq_len = int(saved_cfg.get("max_seq_len", 50))
    sequence_col = str(saved_cfg.get("sequence_col", "watch_seq"))
    user_ids = np.sort(user_sequences["user_id"].dropna().astype("int64").unique())
    user_ids, user_vectors = encode_users(
        model,
        user_ids=user_ids,
        user_sequences=user_sequences,
        encoders=encoders,
        max_seq_len=max_seq_len,
        sequence_col=sequence_col,
        device=device,
    )
    if logger:
        logger.info(
            "Rebuilt user embeddings from checkpoint: checkpoint=%s rows=%d dim=%d",
            ckpt_path,
            user_vectors.shape[0],
            user_vectors.shape[1] if user_vectors.ndim == 2 else 0,
        )
    return user_ids, user_vectors, str(ckpt_path)


def export_video_embeddings(
    item_ids: np.ndarray,
    item_vectors: np.ndarray,
    embeddings_path: str | Path,
    id_map_path: str | Path,
    logger: Any | None = None,
) -> dict[str, Any]:
    embeddings_path = Path(embeddings_path).expanduser().resolve()
    id_map_path = Path(id_map_path).expanduser().resolve()
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, np.asarray(item_vectors, dtype=np.float32))
    payload = {
        "video_ids": np.asarray(item_ids, dtype=np.int64),
        "video_id_to_row": {int(video_id): int(row) for row, video_id in enumerate(item_ids)},
    }
    with id_map_path.open("wb") as f:
        pickle.dump(payload, f)

    if logger:
        logger.info(
            "Exported FAISS-ready video embeddings: path=%s rows=%d dim=%d",
            embeddings_path,
            item_vectors.shape[0],
            item_vectors.shape[1] if item_vectors.ndim == 2 else 0,
        )
        logger.info("Exported FAISS video id map: path=%s entries=%d", id_map_path, len(item_ids))

    return {
        "embeddings_path": str(embeddings_path),
        "id_map_path": str(id_map_path),
    }


def build_flat_index(
    item_vectors: np.ndarray,
    logger: Any | None = None,
) -> Any:
    faiss_lib = require_faiss()
    values = np.asarray(item_vectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"item_vectors must be non-empty 2D float32, got shape={values.shape}")
    index = faiss_lib.IndexFlatIP(values.shape[1])
    index.add(values)
    if logger:
        logger.info("Built FAISS flat index: rows=%d dim=%d", values.shape[0], values.shape[1])
    return index


def build_hnsw_index(
    item_vectors: np.ndarray,
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> Any:
    faiss_lib = require_faiss()
    ann_cfg = cfg["faiss"]["ann"]["hnsw"]
    values = np.asarray(item_vectors, dtype=np.float32)
    m = int(ann_cfg.get("m", 32))
    ef_construction = int(ann_cfg.get("ef_construction", 200))
    ef_search = int(ann_cfg.get("ef_search", 128))
    index = faiss_lib.IndexHNSWFlat(values.shape[1], m, faiss_lib.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(values)
    if logger:
        logger.info(
            "Built FAISS HNSW index: rows=%d dim=%d m=%d ef_construction=%d ef_search=%d",
            values.shape[0],
            values.shape[1],
            m,
            ef_construction,
            ef_search,
        )
    return index


def build_ivf_index(
    item_vectors: np.ndarray,
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> Any:
    faiss_lib = require_faiss()
    ann_cfg = cfg["faiss"]["ann"]["ivf"]
    values = np.asarray(item_vectors, dtype=np.float32)
    if values.shape[0] == 0:
        raise ValueError("Cannot build IVF index with empty item vectors.")

    requested_nlist = int(ann_cfg.get("nlist", 256))
    nlist = max(1, min(requested_nlist, values.shape[0]))
    nprobe = int(ann_cfg.get("nprobe", 32))
    train_sample_size = int(ann_cfg.get("train_sample_size", values.shape[0]))
    quantizer = faiss_lib.IndexFlatIP(values.shape[1])
    index = faiss_lib.IndexIVFFlat(quantizer, values.shape[1], nlist, faiss_lib.METRIC_INNER_PRODUCT)

    if not index.is_trained:
        if train_sample_size > 0 and values.shape[0] > train_sample_size:
            rng = np.random.default_rng(int(cfg["project"].get("random_seed", 2026)))
            sample_idx = rng.choice(values.shape[0], size=train_sample_size, replace=False)
            train_vectors = values[sample_idx]
        else:
            train_vectors = values
        index.train(train_vectors)
    index.add(values)
    index.nprobe = max(1, min(nprobe, nlist))

    if logger:
        logger.info(
            "Built FAISS IVF index: rows=%d dim=%d nlist=%d nprobe=%d trained_on=%d",
            values.shape[0],
            values.shape[1],
            nlist,
            index.nprobe,
            train_vectors.shape[0],
        )
    return index


def build_ann_index(
    item_vectors: np.ndarray,
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> tuple[Any, str]:
    index_type = str(cfg["faiss"]["ann"].get("index_type", "hnsw")).lower()
    if index_type == "hnsw":
        return build_hnsw_index(item_vectors, cfg, logger=logger), "hnsw"
    if index_type == "ivf":
        return build_ivf_index(item_vectors, cfg, logger=logger), "ivf"
    raise ValueError(f"Unsupported faiss ann.index_type: {index_type}")


def resolve_ann_index_path(cfg: Mapping[str, Any], out_dir: Path) -> Path:
    output_cfg = cfg["faiss"]["output"]
    ann_type = str(cfg["faiss"]["ann"].get("index_type", "hnsw")).lower()
    configured_name = str(output_cfg.get("ann_index_file", "")).strip()
    default_names = {"faiss_hnsw.index", "faiss_ivf.index", ""}
    if configured_name not in default_names:
        return out_dir / configured_name
    return out_dir / f"faiss_{ann_type}.index"


def save_index(index: Any, path: str | Path) -> Path:
    faiss_lib = require_faiss()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss_lib.write_index(index, str(path))
    return path


def load_index(path: str | Path) -> Any:
    faiss_lib = require_faiss()
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss_lib.read_index(str(path))


def index_file_size_bytes(path: str | Path) -> int:
    path = Path(path).expanduser().resolve()
    return int(path.stat().st_size) if path.exists() else 0


def build_faiss_artifacts(
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    _similarity, normalize = _resolve_similarity(cfg)
    out_dir = faiss_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids, item_vectors, source_path = load_or_build_item_embeddings(cfg, logger=logger)
    if normalize:
        item_vectors = l2_normalize(item_vectors)
    else:
        item_vectors = np.asarray(item_vectors, dtype=np.float32)

    output_cfg = cfg["faiss"]["output"]
    embeddings_path = out_dir / output_cfg["video_embeddings_file"]
    id_map_path = out_dir / output_cfg["video_id_map_file"]
    flat_index_path = out_dir / output_cfg["flat_index_file"]
    ann_index_path = resolve_ann_index_path(cfg, out_dir)
    build_report_path = out_dir / output_cfg["build_report_file"]

    export_info = export_video_embeddings(
        item_ids=item_ids,
        item_vectors=item_vectors,
        embeddings_path=embeddings_path,
        id_map_path=id_map_path,
        logger=logger,
    )

    t0 = time.perf_counter()
    flat_index = build_flat_index(item_vectors, logger=logger)
    flat_build_seconds = float(time.perf_counter() - t0)
    save_index(flat_index, flat_index_path)

    t1 = time.perf_counter()
    ann_index, ann_index_type = build_ann_index(item_vectors, cfg, logger=logger)
    ann_build_seconds = float(time.perf_counter() - t1)
    save_index(ann_index, ann_index_path)

    report = {
        "source_embeddings_path": source_path,
        "exported_embeddings_path": export_info["embeddings_path"],
        "exported_id_map_path": export_info["id_map_path"],
        "num_items": int(item_vectors.shape[0]),
        "embedding_dim": int(item_vectors.shape[1]) if item_vectors.ndim == 2 else 0,
        "similarity": str(cfg["faiss"].get("similarity", "cosine")).lower(),
        "normalized_before_index": bool(normalize),
        "flat_index": {
            "type": "IndexFlatIP",
            "path": str(flat_index_path),
            "build_seconds": flat_build_seconds,
            "index_size_bytes": index_file_size_bytes(flat_index_path),
            "ntotal": int(getattr(flat_index, "ntotal", 0)),
        },
        "ann_index": {
            "type": ann_index_type,
            "path": str(ann_index_path),
            "build_seconds": ann_build_seconds,
            "index_size_bytes": index_file_size_bytes(ann_index_path),
            "ntotal": int(getattr(ann_index, "ntotal", 0)),
        },
    }
    build_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved FAISS build report: %s", build_report_path)
    return report
