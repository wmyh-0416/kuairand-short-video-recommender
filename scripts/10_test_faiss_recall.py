from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.faiss_index import (
    faiss_dir,
    index_file_size_bytes,
    l2_normalize,
    load_or_build_item_embeddings,
    load_or_build_user_embeddings,
    metrics_dir,
)
from src.recall.faiss_service import FaissRecallService
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FAISS flat vs ANN recall latency and overlap.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "faiss.yaml"),
        help="Path to FAISS YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument("--num-queries", type=int, default=None, help="Override faiss.eval.num_queries.")
    parser.add_argument("--top-k", type=int, default=None, help="Override faiss.eval.top_k.")
    return parser.parse_args()


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _sample_query_embeddings(
    cfg: dict[str, Any],
    num_queries: int,
    normalize: bool,
    logger: Any | None = None,
) -> tuple[np.ndarray, list[str | int], str]:
    try:
        user_ids, user_vectors, source = load_or_build_user_embeddings(cfg, logger=logger)
        if user_vectors.shape[0] == 0:
            raise ValueError("user_vectors are empty")
        rng = np.random.default_rng(int(cfg["faiss"]["eval"].get("random_seed", 2026)))
        sample_size = min(num_queries, user_vectors.shape[0])
        sample_idx = rng.choice(user_vectors.shape[0], size=sample_size, replace=False)
        queries = np.asarray(user_vectors[sample_idx], dtype=np.float32)
        query_ids = [int(user_ids[idx]) for idx in sample_idx]
        if normalize:
            queries = l2_normalize(queries)
        return queries, query_ids, source
    except Exception as exc:
        if not bool(cfg["faiss"]["eval"].get("synthetic_if_missing", True)):
            raise
        item_ids, item_vectors, source = load_or_build_item_embeddings(cfg, logger=logger)
        rng = np.random.default_rng(int(cfg["faiss"]["eval"].get("random_seed", 2026)))
        dim = int(item_vectors.shape[1])
        queries = rng.normal(size=(num_queries, dim)).astype(np.float32)
        if normalize:
            queries = l2_normalize(queries)
        query_ids = [f"synthetic_{idx}" for idx in range(num_queries)]
        if logger:
            logger.warning(
                "Falling back to synthetic query embeddings because user embeddings are unavailable: %s",
                exc,
            )
        return queries, query_ids, f"synthetic_from::{source}"


def _run_service_queries(
    service: FaissRecallService,
    queries: np.ndarray,
    query_ids: list[str | int],
    top_k: int,
    logger: Any | None = None,
    sample_results: int = 3,
) -> tuple[list[list[int]], list[float], list[dict[str, Any]]]:
    ids_by_query: list[list[int]] = []
    latencies_ms: list[float] = []
    samples: list[dict[str, Any]] = []
    for idx, (query_id, query_vec) in enumerate(zip(query_ids, queries)):
        t0 = time.perf_counter()
        results = service.recall(query_vec, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(float(latency_ms))
        ids_by_query.append([int(row["video_id"]) for row in results])
        if idx < sample_results:
            samples.append(
                {
                    "query_id": query_id,
                    "latency_ms": float(latency_ms),
                    "results": results[: min(10, len(results))],
                }
            )
        if logger and (idx + 1) % 50 == 0:
            logger.info("Queried %d/%d FAISS requests", idx + 1, len(query_ids))
    return ids_by_query, latencies_ms, samples


def _mean_overlap_at_k(flat_ids: list[list[int]], ann_ids: list[list[int]], k: int) -> dict[str, float]:
    overlaps: list[float] = []
    for base, test in zip(flat_ids, ann_ids):
        base_top = base[:k]
        test_top = test[:k]
        if not base_top:
            overlaps.append(0.0)
            continue
        overlaps.append(float(len(set(base_top) & set(test_top)) / max(min(k, len(base_top)), 1)))
    values = np.asarray(overlaps, dtype=np.float64) if overlaps else np.asarray([0.0], dtype=np.float64)
    return {
        "mean_overlap@k": float(values.mean()),
        "p50_overlap@k": float(np.percentile(values, 50)),
        "p95_overlap@k": float(np.percentile(values, 95)),
        "min_overlap@k": float(values.min()),
        "max_overlap@k": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.processed_dir is not None:
        cfg["paths"]["processed_dir"] = args.processed_dir
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
        cfg["paths"]["logs_dir"] = str(Path(args.artifacts_dir) / "logs")
    if args.num_queries is not None:
        cfg.setdefault("faiss", {}).setdefault("eval", {})["num_queries"] = args.num_queries
    if args.top_k is not None:
        cfg.setdefault("faiss", {}).setdefault("eval", {})["top_k"] = args.top_k

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))
    logger = setup_logger(
        name="kuairand_rec.test_faiss_recall",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "10_test_faiss_recall.log",
    )

    out_dir = faiss_dir(cfg)
    metric_dir = metrics_dir(cfg)
    metric_dir.mkdir(parents=True, exist_ok=True)
    output_cfg = cfg["faiss"]["output"]
    normalize = str(cfg["faiss"].get("similarity", "cosine")).lower() == "cosine"
    top_k = int(cfg["faiss"]["eval"].get("top_k", 500))
    num_queries = int(cfg["faiss"]["eval"].get("num_queries", 200))
    sample_results = int(cfg["faiss"]["eval"].get("sample_results", 3))

    flat_path = out_dir / output_cfg["flat_index_file"]
    ann_path = out_dir / output_cfg["ann_index_file"]
    id_map_path = out_dir / output_cfg["video_id_map_file"]

    try:
        flat_service = FaissRecallService(flat_path, id_map_path, normalize=normalize)
        ann_service = FaissRecallService(ann_path, id_map_path, normalize=normalize)
    except ImportError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    queries, query_ids, query_source = _sample_query_embeddings(
        cfg=cfg,
        num_queries=num_queries,
        normalize=normalize,
        logger=logger,
    )
    logger.info(
        "Testing FAISS recall: queries=%d top_k=%d query_source=%s normalize=%s",
        len(query_ids),
        top_k,
        query_source,
        normalize,
    )

    flat_ids, flat_latencies_ms, flat_samples = _run_service_queries(
        flat_service,
        queries,
        query_ids,
        top_k=top_k,
        logger=logger,
        sample_results=sample_results,
    )
    ann_ids, ann_latencies_ms, ann_samples = _run_service_queries(
        ann_service,
        queries,
        query_ids,
        top_k=top_k,
        logger=logger,
        sample_results=sample_results,
    )

    recall_report = {
        "query_source": query_source,
        "num_queries": len(query_ids),
        "top_k": top_k,
        "flat": {
            "index_path": str(flat_path),
            "latency_ms": _latency_stats(flat_latencies_ms),
            "sample_results": flat_samples,
        },
        "ann": {
            "index_path": str(ann_path),
            "latency_ms": _latency_stats(ann_latencies_ms),
            "sample_results": ann_samples,
        },
    }
    recall_report_path = metric_dir / output_cfg["recall_report_file"]
    recall_report_path.write_text(json.dumps(recall_report, indent=2, sort_keys=True), encoding="utf-8")

    overlap = _mean_overlap_at_k(flat_ids, ann_ids, top_k)
    flat_latency = _latency_stats(flat_latencies_ms)
    ann_latency = _latency_stats(ann_latencies_ms)
    benchmark = {
        "query_source": query_source,
        "num_queries": len(query_ids),
        "top_k": top_k,
        "flat": {
            "index_type": "IndexFlatIP",
            "latency_ms": flat_latency,
            "index_size_bytes": index_file_size_bytes(flat_path),
        },
        "ann": {
            "index_type": str(cfg["faiss"]["ann"].get("index_type", "hnsw")).lower(),
            "latency_ms": ann_latency,
            "index_size_bytes": index_file_size_bytes(ann_path),
        },
        "comparison": {
            **overlap,
            "latency_speedup_mean": float(flat_latency["mean"] / max(ann_latency["mean"], 1e-12)),
            "latency_speedup_p95": float(flat_latency["p95"] / max(ann_latency["p95"], 1e-12)),
        },
    }
    benchmark_path = metric_dir / output_cfg["benchmark_file"]
    benchmark_path.write_text(json.dumps(benchmark, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("Saved FAISS recall report: %s", recall_report_path)
    logger.info("Saved FAISS benchmark report: %s", benchmark_path)
    logger.info("FAISS benchmark summary: %s", benchmark["comparison"])


if __name__ == "__main__":
    main()
