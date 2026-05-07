from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.faiss_index import build_faiss_artifacts
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS flat and ANN indices from two-tower embeddings.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "faiss.yaml"),
        help="Path to FAISS YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument(
        "--ann-index-type",
        choices=["hnsw", "ivf"],
        default=None,
        help="Override faiss.ann.index_type.",
    )
    parser.add_argument(
        "--similarity",
        choices=["cosine", "inner_product", "ip"],
        default=None,
        help="Override faiss.similarity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.processed_dir is not None:
        cfg["paths"]["processed_dir"] = args.processed_dir
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
        cfg["paths"]["logs_dir"] = str(Path(args.artifacts_dir) / "logs")
    if args.ann_index_type is not None:
        cfg.setdefault("faiss", {}).setdefault("ann", {})["index_type"] = args.ann_index_type
        cfg["faiss"]["output"]["ann_index_file"] = f"faiss_{args.ann_index_type}.index"
    if args.similarity is not None:
        cfg.setdefault("faiss", {})["similarity"] = args.similarity

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))
    logger = setup_logger(
        name="kuairand_rec.build_faiss_index",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "09_build_faiss_index.log",
    )

    try:
        report = build_faiss_artifacts(cfg, logger=logger)
    except ImportError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    logger.info("FAISS build summary: %s", report)


if __name__ == "__main__":
    main()
