from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prerank.infer import generate_prerank_topk
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score recall candidates and generate prerank topK.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "prerank.yaml"),
        help="Path to prerank YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to score.",
    )
    parser.add_argument("--candidate-rows", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--topk", type=int, default=None, help="Override prerank topK per user.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.processed_dir is not None:
        cfg["paths"]["processed_dir"] = args.processed_dir
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
        cfg["paths"]["logs_dir"] = str(Path(args.artifacts_dir) / "logs")
    if args.topk is not None:
        cfg["prerank"]["topk"] = args.topk

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))
    logger = setup_logger(
        name="kuairand_rec.generate_prerank_topk",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "05_generate_prerank_topk.log",
    )
    logger.info("Generating prerank topK for splits=%s topk=%s", args.splits, cfg["prerank"].get("topk", 100))
    metrics = generate_prerank_topk(
        cfg=cfg,
        splits=args.splits,
        candidate_rows=args.candidate_rows,
        logger=logger,
    )
    logger.info("Prerank topK metrics summary: %s", metrics)


if __name__ == "__main__":
    main()
