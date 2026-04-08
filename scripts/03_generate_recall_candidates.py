from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.generate import generate_all_splits
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate recall candidates for train/val/test.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "recall.yaml"),
        help="Path to recall YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to generate.",
    )
    parser.add_argument(
        "--disable-twotower",
        action="store_true",
        help="Skip two-tower recall candidate generation.",
    )
    parser.add_argument(
        "--disable-graph-emb",
        action="store_true",
        help="Skip graph embedding recall candidate generation.",
    )
    parser.add_argument(
        "--final-topk",
        type=int,
        default=None,
        help="Override merged final topK per user.",
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
    if args.disable_twotower:
        cfg["recall"]["twotower"]["enabled"] = False
    if args.disable_graph_emb:
        cfg["recall"]["graph_emb"]["enabled"] = False
    if args.final_topk is not None:
        cfg["recall"]["merge"]["final_topk"] = args.final_topk

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))

    logger = setup_logger(
        name="kuairand_rec.generate_recall",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "03_generate_recall_candidates.log",
    )
    logger.info("Generating recall candidates for splits: %s", args.splits)
    metrics = generate_all_splits(cfg, splits=args.splits, logger=logger)
    for split, split_metrics in metrics.items():
        compact = {
            key: value
            for key, value in split_metrics.items()
            if key.startswith("recall@") or key in {"num_candidates", "coverage_all"}
        }
        logger.info("%s metrics summary: %s", split, compact)


if __name__ == "__main__":
    main()
