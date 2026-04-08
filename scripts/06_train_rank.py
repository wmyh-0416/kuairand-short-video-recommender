from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rank.infer import generate_ranked_candidates
from src.rank.train import train_rank_model
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DIN-style multi-task ranker and generate ranked candidates.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "rank.yaml"),
        help="Path to rank YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument("--train-rows", type=int, default=None, help="Optional train topK row limit for smoke tests.")
    parser.add_argument("--val-rows", type=int, default=None, help="Optional val topK row limit for smoke tests.")
    parser.add_argument("--candidate-rows", type=int, default=None, help="Optional inference row limit for smoke tests.")
    parser.add_argument("--epochs", type=int, default=None, help="Override rank.model.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override rank.model.batch_size.")
    parser.add_argument("--skip-infer", action="store_true", help="Only train the rank model; do not generate ranked candidates.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        choices=["train", "val", "test"],
        help="Splits to rank after training. Defaults to rank.inference.splits.",
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
    if args.epochs is not None:
        cfg["rank"]["model"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["rank"]["model"]["batch_size"] = args.batch_size

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))
    logger = setup_logger(
        name="kuairand_rec.train_rank",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "06_train_rank.log",
    )
    logger.info("Training DIN-style multi-task ranker.")
    metrics = train_rank_model(cfg, train_rows=args.train_rows, val_rows=args.val_rows, logger=logger)
    logger.info("Rank training metrics summary: %s", metrics)

    if not args.skip_infer:
        splits = args.splits if args.splits is not None else list(cfg["rank"]["inference"].get("splits", ["val", "test"]))
        logger.info("Generating ranked candidates for splits=%s", splits)
        infer_metrics = generate_ranked_candidates(
            cfg,
            splits=splits,
            candidate_rows=args.candidate_rows,
            logger=logger,
        )
        logger.info("Rank inference metrics summary: %s", infer_metrics)


if __name__ == "__main__":
    main()
