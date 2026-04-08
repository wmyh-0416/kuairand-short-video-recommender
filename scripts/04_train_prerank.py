from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prerank.train import train_prerank
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prerank model from recall candidates.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "prerank.yaml"),
        help="Path to prerank YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument("--train-rows", type=int, default=None, help="Optional train candidate row limit.")
    parser.add_argument("--val-rows", type=int, default=None, help="Optional val candidate row limit.")
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "mlp"],
        default=None,
        help="Override prerank.model.type.",
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
    if args.model_type is not None:
        cfg["prerank"]["model"]["type"] = args.model_type

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))
    logger = setup_logger(
        name="kuairand_rec.train_prerank",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "04_train_prerank.log",
    )
    logger.info("Training prerank model type=%s", cfg["prerank"]["model"].get("type", "lightgbm"))
    metrics = train_prerank(
        cfg=cfg,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        logger=logger,
    )
    logger.info("Prerank training metrics summary: %s", metrics)


if __name__ == "__main__":
    main()
