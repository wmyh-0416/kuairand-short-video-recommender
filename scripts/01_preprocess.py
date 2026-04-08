from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import run_preprocess
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess KuaiRand-Pure data.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "preprocess.yaml"),
        help="Path to preprocessing YAML config.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional row limit per log file for smoke tests.",
    )
    parser.add_argument(
        "--include-random-log",
        action="store_true",
        help="Include log_random_4_22_to_5_08_pure.csv in preprocessing.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Override output processed directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.nrows is not None:
        cfg.setdefault("preprocess", {}).setdefault("read", {})["nrows"] = args.nrows
    if args.include_random_log:
        cfg.setdefault("data", {})["include_random_log"] = True
    if args.processed_dir is not None:
        cfg.setdefault("paths", {})["processed_dir"] = args.processed_dir

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))

    log_file = logs_dir(cfg) / "01_preprocess.log"
    logger = setup_logger(
        name="kuairand_rec.preprocess",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=log_file,
    )
    logger.info("Loaded config: %s", Path(args.config).resolve())
    logger.info("Project root: %s", PROJECT_ROOT)

    written = run_preprocess(cfg, logger=logger)
    logger.info("Written outputs:")
    for name, path in written.items():
        logger.info("  %s -> %s", name, path)


if __name__ == "__main__":
    main()
