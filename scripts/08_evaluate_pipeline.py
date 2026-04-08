from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rerank.run import build_pipeline_report
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize recall -> prerank -> rank -> rerank metrics.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "rerank.yaml"),
        help="Path to rerank YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.processed_dir is not None:
        cfg["paths"]["processed_dir"] = args.processed_dir
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
        cfg["paths"]["logs_dir"] = str(Path(args.artifacts_dir) / "logs")

    ensure_project_dirs(cfg)
    logger = setup_logger(
        name="kuairand_rec.evaluate_pipeline",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "08_evaluate_pipeline.log",
    )
    report = build_pipeline_report(cfg, logger=logger)
    logger.info("Pipeline report summary: %s", report.get("pipeline_summary", {}))


if __name__ == "__main__":
    main()
