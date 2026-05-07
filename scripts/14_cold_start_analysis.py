from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import run_cold_start_analysis
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cold-start analysis and heuristic enhancement replay.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "cold_start.yaml"),
        help="Path to cold-start YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")
    return parser.parse_args()


def apply_runtime_paths(
    cfg: dict[str, Any],
    *,
    project_root: Path,
    processed_dir: Path | None,
    artifacts_dir: Path | None,
) -> dict[str, Any]:
    cfg.setdefault("project", {})
    cfg.setdefault("paths", {})
    cfg["project"]["root_dir"] = str(project_root.resolve())
    if processed_dir is not None:
        cfg["paths"]["processed_dir"] = str(processed_dir.resolve())
    if artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = str(artifacts_dir.resolve())
        cfg["paths"]["logs_dir"] = str((artifacts_dir / "logs").resolve())
    return cfg


def load_component_configs(main_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs = main_cfg.get("component_configs", {})
    loaded: dict[str, dict[str, Any]] = {}
    for name, ref in refs.items():
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = (PROJECT_ROOT / ref_path).resolve()
        component_cfg = load_config(ref_path)
        component_cfg = apply_runtime_paths(
            component_cfg,
            project_root=PROJECT_ROOT,
            processed_dir=Path(main_cfg["paths"]["processed_dir"]),
            artifacts_dir=Path(main_cfg["paths"]["artifacts_dir"]),
        )
        loaded[str(name)] = component_cfg
    return loaded


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_runtime_paths(
        cfg,
        project_root=PROJECT_ROOT,
        processed_dir=Path(args.processed_dir).resolve() if args.processed_dir else None,
        artifacts_dir=Path(args.artifacts_dir).resolve() if args.artifacts_dir else None,
    )
    ensure_project_dirs(cfg)
    logger = setup_logger(
        name="kuairand_rec.cold_start",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "14_cold_start_analysis.log",
    )
    component_cfgs = load_component_configs(cfg)
    report = run_cold_start_analysis(cfg=cfg, component_cfgs=component_cfgs, logger=logger)

    for segment in ["new_user", "low_active_user", "medium_active_user", "high_active_user"]:
        lift = report["lift_by_user_segment"].get(segment, {})
        logger.info(
            "Cold-start segment=%s hit@10_lift=%s recall@50_lift=%s long_view@10_lift=%s",
            segment,
            "n/a" if lift.get("hit_rate@10 lift") is None else f"{float(lift['hit_rate@10 lift']):.4%}",
            "n/a" if lift.get("recall@50 lift") is None else f"{float(lift['recall@50 lift']):.4%}",
            "n/a" if lift.get("long_view_rate@10 lift") is None else f"{float(lift['long_view_rate@10 lift']):.4%}",
        )


if __name__ == "__main__":
    main()
