from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_offline_ab_test
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline log-replay A/B test simulation.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ab_test.yaml"),
        help="Path to A/B test YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts directory.")
    parser.add_argument("--control", default=None, help="Override control policy type.")
    parser.add_argument("--treatment", default=None, help="Override treatment policy type.")
    parser.add_argument("--split", default=None, help="Override evaluation split.")
    parser.add_argument("--top-k", nargs="+", type=int, default=None, help="Override evaluated top-k values.")
    return parser.parse_args()


def apply_runtime_paths(cfg: dict[str, Any], *, project_root: Path, processed_dir: Path | None, artifacts_dir: Path | None) -> dict[str, Any]:
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

    if args.control:
        cfg["policies"]["control"]["type"] = str(args.control)
        cfg["policies"]["control"]["name"] = str(args.control)
    if args.treatment:
        cfg["policies"]["treatment"]["type"] = str(args.treatment)
        cfg["policies"]["treatment"]["name"] = str(args.treatment)
    if args.split:
        cfg["experiment"]["split"] = str(args.split)
    if args.top_k:
        cfg["experiment"]["top_k"] = [int(value) for value in args.top_k]

    ensure_project_dirs(cfg)
    logger = setup_logger(
        name="kuairand_rec.ab_test",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "13_run_ab_test.log",
    )
    component_cfgs = load_component_configs(cfg)
    report = run_offline_ab_test(cfg=cfg, component_cfgs=component_cfgs, logger=logger)

    primary_metric = cfg["metrics"]["primary"]
    control_value = report["control"]["metrics"].get(primary_metric)
    treatment_value = report["treatment"]["metrics"].get(primary_metric)
    relative_lift = report["relative_lift"].get(primary_metric)
    logger.info(
        "Offline A/B summary | primary=%s control=%.6f treatment=%.6f relative_lift=%s",
        primary_metric,
        float(control_value or 0.0),
        float(treatment_value or 0.0),
        "n/a" if relative_lift is None else f"{float(relative_lift):.4%}",
    )


if __name__ == "__main__":
    main()
