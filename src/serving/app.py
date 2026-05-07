from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.serving.cache import CacheManager
from src.serving.feature_store import ServingFeatureStore
from src.serving.monitoring import MetricsRegistry
from src.serving.model_loader import PROJECT_ROOT, apply_runtime_paths, load_component_configs
from src.serving.recommender import OnlineRecommender
from src.serving.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_project_dirs, logs_dir


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "serving.yaml"


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _apply_serving_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    serving_cfg = cfg.setdefault("serving", {})
    api_cfg = serving_cfg.setdefault("api", {})
    cache_cfg = serving_cfg.setdefault("cache", {})

    env_host = os.environ.get("SERVING_HOST")
    env_port = os.environ.get("SERVING_PORT")
    env_redis_url = os.environ.get("REDIS_URL")
    env_use_redis = _env_bool("SERVING_USE_REDIS")

    if env_host:
        api_cfg["host"] = env_host
    if env_port:
        try:
            api_cfg["port"] = int(env_port)
        except ValueError:
            pass
    if env_redis_url:
        cache_cfg["redis_url"] = env_redis_url
    if env_use_redis is not None:
        cache_cfg["use_redis"] = bool(env_use_redis)
    return cfg


def _resolve_runtime_cfg(
    config_path: Optional[Union[str, Path]],
    processed_dir: Optional[Union[str, Path]],
    artifacts_dir: Optional[Union[str, Path]],
) -> Tuple[dict[str, Any], Path, Path, Path]:
    config_path = Path(config_path or os.environ.get("KUAIRAND_SERVING_CONFIG", DEFAULT_CONFIG_PATH)).expanduser().resolve()
    cfg = load_config(config_path)
    project_root = config_path.parents[1]
    processed_path = Path(processed_dir or os.environ.get("KUAIRAND_PROCESSED_DIR") or (project_root / "processed")).expanduser().resolve()
    artifacts_path = Path(artifacts_dir or os.environ.get("KUAIRAND_ARTIFACTS_DIR") or (project_root / "artifacts")).expanduser().resolve()
    runtime_cfg = apply_runtime_paths(
        cfg,
        project_root=project_root,
        processed_dir=processed_path,
        artifacts_dir_path=artifacts_path,
    )
    runtime_cfg = _apply_serving_env_overrides(runtime_cfg)
    return runtime_cfg, config_path, processed_path, artifacts_path


def create_app(
    config_path: Optional[Union[str, Path]] = None,
    processed_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> FastAPI:
    serving_cfg, serving_config_path, processed_path, artifacts_path = _resolve_runtime_cfg(
        config_path=config_path,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
    )
    ensure_project_dirs(serving_cfg)
    logger = setup_logger(
        name="kuairand_rec.serving",
        level=serving_cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(serving_cfg) / "11_serving.log",
    )
    monitoring_cfg = serving_cfg["serving"].get("monitoring", {})
    metrics_registry = MetricsRegistry(
        enabled=bool(monitoring_cfg.get("enabled", True)),
        latency_window_size=int(monitoring_cfg.get("latency_window_size", 5000)),
    )
    component_cfgs = load_component_configs(
        serving_cfg,
        project_root=serving_config_path.parents[1],
        processed_dir=processed_path,
        artifacts_dir_path=artifacts_path,
    )
    feature_store = ServingFeatureStore(serving_cfg=serving_cfg, component_cfgs=component_cfgs, logger=logger)
    cache_manager = CacheManager(serving_cfg, logger=logger, metrics_registry=metrics_registry)
    from src.serving.model_loader import ServingModelLoader

    model_loader = ServingModelLoader(
        serving_cfg=serving_cfg,
        component_cfgs=component_cfgs,
        feature_store=feature_store,
        logger=logger,
    )
    recommender = OnlineRecommender(
        serving_cfg=serving_cfg,
        model_loader=model_loader,
        feature_store=feature_store,
        cache_manager=cache_manager,
        metrics_registry=metrics_registry,
        logger=logger,
    )

    app = FastAPI(title="KuaiRand Serving API", version="0.1.0")
    cors_cfg = serving_cfg["serving"].get("cors", {})
    if bool(cors_cfg.get("enabled", False)):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_cfg.get("allow_origins", [])),
            allow_credentials=bool(cors_cfg.get("allow_credentials", True)),
            allow_methods=list(cors_cfg.get("allow_methods", ["*"])),
            allow_headers=list(cors_cfg.get("allow_headers", ["*"])),
        )
    app.state.recommender = recommender
    app.state.metrics_registry = metrics_registry

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        return recommender.health()

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(request: RecommendationRequest) -> dict[str, Any]:
        try:
            return recommender.recommend(request)
        except ValueError as exc:
            recommender.record_recommend_error(request, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            recommender.record_recommend_error(request, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected runtime failure.
            recommender.record_recommend_error(request, exc)
            logger.exception("Unhandled recommend failure: %s", exc)
            raise HTTPException(status_code=500, detail="Internal recommendation error.") from exc

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> dict[str, Any]:
        try:
            return recommender.feedback(request)
        except ValueError as exc:
            recommender.record_feedback_error(request, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected runtime failure.
            recommender.record_feedback_error(request, exc)
            logger.exception("Unhandled feedback failure: %s", exc)
            raise HTTPException(status_code=500, detail="Internal feedback error.") from exc

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> dict[str, Any]:
        return recommender.metrics_snapshot()

    @app.get("/metrics/prometheus", response_class=PlainTextResponse)
    def metrics_prometheus() -> str:
        if not bool(monitoring_cfg.get("enable_prometheus", True)):
            raise HTTPException(status_code=404, detail="Prometheus metrics endpoint is disabled.")
        recommender.metrics_registry.incr("request_count")
        return recommender.metrics_registry.to_prometheus_text()

    return app
