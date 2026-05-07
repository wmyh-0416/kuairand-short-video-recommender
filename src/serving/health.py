from __future__ import annotations

from typing import Any


def build_health_response(model_loader: Any, cache_manager: Any) -> dict[str, Any]:
    loaded_components = dict(model_loader.loaded_components)
    models_loaded = bool(all(loaded_components.values()))
    degraded_mode = not models_loaded
    status = "ok" if model_loader.has_any_recall_backend() else "unavailable"
    return {
        "status": status,
        "models_loaded": models_loaded,
        "faiss_loaded": bool(loaded_components.get("faiss", False)),
        "redis_connected": bool(cache_manager.redis_connected),
        "degraded_mode": degraded_mode,
        "loaded_components": loaded_components,
        "component_errors": dict(model_loader.component_errors),
    }
