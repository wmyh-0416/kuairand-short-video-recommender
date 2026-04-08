from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.prerank.dataset import (
    build_train_val_matrices,
    load_processed_tables,
    load_recall_candidates,
    metrics_dir,
    prerank_dir,
)
from src.prerank.lightgbm_model import save_model_bundle, train_lightgbm_model
from src.prerank.mlp_model import train_mlp_model


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _predict_for_metrics(model_type: str, model_payload: Any, x: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    if model_type == "lightgbm":
        from src.prerank.lightgbm_model import predict_lightgbm

        return predict_lightgbm(model_payload, x)
    if model_type == "mlp":
        from src.prerank.mlp_model import predict_mlp

        return predict_mlp(model_payload, x, cfg)
    raise ValueError(f"Unsupported prerank model type: {model_type}")


def train_prerank(
    cfg: Mapping[str, Any],
    train_rows: int | None = None,
    val_rows: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    out_dir = prerank_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out_dir = metrics_dir(cfg)
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Loading recall candidates for prerank training.")
    train_candidates = load_recall_candidates(cfg, "train", nrows=train_rows)
    val_candidates = load_recall_candidates(cfg, "val", nrows=val_rows)
    processed_tables = load_processed_tables(cfg)

    x_train, y_train, x_val, y_val, spec, train_samples, val_samples = build_train_val_matrices(
        cfg=cfg,
        train_candidates=train_candidates,
        val_candidates=val_candidates,
        processed_tables=processed_tables,
    )
    output_cfg = cfg["prerank"]["output"]
    train_samples.to_parquet(out_dir / output_cfg["train_samples_file"], index=False)
    val_samples.to_parquet(out_dir / output_cfg["val_samples_file"], index=False)
    if logger:
        logger.info(
            "Built prerank samples: train=%s positives=%d val=%s positives=%d",
            train_samples.shape,
            int(y_train.sum()),
            val_samples.shape,
            int(y_val.sum()),
        )

    model_type = str(cfg["prerank"]["model"].get("type", "lightgbm")).lower()
    if model_type == "lightgbm":
        model_payload = train_lightgbm_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            cfg=cfg,
            categorical_columns=spec.categorical_columns,
            logger=logger,
        )
    elif model_type == "mlp":
        model_payload = train_mlp_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            cfg=cfg,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported prerank model type: {model_type}")

    train_pred = _predict_for_metrics(model_type, model_payload, x_train, cfg)
    val_pred = _predict_for_metrics(model_type, model_payload, x_val, cfg)
    metrics = {
        "model_type": model_type,
        "train_rows": int(len(y_train)),
        "train_positive": int(y_train.sum()),
        "val_rows": int(len(y_val)),
        "val_positive": int(y_val.sum()),
        "train_auc": _safe_auc(y_train, train_pred),
        "val_auc": _safe_auc(y_val, val_pred),
        "num_features": int(len(spec.feature_columns)),
        "categorical_features": spec.categorical_columns,
    }

    model_path = out_dir / output_cfg["model_file"]
    save_model_bundle(
        {
            "model_type": model_type,
            "model": model_payload,
            "feature_spec": spec,
            "cfg_prerank": dict(cfg["prerank"]),
        },
        model_path,
    )
    if logger:
        logger.info("Saved prerank model bundle: %s", model_path)

    metrics_path = metrics_out_dir / output_cfg["metrics_file"]
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing["train"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved prerank training metrics: %s metrics=%s", metrics_path, metrics)
    return metrics
