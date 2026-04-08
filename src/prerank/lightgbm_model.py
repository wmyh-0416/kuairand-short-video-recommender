from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "LightGBM is required for prerank.model.type=lightgbm. "
            "Install lightgbm or switch configs/prerank.yaml prerank.model.type to mlp."
        ) from exc
    return lgb


def train_lightgbm_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Mapping[str, Any],
    categorical_columns: list[str],
    logger: Any | None = None,
) -> Any:
    lgb = _require_lightgbm()
    params = dict(cfg["prerank"]["model"].get("lightgbm", {}))
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 0) or 0)
    log_period = int(params.pop("log_period", 50))
    params.setdefault("objective", "binary")
    params.setdefault("random_state", int(cfg["project"].get("random_seed", 2026)))

    model = lgb.LGBMClassifier(**params)
    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=logger is not None))
    if log_period > 0:
        callbacks.append(lgb.log_evaluation(period=log_period))

    fit_kwargs: dict[str, Any] = {
        "eval_set": [(x_val, y_val)],
        "eval_metric": "auc",
        "categorical_feature": [col for col in categorical_columns if col in x_train.columns],
    }
    if callbacks:
        fit_kwargs["callbacks"] = callbacks

    if logger:
        logger.info(
            "Training LightGBM prerank: train=%s val=%s features=%d categorical=%d",
            x_train.shape,
            x_val.shape,
            x_train.shape[1],
            len(fit_kwargs["categorical_feature"]),
        )
    model.fit(x_train, y_train, **fit_kwargs)
    return model


def predict_lightgbm(model: Any, x: pd.DataFrame) -> np.ndarray:
    pred = model.predict_proba(x)[:, 1]
    return np.asarray(pred, dtype=np.float32)


def save_model_bundle(bundle: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(bundle), f)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as f:
        return pickle.load(f)
