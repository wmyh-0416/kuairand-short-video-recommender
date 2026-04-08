from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError("PyTorch is required for prerank.model.type=mlp.") from exc
    return torch


def resolve_device(device_cfg: str = "auto") -> str:
    torch = _require_torch()
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> Any:
    torch = _require_torch()
    import torch.nn as nn

    layers: list[Any] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


def train_mlp_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: Mapping[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    mlp_cfg = cfg["prerank"]["model"].get("mlp", {})
    device = resolve_device(str(mlp_cfg.get("device", "auto")))
    model = _build_mlp(
        input_dim=x_train.shape[1],
        hidden_dims=list(mlp_cfg.get("hidden_dims", [256, 128, 64])),
        dropout=float(mlp_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mlp_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-6)),
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.as_tensor(x_train.to_numpy(dtype=np.float32)),
        torch.as_tensor(y_train.to_numpy(dtype=np.float32)).view(-1, 1),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(mlp_cfg.get("batch_size", 4096)),
        shuffle=True,
        num_workers=int(mlp_cfg.get("num_workers", 0)),
    )
    val_x = torch.as_tensor(x_val.to_numpy(dtype=np.float32), device=device)
    val_y = torch.as_tensor(y_val.to_numpy(dtype=np.float32), device=device).view(-1, 1)

    best_state = None
    best_val_loss = float("inf")
    for epoch in range(int(mlp_cfg.get("epochs", 5))):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if logger:
            logger.info(
                "MLP prerank epoch %d/%d train_loss=%.6f val_loss=%.6f device=%s",
                epoch + 1,
                int(mlp_cfg.get("epochs", 5)),
                float(np.mean(losses)) if losses else 0.0,
                val_loss,
                device,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": x_train.shape[1],
        "hidden_dims": list(mlp_cfg.get("hidden_dims", [256, 128, 64])),
        "dropout": float(mlp_cfg.get("dropout", 0.1)),
    }


def predict_mlp(model_payload: Mapping[str, Any], x: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    torch = _require_torch()
    mlp_cfg = cfg["prerank"]["model"].get("mlp", {})
    device = resolve_device(str(mlp_cfg.get("device", "auto")))
    model = _build_mlp(
        input_dim=int(model_payload["input_dim"]),
        hidden_dims=list(model_payload["hidden_dims"]),
        dropout=float(model_payload["dropout"]),
    ).to(device)
    model.load_state_dict(model_payload["state_dict"])
    model.eval()

    batch_size = int(mlp_cfg.get("batch_size", 4096))
    preds: list[np.ndarray] = []
    values = x.to_numpy(dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start : start + batch_size], device=device)
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy().reshape(-1)
            preds.append(pred.astype(np.float32))
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def save_mlp_bundle(bundle: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dict(bundle), f)
