from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from src.rank.dataset import metrics_dir, prepare_train_val_datasets, rank_dir
from src.rank.model import build_model, compute_rank_score, resolve_device


def _move_batch(batch: Mapping[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def multitask_loss(
    preds: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    tasks: list[str],
    task_weights: Mapping[str, float],
) -> torch.Tensor:
    losses = []
    for idx, task in enumerate(tasks):
        weight = float(task_weights.get(task, 1.0))
        losses.append(weight * F.binary_cross_entropy_with_logits(preds[task], labels[:, idx]))
    return torch.stack(losses).sum()


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6), labels=[0, 1]))


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    model.eval()
    tasks = [str(x) for x in cfg["rank"]["tasks"]]
    labels_all: list[np.ndarray] = []
    pred_by_task = {task: [] for task in tasks}
    rank_scores: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            preds = model(batch)
            loss = multitask_loss(preds, batch["labels"].float(), tasks, cfg["rank"].get("task_weights", {}))
            losses.append(float(loss.detach().cpu()))
            labels_all.append(batch["labels"].detach().cpu().numpy())
            for task in tasks:
                pred_by_task[task].append(torch.sigmoid(preds[task]).detach().cpu().numpy())
            rank_scores.append(compute_rank_score(preds, cfg).detach().cpu().numpy())

    labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0, len(tasks)))
    metrics: dict[str, Any] = {"loss": float(np.mean(losses)) if losses else None}
    for idx, task in enumerate(tasks):
        pred = np.concatenate(pred_by_task[task], axis=0) if pred_by_task[task] else np.array([])
        y = labels_np[:, idx] if labels_np.size else np.array([])
        metrics[f"{task}_auc"] = _safe_auc(y, pred) if len(y) else None
        metrics[f"{task}_logloss"] = _safe_logloss(y, pred) if len(y) else None
    if rank_scores:
        rank_pred = np.concatenate(rank_scores, axis=0)
        rank_label = labels_np.max(axis=1) if labels_np.size else np.array([])
        metrics["rank_auc"] = _safe_auc(rank_label, rank_pred) if len(rank_label) else None
    return metrics


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    spec: Any,
    cfg: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_spec": spec,
            "cfg_rank": dict(cfg["rank"]),
            "metrics": dict(metrics),
        },
        path,
    )


def train_rank_model(
    cfg: Mapping[str, Any],
    train_rows: int | None = None,
    val_rows: int | None = None,
    logger: Any | None = None,
) -> dict[str, Any]:
    out_dir = rank_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out_dir = metrics_dir(cfg)
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, spec, _store, train_frame, val_frame = prepare_train_val_datasets(
        cfg,
        train_rows=train_rows,
        val_rows=val_rows,
        logger=logger,
    )
    device = resolve_device(str(cfg["rank"]["model"].get("device", "auto")))
    model = build_model(spec, cfg).to(device)
    model_cfg = cfg["rank"]["model"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(model_cfg.get("batch_size", 4096)),
        shuffle=True,
        num_workers=int(model_cfg.get("num_workers", 2)),
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(model_cfg.get("batch_size", 4096)),
        shuffle=False,
        num_workers=int(model_cfg.get("num_workers", 2)),
        pin_memory=device == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-6)),
    )
    tasks = [str(x) for x in cfg["rank"]["tasks"]]
    best_metric = -float("inf")
    best_metrics: dict[str, Any] = {}
    bad_epochs = 0
    patience = int(model_cfg.get("early_stopping_patience", 2))
    grad_clip = float(model_cfg.get("grad_clip_norm", 5.0))
    ckpt_path = out_dir / cfg["rank"]["output"]["best_model_file"]

    if logger:
        logger.info(
            "Training DIN ranker on device=%s train_rows=%d val_rows=%d numeric=%d other_cats=%d",
            device,
            len(train_ds),
            len(val_ds),
            len(spec.numeric_columns),
            len(spec.other_categorical_columns),
        )

    for epoch in range(int(model_cfg.get("epochs", 4))):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch)
            loss = multitask_loss(preds, batch["labels"].float(), tasks, cfg["rank"].get("task_weights", {}))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_loader(model, val_loader, cfg, device)
        val_rank_auc = val_metrics.get("rank_auc") or 0.0
        train_loss = float(np.mean(losses)) if losses else 0.0
        if logger:
            logger.info(
                "DIN rank epoch %d/%d train_loss=%.6f val_metrics=%s",
                epoch + 1,
                int(model_cfg.get("epochs", 4)),
                train_loss,
                val_metrics,
            )
        if val_rank_auc > best_metric:
            best_metric = val_rank_auc
            best_metrics = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
            save_checkpoint(ckpt_path, model, spec, cfg, best_metrics)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if logger:
                    logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    metrics = {
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "train_positive_any": int(train_frame["rank_label"].sum()),
        "val_positive_any": int(val_frame["rank_label"].sum()),
        "best": best_metrics,
    }
    metrics_path = metrics_out_dir / cfg["rank"]["output"]["metrics_file"]
    existing: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing["train"] = metrics
    metrics_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    if logger:
        logger.info("Saved rank checkpoint: %s", ckpt_path)
        logger.info("Saved rank train metrics: %s", metrics_path)
    return metrics
