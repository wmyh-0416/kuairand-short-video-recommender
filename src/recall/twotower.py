from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.recall.twotower_dataset import (
    CategoricalEncoder,
    TwoTowerTrainDataset,
    build_item_feature_table,
    build_user_feature_table,
    require_torch,
    twotower_collate_fn,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    torch = None
    nn = None
    F = None
    DataLoader = None


def resolve_device(device_cfg: str = "auto") -> str:
    require_torch()
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> Any:
    require_torch()
    layers: list[Any] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class UserTower(nn.Module if nn is not None else object):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.mlp = _make_mlp(embedding_dim * 2, hidden_dims, embedding_dim, dropout)

    def forward(self, user_idx: Any, seq_item_idx: Any) -> Any:
        user_emb = self.user_embedding(user_idx)
        seq_emb = self.item_embedding(seq_item_idx)
        mask = (seq_item_idx > 0).float().unsqueeze(-1)
        seq_sum = (seq_emb * mask).sum(dim=1)
        seq_len = mask.sum(dim=1).clamp_min(1.0)
        seq_pool = seq_sum / seq_len
        out = self.mlp(torch.cat([user_emb, seq_pool], dim=-1))
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module if nn is not None else object):
    def __init__(
        self,
        num_items: int,
        num_authors: int,
        num_tags: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        require_torch()
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(num_tags, embedding_dim, padding_idx=0)
        self.mlp = _make_mlp(embedding_dim * 3, hidden_dims, embedding_dim, dropout)

    def forward(self, item_idx: Any, author_idx: Any, tag_idx: Any) -> Any:
        item_emb = self.item_embedding(item_idx)
        author_emb = self.author_embedding(author_idx)
        tag_emb = self.tag_embedding(tag_idx)
        out = self.mlp(torch.cat([item_emb, author_emb, tag_emb], dim=-1))
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module if nn is not None else object):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_authors: int,
        num_tags: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        require_torch()
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        self.temperature = float(temperature)
        self.user_tower = UserTower(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            num_authors=num_authors,
            num_tags=num_tags,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def encode_user(self, user_idx: Any, seq_item_idx: Any) -> Any:
        return self.user_tower(user_idx, seq_item_idx)

    def encode_item(self, item_idx: Any, author_idx: Any, tag_idx: Any) -> Any:
        return self.item_tower(item_idx, author_idx, tag_idx)

    def forward(self, batch: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        user_vec = self.encode_user(batch["user_idx"], batch["seq_item_idx"])
        item_vec = self.encode_item(batch["item_idx"], batch["author_idx"], batch["tag_idx"])
        logits = user_vec @ item_vec.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return logits, labels, user_vec


def train_one_epoch(
    model: TwoTowerModel,
    dataloader: Any,
    optimizer: Any,
    device: str,
) -> float:
    require_torch()
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits, labels, _ = model(batch)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def train_twotower(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    user_sequences: pd.DataFrame,
    encoders: Mapping[str, CategoricalEncoder],
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> TwoTowerModel:
    require_torch()
    tower_cfg = cfg["recall"]["twotower"]
    device = resolve_device(str(tower_cfg.get("device", "auto")))

    dataset = TwoTowerTrainDataset(
        interactions=train_df,
        item_features=item_features,
        user_sequences=user_sequences,
        encoders=encoders,
        label_col=tower_cfg.get("label_col", "is_positive"),
        sequence_col=tower_cfg.get("sequence_col", "watch_seq"),
        max_seq_len=int(tower_cfg.get("max_seq_len", 50)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(tower_cfg.get("batch_size", 2048)),
        shuffle=True,
        num_workers=int(tower_cfg.get("num_workers", 0)),
        collate_fn=twotower_collate_fn,
        drop_last=True,
    )

    model = TwoTowerModel(
        num_users=encoders["user"].size,
        num_items=encoders["video"].size,
        num_authors=encoders["author"].size,
        num_tags=encoders["tag"].size,
        embedding_dim=int(tower_cfg.get("embedding_dim", 64)),
        hidden_dims=[int(x) for x in tower_cfg.get("hidden_dims", [128, 64])],
        dropout=float(tower_cfg.get("dropout", 0.1)),
        temperature=float(tower_cfg.get("temperature", 0.07)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tower_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(tower_cfg.get("weight_decay", 1e-6)),
    )

    epochs = int(tower_cfg.get("epochs", 5))
    if logger:
        logger.info(
            "Training two-tower: samples=%d batch_size=%d epochs=%d device=%s",
            len(dataset),
            int(tower_cfg.get("batch_size", 2048)),
            epochs,
            device,
        )
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        if logger:
            logger.info("Two-tower epoch %d/%d loss=%.6f", epoch, epochs, loss)

    return model


@torch.no_grad() if torch is not None else (lambda f: f)
def encode_items(
    model: TwoTowerModel,
    item_features: pd.DataFrame,
    encoders: Mapping[str, CategoricalEncoder],
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    require_torch()
    model.eval()
    item_table = build_item_feature_table(item_features, encoders).sort_values("video_id")
    item_ids = item_table["video_id"].astype("int64").to_numpy()
    vectors: list[np.ndarray] = []

    for start in range(0, len(item_table), batch_size):
        batch = item_table.iloc[start : start + batch_size]
        item_idx = torch.as_tensor(batch["item_idx"].to_numpy(), dtype=torch.long, device=device)
        author_idx = torch.as_tensor(batch["author_idx"].to_numpy(), dtype=torch.long, device=device)
        tag_idx = torch.as_tensor(batch["tag_idx"].to_numpy(), dtype=torch.long, device=device)
        vec = model.encode_item(item_idx, author_idx, tag_idx).detach().cpu().numpy()
        vectors.append(vec.astype("float32"))

    return item_ids, np.vstack(vectors)


@torch.no_grad() if torch is not None else (lambda f: f)
def encode_users(
    model: TwoTowerModel,
    user_ids: np.ndarray,
    user_sequences: pd.DataFrame,
    encoders: Mapping[str, CategoricalEncoder],
    max_seq_len: int,
    sequence_col: str = "watch_seq",
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    require_torch()
    model.eval()
    user_table = build_user_feature_table(
        user_ids=user_ids,
        user_sequences=user_sequences,
        encoders=encoders,
        max_seq_len=max_seq_len,
        sequence_col=sequence_col,
    ).sort_values("user_id")
    out_user_ids = user_table["user_id"].astype("int64").to_numpy()
    vectors: list[np.ndarray] = []

    for start in range(0, len(user_table), batch_size):
        batch = user_table.iloc[start : start + batch_size]
        user_idx = torch.as_tensor(batch["user_idx"].to_numpy(), dtype=torch.long, device=device)
        seq_idx = torch.as_tensor(np.stack(batch["seq_item_idx"].to_numpy()), dtype=torch.long, device=device)
        vec = model.encode_user(user_idx, seq_idx).detach().cpu().numpy()
        vectors.append(vec.astype("float32"))

    return out_user_ids, np.vstack(vectors) if vectors else np.empty((0, 0), dtype="float32")


def retrieve_topk_numpy(
    user_ids: np.ndarray,
    user_vectors: np.ndarray,
    item_ids: np.ndarray,
    item_vectors: np.ndarray,
    topk: int,
) -> pd.DataFrame:
    if user_vectors.size == 0 or item_vectors.size == 0:
        return pd.DataFrame(columns=["user_id", "video_id", "recall_source", "source_score", "source_rank"])

    scores = user_vectors @ item_vectors.T
    k = min(topk, item_vectors.shape[0])
    top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]

    rows: list[dict[str, object]] = []
    for row_idx, user_id in enumerate(user_ids):
        idx = top_idx[row_idx]
        sorted_idx = idx[np.argsort(-scores[row_idx, idx])]
        for rank, item_pos in enumerate(sorted_idx, start=1):
            rows.append(
                {
                    "user_id": int(user_id),
                    "video_id": int(item_ids[item_pos]),
                    "recall_source": "twotower",
                    "source_score": float(scores[row_idx, item_pos]),
                    "source_rank": int(rank),
                }
            )
    return pd.DataFrame(rows)


def save_twotower_checkpoint(
    model: TwoTowerModel,
    encoders: Mapping[str, CategoricalEncoder],
    path: str | Path,
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> Path:
    require_torch()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoders": {name: encoder.to_dict() for name, encoder in encoders.items()},
            "twotower_config": dict(cfg["recall"]["twotower"]),
        },
        path,
    )
    if logger:
        logger.info("Saved two-tower checkpoint: %s", path)
    return path


def load_twotower_checkpoint(path: str | Path, device: str = "cpu") -> tuple[TwoTowerModel, dict[str, CategoricalEncoder], dict[str, Any]]:
    require_torch()
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device)
    encoders = {
        name: CategoricalEncoder.from_dict(data)
        for name, data in checkpoint["encoders"].items()
    }
    tower_cfg = checkpoint["twotower_config"]
    model = TwoTowerModel(
        num_users=encoders["user"].size,
        num_items=encoders["video"].size,
        num_authors=encoders["author"].size,
        num_tags=encoders["tag"].size,
        embedding_dim=int(tower_cfg.get("embedding_dim", 64)),
        hidden_dims=[int(x) for x in tower_cfg.get("hidden_dims", [128, 64])],
        dropout=float(tower_cfg.get("dropout", 0.1)),
        temperature=float(tower_cfg.get("temperature", 0.07)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, encoders, tower_cfg
