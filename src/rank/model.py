from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rank.features import RankFeatureSpec


def resolve_device(device_cfg: str = "auto") -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class DINMultiTaskRanker(nn.Module):
    """DIN-style target-aware sequence attention with multi-task heads."""

    def __init__(self, spec: RankFeatureSpec, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["rank"]["model"]
        self.tasks = list(spec.tasks)
        self.other_categorical_columns = list(spec.other_categorical_columns)
        self.embedding_dim = int(model_cfg.get("embedding_dim", 64))
        other_dim = int(model_cfg.get("other_cat_embedding_dim", 8))
        numeric_dim = len(spec.numeric_columns)
        numeric_projection_dim = int(model_cfg.get("numeric_projection_dim", 64))
        dropout = float(model_cfg.get("dropout", 0.15))

        self.user_embedding = nn.Embedding(spec.vocab_size("user_id"), self.embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(spec.vocab_size("video_id"), self.embedding_dim, padding_idx=0)
        self.author_embedding = nn.Embedding(spec.vocab_size("author_id"), self.embedding_dim, padding_idx=0)
        self.tag_embedding = nn.Embedding(spec.vocab_size("tag"), self.embedding_dim, padding_idx=0)
        self.other_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(spec.vocab_size(col), other_dim, padding_idx=0)
                for col in self.other_categorical_columns
            }
        )

        self.attention_mlp = _make_mlp(
            input_dim=self.embedding_dim * 4,
            hidden_dims=list(model_cfg.get("attention_hidden_dims", [128, 64])),
            output_dim=1,
            dropout=dropout,
        )
        self.numeric_projection = _make_mlp(
            input_dim=max(numeric_dim, 1),
            hidden_dims=[],
            output_dim=numeric_projection_dim,
            dropout=0.0,
        )
        other_total_dim = other_dim * len(self.other_categorical_columns)
        shared_input_dim = self.embedding_dim * 5 + numeric_projection_dim + other_total_dim
        self.shared_mlp = _make_mlp(
            input_dim=shared_input_dim,
            hidden_dims=list(model_cfg.get("shared_hidden_dims", [256, 128])),
            output_dim=int(model_cfg.get("shared_hidden_dims", [256, 128])[-1]),
            dropout=dropout,
        )
        shared_out_dim = int(model_cfg.get("shared_hidden_dims", [256, 128])[-1])
        task_hidden = list(model_cfg.get("task_hidden_dims", [64]))
        self.task_heads = nn.ModuleDict(
            {task: _make_mlp(shared_out_dim, task_hidden, 1, dropout=dropout) for task in self.tasks}
        )

    def _target_embedding(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.item_embedding(batch["video_id"])
            + self.author_embedding(batch["author_id"])
            + self.tag_embedding(batch["tag"])
        )

    def _attention_pool(self, target_emb: torch.Tensor, hist_item_seq: torch.Tensor) -> torch.Tensor:
        hist_emb = self.item_embedding(hist_item_seq)
        target = target_emb.unsqueeze(1).expand_as(hist_emb)
        att_input = torch.cat([hist_emb, target, hist_emb - target, hist_emb * target], dim=-1)
        scores = self.attention_mlp(att_input).squeeze(-1)
        mask = hist_item_seq > 0
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (weights.unsqueeze(-1) * hist_emb).sum(dim=1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        user_emb = self.user_embedding(batch["user_id"])
        target_emb = self._target_embedding(batch)
        interest_emb = self._attention_pool(target_emb, batch["hist_item_seq"])
        numeric = batch["numeric"]
        if numeric.shape[1] == 0:
            numeric = torch.zeros((numeric.shape[0], 1), device=numeric.device, dtype=numeric.dtype)
        numeric_emb = self.numeric_projection(numeric)

        if self.other_categorical_columns:
            other_cats = batch["other_cats"]
            other_embs = [
                self.other_embeddings[col](other_cats[:, idx])
                for idx, col in enumerate(self.other_categorical_columns)
            ]
            other_emb = torch.cat(other_embs, dim=-1)
        else:
            other_emb = torch.zeros((numeric.shape[0], 0), device=numeric.device)

        shared_input = torch.cat(
            [
                user_emb,
                target_emb,
                interest_emb,
                target_emb * interest_emb,
                torch.abs(target_emb - interest_emb),
                numeric_emb,
                other_emb,
            ],
            dim=-1,
        )
        shared = self.shared_mlp(shared_input)
        return {task: self.task_heads[task](shared).squeeze(-1) for task in self.tasks}


def build_model(spec: RankFeatureSpec, cfg: Mapping[str, Any]) -> DINMultiTaskRanker:
    model_type = str(cfg["rank"]["model"].get("type", "din_multitask")).lower()
    if model_type != "din_multitask":
        raise ValueError(f"Only din_multitask is implemented as the main rank model, got: {model_type}")
    return DINMultiTaskRanker(spec, cfg)


def compute_rank_score(preds: Mapping[str, torch.Tensor], cfg: Mapping[str, Any]) -> torch.Tensor:
    weights = cfg["rank"].get("rank_score_weights", {})
    score = None
    for task, logits in preds.items():
        weight = float(weights.get(task, 1.0))
        value = weight * torch.sigmoid(logits)
        score = value if score is None else score + value
    return score if score is not None else torch.zeros(0)
