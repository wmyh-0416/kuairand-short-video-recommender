from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - only used in minimal envs.
    torch = None
    nn = None
    F = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for graph_emb training. "
            "Install torch or set recall.graph_emb.enabled=false."
        )


def resolve_device(device_cfg: str = "auto") -> str:
    require_torch()
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def _dedupe_keep_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_positive_item_sequences(
    train_df: pd.DataFrame,
    label_col: str,
    max_user_history_len: int,
) -> dict[int, list[int]]:
    """Build train-only user positive item histories for graph construction.

    This function must only receive the train split. It intentionally ignores
    val/test interactions so graph training and val/test recall do not leak
    future behavior.
    """
    if label_col not in train_df.columns:
        raise KeyError(f"Missing graph positive label column: {label_col}")

    pos = train_df.loc[train_df[label_col] > 0, ["user_id", "video_id", "time_ms"]].copy()
    pos = pos.sort_values(["user_id", "time_ms", "video_id"])
    sequences: dict[int, list[int]] = {}
    for user_id, group in pos.groupby("user_id", sort=False):
        seq = _dedupe_keep_order(group["video_id"].tail(max_user_history_len).tolist())
        if seq:
            sequences[int(user_id)] = seq
    return sequences


def build_item_graph(
    train_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build an undirected item-item graph from train positive co-occurrence.

    Nodes are videos. Two videos get an edge if they co-occur within the same
    user's train positive history window. Edge weight is down-weighted for long
    user histories and decays with sequence distance.
    """
    graph_cfg = cfg["recall"]["graph_emb"]
    label_col = cfg["recall"].get("positive_label_col", "is_positive")
    max_user_history_len = int(graph_cfg.get("max_user_history_len", 100))
    window_size = int(graph_cfg.get("graph_window_size", 5))
    min_edge_weight = float(graph_cfg.get("min_edge_weight", 1.0))
    max_neighbors = int(graph_cfg.get("max_neighbors_per_item", 200))

    sequences = build_positive_item_sequences(train_df, label_col, max_user_history_len)
    edge_weights: dict[int, Counter[int]] = defaultdict(Counter)

    for seq in sequences.values():
        if len(seq) < 2:
            continue
        user_weight = 1.0 / math.log2(2.0 + len(seq))
        for i, src in enumerate(seq):
            right = min(len(seq), i + window_size + 1)
            for j in range(i + 1, right):
                dst = seq[j]
                if src == dst:
                    continue
                distance = j - i
                weight = user_weight / float(distance)
                edge_weights[src][dst] += weight
                edge_weights[dst][src] += weight

    rows: list[dict[str, object]] = []
    for src, neighbors in edge_weights.items():
        ranked = [
            (int(dst), float(weight))
            for dst, weight in neighbors.items()
            if float(weight) >= min_edge_weight
        ]
        ranked.sort(key=lambda kv: (kv[1], -kv[0]), reverse=True)
        for rank, (dst, weight) in enumerate(ranked[:max_neighbors], start=1):
            rows.append(
                {
                    "video_id": int(src),
                    "neighbor_video_id": int(dst),
                    "edge_weight": float(weight),
                    "neighbor_rank": int(rank),
                }
            )

    graph_df = pd.DataFrame(rows)
    if logger:
        logger.info(
            "Built graph_emb item graph: users=%d nodes=%d edges=%d",
            len(sequences),
            graph_df["video_id"].nunique() if not graph_df.empty else 0,
            graph_df.shape[0],
        )
    return graph_df


def save_graph_neighbors(
    graph_df: pd.DataFrame,
    path: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_df.to_parquet(path, index=False)
    if logger:
        logger.info("Saved graph_emb neighbors: %s rows=%d", path, graph_df.shape[0])
    return path


def load_graph_neighbors(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path).expanduser().resolve())


def _build_node_index(graph_df: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    node_ids = np.unique(
        np.concatenate(
            [
                graph_df["video_id"].astype("int64").to_numpy(),
                graph_df["neighbor_video_id"].astype("int64").to_numpy(),
            ]
        )
    )
    node_ids.sort()
    node_to_idx = {int(node_id): int(idx) for idx, node_id in enumerate(node_ids)}
    return node_ids, node_to_idx


def _build_neighbor_arrays(
    graph_df: pd.DataFrame,
    node_to_idx: Mapping[int, int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    neighbor_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for src, group in graph_df.groupby("video_id", sort=False):
        dst_idx = np.asarray(
            [node_to_idx[int(v)] for v in group["neighbor_video_id"].tolist()],
            dtype=np.int64,
        )
        weights = pd.to_numeric(group["edge_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if weights.sum() <= 0:
            probs = np.ones(len(weights), dtype=np.float64) / max(len(weights), 1)
        else:
            probs = weights / weights.sum()
        neighbor_map[node_to_idx[int(src)]] = (dst_idx, probs)
    return neighbor_map


def generate_random_walks(
    graph_df: pd.DataFrame,
    node_to_idx: Mapping[int, int],
    walk_length: int,
    num_walks_per_node: int,
    random_seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(random_seed)
    neighbor_map = _build_neighbor_arrays(graph_df, node_to_idx)
    node_indices = np.asarray(sorted(node_to_idx.values()), dtype=np.int64)
    walks: list[list[int]] = []

    for _ in range(num_walks_per_node):
        shuffled = node_indices.copy()
        rng.shuffle(shuffled)
        for start in shuffled:
            walk = [int(start)]
            current = int(start)
            for _step in range(max(walk_length - 1, 0)):
                entry = neighbor_map.get(current)
                if entry is None:
                    break
                neighbors, probs = entry
                current = int(rng.choice(neighbors, p=probs))
                walk.append(current)
            if len(walk) > 1:
                walks.append(walk)
    return walks


def build_skipgram_pairs(
    walks: list[list[int]],
    context_window: int,
    max_pairs: int | None,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []
    for walk in walks:
        for i, center in enumerate(walk):
            left = max(0, i - context_window)
            right = min(len(walk), i + context_window + 1)
            for j in range(left, right):
                if i == j:
                    continue
                centers.append(int(center))
                contexts.append(int(walk[j]))

    if not centers:
        raise ValueError("No graph_emb skip-gram pairs generated.")

    center_arr = np.asarray(centers, dtype=np.int64)
    context_arr = np.asarray(contexts, dtype=np.int64)
    if max_pairs is not None and max_pairs > 0 and len(center_arr) > max_pairs:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(center_arr), size=int(max_pairs), replace=False)
        center_arr = center_arr[idx]
        context_arr = context_arr[idx]
    return center_arr, context_arr


class GraphSkipGramDataset(Dataset):
    def __init__(self, centers: np.ndarray, contexts: np.ndarray) -> None:
        require_torch()
        self.centers = centers.astype(np.int64, copy=False)
        self.contexts = contexts.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.centers))

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return int(self.centers[idx]), int(self.contexts[idx])


class GraphSkipGramModel(nn.Module if nn is not None else object):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        require_torch()
        super().__init__()
        self.input_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.output_embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.xavier_uniform_(self.output_embedding.weight)

    def forward(self, center_idx: Any, context_idx: Any, negative_idx: Any) -> Any:
        center_vec = self.input_embedding(center_idx)
        context_vec = self.output_embedding(context_idx)
        pos_logits = (center_vec * context_vec).sum(dim=-1)
        pos_loss = F.logsigmoid(pos_logits)

        neg_vec = self.output_embedding(negative_idx)
        neg_logits = torch.bmm(neg_vec.neg(), center_vec.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.logsigmoid(neg_logits).sum(dim=-1)
        return -(pos_loss + neg_loss).mean()

    def export_embeddings(self) -> np.ndarray:
        with torch.no_grad():
            emb = self.input_embedding.weight.detach()
            emb = F.normalize(emb, p=2, dim=-1)
            return emb.cpu().numpy().astype("float32")


def train_graph_embedding(
    train_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, GraphSkipGramModel]:
    require_torch()
    graph_cfg = cfg["recall"]["graph_emb"]
    random_seed = int(graph_cfg.get("random_seed", cfg["project"].get("random_seed", 2026)))
    graph_df = build_item_graph(train_df, cfg, logger=logger)
    if graph_df.empty:
        raise ValueError("graph_emb item graph is empty; cannot train graph embedding.")

    node_ids, node_to_idx = _build_node_index(graph_df)
    walks = generate_random_walks(
        graph_df,
        node_to_idx=node_to_idx,
        walk_length=int(graph_cfg.get("walk_length", 16)),
        num_walks_per_node=int(graph_cfg.get("num_walks_per_node", 3)),
        random_seed=random_seed,
    )
    centers, contexts = build_skipgram_pairs(
        walks,
        context_window=int(graph_cfg.get("context_window", 4)),
        max_pairs=graph_cfg.get("max_pairs", 3_000_000),
        random_seed=random_seed,
    )

    device = resolve_device(str(graph_cfg.get("device", "auto")))
    model = GraphSkipGramModel(
        num_nodes=len(node_ids),
        embedding_dim=int(graph_cfg.get("embedding_dim", 64)),
    ).to(device)
    dataset = GraphSkipGramDataset(centers, contexts)
    dataloader = DataLoader(
        dataset,
        batch_size=int(graph_cfg.get("batch_size", 4096)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(graph_cfg.get("learning_rate", 0.003)),
        weight_decay=float(graph_cfg.get("weight_decay", 1e-6)),
    )
    epochs = int(graph_cfg.get("epochs", 2))
    neg_samples = int(graph_cfg.get("negative_samples", 5))
    num_nodes = len(node_ids)

    if logger:
        logger.info(
            "Training graph_emb: nodes=%d edges=%d walks=%d pairs=%d epochs=%d device=%s",
            len(node_ids),
            graph_df.shape[0],
            len(walks),
            len(dataset),
            epochs,
            device,
        )

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_examples = 0
        for center_idx, context_idx in dataloader:
            center_idx = center_idx.to(device)
            context_idx = context_idx.to(device)
            negative_idx = torch.randint(
                low=0,
                high=num_nodes,
                size=(center_idx.size(0), neg_samples),
                device=device,
            )
            loss = model(center_idx, context_idx, negative_idx)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(center_idx.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

        if logger:
            logger.info(
                "graph_emb epoch %d/%d loss=%.6f",
                epoch,
                epochs,
                total_loss / max(total_examples, 1),
            )

    item_vectors = model.export_embeddings()
    return graph_df, node_ids.astype("int64"), item_vectors, model


def save_graph_embedding_checkpoint(
    model: GraphSkipGramModel,
    item_ids: np.ndarray,
    cfg: Mapping[str, Any],
    path: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    require_torch()
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "item_ids": item_ids.astype("int64"),
            "graph_emb_config": dict(cfg["recall"]["graph_emb"]),
        },
        path,
    )
    if logger:
        logger.info("Saved graph_emb checkpoint: %s", path)
    return path


def save_graph_item_embeddings(
    item_ids: np.ndarray,
    item_vectors: np.ndarray,
    path: str | Path,
    logger: logging.Logger | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        item_ids=item_ids.astype("int64"),
        item_vectors=item_vectors.astype("float32"),
    )
    if logger:
        logger.info("Saved graph_emb item embeddings: %s shape=%s", path, item_vectors.shape)
    return path


def load_graph_item_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(Path(path).expanduser().resolve())
    return payload["item_ids"].astype("int64"), payload["item_vectors"].astype("float32")


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return (matrix / denom).astype("float32")


def _build_seen_map(history_df: pd.DataFrame | None) -> dict[int, set[int]]:
    if history_df is None or history_df.empty:
        return {}
    return {
        int(user_id): set(int(v) for v in group["video_id"].tolist())
        for user_id, group in history_df[["user_id", "video_id"]].drop_duplicates().groupby("user_id", sort=False)
    }


def _build_user_history_from_train(
    train_df: pd.DataFrame,
    label_col: str,
    max_user_history_len: int,
) -> dict[int, list[int]]:
    return build_positive_item_sequences(
        train_df=train_df,
        label_col=label_col,
        max_user_history_len=max_user_history_len,
    )


def generate_graph_emb_candidates(
    user_ids: Iterable[int],
    train_df: pd.DataFrame,
    item_ids: np.ndarray,
    item_vectors: np.ndarray,
    cfg: Mapping[str, Any],
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    graph_cfg = cfg["recall"]["graph_emb"]
    label_col = cfg["recall"].get("positive_label_col", "is_positive")
    topk = int(graph_cfg.get("candidate_topk", 300))
    max_history_len = int(graph_cfg.get("max_user_history_len", 100))
    recency_alpha = float(graph_cfg.get("recency_alpha", 0.92))
    exclude_seen = bool(graph_cfg.get("exclude_seen", True))

    item_ids = item_ids.astype("int64")
    item_vectors = _normalize_matrix(item_vectors.astype("float32"))
    item_index = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
    user_history = _build_user_history_from_train(train_df, label_col, max_history_len)
    seen_map = _build_seen_map(history_df) if exclude_seen else {}

    rows: list[dict[str, object]] = []
    for user_id_raw in user_ids:
        user_id = int(user_id_raw)
        history = user_history.get(user_id, [])[-max_history_len:]
        history_indices = [item_index[item] for item in history if item in item_index]
        if not history_indices:
            continue

        weights = np.asarray(
            [recency_alpha ** rank for rank in range(len(history_indices) - 1, -1, -1)],
            dtype="float32",
        )
        history_vectors = item_vectors[np.asarray(history_indices, dtype=np.int64)]
        user_vector = (history_vectors * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)
        user_vector = user_vector / max(float(np.linalg.norm(user_vector)), 1e-12)

        scores = item_vectors @ user_vector.astype("float32")
        fetch_k = min(len(item_ids), topk * 3 if exclude_seen and history_df is not None else topk)
        top_idx = np.argpartition(-scores, kth=fetch_k - 1)[:fetch_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        seen = seen_map.get(user_id, set())
        kept = 0
        for item_pos in top_idx:
            video_id = int(item_ids[item_pos])
            if video_id in seen:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "video_id": video_id,
                    "recall_source": "graph_emb",
                    "source_score": float(scores[item_pos]),
                    "source_rank": int(kept + 1),
                }
            )
            kept += 1
            if kept >= topk:
                break

    return pd.DataFrame(rows)
