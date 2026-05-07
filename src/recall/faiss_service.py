from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.recall.faiss_index import l2_normalize, load_index, load_video_id_map


class FaissRecallService:
    def __init__(self, index_path: str | Path, id_map_path: str | Path, normalize: bool = True) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.id_map_path = Path(id_map_path).expanduser().resolve()
        self.index = load_index(self.index_path)
        self.video_ids = load_video_id_map(self.id_map_path)
        self.normalize = bool(normalize)
        ntotal = int(getattr(self.index, "ntotal", 0))
        if ntotal != len(self.video_ids):
            raise ValueError(
                "FAISS index size does not match video id map: "
                f"index.ntotal={ntotal} vs len(video_ids)={len(self.video_ids)}"
            )

    def _prepare_query(self, user_embedding: Any) -> np.ndarray:
        query = np.asarray(user_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError(f"user_embedding must be a 1D vector or single-row 2D array, got shape={query.shape}")
        if self.normalize:
            query = l2_normalize(query)
        return query.astype(np.float32, copy=False)

    def recall(self, user_embedding: Any, top_k: int = 500) -> list[dict[str, float | int]]:
        if int(top_k) <= 0:
            raise ValueError(f"top_k must be positive, got: {top_k}")
        query = self._prepare_query(user_embedding)
        scores, indices = self.index.search(query, int(top_k))

        results: list[dict[str, float | int]] = []
        for score, row_idx in zip(scores[0], indices[0]):
            if int(row_idx) < 0:
                continue
            results.append(
                {
                    "video_id": int(self.video_ids[int(row_idx)]),
                    "recall_score": float(score),
                }
            )
        return results
