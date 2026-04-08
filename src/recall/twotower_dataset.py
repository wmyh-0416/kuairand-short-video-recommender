from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


def require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the two-tower recall model. "
            "Install torch or set recall.twotower.enabled=false."
        )


@dataclass
class CategoricalEncoder:
    name: str
    mapping: dict[str, int]

    @classmethod
    def fit(cls, name: str, values: Iterable[object]) -> "CategoricalEncoder":
        normalized = sorted({cls._key(v) for v in values if pd.notna(v)})
        # 0 is reserved for padding/OOV.
        mapping = {value: idx + 1 for idx, value in enumerate(normalized)}
        return cls(name=name, mapping=mapping)

    @staticmethod
    def _key(value: object) -> str:
        return str(value)

    @property
    def size(self) -> int:
        return len(self.mapping) + 1

    def transform_one(self, value: object) -> int:
        return int(self.mapping.get(self._key(value), 0))

    def transform(self, values: Iterable[object]) -> np.ndarray:
        return np.asarray([self.transform_one(v) for v in values], dtype=np.int64)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "mapping": self.mapping}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CategoricalEncoder":
        return cls(
            name=str(data["name"]),
            mapping={str(k): int(v) for k, v in dict(data["mapping"]).items()},
        )


def save_encoders(encoders: Mapping[str, CategoricalEncoder], path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: encoder.to_dict() for name, encoder in encoders.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_encoders(path: str | Path) -> dict[str, CategoricalEncoder]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return {name: CategoricalEncoder.from_dict(data) for name, data in payload.items()}


def build_twotower_encoders(
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    user_sequences: pd.DataFrame,
) -> dict[str, CategoricalEncoder]:
    user_values = pd.concat(
        [
            train_df["user_id"],
            user_sequences["user_id"] if "user_id" in user_sequences.columns else pd.Series(dtype="int64"),
        ],
        ignore_index=True,
    )
    item_values = pd.concat(
        [
            item_features["video_id"],
            train_df["video_id"],
        ],
        ignore_index=True,
    )
    return {
        "user": CategoricalEncoder.fit("user", user_values),
        "video": CategoricalEncoder.fit("video", item_values),
        "author": CategoricalEncoder.fit("author", item_features["author_id"]),
        "tag": CategoricalEncoder.fit("tag", item_features["tag"].astype("string").fillna("unknown")),
    }


def sequence_to_indices(
    seq: object,
    item_encoder: CategoricalEncoder,
    max_seq_len: int,
) -> np.ndarray:
    if seq is None:
        values: list[object] = []
    elif isinstance(seq, np.ndarray):
        values = seq.tolist()
    elif isinstance(seq, list):
        values = seq
    elif isinstance(seq, tuple):
        values = list(seq)
    else:
        values = []

    values = values[-max_seq_len:]
    encoded = item_encoder.transform(values)
    out = np.zeros(max_seq_len, dtype=np.int64)
    if len(encoded) > 0:
        out[-len(encoded) :] = encoded
    return out


def build_item_feature_table(
    item_features: pd.DataFrame,
    encoders: Mapping[str, CategoricalEncoder],
) -> pd.DataFrame:
    item_df = item_features[["video_id", "author_id", "tag"]].drop_duplicates("video_id").copy()
    item_df["tag"] = item_df["tag"].astype("string").fillna("unknown")
    item_df["item_idx"] = encoders["video"].transform(item_df["video_id"])
    item_df["author_idx"] = encoders["author"].transform(item_df["author_id"])
    item_df["tag_idx"] = encoders["tag"].transform(item_df["tag"])
    return item_df[["video_id", "item_idx", "author_idx", "tag_idx"]]


def build_user_feature_table(
    user_ids: Iterable[int],
    user_sequences: pd.DataFrame,
    encoders: Mapping[str, CategoricalEncoder],
    max_seq_len: int,
    sequence_col: str = "watch_seq",
) -> pd.DataFrame:
    seq_map: dict[int, object] = {}
    if not user_sequences.empty and sequence_col in user_sequences.columns:
        seq_map = {
            int(row.user_id): getattr(row, sequence_col)
            for row in user_sequences[["user_id", sequence_col]].itertuples(index=False)
        }

    rows: list[dict[str, object]] = []
    for user_id_raw in user_ids:
        user_id = int(user_id_raw)
        rows.append(
            {
                "user_id": user_id,
                "user_idx": encoders["user"].transform_one(user_id),
                "seq_item_idx": sequence_to_indices(
                    seq_map.get(user_id),
                    encoders["video"],
                    max_seq_len=max_seq_len,
                ),
            }
        )
    return pd.DataFrame(rows)


class TwoTowerTrainDataset(Dataset):
    def __init__(
        self,
        interactions: pd.DataFrame,
        item_features: pd.DataFrame,
        user_sequences: pd.DataFrame,
        encoders: Mapping[str, CategoricalEncoder],
        label_col: str = "is_positive",
        sequence_col: str = "watch_seq",
        max_seq_len: int = 50,
    ) -> None:
        require_torch()
        if label_col not in interactions.columns:
            raise KeyError(f"Missing label column: {label_col}")

        positives = interactions.loc[
            interactions[label_col] > 0,
            ["user_id", "video_id"],
        ].drop_duplicates()
        if positives.empty:
            raise ValueError("No positive samples found for two-tower training.")

        item_table = build_item_feature_table(item_features, encoders).set_index("video_id")
        seq_table = build_user_feature_table(
            positives["user_id"].unique(),
            user_sequences,
            encoders,
            max_seq_len=max_seq_len,
            sequence_col=sequence_col,
        ).set_index("user_id")

        rows: list[dict[str, object]] = []
        for user_id, video_id in positives.itertuples(index=False):
            user_id_int = int(user_id)
            video_id_int = int(video_id)
            if video_id_int not in item_table.index:
                continue
            item_row = item_table.loc[video_id_int]
            if user_id_int in seq_table.index:
                seq_idx = seq_table.loc[user_id_int, "seq_item_idx"]
                user_idx = int(seq_table.loc[user_id_int, "user_idx"])
            else:
                seq_idx = np.zeros(max_seq_len, dtype=np.int64)
                user_idx = encoders["user"].transform_one(user_id_int)

            rows.append(
                {
                    "user_idx": user_idx,
                    "seq_item_idx": seq_idx,
                    "item_idx": int(item_row["item_idx"]),
                    "author_idx": int(item_row["author_idx"]),
                    "tag_idx": int(item_row["tag_idx"]),
                }
            )

        if not rows:
            raise ValueError("No valid two-tower samples after joining item features.")
        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {
            "user_idx": int(sample["user_idx"]),
            "seq_item_idx": np.asarray(sample["seq_item_idx"], dtype=np.int64),
            "item_idx": int(sample["item_idx"]),
            "author_idx": int(sample["author_idx"]),
            "tag_idx": int(sample["tag_idx"]),
        }


def twotower_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    require_torch()
    return {
        "user_idx": torch.as_tensor([x["user_idx"] for x in batch], dtype=torch.long),
        "seq_item_idx": torch.as_tensor(np.stack([x["seq_item_idx"] for x in batch]), dtype=torch.long),
        "item_idx": torch.as_tensor([x["item_idx"] for x in batch], dtype=torch.long),
        "author_idx": torch.as_tensor([x["author_idx"] for x in batch], dtype=torch.long),
        "tag_idx": torch.as_tensor([x["tag_idx"] for x in batch], dtype=torch.long),
    }
