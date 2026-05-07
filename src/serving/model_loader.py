from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from src.prerank.lightgbm_model import load_model_bundle, predict_lightgbm
from src.prerank.mlp_model import predict_mlp
from src.rank.infer import load_rank_checkpoint, predict_rank_scores
from src.rank.model import resolve_device as resolve_rank_device
from src.recall.faiss_service import FaissRecallService
from src.recall.twotower import encode_users, load_twotower_checkpoint, resolve_device as resolve_twotower_device
from src.utils.config import load_config
from src.utils.paths import artifact_path, artifacts_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def apply_runtime_paths(
    cfg: Mapping[str, Any],
    *,
    project_root: Path,
    processed_dir: Path,
    artifacts_dir_path: Path,
) -> dict[str, Any]:
    out = deepcopy(dict(cfg))
    out.setdefault("project", {})
    out.setdefault("paths", {})
    out["project"]["root_dir"] = str(project_root.resolve())
    out["paths"]["raw_data_dir"] = str((project_root / "KuaiRand-Pure").resolve())
    out["paths"]["processed_dir"] = str(processed_dir.resolve())
    out["paths"]["artifacts_dir"] = str(artifacts_dir_path.resolve())
    out["paths"]["logs_dir"] = str((artifacts_dir_path / "logs").resolve())
    return out


def load_component_configs(
    serving_cfg: Mapping[str, Any],
    *,
    project_root: Path,
    processed_dir: Path,
    artifacts_dir_path: Path,
) -> dict[str, dict[str, Any]]:
    config_refs = serving_cfg["serving"]["component_configs"]
    loaded: dict[str, dict[str, Any]] = {}
    for name, ref in config_refs.items():
        ref_path = Path(ref)
        if not ref_path.is_absolute():
            ref_path = (project_root / ref_path).resolve()
        component_cfg = load_config(ref_path)
        loaded[str(name)] = apply_runtime_paths(
            component_cfg,
            project_root=project_root,
            processed_dir=processed_dir,
            artifacts_dir_path=artifacts_dir_path,
        )
    return loaded


class ServingModelLoader:
    def __init__(
        self,
        serving_cfg: Mapping[str, Any],
        component_cfgs: Mapping[str, Mapping[str, Any]],
        feature_store: Any,
        logger: Optional[Any] = None,
    ) -> None:
        self.serving_cfg = serving_cfg
        self.component_cfgs = component_cfgs
        self.feature_store = feature_store
        self.logger = logger

        self.loaded_components: dict[str, bool] = {
            "faiss": False,
            "prerank": False,
            "rank": False,
            "rerank": False,
        }
        self.component_errors: dict[str, str] = {}

        self.faiss_service: Optional[FaissRecallService] = None
        self.prerank_bundle: Optional[dict[str, Any]] = None
        self.rank_model: Optional[Any] = None
        self.rank_spec: Optional[Any] = None
        self.rank_runtime_cfg: Optional[dict[str, Any]] = None
        self.rank_device: str = "cpu"
        self.rerank_cfg: Optional[dict[str, Any]] = None

        self.twotower_model: Optional[Any] = None
        self.twotower_encoders: Optional[dict[str, Any]] = None
        self.twotower_saved_cfg: Optional[dict[str, Any]] = None
        self.twotower_device: str = "cpu"

        self.user_embedding_ids: Optional[np.ndarray] = None
        self.user_embedding_vectors: Optional[np.ndarray] = None
        self.user_embedding_index: dict[int, int] = {}
        self.mean_user_embedding: Optional[np.ndarray] = None

        self._load_all()

    def _log_warning(self, message: str, *args: Any) -> None:
        if self.logger:
            self.logger.warning(message, *args)

    def _log_info(self, message: str, *args: Any) -> None:
        if self.logger:
            self.logger.info(message, *args)

    def _mark_error(self, component: str, exc: Exception) -> None:
        self.loaded_components[component] = False
        self.component_errors[component] = str(exc)
        self._log_warning("Serving component %s is unavailable: %s", component, exc)

    def _load_all(self) -> None:
        self._load_faiss()
        self._load_user_embeddings()
        self._load_twotower_checkpoint()
        self._load_prerank()
        self._load_rank()
        self._load_rerank()

    def _load_faiss(self) -> None:
        faiss_cfg = self.serving_cfg["serving"]["faiss"]
        if not bool(faiss_cfg.get("enabled", True)):
            return
        try:
            index_type = str(faiss_cfg.get("index_type", "hnsw")).lower()
            index_path = artifact_path(self.serving_cfg, faiss_cfg["index_files"][index_type])
            id_map_path = artifact_path(self.serving_cfg, faiss_cfg["id_map_file"])
            self.faiss_service = FaissRecallService(
                index_path=index_path,
                id_map_path=id_map_path,
                normalize=bool(faiss_cfg.get("normalize", True)),
            )
            self.loaded_components["faiss"] = True
            self._log_info("Loaded FAISS recall service: index=%s id_map=%s", index_path, id_map_path)
        except Exception as exc:
            self._mark_error("faiss", exc)

    def _load_user_embeddings(self) -> None:
        try:
            user_emb_path = artifact_path(self.serving_cfg, self.serving_cfg["serving"]["faiss"]["user_embeddings_file"])
            if not user_emb_path.exists():
                raise FileNotFoundError(f"Two-tower user embedding file not found: {user_emb_path}")
            payload = np.load(user_emb_path)
            required = {"user_ids", "user_vectors"}
            missing = required - set(payload.files)
            if missing:
                raise KeyError(f"Two-tower user embeddings missing arrays: {sorted(missing)}")
            user_ids = np.asarray(payload["user_ids"], dtype=np.int64)
            user_vectors = np.asarray(payload["user_vectors"], dtype=np.float32)
            self.user_embedding_ids = np.asarray(user_ids, dtype=np.int64)
            self.user_embedding_vectors = np.asarray(user_vectors, dtype=np.float32)
            self.user_embedding_index = {int(user_id): int(idx) for idx, user_id in enumerate(self.user_embedding_ids)}
            if len(self.user_embedding_vectors):
                self.mean_user_embedding = self.user_embedding_vectors.mean(axis=0).astype(np.float32)
            self._log_info("Loaded online user embeddings: path=%s rows=%d", user_emb_path, len(self.user_embedding_index))
        except Exception as exc:
            self.component_errors["twotower_user_embeddings"] = str(exc)
            self._log_warning("Online user embeddings are unavailable: %s", exc)

    def _load_twotower_checkpoint(self) -> None:
        try:
            recall_cfg = self.component_cfgs["recall"]
            ckpt_path = (
                artifacts_dir(recall_cfg)
                / recall_cfg["recall"]["output"]["recall_dir"]
                / self.serving_cfg["serving"]["faiss"]["twotower_checkpoint_file"]
            )
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Two-tower checkpoint not found: {ckpt_path}")
            device = resolve_twotower_device("auto")
            model, encoders, saved_cfg = load_twotower_checkpoint(ckpt_path, device=device)
            self.twotower_model = model
            self.twotower_encoders = encoders
            self.twotower_saved_cfg = saved_cfg
            self.twotower_device = device
            self._log_info("Loaded two-tower checkpoint for online encoding: %s", ckpt_path)
        except Exception as exc:
            self.component_errors["twotower_checkpoint"] = str(exc)
            self._log_warning("Online two-tower checkpoint is unavailable: %s", exc)

    def _load_prerank(self) -> None:
        if not bool(self.serving_cfg["serving"]["components"].get("prerank", True)):
            return
        try:
            prerank_cfg = self.component_cfgs["prerank"]
            model_path = (
                artifacts_dir(prerank_cfg)
                / prerank_cfg["prerank"]["output"]["prerank_dir"]
                / prerank_cfg["prerank"]["output"]["model_file"]
            )
            if not model_path.exists():
                raise FileNotFoundError(f"Prerank model not found: {model_path}")
            self.prerank_bundle = load_model_bundle(model_path)
            self.loaded_components["prerank"] = True
            self._log_info("Loaded prerank model bundle: %s", model_path)
        except Exception as exc:
            self._mark_error("prerank", exc)

    def _load_rank(self) -> None:
        if not bool(self.serving_cfg["serving"]["components"].get("rank", True)):
            return
        try:
            rank_cfg = self.component_cfgs["rank"]
            device = resolve_rank_device(str(rank_cfg["rank"]["model"].get("device", "auto")))
            model, spec, runtime_cfg = load_rank_checkpoint(rank_cfg, device=device)
            self.rank_model = model
            self.rank_spec = spec
            self.rank_runtime_cfg = runtime_cfg
            self.rank_device = device
            self.loaded_components["rank"] = True
            self._log_info("Loaded DIN rank checkpoint on device=%s", device)
        except Exception as exc:
            self._mark_error("rank", exc)

    def _load_rerank(self) -> None:
        if not bool(self.serving_cfg["serving"]["components"].get("rerank", True)):
            return
        try:
            self.rerank_cfg = dict(self.component_cfgs["rerank"])
            self.loaded_components["rerank"] = True
            self._log_info("Loaded rerank config for online serving.")
        except Exception as exc:
            self._mark_error("rerank", exc)

    def has_any_recall_backend(self) -> bool:
        return self.faiss_service is not None or self.feature_store.popular_items is not None

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        idx = self.user_embedding_index.get(int(user_id))
        if idx is None or self.user_embedding_vectors is None:
            return None
        return np.asarray(self.user_embedding_vectors[idx], dtype=np.float32)

    def encode_user_embedding(self, user_id: int, user_sequences: pd.DataFrame) -> Optional[np.ndarray]:
        if self.twotower_model is None or self.twotower_encoders is None or self.twotower_saved_cfg is None:
            return None
        max_seq_len = int(self.twotower_saved_cfg.get("max_seq_len", 50))
        sequence_col = str(self.twotower_saved_cfg.get("sequence_col", "watch_seq"))
        user_ids, user_vectors = encode_users(
            self.twotower_model,
            user_ids=np.asarray([int(user_id)], dtype=np.int64),
            user_sequences=user_sequences,
            encoders=self.twotower_encoders,
            max_seq_len=max_seq_len,
            sequence_col=sequence_col,
            device=self.twotower_device,
        )
        if len(user_ids) == 0 or user_vectors.size == 0:
            return None
        return np.asarray(user_vectors[0], dtype=np.float32)

    def recall_faiss(self, user_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.faiss_service is None:
            raise RuntimeError("FAISS recall service is unavailable.")
        return self.faiss_service.recall(user_embedding, top_k=top_k)

    def predict_prerank_scores(self, x: pd.DataFrame) -> np.ndarray:
        if self.prerank_bundle is None:
            raise RuntimeError("Prerank model bundle is unavailable.")
        model_type = str(self.prerank_bundle["model_type"])
        model_payload = self.prerank_bundle["model"]
        if model_type == "lightgbm":
            return predict_lightgbm(model_payload, x)
        if model_type == "mlp":
            return predict_mlp(model_payload, x, self.component_cfgs["prerank"])
        raise ValueError(f"Unsupported prerank model type: {model_type}")

    def predict_rank_frame(self, frame: pd.DataFrame, request_store: Any) -> pd.DataFrame:
        if self.rank_model is None or self.rank_spec is None or self.rank_runtime_cfg is None:
            raise RuntimeError("Rank model is unavailable.")
        from src.rank.features import transform_rank_frame

        arrays = transform_rank_frame(frame, self.rank_spec, request_store, self.rank_runtime_cfg, include_labels=True)
        return predict_rank_scores(
            frame=frame,
            arrays=arrays,
            spec=self.rank_spec,
            model=self.rank_model,
            cfg=self.rank_runtime_cfg,
            device=self.rank_device,
        )
