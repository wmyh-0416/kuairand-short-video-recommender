from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.itemcf import save_itemcf_neighbors, train_itemcf
from src.recall.popular import save_popular_items, train_popular_recall
from src.recall.twotower_dataset import build_twotower_encoders
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.paths import artifacts_dir, ensure_project_dirs, logs_dir, processed_path
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recall-layer models.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "recall.yaml"),
        help="Path to recall YAML config.",
    )
    parser.add_argument("--processed-dir", default=None, help="Override processed data directory.")
    parser.add_argument("--artifacts-dir", default=None, help="Override artifacts output directory.")
    parser.add_argument(
        "--train-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--disable-twotower",
        action="store_true",
        help="Skip two-tower training even if enabled in config.",
    )
    parser.add_argument(
        "--disable-graph-emb",
        action="store_true",
        help="Skip graph embedding recall training even if enabled in config.",
    )
    parser.add_argument(
        "--twotower-epochs",
        type=int,
        default=None,
        help="Override two-tower epochs for smoke tests.",
    )
    parser.add_argument(
        "--graph-emb-epochs",
        type=int,
        default=None,
        help="Override graph embedding epochs for smoke tests.",
    )
    return parser.parse_args()


def _recall_dir(cfg: dict) -> Path:
    return artifacts_dir(cfg) / cfg["recall"]["output"]["recall_dir"]


def _load_train_df(cfg: dict, train_rows: int | None) -> pd.DataFrame:
    path = processed_path(cfg, Path(cfg["recall"]["processed"]["splits_dir"]) / "train.parquet")
    df = pd.read_parquet(path)
    if train_rows is not None:
        df = df.head(train_rows).copy()
    return df


def _load_item_features(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(processed_path(cfg, cfg["recall"]["processed"]["item_features_file"]))


def _load_user_sequences(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(processed_path(cfg, cfg["recall"]["processed"]["user_sequences_file"]))


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _train_twotower_if_available(
    cfg: dict,
    train_df: pd.DataFrame,
    item_features: pd.DataFrame,
    user_sequences: pd.DataFrame,
    logger,
) -> None:
    tower_cfg = cfg["recall"]["twotower"]
    if not tower_cfg.get("enabled", True):
        logger.info("Two-tower recall is disabled by config.")
        return
    if not _torch_available():
        logger.warning("PyTorch is not installed; skip two-tower recall training.")
        return

    from src.recall.twotower import (
        encode_items,
        encode_users,
        resolve_device,
        save_twotower_checkpoint,
        train_twotower,
    )

    recall_dir = _recall_dir(cfg)
    encoders = build_twotower_encoders(train_df, item_features, user_sequences)
    model = train_twotower(
        train_df=train_df,
        item_features=item_features,
        user_sequences=user_sequences,
        encoders=encoders,
        cfg=cfg,
        logger=logger,
    )

    ckpt_path = recall_dir / cfg["recall"]["output"]["twotower_checkpoint_file"]
    save_twotower_checkpoint(model, encoders, ckpt_path, cfg=cfg, logger=logger)

    device = resolve_device(str(tower_cfg.get("device", "auto")))
    item_ids, item_vectors = encode_items(
        model,
        item_features=item_features,
        encoders=encoders,
        device=device,
    )
    item_emb_path = recall_dir / cfg["recall"]["output"]["twotower_item_embeddings_file"]
    np.savez_compressed(item_emb_path, item_ids=item_ids, item_vectors=item_vectors)
    logger.info("Saved two-tower item embeddings: %s shape=%s", item_emb_path, item_vectors.shape)

    user_ids = np.sort(train_df["user_id"].dropna().astype("int64").unique())
    max_seq_len = int(tower_cfg.get("max_seq_len", 50))
    sequence_col = str(tower_cfg.get("sequence_col", "watch_seq"))
    user_ids, user_vectors = encode_users(
        model,
        user_ids=user_ids,
        user_sequences=user_sequences,
        encoders=encoders,
        max_seq_len=max_seq_len,
        sequence_col=sequence_col,
        device=device,
    )
    user_emb_path = recall_dir / cfg["recall"]["output"]["twotower_user_embeddings_file"]
    np.savez_compressed(user_emb_path, user_ids=user_ids, user_vectors=user_vectors)
    logger.info("Saved two-tower train user embeddings: %s shape=%s", user_emb_path, user_vectors.shape)


def _train_graph_emb_if_available(
    cfg: dict,
    train_df: pd.DataFrame,
    logger,
) -> None:
    graph_cfg = cfg["recall"]["graph_emb"]
    if not graph_cfg.get("enabled", True):
        logger.info("Graph embedding recall is disabled by config.")
        return
    if not _torch_available():
        logger.warning("PyTorch is not installed; skip graph embedding recall training.")
        return

    from src.recall.graph_emb import (
        save_graph_embedding_checkpoint,
        save_graph_item_embeddings,
        save_graph_neighbors,
        train_graph_embedding,
    )

    recall_dir = _recall_dir(cfg)
    try:
        graph_df, item_ids, item_vectors, model = train_graph_embedding(
            train_df=train_df,
            cfg=cfg,
            logger=logger,
        )
    except ValueError as exc:
        logger.warning("Skip graph embedding recall training: %s", exc)
        return
    save_graph_neighbors(
        graph_df,
        recall_dir / cfg["recall"]["output"]["graph_emb_neighbors_file"],
        logger=logger,
    )
    save_graph_embedding_checkpoint(
        model,
        item_ids=item_ids,
        cfg=cfg,
        path=recall_dir / cfg["recall"]["output"]["graph_emb_checkpoint_file"],
        logger=logger,
    )
    save_graph_item_embeddings(
        item_ids=item_ids,
        item_vectors=item_vectors,
        path=recall_dir / cfg["recall"]["output"]["graph_emb_item_embeddings_file"],
        logger=logger,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.processed_dir is not None:
        cfg["paths"]["processed_dir"] = args.processed_dir
    if args.artifacts_dir is not None:
        cfg["paths"]["artifacts_dir"] = args.artifacts_dir
        cfg["paths"]["logs_dir"] = str(Path(args.artifacts_dir) / "logs")
    if args.disable_twotower:
        cfg["recall"]["twotower"]["enabled"] = False
    if args.disable_graph_emb:
        cfg["recall"]["graph_emb"]["enabled"] = False
    if args.twotower_epochs is not None:
        cfg["recall"]["twotower"]["epochs"] = args.twotower_epochs
    if args.graph_emb_epochs is not None:
        cfg["recall"]["graph_emb"]["epochs"] = args.graph_emb_epochs

    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["random_seed"]))

    logger = setup_logger(
        name="kuairand_rec.train_recall",
        level=cfg.get("runtime", {}).get("log_level", "INFO"),
        log_file=logs_dir(cfg) / "02_train_recall.log",
    )
    recall_dir = _recall_dir(cfg)
    recall_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training recall artifacts into: %s", recall_dir)

    train_df = _load_train_df(cfg, args.train_rows)
    item_features = _load_item_features(cfg)
    user_sequences = _load_user_sequences(cfg)
    logger.info(
        "Loaded train split rows=%d users=%d items=%d",
        train_df.shape[0],
        train_df["user_id"].nunique(),
        train_df["video_id"].nunique(),
    )

    if cfg["recall"]["popular"].get("enabled", True):
        popular_items = train_popular_recall(train_df, cfg=cfg, logger=logger)
        save_popular_items(
            popular_items,
            recall_dir / cfg["recall"]["output"]["popular_items_file"],
            logger=logger,
        )
    else:
        logger.info("Popular recall is disabled by config.")

    if cfg["recall"]["itemcf"].get("enabled", True):
        itemcf_neighbors = train_itemcf(train_df, cfg=cfg, logger=logger)
        save_itemcf_neighbors(
            itemcf_neighbors,
            recall_dir / cfg["recall"]["output"]["itemcf_neighbors_file"],
            logger=logger,
        )
    else:
        logger.info("ItemCF recall is disabled by config.")

    _train_twotower_if_available(
        cfg=cfg,
        train_df=train_df,
        item_features=item_features,
        user_sequences=user_sequences,
        logger=logger,
    )
    _train_graph_emb_if_available(
        cfg=cfg,
        train_df=train_df,
        logger=logger,
    )
    logger.info("Recall training finished.")


if __name__ == "__main__":
    main()
