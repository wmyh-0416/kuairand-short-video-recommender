# Project Report

## Background

Short-video recommendation is a good systems problem because it mixes large-scale retrieval, ranking, feedback signals, and online serving constraints. A single model is usually not enough: the system needs staged filtering, multi-objective learning, diversity control, and practical deployment logic.

This project uses **KuaiRand-Pure** to build a more production-style recommendation stack rather than only training a single baseline model.

## Problem Definition

The goal is to recommend a ranked list of short videos for each user under realistic constraints:

- large candidate space
- multiple feedback labels such as long watch, finish, and like
- need for efficient retrieval
- need for online serving and user-state updates
- need for evaluation beyond one offline metric

The project is designed as an **end-to-end system project**, not only a modeling exercise.

## Dataset

The project uses KuaiRand-Pure, which contains short-video exposure logs and item metadata. Compared with movie-rating datasets, it is much closer to a feed recommendation scenario because:

- feedback is implicit and multi-objective
- logs are exposure-based
- items have tags, authors, upload time, and statistics

### Current local processed sizes

| Split | Rows | Users | Unique items |
|---|---:|---:|---:|
| train | 1,208,280 | 26,469 | 7,542 |
| val | 94,784 | 21,184 | 5,564 |
| test | 133,545 | 22,709 | 5,722 |

The local item feature catalog currently has `7,583` rows.

### Label design

The pipeline uses:

- `like`
- `finish`
- `long_watch`
- `is_positive = long_watch OR finish OR like`

### Leakage prevention

The dataset is split by time:

- train: `date <= 20220424`
- val: `20220425 ~ 20220430`
- test: `date >= 20220501`

The ranking stages also use point-in-time historical sequences rather than mixing future behavior into earlier features.

## Methodology

### Multi-channel recall

Implemented recall branches:

- `popular`
- `itemcf`
- `twotower`
- `graph_emb`
- `twotower_faiss`

Recall branches are merged into unified candidate files and passed downstream.

### FAISS ANN

Two-Tower item embeddings are exported to FAISS indices:

- `IndexFlatIP`
- `HNSW`
- `IVF`

This turns the Two-Tower branch into a more realistic vector retrieval system rather than an offline brute-force demo.

### Pre-rank

The pre-rank stage uses LightGBM to compress candidates before deep ranking. It consumes merged recall candidates and engineered user/item/source features.

### Rank

The rank stage uses a DIN-style multi-task model to predict:

- `long_watch`
- `finish`
- `like`

These task scores are fused into a final `rank_score`.

### Rerank

The rerank layer is rule-based and adjusts the final slate with:

- author frequency control
- tag diversity
- freshness bonus

## System Architecture

```text
KuaiRand-Pure logs
  -> preprocess
  -> multi-channel recall
  -> FAISS ANN retrieval
  -> LightGBM prerank
  -> DIN multi-task rank
  -> rule-based rerank
  -> FastAPI serving
  -> feedback / cache / monitoring
```

## Offline Experiments

### Recall / pre-rank / rank / rerank

Key current local results:

| Stage | Metric | Val | Test |
|---|---|---:|---:|
| Recall | Recall@100 | 0.2476 | 0.2343 |
| Recall | Recall@200 | 0.3151 | 0.2973 |
| Recall | Coverage | 0.9995 | 0.9996 |
| Pre-rank | Recall retained@100 | 0.6600 | 0.6315 |
| Rank | rank_auc | 0.5753 | 0.5845 |
| Rank | NDCG@20 | 0.0295 | 0.0359 |
| Rerank | NDCG@20 delta | +0.0000117 | +0.0001823 |
| Rerank | avg unique tags / user delta | +0.3091 | +0.3023 |

Interpretation:

- recall and pre-rank do most of the heavy filtering work
- rank improves final ordering but still leaves room for stronger sequence models
- rerank improves slate quality mostly through diversity, not big relevance jumps

### FAISS benchmark

Current local FAISS benchmark:

| Index | Mean latency (ms) | P95 latency (ms) | Mean overlap@500 vs FlatIP |
|---|---:|---:|---:|
| FlatIP | 0.8636 | 0.8543 | 1.0000 |
| HNSW | 0.7151 | 0.8091 | 0.8352 |

This shows the HNSW ANN branch is faster in the current local setup while preserving most of the exact TopK set.

## Online Serving

The project includes a FastAPI serving layer with:

- `/health`
- `/recommend`
- `/feedback`
- `/metrics`
- `/metrics/prometheus`

The online path reuses offline artifacts instead of retraining models in the request path.

### Realtime feedback loop

`/feedback` updates:

- recent viewed videos
- recent positive videos
- liked videos
- recent tags
- counters such as skip, like, and long view

This state is used in the next `/recommend` call for recent-view filtering and state-aware behavior.

### Degraded mode

The service can still start if some components are missing:

- Redis missing: memory fallback
- rank missing: return prerank order
- rerank missing: return rank order
- some recall assets missing: try fallback recall

## Monitoring

The serving layer exposes:

- request counts
- error counts
- stage-level latency
- cache hit / miss
- degraded mode counts
- request log JSONL

### Current local benchmark

Current local benchmark report:

- requests: `50`
- concurrency: `5`
- success: `50/50`
- QPS: `1.74`
- P95 latency: `3245 ms`
- cache hit rate: `0.02`

The main bottlenecks are prerank and rank inference.

## A/B Simulation

The project includes **offline log-replay A/B simulation**, not real online experimentation.

Current comparison:

- control: `popular`
- treatment: `full_pipeline`

### Key result

The primary metric `long_view_rate@10` did not significantly beat the popular baseline:

- control: `0.007492`
- treatment: `0.006921`
- relative lift: `-7.62%`
- bootstrap 95% CI crosses zero

However, treatment improved:

- `recall@50` by `52.0%`
- `ndcg@50` by `14.6%`
- `coverage@10` very strongly

This is a useful discussion point: the full pipeline improves depth and coverage, but not all proxy quality metrics move in the desired direction under offline replay.

## Cold-start Analysis

The project also includes a cold-start cohort analysis with heuristic enhancement:

- `global_popular`
- `category_popular`
- `freshness_boost`

### User-side result

The enhancement helps cold users the most:

- `new_user hit_rate@10 +48.8%`
- `new_user recall@50 +15.5%`
- `new_user long_view_rate@10 +64.1%`

### Trade-off

The improvement comes with lower coverage and stronger concentration on popular content. Item cold-start is still weak because the current logic mostly improves user cold-start rather than generating strong cold-item candidates.

## Limitations

- Offline A/B replay is not real online A/B
- Serving benchmark is local only
- Docker Compose was configured but not fully verified on the current machine because Docker was unavailable
- Realtime feedback is a serving-state simulation, not a retraining loop
- Some artifacts may exist as Git LFS pointers instead of materialized parquet files
- Item cold-start remains weak
- The current feature store is still local parquet-based

## Future Work

- validate Redis + Docker in a real environment
- add Prometheus / Grafana dashboards
- build content-based cold-item recall
- add IPS / DR debiased offline evaluation
- add model retraining and refresh loop
- compare DIN against Transformer / DCN / DeepFM rankers
