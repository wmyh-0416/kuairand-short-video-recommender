# System Design Notes

## 1. Why a multi-stage recommendation architecture

Short-video recommendation has a large candidate space and multiple feedback types. Running a heavy ranker over the entire catalog is expensive and unnecessary, so the system is split into recall, prerank, rank, and rerank. Each stage solves a different problem: recall expands coverage, prerank filters obvious negatives, rank optimizes relevance, and rerank injects business or presentation constraints.

## 2. Why use FAISS / ANN in retrieval

Two-Tower embeddings are useful only if they can be searched efficiently over the full item catalog. FAISS gives a practical vector retrieval layer so the project is not limited to brute-force numpy matrix multiplication. It also makes the repository closer to how production retrieval systems are usually discussed in interviews.

## 3. IndexFlatIP vs HNSW vs IVF trade-off

`IndexFlatIP` is the exact baseline: simple and correct, but it does full scan search. `HNSW` is an approximate graph-based index with good latency-quality trade-offs and little training overhead. `IVF` clusters vectors first and then searches a subset of partitions, which is attractive for larger catalogs but depends more on index training and probe settings.

In the current local benchmark, HNSW improved mean retrieval latency versus FlatIP while keeping strong overlap@500, but it used more index memory.

## 4. Why LightGBM for prerank

The prerank stage needs a cheap model that can score large candidate sets quickly and handle dense plus categorical engineered features. LightGBM is a strong fit because it trains fast, is easy to debug, and usually works well as a first-stage rank filter before deep ranking. It is also easier to serve than a larger deep model.

## 5. Why a DIN-style multi-task ranker

The rank stage needs sequence awareness because short-video preferences depend heavily on recent watch behavior. DIN-style attention lets the model compare the target item with the user's recent history more directly than a plain MLP. The multi-task setup is useful because the project cares about long watch, finish, and like at the same time rather than optimizing a single binary click target.

## 6. What rerank is doing

The rerank layer is intentionally lightweight and rule-based. It does not try to relearn the whole ranking problem. Instead, it adjusts the final slate with author frequency control, tag diversity, and freshness bonuses so the feed is less repetitive while staying close to the ranking model's relevance order.

## 7. How online serving is designed

The online path wraps offline artifacts rather than retraining anything on request. A request goes through user-state lookup, recall, prerank, rank, and rerank, and then returns a JSON response with stage scores and latency. This keeps serving logic separate from training logic and makes degraded-mode fallbacks easier to reason about.

## 8. How UserState is updated

`/feedback` updates a lightweight `UserState` object containing recent viewed items, positive items, liked items, recent tags, and simple counters such as skip count and long-view count. This state is then used by `/recommend` to filter recently viewed videos and provide a more realistic feedback loop. It is a serving-state simulation, not a full online learning pipeline.

## 9. How Redis cache is used

Redis is optional. When available, it stores user state and recommendation cache entries keyed by namespace and user ID. When Redis is unavailable, the system falls back to in-process memory so the service can still run locally without external dependencies.

## 10. How degraded mode is designed

The service avoids failing hard when optional components are missing. If FAISS is unavailable, the service tries other recall fallbacks. If prerank or rank artifacts are missing, the pipeline skips the unavailable stage and continues with earlier scores. `/health` exposes which components are loaded and why degraded mode is active.

## 11. How time leakage is avoided

The project uses date-based splits instead of random splits. Ranking history is constructed from earlier interactions only, and later interactions are not allowed to influence earlier-stage user features. This is important because recommendation metrics can look unrealistically strong if the model accidentally sees future behavior.

## 12. Offline evaluation vs online metrics

Offline metrics such as Recall@K, NDCG@K, AUC, and long_view_rate@K are useful for debugging model behavior and comparing policies under fixed logged data. Online metrics measure live user response under actual traffic and exposure. Offline gains can be directionally useful, but they do not guarantee online improvement because of exposure bias, policy mismatch, and feedback loop effects.

## 13. Why the A/B simulation is limited

The A/B module is an offline log-replay simulation, not a randomized online experiment. It uses logged test interactions as ground truth and compares policies by user-level bucketing, but it cannot estimate causal treatment effects reliably. That is why the README and reports explicitly say it is not real online A/B.

## 14. Cold-start strategy and trade-off

The cold-start enhancement is heuristic by design: global popular backfill, category-popular backfill, and lightweight freshness boosting. It helps new users because those users lack enough history for deeper personalization. The trade-off is lower coverage and continued concentration on popular items, which is exactly what the current analysis shows.

## 15. Current system bottlenecks

- serving latency is still high for a realistic production target, with prerank and rank dominating the local benchmark
- item cold-start remains weak because the current enhancement mostly helps user cold-start, not cold-item retrieval
- there is no real online traffic, so evaluation is still fundamentally offline
- the feature store is still local parquet-based instead of an online serving feature platform

## 16. Why FAISS was connected to the existing pipeline instead of replacing it

The point of the project is to preserve the multi-stage structure and improve one layer at a time. FAISS was added as a retrieval backend for the Two-Tower branch rather than replacing the whole recall pipeline. That keeps comparison cleaner and allows the project to retain `popular`, `itemcf`, and `graph_emb` for diversity and robustness.

## 17. Why monitoring matters in this project

Even in a local project, service-level observability makes the system more interview-ready. The benchmark and metrics endpoints show where latency comes from, whether cache is working, and whether degraded mode is silently masking failures. This turns the repository from pure modeling code into a system that can be discussed like an actual service.

## 18. What would change in a more production-like version

The next serious upgrades would be a real Redis environment, a proper feature store, asynchronous feedback ingestion, automated model refresh, Prometheus/Grafana dashboards, and more realistic serving optimization. On the modeling side, content-based cold-item recall, debiased evaluation, and stronger rankers such as Transformer / DCN / DeepFM would be natural next steps.
