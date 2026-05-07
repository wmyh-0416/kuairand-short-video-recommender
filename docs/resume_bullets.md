# Resume Bullets

## 中文简历 Bullet

### 算法工程师版本

- 基于 KuaiRand-Pure 构建端到端短视频推荐系统，完成 `preprocess -> multi-recall -> LightGBM 粗排 -> DIN 多任务精排 -> rerank` 全链路；当前本地 test artifact 上 `Recall@100=0.2343`，rerank 后 `NDCG@20=0.0462`。
- 在召回层实现 `popular / itemcf / twotower / graph_emb / twotower_faiss` 多路融合，并将 Two-Tower item embedding 接入 FAISS ANN；HNSW 相比 `IndexFlatIP` 平均延迟从 `0.864ms` 降到 `0.715ms`，`Top500 mean overlap=0.835`。
- 设计离线 log-replay A/B simulation，比对 `popular` 与 `full_pipeline`；结果显示 primary metric `long_view_rate@10` 未显著优于 popular， 但 `recall@50 +52.0%`、`ndcg@50 +14.6%`、`coverage@10 +69.6x`，形成更完整的指标 trade-off 论证。
- 补充冷启动分层分析，对 `new / low / medium / high active` 用户分别评估；启发式增强策略使 `new_user hit_rate@10 +48.8%`、`long_view_rate@10 +64.1%`，同时识别出 `coverage` 下降和 `item cold-start` 仍弱的问题。

### 机器学习工程师版本

- 将 KuaiRand-Pure 曝光日志改造成多目标推荐训练集，构造 `is_positive / long_watch / finish / like` 标签，并采用严格时间切分与 point-in-time 历史特征，避免排序阶段时间泄漏。
- 训练 Two-Tower 召回模型、LightGBM 粗排模型和 DIN-style 多任务排序模型；本地 test artifact 上 `rank_auc=0.5845`，子任务 `finish_auc=0.6663`、`like_auc=0.6219`。
- 在离线评估之外补充 FastAPI serving、feedback state、offline A/B simulation 和 cold-start cohort analysis，使项目不仅覆盖模型训练，也覆盖模型使用、实验解释和失败案例分析。
- 对冷启动、A/B、Serving benchmark 的结果保持诚实表达：例如 full pipeline 在 offline replay 中 primary metric 未显著赢 popular，serving benchmark 仍暴露出 prerank/rank 推理延迟瓶颈。

### 后端 / MLOps 偏工程版本

- 将离线推荐产物封装为 FastAPI 在线服务，提供 `/health /recommend /feedback /metrics /metrics/prometheus` 接口，并支持缺失模型时的 degraded mode。
- 设计 `UserState + recommendation cache + feedback invalidation` 的实时闭环，支持 Redis 可选接入与 memory fallback；feedback 后推荐结果可即时变化。
- 实现结构化请求日志、进程内 MetricsRegistry、本地 benchmark 与 Docker Compose 配置；本地 50 请求 benchmark 下 `QPS=1.74`、`P95 latency=3245ms`，并定位 prerank/rank 为主要延迟热点。
- 编写部署文档、系统设计文档、面试问答和项目报告，使项目从“能跑”扩展为“可展示、可复盘、可面试讲解”的完整工程仓库。

## English Resume Bullets

### Recommendation Algorithm Engineer

- Built an end-to-end short-video recommendation system on KuaiRand-Pure with `preprocess -> multi-recall -> LightGBM prerank -> DIN-style multi-task rank -> rule-based rerank`; current local test artifacts show `Recall@100=0.2343` and reranked `NDCG@20=0.0462`.
- Implemented and merged `popular`, `itemcf`, `twotower`, `graph_emb`, and `twotower_faiss` recall channels, and integrated FAISS ANN retrieval for Two-Tower embeddings; HNSW reduced mean retrieval latency from `0.864 ms` to `0.715 ms` with `mean overlap@500=0.835` against FlatIP.
- Added offline log-replay A/B simulation comparing `popular` vs `full_pipeline`; the primary metric `long_view_rate@10` did not significantly outperform the baseline, but treatment improved `recall@50` by `52.0%` and `ndcg@50` by `14.6%`, providing a clear trade-off discussion.
- Added cold-start cohort analysis and heuristic enhancement; for `new_user`, the enhanced pipeline improved `hit_rate@10` by `48.8%` and `long_view_rate@10` by `64.1%`, while also exposing lower coverage and weak item cold-start behavior.

### Machine Learning Engineer

- Converted KuaiRand-Pure exposure logs into a time-aware multi-objective recommendation dataset with `is_positive`, `long_watch`, `finish`, and `like` supervision, and enforced point-in-time history construction to reduce leakage risk.
- Trained a Two-Tower retriever, LightGBM preranker, and DIN-style multi-task ranker; current local test artifacts show `rank_auc=0.5845`, `finish_auc=0.6663`, and `like_auc=0.6219`.
- Extended the project beyond offline modeling by adding online serving, realtime user-state simulation, offline experimentation, and cold-start analysis, making the system useful for both modeling and systems-oriented interviews.
- Reported negative or mixed results honestly, including the non-significant offline A/B primary metric and high local serving latency, instead of overstating offline gains as online wins.

### ML Systems / MLOps Engineer

- Wrapped offline recommendation artifacts into a FastAPI serving stack with `/health`, `/recommend`, `/feedback`, `/metrics`, and `/metrics/prometheus`, plus degraded-mode fallbacks when specific models or assets are unavailable.
- Implemented a realtime feedback loop with `UserState`, recent-view filtering, recommendation cache invalidation, and optional Redis-backed state with in-memory fallback.
- Added structured request logging, in-process metrics collection, local load benchmarking, and Docker Compose deployment files; the current local benchmark reached `1.74 QPS` at `P95=3245 ms`, highlighting prerank/rank inference as the main bottlenecks.
- Produced deployment, system-design, interview, and project-report documentation so the repository functions as a full ML systems portfolio project rather than only a training codebase.

## 30-Second Project Intro

### 中文

这是一个基于 KuaiRand-Pure 的端到端短视频推荐系统项目。我没有只做单模型，而是完整实现了多阶段推荐链路：多路召回、LightGBM 粗排、DIN 多任务精排、规则重排，以及 FastAPI 在线服务、反馈闭环、FAISS ANN、离线 A/B simulation 和冷启动分析。项目里既有算法模块，也有 serving、监控、实验和 trade-off 分析，比较适合推荐算法和 ML Systems 面试。

### English

This is an end-to-end short-video recommendation system built on KuaiRand-Pure. Instead of stopping at a single model, I implemented a full multi-stage stack: multi-channel recall, LightGBM preranking, DIN-style multi-task ranking, rule-based reranking, plus FAISS ANN retrieval, FastAPI serving, realtime feedback simulation, offline A/B replay, and cold-start analysis. The project is useful for both recommendation algorithm and ML systems interviews because it covers modeling, serving, evaluation, and failure analysis.

## 2-Minute Project Intro

### 中文

这个项目的目标不是复现一个单点模型，而是把公开数据集做成一个更接近工业推荐系统的完整项目。数据集是 KuaiRand-Pure，我先做了严格的时间切分和标签构造，把日志转成 `is_positive`、`long_watch`、`finish`、`like` 等多目标监督信号，并保证排序阶段只使用 point-in-time 历史，尽量避免时间泄漏。

在离线 pipeline 里，我实现了 `popular / itemcf / twotower / graph_emb` 多路召回，并把 Two-Tower item embedding 接到了 FAISS ANN，支持 FlatIP、HNSW 和 IVF。召回之后用 LightGBM 做粗排，把候选从 500 压到 100，再用 DIN-style 多任务排序模型同时预测 long_watch、finish 和 like，最后再做 rule-based rerank，补作者频控、内容多样性和 freshness。

除了离线效果，我还把这套产物封装成 FastAPI 在线服务，支持 `/recommend`、`/feedback`、`/metrics`、`/metrics/prometheus`，并加了 UserState、recent viewed filtering、recommendation cache、Redis optional fallback 和 degraded mode。然后我又补了 offline A/B simulation 和 cold-start analysis。比较有意思的是，full pipeline 在 offline replay 里 primary metric 没显著赢过 popular，但 recall@50、ndcg@50 和 coverage 提升明显；冷启动增强对 new_user 帮助很大，但 coverage 下降，而且 item cold-start 仍然偏弱。这些结果让我能在面试里不仅讲“做了什么”，还可以讲“哪里有效、哪里没那么有效，以及为什么”。

### English

The goal of this project was not to reproduce a single recommendation model, but to turn a public dataset into a more production-style recommendation system. I used KuaiRand-Pure and first built a strict time-based preprocessing pipeline with multi-target labels such as `is_positive`, `long_watch`, `finish`, and `like`, while keeping ranking features point-in-time to reduce leakage risk.

On the offline side, I implemented multi-channel recall with `popular`, `itemcf`, `twotower`, and `graph_emb`, and then integrated FAISS ANN retrieval on top of Two-Tower embeddings with FlatIP, HNSW, and IVF options. After recall, I used LightGBM for preranking to compress candidates, then a DIN-style multi-task ranker to predict long watch, finish, and like, and finally a rule-based rerank layer for diversity, author frequency control, and freshness.

I also wrapped the offline artifacts into a FastAPI serving layer with `/recommend`, `/feedback`, `/metrics`, and `/metrics/prometheus`, plus realtime `UserState`, recent-view filtering, recommendation cache invalidation, optional Redis fallback, and degraded mode when assets are missing. On top of that, I added offline A/B replay and cold-start analysis. One important outcome is that the full pipeline did not significantly beat the popular baseline on the offline replay primary metric, but it did improve deeper recall, NDCG, and coverage. The cold-start heuristics helped new users a lot, but reduced coverage and still left item cold-start weak. That gives me a much stronger and more honest interview story about what worked, what did not, and why.
