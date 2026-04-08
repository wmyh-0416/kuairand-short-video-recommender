# KuaiRand 四层短视频推荐系统

一个基于 **KuaiRand-Pure** 数据集实现的、面向短视频 Feed 场景的四层推荐系统项目。项目目标不是做单模型 baseline，而是用个人可落地的工程复杂度，复现更贴近工业推荐系统的 **预处理 -> 多路召回 -> 粗排 -> 精排 -> 重排** 全链路。

## 项目背景

短视频推荐和传统 MovieLens 风格推荐有两个核心差异：

- 候选空间更大，必须依赖多路召回和层层过滤，而不是直接在全量 item 上做排序。
- 用户反馈更丰富，除了点击，还包括 **观看时长、完播、点赞** 等多种目标，排序阶段更适合做多任务学习。

因此，这个项目没有做“召回 + 排序”两段式 toy demo，而是实现了更接近真实 Feed 推荐系统的四层架构：

- 预处理层：标签构造、时间切分、用户序列、公共特征表
- 多路召回层：popular / itemcf / twotower / graph_emb
- 粗排层：LightGBM 过滤低质量候选
- 精排层：DIN-style 多任务精排
- 重排层：规则式 rerank，补充作者频控、内容多样性与新鲜度策略

项目强调的是 **工业推荐分层思维**，而不是“把一个深度模型硬套到公开数据集上”。

## 数据集

项目使用 [KuaiRand-Pure](https://zenodo.org/records/10439422)。

选择这个数据集的原因：

- 它来自短视频场景，天然适合 Feed 推荐建模。
- 日志里包含 `play_time_ms`、`duration_ms`、`is_like`、`long_view` 等字段，能够支持短视频里常见的多目标标签设计。
- item 侧包含 `author_id`、`tag`、上传时间、统计特征，便于做作者维度、内容类别维度和新鲜度建模。

本项目采用 **时间切分**，避免随机切分带来的信息泄漏：

- `train`: `date <= 20220424`
- `val`: `20220425 ~ 20220430`
- `test`: `date >= 20220501`

标签定义：

- `like = is_like`
- `finish = play_time_ms / duration_ms >= 0.95`
- `long_watch = long_view`
- `is_positive = long_watch OR finish OR like`

这里的 `is_positive` 是短视频隐式反馈场景下的“有效正反馈”定义，不等价于“明确喜欢”，但适合作为召回和粗排的监督信号。

## 整体架构

```text
raw KuaiRand-Pure csv
  -> preprocess
  -> multi-recall
  -> prerank
  -> DIN-style multi-task rank
  -> rule-based rerank
  -> final feed list
```

### 1. 预处理

预处理阶段统一完成：

- 原始日志读取与字段标准化
- `watch_ratio / finish / long_watch / like / is_positive` 标签构造
- train / val / test 时间切分
- 用户基础特征、item 基础特征、item 统计特征落盘
- 用户行为序列构造

核心产物：

- `processed/interactions.parquet`
- `processed/user_features.parquet`
- `processed/item_features.parquet`
- `processed/user_sequences.parquet`
- `processed/splits/{train,val,test}.parquet`

### 2. 多路召回

召回层最终实现了 4 条路径：

- `popular`
  - 基于 train split 的播放、长观看、完播、点赞统计
- `itemcf`
  - 基于 train split 正反馈序列构建 item-item 共现相似度
- `twotower`
  - 用户塔使用 `user_id + recent watch sequence`
  - item 塔使用 `video_id + author_id + tag`
  - PyTorch 训练，支持 GPU
- `graph_emb`
  - 基于 train split 正反馈构建 item-item 图
  - 用随机游走 + skip-gram 学习 item embedding
  - 作为补充召回路接入 merge

召回结果通过统一 merge 合并：

- 按 source quota 做 per-source 控制
- 计算 `merged_score`
- 保留各路 `source_score / source_rank`
- 最终输出 `train/val/test_candidates.parquet`

### 3. 粗排

粗排层基于召回候选训练，不做全量 user-item 训练。

第一版主模型选用 **LightGBM**：

- 训练样本来自 `artifacts/recall/*_candidates.parquet`
- 正样本：候选里 `label = 1`
- 负样本：候选里 `label = 0`，并按用户负采样
- 特征为轻量工程特征：
  - `user_id / video_id / author_id / tag`
  - user/item 基础与统计特征
  - recall source、merged score、source count
  - 各路 source score / rank
  - freshness_days

粗排目标是把每用户候选从 500 压到 100，同时尽量保留正样本。

### 4. 精排

精排层主模型不是纯 MLP，而是 **DIN-style 多任务精排模型**。

设计思路：

- 用 `train split` 构建 leakage-safe 用户历史序列
- 对候选 item 与用户历史 item 做 target-aware attention
- 得到用户兴趣向量后，与静态特征拼接进入 shared tower
- 输出 3 个任务：
  - `long_watch`
  - `finish`
  - `like`

最终 `rank_score` 由三个任务分数加权融合：

- `0.45 * p(long_watch)`
- `0.25 * p(finish)`
- `0.30 * p(like)`

为什么选 DIN：

- 相比纯 MLP，更适合建模“当前候选视频”和“用户最近看过什么”之间的匹配关系
- 相比 Transformer，工程复杂度更可控，更适合个人项目第一版落地

### 5. 重排

重排层使用规则式 greedy rerank，不额外训练学习型 slate 模型。

当前实现的策略包括：

- 作者频控
  - 连续作者惩罚
  - topN 内作者最大曝光数约束
- tag 多样性
  - 连续同 tag 惩罚
  - 重复 tag 惩罚
  - 新 tag bonus
- 新鲜度调节
  - 基于 `freshness_days` 做指数衰减 bonus

设计原则是：

- 不推翻 DIN 的相关性排序
- 只在最后一层做轻量业务约束补充
- 兼顾排序质量、多样性和 feed 展示体验

## 结果总结

以下结果基于当前正式跑通的本地实验产物。

### 召回层

最终采用 `popular + itemcf + twotower + graph_emb`，其中 `graph_emb` 经过权重和 quota 调整后作为补充召回路。

| split | Recall@50 | Recall@100 | Recall@200 | Coverage |
|---|---:|---:|---:|---:|
| val | 0.1832 | 0.2476 | 0.3151 | 0.9995 |
| test | 0.1731 | 0.2343 | 0.2973 | 0.9996 |

### 粗排层

LightGBM 粗排结果：

- `val_auc = 0.8149`
- `train_auc = 0.8648`
- 候选压缩率：`500 -> 100`，整体保留 `20%` 候选
- `val recall_retained@100 = 0.6600`
- `test recall_retained@100 = 0.6315`

如果按与召回层同口径比较粗排 top100 的绝对 Recall：

- `val`: 从 `0.2476` 提升到 `0.3523`
- `test`: 从 `0.2343` 提升到 `0.3317`

这说明粗排在缩小候选集的同时，确实把更可能命中的 item 提前了。

### 精排层

DIN-style 多任务精排的关键指标：

| split | rank_auc | long_watch_auc | finish_auc | like_auc | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| val | 0.5753 | 0.5746 | 0.6596 | 0.6201 | 0.0295 |
| test | 0.5845 | 0.5838 | 0.6663 | 0.6219 | 0.0359 |

### 重排层

规则式 rerank 在尽量不损失质量的前提下，改善了多样性与 feed 生态：

- `val`
  - `NDCG@20: +0.0000117`
  - `Recall@20: +0.000479`
  - `avg_unique_tags_per_user: +0.309`
  - `adjacent_same_tag_rate: -0.0172`
- `test`
  - `NDCG@20: +0.000182`
  - `Recall@20: +0.001631`
  - `avg_unique_tags_per_user: +0.302`
  - `adjacent_same_tag_rate: -0.0171`

这个结果符合重排层的定位：不追求大幅提升模型指标，而是用较小代价换取更好的内容分布与展示体验。

## 如何运行

推荐使用已经配置好的 A100 环境：

```bash
cd /scratch/ym3447/Rec
source /scratch/ym3447/Rec/.venv-a100/bin/activate
```

按阶段运行：

```bash
python scripts/01_preprocess.py
python scripts/02_train_recall.py
python scripts/03_generate_recall_candidates.py
python scripts/04_train_prerank.py
python scripts/05_generate_prerank_topk.py
python scripts/06_train_rank.py
python scripts/07_run_rerank.py
python scripts/08_evaluate_pipeline.py
```

说明：

- PyTorch 相关训练默认在 `.venv-a100` 下走 `device: auto -> cuda`
- LightGBM 粗排为 CPU 训练
- 重排为规则式 CPU 逻辑

## 项目结构

```text
Rec/
  configs/
  processed/
  artifacts/
  scripts/
  src/
    data/
    recall/
    prerank/
    rank/
    rerank/
    utils/
  docs/
```

## 项目亮点

- 不是单模型 demo，而是完整的 **四层推荐系统 pipeline**
- 明确体现 **多路召回 -> 粗排 -> 精排 -> 重排** 的工业推荐分层思想
- 召回层不止做 popular / itemcf，还实现了 `twotower + graph_emb`
- 粗排层使用 LightGBM 处理海量候选，贴近真实工程实践
- 精排层使用 **DIN-style 多任务模型**，而不是简单 MLP
- 重排层补上了作者频控、tag 多样性、新鲜度调节等 Feed 策略逻辑
- 中间候选、模型、指标均落盘，可复现、可扩展、可用于面试讲解

## 这个项目不是什么

这个项目不是线上大规模推荐系统复现，也没有实现：

- 实时特征服务
- ANN 服务化部署
- 分布式训练与在线推断
- A/B test 与在线指标闭环

但它已经比较完整地体现了：

- 工业推荐系统为什么要分层
- 每层各自解决什么问题
- 为什么短视频推荐要做多目标学习和重排约束

这也是它适合用于推荐算法岗简历和面试讲解的原因。

## 后续可扩展方向

- 用更强的序列模型替换 DIN，例如 Transformer ranker
- 在召回层加入更稳健的 ANN 检索与向量索引
- 增加 debias / IPS / DR 等曝光偏差处理
- 把重排从规则式扩展到 learning-to-rerank
- 引入 context 特征，如时间段、tab、场景上下文
- 做离线 ablation，系统比较 graph_emb、twotower、多任务 loss、rerank 约束的贡献

## 文档

- 结果总结：[docs/results_summary.md](docs/results_summary.md)
- 中文简历版：[docs/resume_cn.md](docs/resume_cn.md)
- 英文简历版：[docs/resume_en.md](docs/resume_en.md)
- 面试讲稿：[docs/interview_notes.md](docs/interview_notes.md)
- 常见追问：[docs/qa_notes.md](docs/qa_notes.md)
