# Interview Q&A

## Data

### 1. 为什么不能随机切分？

推荐系统日志有明显时间顺序，随机切分会把未来行为泄漏到训练里，导致指标虚高。这个项目用的是严格时间切分，训练、验证、测试分别对应不同日期区间。

### 2. 怎么构造正负样本？

召回和粗排阶段把 `is_positive` 作为主正样本定义，负样本来自曝光但没有形成有效正反馈的 user-item 对。粗排和精排还做了用户级负采样，控制样本规模和正负比例。

### 3. `long_view` label 怎么定义？

这个项目优先复用 KuaiRand-Pure 里的 `long_view` / `long_watch` 字段；如果某个分析模块缺这个字段，才会回退到 `watch_time / duration >= 0.7` 的规则近似。

### 4. 如何避免数据泄漏？

核心做法有两层：一是时间切分，二是 point-in-time 历史特征。比如 ranking 的用户序列只用过去行为，不允许测试期交互反向污染训练特征。

### 5. 点播 / 曝光日志有什么偏差？

它有明显的 exposure bias，因为用户只能反馈系统已经曝光过的内容。热门内容更容易被看到，所以 offline 指标不等于真实偏好，也不等于线上因果效果。

## Recall

### 6. ItemCF 原理是什么？

ItemCF 根据用户历史行为构建 item-item 共现关系，相似 item 会互相推荐。这个项目里它主要提供强解释性和稳定性，尤其适合 warm user 的局部兴趣扩展。

### 7. Two-Tower 怎么训练？

用户塔输入 `user_id + recent watch sequence`，物品塔输入 `video_id + author_id + tag`。训练目标是把匹配的 user/item 表示拉近，不匹配的拉远，最后输出可做向量检索的 embedding。

### 8. 负采样怎么做？

离线训练阶段使用用户级负采样，从召回候选或曝光样本里保留一定比例负样本。这样能控制数据规模，也能避免极端类别不平衡。

### 9. FAISS 是什么？

FAISS 是一个向量检索库，可以把 embedding 建成索引并高效做 TopK 搜索。这个项目用它把 Two-Tower 从“离线算分”升级成更像真实召回系统的 ANN retrieval。

### 10. ANN 是什么？

ANN 是 approximate nearest neighbor，也就是近似近邻搜索。它比精确全量扫描更快，但可能丢失一部分精确邻居，所以需要用 overlap@K 或 recall@K 评估近似误差。

### 11. `IndexFlatIP / HNSW / IVF` 区别是什么？

`IndexFlatIP` 是精确内积搜索，正确但全量扫描。`HNSW` 是图索引，延迟和效果通常平衡得比较好。`IVF` 先聚类再搜局部桶，适合更大规模，但要调训练和 probe 参数。

### 12. overlap@K 怎么解释？

它表示 ANN TopK 和精确 Flat TopK 的重合率。这个项目里 mean overlap@500 大约是 `0.835`，说明 HNSW 并不完全等价于精确检索，但保留了大部分高质量候选。

### 13. 为什么召回层不用精排模型？

因为召回面对的是全量或近全量候选，目标是快速缩小候选集，而不是做最精细的排序。用精排模型做召回在计算上太重，也不符合多阶段架构的职责分离。

## Ranking

### 14. 粗排和精排的区别？

粗排目标是低成本过滤明显低质量候选，通常更看重速度和大规模吞吐。精排目标是对较小候选集做更细致的相关性建模，所以可以用更复杂的模型。

### 15. 为什么 LightGBM 适合粗排？

LightGBM 对表格特征友好，训练和推理都比较稳定，而且可解释性比深模型更强。作为候选过滤层，它通常是性价比很高的第一选择。

### 16. DIN 的核心思想是什么？

DIN 的关键是把当前候选 item 和用户历史序列做目标感知的注意力匹配，而不是简单平均历史 embedding。这样能更直接建模“当前视频和最近兴趣是否匹配”。

### 17. 多任务 ranker 怎么设计？

这个项目同时预测 `long_watch`、`finish`、`like` 三个任务，再按设定权重融合成 `rank_score`。这样做是因为短视频里用户价值不止点击，一个任务无法完整覆盖质量目标。

### 18. 为什么要 rerank？

因为单纯 relevance 排序容易让 feed 变得重复，比如作者重复、tag 重复、内容不新鲜。rerank 用很轻的规则层在最后一步做多样性和展示体验修正。

### 19. 多样性和准确率如何 trade-off？

这是典型的业务约束问题。这个项目里 rerank 对 NDCG 的提升很小，但 `avg_unique_tags_per_user` 提升明显，说明它是在相对小的准确率代价下换取更好的 slate 多样性。

## Online Serving

### 20. FastAPI 在项目里做什么？

它把离线训练产物包装成在线推荐 API，让项目不只停留在 parquet 和 notebook。接口包括健康检查、推荐、反馈和监控指标。

### 21. `/recommend` 流程是什么？

大致是：读用户状态，拿 user embedding，走 FAISS recall，做 prerank、rank、rerank，然后返回 TopK 和分阶段延迟。某个模型缺失时会走 degraded mode。

### 22. `/feedback` 如何更新用户状态？

`/feedback` 会把 `view / like / long view / recent tag` 等信息写进 `UserState`，同时记录日志并清掉该用户的推荐缓存。这样下一次 `/recommend` 会基于更新后的状态重新生成结果。

### 23. Redis 缓存什么？

Redis 主要缓存 `UserState` 和 recommendation cache。推荐缓存按 `user_id + context_hash` 存，用户状态按 namespace 和 user_id 存。

### 24. cache invalidation 怎么做？

每次 `/feedback` 后，服务会删除该用户相关的推荐缓存键。这样可以保证刚看过或刚反馈过的视频不会因为旧缓存而继续被推荐。

### 25. degraded mode 怎么设计？

degraded mode 的原则是“组件缺失时尽量降级，不直接 500”。比如缺 rank 模型就沿用 prerank 顺序，缺 Redis 就回退到内存状态。

### 26. P95 latency 高怎么优化？

当前本地 benchmark 显示主要瓶颈是 prerank 和 rank 阶段。优化方向包括：特征预计算、批量化推理、模型轻量化、缓存更多中间结果，或者把 rank 改成更高效的 serving 形式。

## Experimentation

### 27. `Recall@K / NDCG@K / MAP` 区别？

`Recall@K` 看正样本召回了多少，`NDCG@K` 看正样本排得有多前，`MAP` 更强调命中顺序的整体平均精度。这个项目主要用了 Recall 和 NDCG，因为它们更直观地对应召回层和排序层表现。

### 28. AUC 和排序指标区别？

AUC 主要反映正负样本整体区分能力，但不一定能直接说明 TopK 排序质量。推荐系统更关心前几位的质量，所以通常会同时看 AUC 和 NDCG / Recall@K。

### 29. Offline A/B simulation 为什么不等于线上 A/B？

因为它没有真实随机流量分配，也没有真实曝光后的反事实反馈。它只是基于日志回放对策略做离线比较，不能当成线上 lift 证明。

### 30. bootstrap CI 怎么做？

这个项目对 primary metric 做了用户级 bootstrap，有放回地重采样用户并重复计算 treatment-control 差值。这样能给出一个比单点 lift 更稳的置信区间。

### 31. 为什么 `full_pipeline` primary metric 没显著赢 `popular`？

有几个原因：一是 offline replay 本身受日志曝光偏差影响，二是 `popular` 在短视频场景里常常是很强的 baseline，三是 full pipeline 更偏个性化和覆盖，未必直接优化当前定义的 primary proxy。

### 32. 如何进一步提升 `long_view_rate`？

可以尝试重调多任务权重、加入更强的序列模型、引入上下文特征、做更合理的负采样和 hard negative，以及在 rerank 层单独约束 short watch 风险。

## Cold-start

### 33. new user 怎么处理？

new user 没有稳定历史时，个性化模型很难工作，所以需要 popular 或 category-popular 这类 fallback。这个项目的 heuristic 对 new user 提升很明显，但也牺牲了部分覆盖率。

### 34. new item 怎么处理？

当前项目对 new item 主要只做了轻量 freshness / low-exposure boost，还没有真正的内容召回或多模态表示。结果也说明 item cold-start 仍然弱。

### 35. 为什么冷用户提升但 coverage 降低？

因为补充策略更多依赖热门或类目热门内容，会把流量集中到更稳的候选上。这样 short-term hit 和 long_view 容易提升，但 catalog spread 会下降。

### 36. item cold-start 为什么弱？

因为现在没有专门的新物品召回模块，低曝光物品只是被 rerank 或 heuristic 轻微上推。没有内容 embedding 或 metadata similarity，系统很难真正给冷 item 足够曝光。

### 37. 如何用内容特征改善 new item？

可以基于 `tag / author / metadata / multimodal embedding` 做 content-based recall，把新物品挂到相似成熟内容附近。这样即使没有足够交互数据，新 item 也有机会进入候选集。

## Engineering

### 38. Docker Compose 做什么？

它把 `recommender-api` 和 `redis` 组合成一个最小可部署栈，方便本地或演示环境一键启动。虽然当前机器没有 Docker，所以没做完全联调，但配置和文档已经补齐了。

### 39. Prometheus metrics 有什么用？

Prometheus-compatible endpoint 可以让服务指标更容易被标准监控系统抓取。即使现在还是轻量实现，它也让项目更接近真实服务的可观测性要求。

### 40. 如何设计 feature store？

理想做法是把用户、物品、上下文和统计特征拆成独立在线存储，并支持 point-in-time 读取和版本管理。这个项目目前还是本地 parquet 复用，所以 feature store 仍然是后续升级方向。

### 41. 如何做模型版本管理？

至少要对 checkpoint、特征 schema、配置和评估报告做版本化绑定。这个项目目前通过 `artifacts/`、YAML 配置和产物命名做了基础管理，但还没有完整的 model registry。

### 42. 如何上线真实 A/B？

需要在线流量分桶、曝光日志、埋点回流、实时或准实时指标看板，以及实验开关和回滚机制。离线 replay 只能做预演，真正上线需要系统级实验平台支持。

### 43. 如何做增量训练？

可以把 feedback 日志汇总进每日或小时级训练样本，更新用户状态、热门统计、召回索引和排序模型。更完整的版本还需要特征一致性、模型版本切换和回滚策略。
