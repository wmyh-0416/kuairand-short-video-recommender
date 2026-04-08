# 中文简历版

## 项目名称

基于 KuaiRand-Pure 的四层短视频推荐系统

## 简历描述

- 基于 KuaiRand-Pure 从零实现面向短视频 Feed 场景的四层推荐系统，完整打通 **多路召回、粗排、精排、重排** 全链路，采用时间切分与中间候选落盘，体现工业推荐系统的 cascade 设计。
- 在召回层实现 `popular + itemcf + twotower + graph_emb` 四路召回，其中 Graph Embedding 基于 train split 正反馈构建 item-item 图并学习 embedding，最终将 `val/test Recall@100` 做到 `0.2476 / 0.2343`。
- 基于召回候选构造 LightGBM 粗排训练集，引入 user/item 基础特征、统计特征、source score、freshness 等轻量特征，在将候选从每用户 `500` 压缩到 `100` 的同时，实现 `val AUC=0.8149`，且粗排 top100 绝对 Recall 明显优于召回层原始 top100。
- 实现基于用户历史行为序列的 **DIN-style 多任务精排模型**，联合预测 `long_watch / finish / like`，在 test 集达到 `finish_auc=0.6663`、`like_auc=0.6219`、`NDCG@20=0.0359`，并通过 `device=auto -> cuda` 在 A100 上完成训练与推断。
- 设计规则式重排模块，在 DIN 排序结果上加入作者频控、tag 多样性和 freshness 调节，在 test 集实现 `avg_unique_tags_per_user +0.302`、`adjacent_same_tag_rate -0.0171`，同时 `NDCG@20` 与 `Recall@20` 保持小幅提升。
