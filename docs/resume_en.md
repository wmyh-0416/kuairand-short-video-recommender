# English Resume Version

## Project Title

Four-Stage Short-Video Recommendation System on KuaiRand-Pure

## Resume Bullets

- Built an end-to-end **four-stage recommendation pipeline** for short-video feed recommendation on KuaiRand-Pure, covering preprocessing, multi-recall, pre-ranking, ranking, and reranking with time-based splits and persisted intermediate candidates.
- Implemented a multi-recall layer with `popular`, `itemcf`, `two-tower`, and `graph embedding` recall; trained the graph embedding branch on a train-only positive item graph and achieved `Recall@100 = 0.2476 / 0.2343` on `val / test`.
- Constructed a LightGBM pre-ranking dataset directly from recall candidates rather than full user-item pairs, using lightweight user/item/source features to compress candidates from `500` to `100` per user while reaching `val AUC = 0.8149`.
- Designed and trained a **DIN-style multi-task ranker** with target-aware attention over user watch history to jointly predict `long_watch`, `finish`, and `like`; the final ranker reached `finish_auc = 0.6663`, `like_auc = 0.6219`, and `NDCG@20 = 0.0359` on test.
- Added a rule-based reranking layer with author frequency control, tag diversity, and freshness adjustment; improved `avg_unique_tags_per_user` by `+0.302` and reduced adjacent same-tag rate by `-0.0171` on test while slightly improving `NDCG@20` and `Recall@20`.
