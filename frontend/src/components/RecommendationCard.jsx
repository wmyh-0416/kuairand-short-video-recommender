function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

export default function RecommendationCard({
  item,
  index,
  onFeedback,
  feedbackLoading,
  lastFeedbackVideoId,
}) {
  const isUpdating = feedbackLoading && String(lastFeedbackVideoId) === String(item.video_id);

  return (
    <article className="recommendation-card">
      <div className="recommendation-card__topline">
        <span className="card-rank">#{index + 1}</span>
        <span className="card-video-id">video_id: {item.video_id}</span>
        <span className="card-score">final: {formatNumber(item.score)}</span>
      </div>

      <div className="recommendation-grid">
        <div>
          <span className="meta-label">recall_source</span>
          <span className="meta-value">{item.recall_source || "n/a"}</span>
        </div>
        <div>
          <span className="meta-label">recall_score</span>
          <span className="meta-value">{formatNumber(item.recall_score)}</span>
        </div>
        <div>
          <span className="meta-label">prerank_score</span>
          <span className="meta-value">{formatNumber(item.prerank_score)}</span>
        </div>
        <div>
          <span className="meta-label">rank_score</span>
          <span className="meta-value">{formatNumber(item.rank_score)}</span>
        </div>
      </div>

      <div className="card-reason">
        <span className="meta-label">reason</span>
        <span className="meta-value card-reason__text">{item.reason || "n/a"}</span>
      </div>

      <div className="card-actions">
        <button
          type="button"
          className="action-button action-button--positive"
          disabled={feedbackLoading}
          onClick={() =>
            onFeedback(item, {
              watch_time: 18,
              duration: 20,
              click: 1,
              like: 1,
              actionLabel: "Like",
            })
          }
        >
          {isUpdating ? "Submitting..." : "Like"}
        </button>

        <button
          type="button"
          className="action-button action-button--neutral"
          disabled={feedbackLoading}
          onClick={() =>
            onFeedback(item, {
              watch_time: 16,
              duration: 20,
              click: 1,
              like: 0,
              actionLabel: "Long View",
            })
          }
        >
          {isUpdating ? "Submitting..." : "Long View"}
        </button>

        <button
          type="button"
          className="action-button action-button--negative"
          disabled={feedbackLoading}
          onClick={() =>
            onFeedback(item, {
              watch_time: 1,
              duration: 20,
              click: 0,
              like: 0,
              actionLabel: "Skip",
            })
          }
        >
          {isUpdating ? "Submitting..." : "Skip"}
        </button>
      </div>
    </article>
  );
}
