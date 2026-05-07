export default function FeedbackPanel({ feedback, message, error, refreshed, removedVideoId }) {
  return (
    <section className="panel">
      <div className="panel__header">
        <h3>Feedback</h3>
      </div>

      {message ? <p className="panel-message">{message}</p> : null}
      {refreshed ? <p className="panel-success">Feedback submitted, recommendations refreshed.</p> : null}
      {removedVideoId ? (
        <p className="panel-success">
          recent-viewed filtering applied: video <strong>{removedVideoId}</strong> no longer appears in refreshed TopK.
        </p>
      ) : null}
      {error ? <p className="panel-error">{error}</p> : null}

      {!feedback ? (
        <p className="panel-empty">Submit Like / Long View / Skip on any recommendation card.</p>
      ) : (
        <div className="panel-stack">
          <div className="panel-metric">
            <span>status</span>
            <strong>{feedback.status}</strong>
          </div>
          <div className="panel-metric">
            <span>user_id</span>
            <strong>{feedback.user_id}</strong>
          </div>
          <div className="panel-metric">
            <span>video_id</span>
            <strong>{feedback.video_id}</strong>
          </div>
          <div className="panel-metric">
            <span>history_len</span>
            <strong>{feedback.history_len}</strong>
          </div>
          <div className="panel-metric">
            <span>recent_viewed_count</span>
            <strong>{feedback.recent_viewed_count}</strong>
          </div>
          <div className="panel-metric">
            <span>like_count</span>
            <strong>{feedback.like_count}</strong>
          </div>
          <div className="panel-metric">
            <span>long_view_count</span>
            <strong>{feedback.long_view_count}</strong>
          </div>
          <div className="panel-metric">
            <span>cache_invalidated</span>
            <strong>{String(Boolean(feedback.cache_invalidated))}</strong>
          </div>
        </div>
      )}
    </section>
  );
}
