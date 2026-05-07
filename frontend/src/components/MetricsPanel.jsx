function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function MetricRow({ label, value }) {
  return (
    <div className="panel-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function MetricsPanel({ metrics, loading, error }) {
  return (
    <section className="panel">
      <div className="panel__header">
        <h3>Metrics</h3>
        {loading ? <span className="status-badge status-badge--pending">Loading</span> : null}
      </div>
      {error ? <p className="panel-error">{error}</p> : null}
      {!metrics ? (
        <p className="panel-empty">No metrics loaded yet.</p>
      ) : (
        <div className="panel-stack">
          <MetricRow label="request_count" value={metrics.request_count ?? "n/a"} />
          <MetricRow label="recommend_request_count" value={metrics.recommend_request_count ?? "n/a"} />
          <MetricRow label="feedback_count" value={metrics.feedback_count ?? "n/a"} />
          <MetricRow label="average_latency_ms" value={formatMetric(metrics.average_latency_ms)} />
          <MetricRow label="p95_latency_ms" value={formatMetric(metrics.p95_latency_ms)} />
          <MetricRow label="cache_hit_rate" value={formatMetric(metrics.cache_hit_rate, 4)} />
          <MetricRow label="degraded_mode_count" value={metrics.degraded_mode_count ?? "n/a"} />
          <MetricRow label="cache_hit_count" value={metrics.cache_hit_count ?? "n/a"} />
          <MetricRow label="cache_miss_count" value={metrics.cache_miss_count ?? "n/a"} />
          <MetricRow label="user_state_read_count" value={metrics.user_state_read_count ?? "n/a"} />
          <MetricRow label="user_state_write_count" value={metrics.user_state_write_count ?? "n/a"} />
        </div>
      )}
    </section>
  );
}
