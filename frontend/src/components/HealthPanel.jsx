function Flag({ value }) {
  return (
    <span className={`status-badge ${value ? "status-badge--ok" : "status-badge--warn"}`}>
      {String(Boolean(value))}
    </span>
  );
}

export default function HealthPanel({ health, loading, error }) {
  return (
    <section className="panel">
      <div className="panel__header">
        <h3>Health</h3>
        {loading ? <span className="status-badge status-badge--pending">Loading</span> : null}
      </div>
      {error ? <p className="panel-error">{error}</p> : null}
      {!health ? (
        <p className="panel-empty">No health snapshot loaded yet.</p>
      ) : (
        <div className="panel-stack">
          <div className="panel-metric">
            <span>status</span>
            <strong>{health.status}</strong>
          </div>
          <div className="panel-metric">
            <span>models_loaded</span>
            <Flag value={health.models_loaded} />
          </div>
          <div className="panel-metric">
            <span>faiss_loaded</span>
            <Flag value={health.faiss_loaded} />
          </div>
          <div className="panel-metric">
            <span>redis_connected</span>
            <Flag value={health.redis_connected} />
          </div>
          <div className="panel-metric">
            <span>degraded_mode</span>
            <Flag value={health.degraded_mode} />
          </div>
          <div className="panel-subgrid">
            {Object.entries(health.loaded_components || {}).map(([key, value]) => (
              <div key={key} className="component-chip">
                <span>{key}</span>
                <Flag value={value} />
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
