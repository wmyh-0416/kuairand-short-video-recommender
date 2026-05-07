import { useEffect, useMemo, useState } from "react";

import { API_BASE_URL, getHealth, getMetrics, getRecommendations, submitFeedback } from "./api";
import FeedbackPanel from "./components/FeedbackPanel.jsx";
import HealthPanel from "./components/HealthPanel.jsx";
import MetricsPanel from "./components/MetricsPanel.jsx";
import RecommendationCard from "./components/RecommendationCard.jsx";

function currentHourString() {
  return String(new Date().getHours()).padStart(2, "0");
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function makeRequestId() {
  return `frontend-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [userId, setUserId] = useState("0");
  const [topK, setTopK] = useState(5);
  const [device, setDevice] = useState("web");
  const [hour, setHour] = useState(currentHourString());

  const [recommendation, setRecommendation] = useState(null);
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [feedbackResponse, setFeedbackResponse] = useState(null);

  const [recommendLoading, setRecommendLoading] = useState(false);
  const [healthLoading, setHealthLoading] = useState(false);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  const [recommendError, setRecommendError] = useState("");
  const [healthError, setHealthError] = useState("");
  const [metricsError, setMetricsError] = useState("");
  const [feedbackError, setFeedbackError] = useState("");

  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [feedbackRefreshed, setFeedbackRefreshed] = useState(false);
  const [removedVideoId, setRemovedVideoId] = useState("");
  const [lastFeedbackVideoId, setLastFeedbackVideoId] = useState("");

  const recommendationItems = recommendation?.items || [];
  const latency = recommendation?.latency_ms || null;

  const requestPayload = useMemo(
    () => ({
      user_id: userId,
      top_k: Math.max(1, Number(topK) || 1),
      context: {
        device,
        hour,
      },
    }),
    [device, hour, topK, userId],
  );

  async function refreshHealth() {
    setHealthLoading(true);
    setHealthError("");
    try {
      const payload = await getHealth();
      setHealth(payload);
    } catch (error) {
      setHealthError(error.message);
    } finally {
      setHealthLoading(false);
    }
  }

  async function refreshMetrics() {
    setMetricsLoading(true);
    setMetricsError("");
    try {
      const payload = await getMetrics();
      setMetrics(payload);
    } catch (error) {
      setMetricsError(error.message);
    } finally {
      setMetricsLoading(false);
    }
  }

  async function loadRecommendations(overrides = {}) {
    setRecommendLoading(true);
    setRecommendError("");
    try {
      const payload = {
        ...requestPayload,
        request_id: makeRequestId(),
        ...overrides,
        context: {
          ...requestPayload.context,
          ...(overrides.context || {}),
        },
      };
      const response = await getRecommendations(payload);
      setRecommendation(response);
      return response;
    } catch (error) {
      setRecommendError(error.message);
      throw error;
    } finally {
      setRecommendLoading(false);
    }
  }

  async function handleGetRecommendations() {
    setFeedbackRefreshed(false);
    setRemovedVideoId("");
    await loadRecommendations();
  }

  async function handleFeedback(item, feedbackPreset) {
    setFeedbackLoading(true);
    setFeedbackError("");
    setFeedbackMessage(`${feedbackPreset.actionLabel} submitted for video ${item.video_id}.`);
    setFeedbackRefreshed(false);
    setRemovedVideoId("");
    setLastFeedbackVideoId(String(item.video_id));

    const previousItems = recommendationItems.map((entry) => String(entry.video_id));
    try {
      const feedbackPayload = {
        user_id: userId,
        video_id: String(item.video_id),
        watch_time: feedbackPreset.watch_time,
        duration: feedbackPreset.duration,
        click: feedbackPreset.click,
        like: feedbackPreset.like,
      };
      const feedback = await submitFeedback(feedbackPayload);
      setFeedbackResponse(feedback);
      const refreshed = await loadRecommendations();
      setFeedbackRefreshed(true);

      const refreshedItems = (refreshed?.items || []).map((entry) => String(entry.video_id));
      const wasVisible = previousItems.includes(String(item.video_id));
      const isRemoved = wasVisible && !refreshedItems.includes(String(item.video_id));
      setRemovedVideoId(isRemoved ? String(item.video_id) : "");
      await refreshMetrics();
    } catch (error) {
      setFeedbackError(error.message);
    } finally {
      setFeedbackLoading(false);
    }
  }

  useEffect(() => {
    refreshHealth();
    refreshMetrics();
  }, []);

  return (
    <div className="page-shell">
      <div className="page-backdrop" />
      <main className="page-content">
        <header className="hero">
          <p className="hero-kicker">Interactive Frontend Demo</p>
          <h1>KuaiRand Short Video Recommender Demo</h1>
          <p className="hero-subtitle">
            FAISS Retrieval + LightGBM Prerank + DIN Ranker + Real-time Feedback
          </p>
          <div className="hero-meta">
            <span>API base URL: {API_BASE_URL}</span>
            <span>Frontend: React + Vite</span>
          </div>
        </header>

        <section className="control-panel panel">
          <div className="panel__header">
            <h2>User Control Panel</h2>
            <div className="panel-actions">
              <button type="button" className="secondary-button" onClick={refreshHealth} disabled={healthLoading}>
                Refresh Health
              </button>
              <button type="button" className="secondary-button" onClick={refreshMetrics} disabled={metricsLoading}>
                Refresh Metrics
              </button>
            </div>
          </div>

          <div className="control-grid">
            <label className="field">
              <span>user_id</span>
              <input value={userId} onChange={(event) => setUserId(event.target.value)} />
            </label>

            <label className="field">
              <span>top_k</span>
              <input
                type="number"
                min="1"
                max="20"
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value) || 1)}
              />
            </label>

            <label className="field">
              <span>context.device</span>
              <select value={device} onChange={(event) => setDevice(event.target.value)}>
                <option value="ios">ios</option>
                <option value="android">android</option>
                <option value="web">web</option>
              </select>
            </label>

            <label className="field">
              <span>context.hour</span>
              <input value={hour} onChange={(event) => setHour(event.target.value)} />
            </label>
          </div>

          <div className="panel-actions panel-actions--main">
            <button type="button" className="primary-button" onClick={handleGetRecommendations} disabled={recommendLoading}>
              {recommendLoading ? "Loading..." : "Get Recommendations"}
            </button>
          </div>
          {recommendError ? <p className="panel-error">{recommendError}</p> : null}
        </section>

        <div className="dashboard-grid">
          <div className="left-rail">
            <HealthPanel health={health} loading={healthLoading} error={healthError} />
            <MetricsPanel metrics={metrics} loading={metricsLoading} error={metricsError} />
            <FeedbackPanel
              feedback={feedbackResponse}
              message={feedbackMessage}
              error={feedbackError}
              refreshed={feedbackRefreshed}
              removedVideoId={removedVideoId}
            />
            <section className="panel">
              <div className="panel__header">
                <h3>Latency Panel</h3>
              </div>
              {!latency ? (
                <p className="panel-empty">Fetch recommendations to see latency breakdown.</p>
              ) : (
                <div className="panel-stack">
                  <div className="panel-metric"><span>total</span><strong>{formatNumber(latency.total)}</strong></div>
                  <div className="panel-metric"><span>recall</span><strong>{formatNumber(latency.recall)}</strong></div>
                  <div className="panel-metric"><span>prerank</span><strong>{formatNumber(latency.prerank)}</strong></div>
                  <div className="panel-metric"><span>rank</span><strong>{formatNumber(latency.rank)}</strong></div>
                  <div className="panel-metric"><span>rerank</span><strong>{formatNumber(latency.rerank)}</strong></div>
                </div>
              )}
            </section>
          </div>

          <section className="recommendation-section">
            <div className="section-header">
              <div>
                <p className="section-kicker">Recommendation List</p>
                <h2>TopK Results</h2>
              </div>
              {recommendation ? (
                <div className="section-meta">
                  <span>request_id: {recommendation.request_id}</span>
                  <span>degraded_mode: {String(Boolean(recommendation.degraded_mode))}</span>
                </div>
              ) : null}
            </div>

            {recommendationItems.length === 0 ? (
              <div className="empty-state">
                <p>No recommendations loaded yet.</p>
                <p>Enter a user ID and click <strong>Get Recommendations</strong>.</p>
              </div>
            ) : (
              <div className="recommendation-list">
                {recommendationItems.map((item, index) => (
                  <RecommendationCard
                    key={`${item.video_id}-${index}`}
                    item={item}
                    index={index}
                    onFeedback={handleFeedback}
                    feedbackLoading={feedbackLoading}
                    lastFeedbackVideoId={lastFeedbackVideoId}
                  />
                ))}
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
