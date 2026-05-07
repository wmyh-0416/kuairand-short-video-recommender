const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail =
      typeof payload === "string"
        ? payload
        : payload?.detail || JSON.stringify(payload);
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

export function getHealth() {
  return request("/health", { method: "GET" });
}

export function getMetrics() {
  return request("/metrics", { method: "GET" });
}

export function getPrometheusMetrics() {
  return request("/metrics/prometheus", {
    method: "GET",
    headers: {
      Accept: "text/plain",
    },
  });
}

export function getRecommendations(payload) {
  return request("/recommend", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function submitFeedback(payload) {
  return request("/feedback", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export { API_BASE_URL };
